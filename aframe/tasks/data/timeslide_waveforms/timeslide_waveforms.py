import os
import shutil
from pathlib import Path
from typing import Dict, Literal

import law
import luigi
from luigi.util import inherits

from aframe.parameters import PathParameter, load_prior
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.fetch import FetchTest
from utils import data as data_utils

TsWorkflowRequires = Dict[Literal["test_segments"], law.Task]


# TODO: add descriptions to all parameters
class TimeSlideWaveformsParams(law.Task):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    num_injections = luigi.IntParameter()
    output_dir = PathParameter()
    shifts = luigi.ListParameter()
    spacing = luigi.FloatParameter()
    buffer = luigi.FloatParameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    coalescence_time = luigi.FloatParameter(
        description="Location of the defining point of the signal "
        "within the generated waveform"
    )
    highpass = luigi.FloatParameter()
    snr_threshold = luigi.FloatParameter()
    psd_length = luigi.FloatParameter()
    seed = luigi.IntParameter()
    # verbose = luigi.BoolParameter(default=False)


@inherits(TimeSlideWaveformsParams)
class DeployTimeslideWaveforms(
    AframeDataTask,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
):
    def workflow_requires(self):
        reqs = {}
        reqs["test_segments"] = FetchTest.req(
            self,
            segments_file=self.output_dir / "segments.txt",
            data_dir=self.output_dir / "background",
        )
        return reqs

    # for now, require that testing segments exist.
    # in principle, we could just require that the segments __file__
    # exist since we only need the start and stop times,
    # but there were some instances where gwpy would not
    # successfully download all of the segments.
    # TODO: check if this is still the case with gwpy.
    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.workflow_input()["test_segments"].collection.exists()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_map, i = {}, 0
        for start, end in self.test_segments:
            for j in range(self.shifts_required):
                shift = [(j + 1) * shift for shift in self.shifts]

                if data_utils.is_analyzeable_segment(
                    start, end, shift, self.psd_length
                ):
                    # add psd_length to account for the
                    # burn in of psd calculation
                    branch_map[i] = (
                        start + self.psd_length,
                        end,
                        shift,
                        self.psd_segment,
                    )
                    i += 1
        return branch_map

    @workflow_condition.output
    def output(self):
        return [
            law.LocalFileTarget(self.tmp_dir / "waveforms.hdf5"),
            law.LocalFileTarget(self.tmp_dir / "rejected-parameters.hdf5"),
        ]

    @property
    def shifts_required(self):
        return data_utils.get_num_shifts_from_num_injections(
            self.test_segments,
            self.num_injections,
            self.waveform_duration,
            self.spacing,
            self.max_shift,
            self.buffer,
        )

    @property
    def test_segments(self):
        paths = list(
            self.workflow_input()["test_segments"].collection.targets.values()
        )
        return data_utils.segments_from_paths(paths)

    @property
    def psd_segment(self):
        return list(
            self.workflow_input()["test_segments"].collection.targets.values()
        )[-1]

    @property
    def max_shift(self):
        return max(self.shifts)

    @property
    def tmp_dir(self):
        return self.output_dir / f"tmp-{self.branch}"

    def run(self):
        import io

        import h5py
        from data.timeslide_waveforms.timeslide_waveforms import (
            timeslide_waveforms,
        )

        prior = load_prior(self.prior)
        start, end, shift, psd_segment = self.branch_data
        with psd_segment.open("r") as psd_file:
            psd_file = h5py.File(io.BytesIO(psd_file.read()))
            timeslide_waveforms(
                start,
                end,
                self.ifos,
                shift,
                self.spacing,
                self.buffer,
                prior,
                self.minimum_frequency,
                self.reference_frequency,
                self.sample_rate,
                self.waveform_duration,
                self.waveform_approximant,
                self.coalescence_time,
                self.highpass,
                self.snr_threshold,
                psd_file,
                Path(self.tmp_dir),
                self.seed,
            )


@inherits(DeployTimeslideWaveforms)
class TimeslideWaveforms(AframeDataTask):
    condor_directory = PathParameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"),
            "timeslide_waveforms",
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_output = self.output_dir / "waveforms.hdf5"
        self.rejected_output = self.output_dir / "rejected-parameters.hdf5"

    def output(self):
        return [
            law.LocalFileTarget(self.waveform_output),
            law.LocalFileTarget(self.rejected_output),
        ]

    def requires(self):
        return DeployTimeslideWaveforms.req(
            self,
            request_memory=self.request_memory,
            request_disk=self.request_disk,
            request_cpus=self.request_cpus,
        )

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def waveform_files(self):
        return list(map(Path, [targets[0].path for targets in self.targets]))

    @property
    def rejected_parameter_files(self):
        return list(map(Path, [targets[1].path for targets in self.targets]))

    def run(self):
        from ledger.injections import (
            InjectionParameterSet,
            InterferometerResponseSet,
            waveform_class_factory,
        )

        cls = waveform_class_factory(
            self.ifos,
            InterferometerResponseSet,
            "ResponseSet",
        )

        cls.aggregate(self.waveform_files, self.waveform_output, clean=True)
        InjectionParameterSet.aggregate(
            self.rejected_parameter_files, self.rejected_output, clean=True
        )
        # clean up temporary directories
        for dirname in self.output_dir.glob("tmp-*"):
            shutil.rmtree(dirname)
