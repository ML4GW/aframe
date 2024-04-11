import glob
import importlib
import os
import shutil
from pathlib import Path
from typing import Dict, Literal

import law
import luigi
from luigi.util import inherits

import aframe.utils as utils
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.fetch import FetchTest

TsWorkflowRequires = Dict[Literal["test_segments"], law.Task]


class TimeSlideWaveformsParams(law.Task):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    Tb = luigi.FloatParameter()
    output_dir = luigi.Parameter()
    shifts = luigi.ListParameter()
    spacing = luigi.FloatParameter()
    buffer = luigi.FloatParameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
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
        reqs = super().workflow_requires()
        reqs["test_segments"] = FetchTest.req(
            self,
            segments_file=os.path.join(self.output_dir, "segments.txt"),
            data_dir=os.path.join(self.output_dir, "background"),
            condor_directory=os.path.join(
                os.getenv("AFRAME_CONDOR_DIR"), "test"
            ),
        )
        return reqs

    def requires(self):
        reqs = {}
        reqs["test_segments"] = FetchTest.req(
            self,
            branch=-1,
            segments_file=os.path.join(self.output_dir, "segments.txt"),
            data_dir=os.path.join(self.output_dir, "background"),
            condor_directory=os.path.join(
                os.getenv("AFRAME_CONDOR_DIR"), "test"
            ),
        )
        return reqs

    def load_prior(self):
        module_path, prior = self.prior.rsplit(".", 1)
        module = importlib.import_module(module_path)
        prior = getattr(module, prior)
        return prior

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
                # add psd_length to account for the burn in of psd calculation
                branch_map[i] = (start + self.psd_length, end, shift)
                i += 1
        return branch_map

    @workflow_condition.output
    def output(self):
        return [
            law.LocalFileTarget(os.path.join(self.tmp_dir, "waveforms.hdf5")),
            law.LocalFileTarget(
                os.path.join(self.tmp_dir, "rejected-parameters.hdf5")
            ),
        ]

    @property
    def shifts_required(self):
        return utils.get_num_shifts(
            self.test_segments, self.Tb, self.max_shift
        )

    @property
    def test_segments(self):
        paths = list(
            self.workflow_input()["test_segments"].collection.targets.values()
        )
        return utils.segments_from_paths(paths)

    @property
    def psd_segment(self):
        return list(self.input()["test_segments"].collection.targets.values())[
            -1
        ]

    @property
    def max_shift(self):
        return max(self.shifts)

    @property
    def tmp_dir(self):
        return os.path.join(self.output_dir, f"tmp-{self.branch}")

    def run(self):
        import io

        import h5py
        from data.timeslide_waveforms.timeslide_waveforms import (
            timeslide_waveforms,
        )

        prior = self.load_prior()
        start, end, shift = self.branch_data
        with self.psd_segment.open("r") as psd_file:
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
                self.highpass,
                self.snr_threshold,
                psd_file,
                Path(self.tmp_dir),
                self.seed,
            )


@inherits(DeployTimeslideWaveforms)
class TimeslideWaveforms(AframeDataTask):
    condor_directory = luigi.Parameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"),
            "timeslide_waveforms",
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_output = os.path.join(self.output_dir, "waveforms.hdf5")
        self.rejected_output = os.path.join(
            self.output_dir, "rejected-parameters.hdf5"
        )

    def output(self):
        return [
            law.LocalFileTarget(self.waveform_output),
            law.LocalFileTarget(self.rejected_output),
        ]

    def requires(self):
        return DeployTimeslideWaveforms.req(self)

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
        from ledger.injections import InjectionParameterSet, LigoResponseSet

        LigoResponseSet.aggregate(
            self.waveform_files, self.waveform_output, clean=True
        )
        InjectionParameterSet.aggregate(
            self.rejected_parameter_files, self.rejected_output, clean=True
        )
        # clean up temporary directories
        for dirname in glob.glob(os.path.join(self.output_dir, "tmp-*")):
            shutil.rmtree(dirname)
