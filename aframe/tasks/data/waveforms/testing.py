import shutil
from pathlib import Path
from typing import Dict, Literal

import law
import luigi
from luigi.util import inherits

from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.fetch import FetchTest
from aframe.tasks.data.waveforms.base import WaveformParams
from utils import data as data_utils

TsWorkflowRequires = Dict[Literal["test_segments"], law.Task]


class TestingWaveformsParams(WaveformParams):
    start = luigi.FloatParameter(
        description="Start time of the test data segments to query"
    )
    end = luigi.FloatParameter(
        description="End time of the test data segments to query"
    )
    ifos = luigi.ListParameter(
        description="Interferometers for which waveforms will be generated"
    )
    shifts = luigi.ListParameter(
        description="List of shift multiple to apply to the test data segments"
    )
    spacing = luigi.FloatParameter(
        description="Spacing between injections in seconds"
    )
    buffer = luigi.FloatParameter(
        description="Buffer time between the first (last)"
        "waveform and start (end) the test data segment"
    )
    highpass = luigi.FloatParameter(
        description="Frequency of highpass filter in Hz"
    )
    lowpass = luigi.OptionalFloatParameter(
        description="Frequency of lowpass filter in Hz",
        default="",
    )
    snr_threshold = luigi.FloatParameter(
        description="SNR threshold for rejection sampling"
    )
    psd_length = luigi.FloatParameter(
        description="Length of the PSD segment to use for rejection sampling"
    )
    seed = luigi.IntParameter(
        description="Seed for controlling randomness"
        " of waveform prior sampling"
    )
    jitter = luigi.FloatParameter(
        description="Scale of random jitter to add to injection times",
        default=0.1,
    )
    background_dir = PathParameter(
        description="Directory containing background strain data into "
        "which waveforms will be injected during inference",
        default=paths().test_background_dir,
    )
    output_dir = PathParameter(
        description="Directory where merged waveforms and "
        "rejected parameters will be saved",
        default=paths().test_waveforms_dir,
    )


@inherits(TestingWaveformsParams)
class DeployTestingWaveforms(
    AframeDataTask,
    StaticMemoryWorkflow,
    law.LocalWorkflow,
):
    """
    Deploy condor jobs for generating testing waveforms via rejection sampling.
    """

    condor_directory = PathParameter(
        default=paths().condor_dir / "testing_waveforms"
    )

    def workflow_requires(self):
        reqs = {}
        reqs["test_segments"] = FetchTest.req(
            self,
            segments_file=self.background_dir / "segments.txt",
            data_dir=self.background_dir / "background",
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
        return data_utils.get_num_shifts_from_num_signals(
            self.test_segments,
            self.num_signals,
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

        from data.waveforms.testing import testing_waveforms

        prior = load_prior(self.prior)
        start, end, shift, psd_segment = self.branch_data
        with psd_segment.open("r") as psd_file:
            psd_file = h5py.File(io.BytesIO(psd_file.read()))
            testing_waveforms(
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
                self.right_pad,
                self.highpass,
                self.lowpass,
                self.snr_threshold,
                psd_file,
                Path(self.tmp_dir),
                jitter=self.jitter,
                seed=self.seed,
            )


@inherits(DeployTestingWaveforms)
class TestingWaveforms(AframeDataTask):
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
        return DeployTestingWaveforms.req(
            self,
            request_memory=self.request_memory,
            request_disk=self.request_disk,
            request_cpus=self.request_cpus,
            workflow=self.workflow,
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
