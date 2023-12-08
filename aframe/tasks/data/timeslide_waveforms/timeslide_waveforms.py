import glob
import os
import shutil
from pathlib import Path
from typing import TypedDict

import law
import luigi
from luigi.util import inherits

import aframe.utils as utils
from aframe.base import logger
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.workflow import LDGCondorWorkflow


class TsWorkflowRequires(TypedDict):
    test_segments: law.Task
    train_segments: law.Task


class TimeSlideWaveformsParams(law.Task):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    Tb = luigi.FloatParameter()
    data_dir = luigi.Parameter()
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
    seed = luigi.OptionalParameter(default=None)
    segments_file = luigi.Parameter(default="")
    # verbose = luigi.BoolParameter(default=False)


@inherits(TimeSlideWaveformsParams)
class GenerateTimeslideWaveforms(
    AframeDataTask,
    law.LocalWorkflow,
    LDGCondorWorkflow,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.data_dir, exist_ok=True)
        if not self.segments_file:
            self.segments_file = os.path.join(self.data_dir, "segments.txt")

        if self.job_log and not os.path.isabs(self.job_log):
            os.makedirs(self.log_dir, exist_ok=True)
            self.job_log = os.path.join(self.log_dir, self.job_log)

    # the workflow requires the testing segments
    def workflow_requires(self) -> TsWorkflowRequires:
        raise NotImplementedError

    # each workflow branch requires a training
    # segment file for psd calculatoin
    def requires(self):
        raise NotImplementedError

    # for now, require that testing segments exist.
    # in principle, we could just require that segments
    # exist since we only need the start and stop times,
    # but there were some instances where gwpy would not
    # successfully download all of the segments.
    # TODO: check if this is still the case.
    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.workflow_input()["test_segments"].collection.exists()

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
        return list(
            self.input()["train_segments"].collection.targets.values()
        )[-1].path

    @property
    def max_shift(self):
        return max(self.shifts)

    @property
    def tmp_dir(self):
        return os.path.join(self.output_dir, f"tmp-{self.branch}")

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_map, i = {}, 0
        for start, end in self.test_segments:
            for shift in range(self.shifts_required):
                # add psd_length to account for the burn in of psd calculation
                branch_map[i] = (start + self.psd_length, end, shift + 1)
                i += 1
        return branch_map

    # TODO: this is getting a bit messy. I think we should
    # find a way to annotate arguments that will get passed
    # to the command line like below, and define a generic
    # get_args method that handles parsing them to CLI based
    # on their parameter type. Or, get rid of CLI altogether.
    # I think it's really only useful for the train jobs
    # where the pytorch lightning config.yaml is actually useful
    def get_args(self):
        start, end, shift = self.branch_data

        return [
            "timeslide_waveforms",
            "--start",
            str(start),
            "--end",
            str(end),
            f"--ifos=[{','.join(self.ifos)}]",
            "--shift",
            str(shift),
            "--spacing",
            str(self.spacing),
            "--buffer",
            str(self.buffer),
            "--prior",
            self.prior,
            "--minimum_frequency",
            str(self.minimum_frequency),
            "--reference_frequency",
            str(self.reference_frequency),
            "--sample_rate",
            str(self.sample_rate),
            "--waveform_duration",
            str(self.waveform_duration),
            "--waveform_approximant",
            self.waveform_approximant,
            "--highpass",
            str(self.highpass),
            "--snr_threshold",
            str(self.snr_threshold),
            "--seed",
            str(self.seed),
            # "--verbose",
            # str(self.verbose),
            "--psd_file",
            str(self.psd_segment),
            "--output_dir",
            self.tmp_dir,
        ]

    @workflow_condition.output
    def output(self):
        return [
            law.LocalFileTarget(os.path.join(self.tmp_dir, "waveforms.hdf5")),
            law.LocalFileTarget(
                os.path.join(self.tmp_dir, "rejected-parameters.hdf5")
            ),
        ]

    def run(self):
        from data.cli import main

        logger.debug(
            "Running timeslide-waveforms with args: "
            f"{' '.join(self.get_args())}"
        )
        main(args=self.get_args())


@inherits(TimeSlideWaveformsParams)
class MergeTimeslideWaveforms(AframeDataTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_output = os.path.join(self.output_dir, "waveforms.hdf5")
        self.rejected_output = os.path.join(
            self.output_dir, "rejected-parameters.hdf5"
        )

    def requires(self):
        return GenerateTimeslideWaveforms.req(self)

    def output(self):
        return [
            law.LocalFileTarget(self.waveform_output),
            law.LocalFileTarget(self.rejected_output),
        ]

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def waveform_files(self):
        return map(Path, [targets[0].path for targets in self.targets])

    @property
    def rejected_parameter_files(self):
        return map(Path, [targets[1].path for targets in self.targets])

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
