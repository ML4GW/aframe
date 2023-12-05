import glob
import os
import shutil

import law
import luigi
from luigi.util import inherits

from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.fetch import Fetch
from aframe.tasks.data.timeslide_waveforms import utils
from aframe.tasks.data.workflow import LDGCondorWorkflow


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
    background_dir = luigi.Parameter()
    seed = luigi.OptionalParameter(default=None)
    segments_file = luigi.Parameter(default="")


@inherits(TimeSlideWaveformsParams)
class GenerateTimeslideWaveforms(
    law.LocalWorkflow,
    LDGCondorWorkflow,
    AframeDataTask,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.data_dir, exist_ok=True)
        if not self.segments_file:
            self.segments_file = os.path.join(self.data_dir, "segments.txt")

        if self.job_log and not os.path.isabs(self.job_log):
            os.makedirs(self.log_dir, exist_ok=True)
            self.job_log = os.path.join(self.log_dir, self.job_log)

    # for just require that some testing segments exist.
    # I think we originally put this in b/c gwpy would
    # not always successfully download all of the segments
    # in the segments.txt file, so we needed a way of determining
    # if the segments were actually downloaded. TODO: check if this
    # is still the case.
    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.input()["test"]["collection"].exists()

    @property
    def shifts_required(self):
        return utils.get_num_shifts(self.segments, self.Tb, self.max_shift)

    @property
    def segments(self):
        d = os.path.dirname(self.input()["test"]["collection"].targets[1].path)
        return utils.segments_from_directory(d)

    @property
    def max_shift(self):
        return max(self.shifts)

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_map, i = {}, 0
        for start, end in self.segments:
            for shift in range(self.shifts_required):
                # add psd_length to account for the burn in of psd calculation
                branch_map[i] = (start + self.psd_length, end, shift)
                i += 1
        return branch_map

    # require directory of segments
    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["data"] = Fetch.req(
            self, data_dir=os.path.join(self.data_dir, "background")
        )
        return reqs

    @property
    def tmp_dir(self):
        return os.path.join(self.output_dir, f"tmp-{self.branch}")

    # TODO: this is getting a bit messy. I think we should
    # find a way to annotate arguments that will get passed
    # to the command line like below, and define a generic
    # get_args method that handles parsing them to CLI based
    # on their parameter type.
    def get_args(self):
        start, end, shift = self.branch_data

        return [
            "--start",
            str(start),
            "--end",
            str(end),
            "--ifos",
            f"--data.ifos=[{','.join(self.ifos)}]",
            "--shifts",
            ",".join([str(shift)]),
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
            "--verbose",
            str(self.verbose),
            "--background_dir",
            str(self.background_dir),
            "--output_dir",
            self.tmp_dir,
        ]

    @workflow_condition.output
    def output(self):
        return [
            law.LocalFileTarget(os.path.join(self.tmp_dir, "wavforms.hdf5")),
            law.LocalFileTarget(
                os.path.join(self.tmp_dir, "rejected-parameters.hdf5")
            ),
        ]

    def run(self):
        from data.cli import main

        main(args=self.get_args())


@inherits(TimeSlideWaveformsParams)
class MergeTimeslideWaveforms(AframeDataTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_file = os.path.join(self.output_dir, "waveforms.hdf5")
        self.rejected_file = os.path.join(
            self.output_dir, "rejected-parameters.hdf5"
        )

    def requires(self):
        return GenerateTimeslideWaveforms.req(self)

    def output(self):
        return [
            law.LocalFileTarget(self.waveform_file),
            law.LocalFileTarget(self.rejected_file),
        ]

    @property
    def collection(self):
        return self.input()["collection"]

    @property
    def waveform_files(self):
        return [self.collection[i][0] for i in self.collection]

    @property
    def rejected_parameters(self):
        return [self.collection[i][1] for i in self.collection]

    def run(self):
        from ledger.injections import InjectionParameterSet, LigoResponseSet

        LigoResponseSet.merge(
            self.waveform_files, self.waveform_file, clean=True
        )
        InjectionParameterSet.merge(
            self.rejected_parameters, self.rejected_file, clean=True
        )
        # clean up temporary directories
        for dirname in glob.glob(os.path.join(self.output_dir, "tmp-*")):
            shutil.rmtree(dirname)
