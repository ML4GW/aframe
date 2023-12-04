import os

import law
import luigi

from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.fetch import Fetch
from aframe.tasks.data.timeslide_waveforms import utils
from aframe.tasks.data.workflow import LDGCondorWorkflow


class TimeSlideWaveformsParams(AframeDataTask):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    data_dir = luigi.Parameter()
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
    segments_file = luigi.Parameter(default="")


class TimeslideWaveforms(
    TimeSlideWaveformsParams, law.LocalWorkflow, LDGCondorWorkflow
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
        return self.input()["data"]["collection"].exists()

    @property
    def shifts_required(self):
        return utils.get_num_shifts(self.segments, self.Tb, self.max_shift)

    @property
    def segments(self):
        return utils.segments_from_directory(self.input()["data"])

    @property
    def max_shift(self):
        return max(self.shifts)

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_map, i = {}, 1
        for start, stop in self.segments:
            for shift in self.shifts_required:
                branch_map[i] = (start, stop, shift)
                i += 1
        return

    # require directory of segments
    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["data"] = Fetch.req(self)
        return reqs

    def get_args(self):
        start, stop, shift = self.branch_data
        return []

    @workflow_condition.output
    def output(self):
        fname = f"tmp-{self.branch}.hdf5"
        return law.LocalFileTarget(os.path.join(self.data_dir, fname))

    def run(self):
        from data.cli import main

        main(args=self.get_args())


class MergeTimeslideWaveforms(law.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_file = os.path.join(self.data_dir, "waveforms.hdf5")
        self.rejected_file = os.path.join(
            self.data_dir, "rejected-parameters.hdf5"
        )

    def requires(self):
        return TimeslideWaveforms.req()

    def output(self):
        return [
            law.LocalFileTarget(self.waveform_file),
            law.LocalFileTarget(self.rejected_file),
        ]
