import os

import law
import luigi
from luigi.util import inherits

from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.segments import Query


@inherits(Query)
class Fetch(law.LocalWorkflow, StaticMemoryWorkflow, AframeDataTask):
    data_dir = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    max_duration = luigi.FloatParameter(default=-1)
    prefix = luigi.Parameter(default="background")
    channels = luigi.ListParameter()

    exclude_params_req = {"condor_directory"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.data_dir.startswith("s3://"):
            os.makedirs(self.data_dir, exist_ok=True)

        if self.job_log and not os.path.isabs(self.job_log):
            log_dir = os.path.join(self.data_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.job_log = os.path.join(log_dir, self.job_log)

    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.workflow_input()["segments"].exists()

    def load_segments(self):
        with self.workflow_input()["segments"].open("r") as f:
            segments = f.read().splitlines()[1:]
        return segments

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        segments = self.load_segments()
        branch_map, i = {}, 1
        for segment in segments:
            segment = segment.split("\t")
            start, duration = map(float, segment[1::2])
            step = duration if self.max_duration == -1 else self.max_duration
            num_steps = (duration - 1) // step + 1

            for j in range(int(num_steps)):
                segstart = start + j * step
                segdur = min(start + duration - segstart, step)
                branch_map[i] = (segstart, segdur)
                i += 1
        return branch_map

    def workflow_requires(self):
        reqs = super().workflow_requires()

        kwargs = {}
        if self.job_log:
            log_file = law.LocalFileTarget(self.job_log)
            log_file = log_file.parent.child("query.log", type="f")
            kwargs["job_log"] = log_file.path
        reqs["segments"] = Query.req(
            self, segments_file=self.segments_file, **kwargs
        )
        return reqs

    @workflow_condition.output
    def output(self):
        start, duration = self.branch_data
        start = int(float(start))
        duration = int(float(duration))
        fname = "{}-{}-{}.hdf5".format(self.prefix, start, duration)
        fname = os.path.join(self.data_dir, fname)
        return s3_or_local(fname)

    def run(self):
        import h5py
        from data.fetch.fetch import fetch

        start, duration = self.branch_data
        start = int(float(start))
        duration = int(float(duration))

        X = fetch(
            start,
            start + duration,
            self.channels,
            self.sample_rate,
        )
        size = int(duration * self.sample_rate)
        with self.output().open("w") as f:
            with h5py.File(f, "w") as h5file:
                # write with chunking for dataloading perf increase
                X.write(
                    h5file,
                    format="hdf5",
                    chunks=(min(size, 131072),),
                    compression=None,
                )


# renaming tasks to allow specifying diff params in config files
class FetchTest(Fetch):
    condor_directory = luigi.Parameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"), "test"
        )
    )


class FetchTrain(Fetch):
    condor_directory = luigi.Parameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"), "train"
        )
    )
