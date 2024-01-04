import os

import law
import luigi
from luigi.contrib.s3 import S3Client
from luigi.util import inherits

from aframe.config import s3
from aframe.pipelines.sandbox.config import SandboxConfig
from aframe.targets import s3_or_local
from aframe.tasks import (
    ExportLocal,
    Fetch,
    GenerateTimeslideWaveforms,
    GenerateWaveforms,
    InferLocal,
    MergeTimeslideWaveforms,
    TrainLocal,
    TrainRemote,
)
from aframe.tasks.train.base import TrainParameters
from aframe.tasks.train.config import train_remote

config = SandboxConfig()


# dynamically create local or remote train task
# depending on the specified format of the data and run directories
@inherits(TrainParameters)
class SandboxTrain(law.Task):
    dev = luigi.BoolParameter(default=False, significant=False)

    @property
    def client(self):
        return S3Client(endpoint_url=s3().endpoint_url)

    def requires(self):
        yield Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(
                config.data_local, "condor", "train"
            ),
            data_dir=os.path.join(config.train_data_dir, "background"),
            segments_file=os.path.join(
                config.data_local, "condor", "train", "segments.txt"
            ),
            **config.train_background.to_dict(),
            workflow="htcondor",
        )
        yield GenerateWaveforms.req(
            self,
            image="data.sif",
            output_file=os.path.join(config.train_data_dir, "signals.hdf5"),
            **config.train_waveforms.to_dict(),
        )

    def output(self):
        return s3_or_local(
            os.path.join(config.train_run_dir, "model.pt"), client=self.client
        )

    def run(self):
        # check that data_remote and run_remote
        # are either both specified, or both None
        if (config.data_remote is None) != (config.run_remote is None):
            raise ValueError(
                "Must specify both remote or local data and run directories"
            )

        # run locally if not specified
        local = config.data_remote is None
        if local:
            yield TrainLocal.req(
                self,
                data_dir=config.train_data_dir,
                run_dir=config.train_run_dir,
            )
        else:
            yield TrainRemote.req(
                self,
                image=train_remote().image,
                config="/home/ethan.marx/projects/aframev2/projects/train/config.yaml",  # noqa
                data_dir=config.train_data_dir,
                run_dir=config.train_run_dir,
                request_cpus=train_remote().request_cpus,
                request_gpus=train_remote().request_gpus,
                request_cpu_memory=train_remote().request_cpu_memory,
            )


class SandboxExport(ExportLocal):
    def requires(self):
        # expicitly pass parameters image and config parameters
        # b/c these are common parameters that should
        # not be inherited from the export task
        return SandboxTrain.req(
            self,
            image="train.sif",
            data_dir=config.train_data_dir,
            run_dir=config.train_run_dir,
            **config.train.to_dict(),
        )


class SandboxGenerateTimeslideWaveforms(GenerateTimeslideWaveforms):
    def workflow_requires(self):
        reqs = super().workflow_requires()
        # requires background testing segments
        # to determine number of waveforms to generate
        reqs["test_segments"] = Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(config.data_local, "condor", "test"),
            data_dir=os.path.join(config.data_local, "test", "background"),
            segments_file=os.path.join(
                config.data_local, "test", "segments.txt"
            ),
            **config.test_background.to_dict(),
        )
        return reqs

    def requires(self):
        reqs = super().requires()
        # requires a background training segment for calculating snr
        # TODO: how to specify just the last segment?
        reqs["train_segments"] = Fetch.req(
            self,
            branch=-1,
            image="data.sif",
            condor_directory=os.path.join(
                config.data_local, "condor", "train"
            ),
            data_dir=os.path.join(config.train_data_dir, "background"),
            segments_file=os.path.join(
                config.data_local, "train", "segments.txt"
            ),
            **config.train_background.to_dict(),
        )
        return reqs


class SandboxTimeslideWaveforms(MergeTimeslideWaveforms):
    # timeslide waveform data always should be kept local
    @property
    def data_dir(self):
        return os.path.join(config.data_local, "test")

    @property
    def output_dir(self):
        return os.path.join(config.data_local, "timeslide_waveforms")

    @property
    def condor_directory(self):
        return os.path.join(config.data_local, "condor", "timeslide_waveforms")

    def requires(self):
        return SandboxGenerateTimeslideWaveforms.req(
            self,
            image="data.sif",
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            condor_directory=self.condor_directory,
            **config.timeslide_waveforms.to_dict(),
        )


class SandboxInfer(InferLocal):
    def requires(self):
        reqs = {}
        reqs["export"] = SandboxExport.req(
            self,
            image="export.sif",
            repository_directory=os.path.join(config.run_local, "model_repo"),
            logfile=os.path.join(config.base.log_dir, "export.log"),
            **config.export.to_dict(),
        )

        reqs["data"] = Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(config.data_local, "condor", "test"),
            data_dir=os.path.join(config.data_local, "test", "background"),
            segments_file=os.path.join(
                config.data_local, "test", "segments.txt"
            ),
            **config.test_background.to_dict(),
        )

        reqs["waveforms"] = SandboxTimeslideWaveforms.req(
            self, image="data.sif", **config.timeslide_waveforms.to_dict()
        )

        return reqs


class Sandbox(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        # simply call infer, which will
        # call all necessary downstream tasks!
        yield SandboxInfer.req(
            self,
            output_dir=config.infer.output_dir,
            **config.infer.to_dict(),
        )
