import luigi

from aframe.base import AframeWrapperTask
from aframe.pipelines.config import paths
from aframe.tasks import ExportLocal, TestingWaveforms, Train
from aframe.tasks.infer import InferLocal
from aframe.tasks.plots.sv import SensitiveVolume
from aframe.tasks.train.tune import TuneRemote


class SandboxExport(ExportLocal):
    train_task = luigi.TaskParameter()

    def requires(self):
        return self.train_task.req(
            self,
            data_dir=paths().train_datadir,
            run_dir=paths().train_rundir,
        )


class SandboxInfer(InferLocal):
    train_task = luigi.TaskParameter()

    def requires(self):
        reqs = {}
        reqs["model_repository"] = SandboxExport.req(
            self, repository_directory=paths().results_dir / "model_repo"
        )
        ts_waveforms = TestingWaveforms.req(
            self,
            output_dir=paths().test_datadir,
        )
        fetch = ts_waveforms.requires().workflow_requires()["test_segments"]

        reqs["data"] = fetch
        reqs["waveforms"] = ts_waveforms
        return reqs


class SandboxSV(SensitiveVolume):
    train_task = luigi.TaskParameter()

    def requires(self):
        reqs = {}
        reqs["ts"] = TestingWaveforms.req(
            self,
            output_dir=paths().test_datadir,
        )
        reqs["infer"] = SandboxInfer.req(
            self,
            output_dir=paths().results_dir / "infer",
            train_task=self.train_task,
        )
        return reqs


class _Sandbox(AframeWrapperTask):
    train_task = None

    def requires(self):
        # simply call SV plot task, which will
        # call all necessary downstream tasks!
        return SandboxSV.req(
            self,
            output_dir=paths().results_dir / "plots",
            train_task=self.train_task,
        )


class Sandbox(_Sandbox):
    train_task = Train


class Tune(_Sandbox):
    train_task = TuneRemote
