import luigi

from aframe.base import AframeWrapperTask
from aframe.tasks import TestingWaveforms, Train
from aframe.tasks.infer import Infer
from aframe.tasks.plots.sv import SensitiveVolume
from aframe.tasks.train.tune import TuneTask


class SandboxInfer(Infer):
    train_task = luigi.TaskParameter()


class SandboxSV(SensitiveVolume):
    train_task = luigi.TaskParameter()

    def requires(self):
        reqs = {}
        reqs["ts"] = TestingWaveforms.req(self)
        reqs["infer"] = SandboxInfer.req(self)
        return reqs


class _Sandbox(AframeWrapperTask):
    train_task = luigi.TaskParameter()

    def requires(self):
        # simply call SV plot task, which will
        # call all necessary downstream tasks!
        return SandboxSV.req(self, train_task=self.train_task)


class Sandbox(_Sandbox):
    train_task = Train


class Tune(_Sandbox):
    train_task = TuneTask
