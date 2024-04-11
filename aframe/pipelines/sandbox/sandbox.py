import os

from aframe.base import AframeWrapperTask
from aframe.pipelines.config import paths
from aframe.tasks import ExportLocal, TimeslideWaveforms, Train
from aframe.tasks.infer import InferLocal
from aframe.tasks.plots.sv import SensitiveVolume


class SandboxExport(ExportLocal):
    def requires(self):
        return Train.req(
            self,
            data_dir=paths().train_datadir,
            run_dir=os.path.join(paths().train_rundir),
        )


class SandboxInfer(InferLocal):
    def requires(self):
        reqs = {}
        reqs["model_repository"] = SandboxExport.req(
            self,
            repository_directory=os.path.join(
                paths().results_dir, "model_repo"
            ),
        )
        ts_waveforms = TimeslideWaveforms.req(
            self,
            output_dir=paths().test_datadir,
        )
        fetch = ts_waveforms.requires().workflow_requires()["test_segments"]

        reqs["data"] = fetch
        reqs["waveforms"] = ts_waveforms
        return reqs


class SandboxSV(SensitiveVolume):
    def requires(self):
        reqs = {}
        reqs["ts"] = TimeslideWaveforms.req(
            self,
            output_dir=paths().test_datadir,
        )
        reqs["infer"] = SandboxInfer.req(
            self,
            output_dir=os.path.join(paths().results_dir, "infer"),
        )
        return reqs


class Sandbox(AframeWrapperTask):
    def requires(self):
        # simply call SV plot task, which will
        # call all necessary downstream tasks!
        return SandboxSV.req(
            self,
            output_dir=os.path.join(paths().results_dir, "plots"),
        )
