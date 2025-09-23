import importlib
import os

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter
from aframe.tasks.export.target import ModelRepositoryTarget


class ExportParams(law.Task):
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    batch_file = luigi.Parameter(default="")
    streams_per_gpu = luigi.IntParameter()
    aframe_instances = luigi.IntParameter()
    preproc_instances = luigi.IntParameter()
    clean = luigi.BoolParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    lowpass = luigi.OptionalFloatParameter(default="")
    preprocessor = luigi.Parameter(default="utils.preprocessing.BatchWhitener")
    preprocessor_kwargs = luigi.Parameter(default={})
    fftlength = luigi.OptionalFloatParameter(default="")
    ifos = luigi.ListParameter()
    repository_directory = PathParameter(
        default=paths().results_dir / "model_repo"
    )
    train_task = luigi.TaskParameter()
    platform = luigi.Parameter(
        default="TENSORRT",
        description="Platform to use for exporting model for inference",
    )


@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repository_directory.mkdir(exist_ok=True, parents=True)

    def _import_class(self, path: str):
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def output(self):
        return ModelRepositoryTarget(self.repository_directory, self.platform)

    def requires(self):
        return self.train_task.req(self)

    @property
    def default_image(self):
        return "export.sif"

    @property
    def num_ifos(self):
        return len(self.ifos)

    def run(self):
        from hermes.quiver import Platform

        from export.main import export

        breakpoint()
        cls = self._import_class(self.preprocessor)
        preprocessor = cls(**self.preprocessor_kwargs)
        # convert string to Platform enum
        platform = Platform[self.platform]

        # Assuming a convention for batch file/model file
        # names and locations
        weights = self.input().path
        weights_dir = os.path.dirname(weights)
        batch_file = weights_dir + "/batch.hdf5"

        export(
            weights,
            self.repository_directory,
            batch_file,
            self.num_ifos,
            self.kernel_length,
            self.inference_sampling_rate,
            self.sample_rate,
            self.batch_size,
            self.fduration,
            self.psd_length,
            preprocessor,
            self.streams_per_gpu,
            self.aframe_instances,
            self.preproc_instances,
            platform,
            clean=self.clean,
        )
