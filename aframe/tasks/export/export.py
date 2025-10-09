import os
import sys

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.config import Defaults, paths
from aframe.parameters import PathParameter
from aframe.tasks.export.target import ModelRepositoryTarget
from aframe.tasks.train.utils import stream_command


class ExportParams(law.Task):
    export_config = luigi.Parameter(default=Defaults.EXPORT)
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    batch_file = luigi.Parameter(default="")
    streams_per_gpu = luigi.IntParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    repository_directory = PathParameter(
        default=paths().results_dir / "model_repo"
    )
    train_task = luigi.TaskParameter()
    platform = luigi.Parameter(
        default="TENSORRT",
        description="Platform to use for exporting model for inference",
    )

    model_type = luigi.Parameter(default="export")
    if model_type ==  'mm_export':
        resample_rates = luigi.ListParameter()
        kernel_lengths = luigi.ListParameter()
        high_passes = luigi.ListParameter()
        low_passes = luigi.ListParameter()
        inference_sampling_rates = luigi.ListParameter()
        starting_offsets = luigi.ListParameter()
        classes = luigi.ListParameter()
        layers = luigi.ListParameter()

@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repository_directory.mkdir(exist_ok=True, parents=True)

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

    def get_args(self):
        args = ["--config", self.export_config]
        args.append("--repository_directory=" + str(self.repository_directory))
        args.append("--num_ifos=" + str(self.num_ifos))
        args.append("--sample_rate=" + str(self.sample_rate))
        args.append("--kernel_length=" + str(self.kernel_length))
        args.append("--fduration=" + str(self.fduration))
        args.append(
            "--inference_sampling_rate=" + str(self.inference_sampling_rate)
        )
        args.append("--batch_size=" + str(self.batch_size))
        args.append("--psd_length=" + str(self.psd_length))
        args.append("--streams_per_gpu=" + str(self.streams_per_gpu))
        args.append("--platform=" + str(self.platform))
        args.append("--model_type=" + str(self.model_type))
        if self.model_type == 'mm_export':
            args.append("--resample_rates=" + str(self.batch_size))
            args.append("--kernel_lengths=" + str(self.psd_length))
            args.append("--high_passes=" + str(self.streams_per_gpu))
            args.append("--low_passes=" + str(self.platform))
            args.append("--inference_sampling_rates=" + str(self.batch_size))
            args.append("--starting_offsets=" + str(self.psd_length))
            args.append("--classes=" + str(self.streams_per_gpu))
            args.append("--layers=" + str(self.platform))
        return args

    def run(self):
        # Assuming a convention for batch file/model file
        # names and locations
        if self.model_type == 'mm_export':
            weights = self.input().path
            weights_dir = os.path.dirname(weights)
            batch_file = weights_dir + "/batch.h5"
            args = self.get_args()
            args.append("--weights=" + weights)
            args.append("--batch_file=" + batch_file)
            cmd = [sys.executable, "-m", "export"] + args
            stream_command(cmd)
        else:
            weights = self.input().path
            weights_dir = os.path.dirname(weights)
            batch_file = weights_dir + "/batch.hdf5"
    
            args = self.get_args()
            args.append("--weights=" + weights)
            args.append("--batch_file=" + batch_file)
            cmd = [sys.executable, "-m", "export"] + args
            stream_command(cmd)
