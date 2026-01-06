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
import json

class ExportParams(law.Task):
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
#    export_config = luigi.Parameter(default=Defaults.EXPORT)
    
#    model_type = luigi.Parameter(default="export")
#    if model_type ==  'mm_export':
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
#        args = ["--config", self.export_config]
        args = []
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
        #args.append("--model_type=" + str(self.model_type))
        #if self.model_type == 'mm_export':
        args.append(f"--resample_rates=[{','.join([str(x) for x in self.resample_rates])}]")
        args.append(f"--kernel_lengths=[{','.join([str(x) for x in self.kernel_lengths])}]")
        args.append(f"--high_passes=[{','.join([str(x) for x in self.high_passes])}]")
        args.append(f"--low_passes=[{','.join([str(x) for x in self.low_passes])}]")
        args.append(f"--inference_sampling_rates=[{','.join([str(x) for x in self.inference_sampling_rates])}]")
        args.append(f"--starting_offsets=[{','.join([str(x) for x in self.starting_offsets])}]")
        args.append(f"--classes=[{','.join([str(x) for x in self.classes])}]")
        args.append(f"--layers=[[{','.join([str(x) for x in self.layers[0]])}],[{','.join([str(x) for x in self.layers[1]])}],[{','.join([str(x) for x in self.layers[2]])}],[{','.join([str(x) for x in self.layers[3]])}]]")
        return args

    def run(self):
        # Assuming a convention for batch file/model file
        # names and locations
        weights = self.input().path
        weights_dir = os.path.dirname(weights)
        batch_file = weights_dir + "/batch.hdf5"
        args = self.get_args()
        args.append("--weights=" + weights)
        args.append("--batch_file=" + batch_file)
        cmd = [sys.executable, "-m", "export"] + args
        stream_command(cmd)
