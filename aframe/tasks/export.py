import os
import shlex
import sys

import law
import luigi

from aframe.base import AframeTask, logger
from aframe.config import Defaults
from aframe.utils import stream_command

class ExportLocal(AframeTask):
    config = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = self.config or Defaults.EXPORT

    def output(self):
        return law.LocalFileTarget(self.cfg.repository_directory)
    
    def run(self):
        from export import cli
        cli(
            self.config,
            os.path.join(self.logdir, "export.log"),
            self.cfg.num_ifos,
            self.cfg.kernel_length,
            self.cfg.inference_sampling_rate,
            self.cfg.sample_rate,
            self.cfg.inference_batch_size,
            self.cfg.fduration,
            self.cfg.inference_psd_length,
            self.cfg.fftlength,
            self.cfg.highpass,
            self.cfg.streams_per_gpu,
            self.cfg.aframe_instances,
            self.cfg.platform,
            self.cfg.clean,
            self.cfg.verbose,
        )
        
        
