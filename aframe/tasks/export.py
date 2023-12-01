import law
import luigi

from aframe.base import AframeTask
from aframe.config import Defaults


class ExportLocal(AframeTask):
    config = luigi.Parameter(default="")
    weights = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = self.config or Defaults.EXPORT
        self.data = self.cfg.data
        self.export = self.cfg.export

    def output(self):
        # TODO: custom file target that checks for existence
        # of all necessary model repo directories and files
        return law.LocalFileTarget(self.export.repository_directory)

    def configure_positional_args(self):
        positional = [
            "kernel_length",
            "inference_sampling_rate",
            "sample_rate",
            "fduration",
        ]
        args = [
            "--config",
            self.config,
            "--logfile",
            "./export.log",
            "--weights",
            self.weights,
            "--repository_directory",
            self.export.repository_directory,
            "--num_ifos",
            len(self.cfg.ifos),
            "--batch_size",
            self.data.inference_batch_size,
            "--psd_length",
            self.data.inference_psd_length,
        ]
        for arg in positional:
            args.extend([f"--{arg}", getattr(self.data, arg)])
        return args

    def configure_optional_args(self, args: list[str]) -> list[str]:
        for arg in ["fftlength", "highpass"]:
            if getattr(self.data, arg):
                args.extend([f"--{arg}", getattr(self.data, arg)])

        for arg in [
            "streams_per_gpu",
            "aframe_instances",
            "platform",
            "clean",
        ]:
            if getattr(self.export, arg):
                args.extend([f"--{arg}", getattr(self.export, arg)])

        return args

    def get_args(self):
        args = self.configure_positional_args()
        args = self.configure_optional_args(args)
        return map(str, args)

    def run(self):
        from export.cli import main

        main(args=self.get_args())
