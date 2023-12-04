import law

from aframe.base import AframeGPUTask, logger
from aframe.tasks.export.base import ExportParams


class ExportLocal(AframeGPUTask, ExportParams):
    def output(self):
        # TODO: custom file target that checks for existence
        # of all necessary model repo directories and files
        return law.LocalFileTarget(self.repository_directory)

    def configure_args(self):
        positional = [
            "kernel_length",
            "inference_sampling_rate",
            "sample_rate",
            "fduration",
            "streams_per_gpu",
            "aframe_instances",
            # "clean",
        ]
        args = [
            "--config",
            self.config,
            "--logfile",
            self.logfile,
            "--weights",
            self.weights,
            "--repository_directory",
            self.repository_directory,
            "--num_ifos",
            self.num_ifos,
            "--batch_size",
            self.batch_size,
            "--psd_length",
            self.psd_length,
        ]
        for arg in positional:
            args.extend([f"--{arg}", getattr(self, arg)])
        return args

    def configure_optional_args(self, args: list[str]) -> list[str]:
        for arg in ["fftlength", "highpass", "platform"]:
            try:
                x = getattr(self, arg)
            except AttributeError:
                continue
            else:
                args.extend([f"--{arg}", x])

        return args

    def get_args(self):
        args = self.configure_args()
        args = self.configure_optional_args(args)
        return map(str, args)

    def run(self):
        from export.cli import main

        args = self.get_args()
        logger.debug(f"Running Export with arguments {' '.join(args)}")

        main(args=self.get_args())
