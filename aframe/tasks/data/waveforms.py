import law
import luigi

from aframe.tasks.data.base import AframeDataTask


class GenerateWaveforms(AframeDataTask):
    num_signals = luigi.IntParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    output_file = luigi.Parameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter(default=20)
    reference_frequency = luigi.FloatParameter(default=20)
    waveform_approximant = luigi.Parameter(default="IMRPhenomPv2")

    def output(self):
        return law.LocalFileTarget(self.output_file)

    def get_args(self):
        args = [
            "waveforms",
            "--num_signals",
            str(self.num_signals),
            "--sample_rate",
            str(self.sample_rate),
            "--waveform_duration",
            str(self.waveform_duration),
            "--output_file",
            self.output().path,
            "--minimum_frequency",
            str(self.minimum_frequency),
            "--reference_frequency",
            str(self.reference_frequency),
            "--waveform_approximant",
            str(self.waveform_approximant),
            "--prior",
            str(self.prior),
        ]

        return args

    def run(self):
        from data.cli import main

        main(args=self.get_args())
