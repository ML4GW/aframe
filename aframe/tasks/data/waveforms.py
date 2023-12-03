import law
import luigi

from aframe.tasks.data.base import AframeDataTask


class GenerateWaveforms(AframeDataTask):
    num_signals = luigi.IntParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    output_file = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter(default=20)
    reference_frequency = luigi.FloatParameter(default=20)
    waveform_approximant = luigi.Parameter(default="IMRPhenomPv2")

    def output(self):
        return law.LocalFileTarget(self.output_file)

    @property
    def command(self):
        args = [
            "--num-signals",
            str(self.num_signals),
            "--sample-rate",
            str(self.sample_rate),
            "--waveform-duration",
            str(self.waveform_duration),
            "--output-file",
            self.output().path,
            "--minimum-frequency",
            str(self.minimum_frequency),
            "--reference-frequency",
            str(self.reference_frequency),
            "--waveform-approximant",
            self.waveform_approximant,
        ]

        return self.cli + args
