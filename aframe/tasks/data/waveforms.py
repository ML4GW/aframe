import law
import luigi

from aframe.base import logger
from aframe.tasks.data.base import AframeDataTask


class WaveformParams(law.Task):
    num_signals = luigi.IntParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    output_file = luigi.Parameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter(default=20)
    reference_frequency = luigi.FloatParameter(default=20)
    waveform_approximant = luigi.Parameter(default="IMRPhenomPv2")


class GenerateWaveforms(AframeDataTask, WaveformParams):
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

        logger.debug(f"Running with args: {' '.join(self.get_args())}")
        main(args=self.get_args())


# for validation waveforms, utilize rejection sampling
# to generate waveforms with same distribution as testing set
class ValidationWaveforms(WaveformParams):
    ifos = luigi.ListParameter()
    snr_threshold = luigi.FloatParameter()
    highpass = luigi.FloatParameter()

    @property
    def psds(self):
        raise NotImplementedError

    def output(self):
        return law.LocalFileTarget(self.output_file)

    def run(self):
        import h5py
        import numpy as np
        from data.waveforms.rejection import rejection_sample

        parameters, _ = rejection_sample(
            self.num_signals,
            self.prior,
            self.ifos,
            self.minimum_frequency,
            self.reference_frequency,
            self.sample_rate,
            self.waveform_duration,
            self.waveform_approximant,
            self.highpass,
            self.snr_threshold,
            self.psds,
            return_raw=True,  # return raw polarizations
        )
        signals = []
        for ifo in self.ifos:
            signals.append(parameters[ifo])
        signals = np.stack(signals)

        # TODO: save parameters
        with h5py.File(self.output().path, "w") as f:
            f.create_dataset("signals", data=signals)
