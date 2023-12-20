import importlib

import law
import luigi
from luigi.util import inherits

from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask


class WaveformParams(law.Task):
    num_signals = luigi.IntParameter()
    sample_rate = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    output_file = luigi.Parameter()
    prior = luigi.Parameter()
    minimum_frequency = luigi.FloatParameter(default=20)
    reference_frequency = luigi.FloatParameter(default=50)
    waveform_approximant = luigi.Parameter(default="IMRPhenomPv2")


@inherits(WaveformParams)
class GenerateWaveforms(AframeDataTask):
    def output(self):
        return s3_or_local(self.output_file, client=self.client)

    def load_prior(self):
        module_path, prior = self.prior.rsplit(".", 1)
        module = importlib.import_module(module_path)
        prior = getattr(module, prior)

        return prior

    def run(self):
        from data.waveforms.injection import (
            WaveformGenerator,
            convert_to_detector_frame,
            write_waveforms,
        )

        generator = WaveformGenerator(
            self.waveform_duration,
            self.sample_rate,
            self.minimum_frequency,
            self.reference_frequency,
            waveform_approximant=self.waveform_approximant,
        )
        prior = self.load_prior()
        prior, detector_frame_prior = prior()
        samples = prior.sample(self.num_signals)
        if not detector_frame_prior:
            samples = convert_to_detector_frame(samples)
        signals = generator(samples)
        with self.output().open("w") as f:
            write_waveforms(f, signals, samples, generator)


# for validation waveforms, utilize rejection sampling
# to generate waveforms with same distribution as testing set
@inherits(WaveformParams)
class ValidationWaveforms(law.Task):
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
