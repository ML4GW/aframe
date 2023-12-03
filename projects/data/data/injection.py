from pathlib import Path
from typing import Callable, Dict, List

import h5py
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole as lal_bbh
from bilby.gw.waveform_generator import WaveformGenerator as BilbyGenerator


class WaveformGenerator(BilbyGenerator):
    def __init__(
        self,
        waveform_duration: float,
        sample_rate: float,
        minimum_frequency: float = 20,
        reference_frequency: float = 20,
        frequency_domain_source_model: Callable = lal_bbh,
        parameter_conversion: Callable = convert_to_lal_binary_black_hole_parameters,  # noqa
        waveform_approximant: str = "IMRPhenomPv2",
    ):
        self.sample_rate = sample_rate
        self.waveform_duration = waveform_duration
        self.waveform_approximant = waveform_approximant
        self.minimum_frequency = minimum_frequency
        self.reference_frequency = reference_frequency
        super().__init__(
            duration=waveform_duration,
            sampling_frequency=sample_rate,
            frequency_domain_source_model=frequency_domain_source_model,
            parameter_conversion=parameter_conversion,
            waveform_arguments={
                "waveform_approximant": waveform_approximant,
                "reference_frequency": reference_frequency,
                "minimum_frequency": minimum_frequency,
            },
        )

    @property
    def waveform_size(self):
        return int(self.sampling_frequency * self.duration)

    def __call__(
        self,
        sample_params: Dict[List, str],
    ):
        """Generate raw gravitational-wave signals, pre-interferometer projection.
        Args:
            sample_params:
                Dictionary of GW parameters where key is the parameter name
                and value is a list of the parameters. Typically generated
                from calling`prior.sample()` where `prior` is a bilby
                PriorDict object.
        Returns:
            An (n_samples, 2, waveform_size) array,
            containing both polarizations
            for each of the desired number of samples.
            The waveforms are shifted such that
            the coalescence time lies at the center of the sample
        """

        # convert sample params to list of dictionaries
        sample_params = [
            dict(zip(sample_params, col))
            for col in zip(*sample_params.values())
        ]

        n_samples = len(sample_params)

        num_pols = 2
        signals = np.zeros((n_samples, num_pols, self.waveform_size))

        for i, p in enumerate(sample_params):
            polarizations = self.time_domain_strain(p)
            polarization_names = sorted(polarizations.keys())
            polarizations = np.stack(
                [polarizations[p] for p in polarization_names]
            )

            # center so that coalescence time is middle sample
            dt = self.waveform_duration / 2
            polarizations = np.roll(
                polarizations, int(dt * self.sample_rate), axis=-1
            )
            signals[i] = polarizations

        return signals


def write_waveforms(
    output_file: Path,
    signals: np.ndarray,
    samples: Dict[str, List],
    generator: WaveformGenerator,
):
    """
    Write signals and corresponding to samples to an hdf5 file,
    storing metadata associated with corresponding WaveformGenerator
    """

    with h5py.File(output_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in samples.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": len(signals),
                "sample_rate": generator.sample_rate,
                "waveform_duration": generator.waveform_duration,
                "waveform_approximant": generator.waveform_approximant,
                "reference_frequency:": generator.reference_frequency,
                "minimum_frequency": generator.minimum_frequency,
            }
        )
    return output_file
