from typing import Callable

import torch
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from ledger.injections import BilbyParameterSet

from .generator import WaveformGenerator


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float,
        f_ref: float,
        right_pad: float,
        **kwargs,
    ):
        """
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `aframe.train.train.data.waveforms.generator.WaveformGenerator`.
        Args:
            *args:
                Positional arguments passed to
                `aframe.train.train.data.waveforms.generator.WaveformGenerator`
            approximant:
                A callable that takes parameter tensors
                and returns the cross and plus polarizations.
                For example, `ml4gw.waveforms.IMRPhenomD()`
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            **kwargs:
                Keyword arguments passed to
                `aframe.train.train.data.waveforms.generator.WaveformGenerator`
        """
        super().__init__(*args, **kwargs)
        self.approximant = approximant
        self.f_ref = f_ref
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            self.kernel_length,
            f_min,
            f_ref,
            self.right_pad,
        )

    def convert(self, parameters):
        # TODO: This assumes a detector-frame prior. Remove this
        # when we switch to source-frame prior.
        for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
            if key in parameters:
                parameters[key] *= 1 + parameters["redshift"]
        parameter_set = BilbyParameterSet(**parameters)
        generation_params = parameter_set.generation_params(
            reference_frequency=self.f_ref
        )
        return generation_params

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        hc, hp = waveforms.transpose(1, 0)
        return hc.float(), hp.float()
