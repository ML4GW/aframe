from typing import Callable

import torch
from bilby.core.prior import ConditionalPriorDict, Constraint
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from ledger.injections import BilbyParameterSet
from priors.utils import mass_constraints

from .generator import WaveformGenerator


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        duration: float,
        f_min: float,
        f_ref: float,
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
            duration:
                Duration of the waveform in seconds
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

        # For CBC generation, need to make sure that the mass ratio
        # does not exceed what ml4gw can handle.
        self.training_prior = ConditionalPriorDict(
            self.training_prior, conversion_function=mass_constraints
        )
        self.training_prior["mass_ratio"] = Constraint(0.02, 0.999)

        self.approximant = approximant
        self.f_ref = f_ref
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            duration,
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
        lal_params = parameter_set.convert_to_lal_param_set(self.f_ref)
        generation_params = lal_params.ml4gw_generation_params
        return generation_params

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        hc, hp = waveforms.transpose(1, 0)
        return hc.float(), hp.float()
