from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
from data_structures import (
    ChannelMuter,
    ChannelSwapper,
    GlitchSampler,
    SignalInverter,
    SignalReverser,
)

import ml4gw.gw as gw
from ml4gw.utils.slicing import sample_kernels

if TYPE_CHECKING:
    from data_structures import SnrRescaler


class BBHNetBatchAugmentor(torch.nn.Module):
    def __init__(
        self,
        ifos: List[str],
        sample_rate: float,
        mute_frac: float,
        swap_frac: float,
        downweight: float,
        signal_prob: float,
        glitch_sampler: GlitchSampler,
        dec: Callable,
        psi: Callable,
        phi: Callable,
        trigger_offset: float,
        snr: Optional[Callable] = None,
        rescaler: Optional["SnrRescaler"] = None,
        inverter_prob: Optional[float] = 0.5,
        reverser_prob: Optional[float] = 0.5,
        **polarizations: np.ndarray,
    ):

        super().__init__()
        glitch_prob = glitch_sampler.prob
        signal_prob = signal_prob / (1 - glitch_prob * (1 - downweight)) ** 2
        signal_prob = signal_prob / (
            1 - (swap_frac + mute_frac - (swap_frac * mute_frac))
        )
        self.signal_prob = signal_prob
        self.trigger_offset = trigger_offset
        self.sample_rate = sample_rate
        self.muter = ChannelMuter(frac=mute_frac)
        self.swapper = ChannelSwapper(frac=swap_frac)

        self.signal_inverter = SignalInverter(inverter_prob)
        self.signal_reverser = SignalReverser(reverser_prob)
        self.dec = dec
        self.psi = psi
        self.phi = phi
        self.snr = snr
        self.rescaler = rescaler

        # store ifo geometries
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

    def sample_responses(self, N: int, kernel_size: int):
        idx = torch.randperm(self.num_waveforms)[:N]
        polarizations = {}
        for polarization, waveforms in self.polarizations.items():
            waveforms = waveforms[idx]
            polarizations[polarization] = waveforms

        dec, psi, phi = self.dec(N), self.psi(N), self.phi(N)
        responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        if self.rescaler is not None:
            target_snrs = self.snr(N)
            responses = self.rescaler(responses, target_snrs)

        kernels = sample_kernels(
            responses,
            kernel_size=kernel_size,
            max_center_offset=self.trigger_offset,
            coincident=True,
        )
        return kernels

    def forward(self, X, y):
        # insert glitches and apply inversion / flip augementations
        X, y = self.glitch_sampler(X, y)
        X = self.signal_inverter(X)
        X = self.signal_reverser(X)

        # calculate number of waveforms to generate
        # based on waveform prob, mute prob, and swap prob and downweight
        # likelihood of injecting a signal on top of a glitch.
        # y == -2 means one glitch, y == -6 means two
        probs = torch.ones_like(y) * self.signal_prob
        probs[y < 0] *= self.downweight
        probs[y < -4] *= self.downweight
        rvs = torch.rand(size=X.shape[:1], device=probs.device)
        mask = rvs < probs[:, 0]

        # sample the desired number of responses and inject them
        N = mask.sum().item()
        responses = self.sample_responses(N, X.shape[-1])
        responses.to(X.device)

        responses, swap_indices = self.channel_swapper(responses)
        waveforms, mute_indices = self.channel_muter(responses)
        X[mask] += waveforms

        # set response augmentation labels to noise
        mask[mask][mute_indices] = False
        mask[mask][swap_indices] = False

        # curriculum learning step
        self.snr.step()

        return X, y
