from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
from train.data_structures import (
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
        trigger_distance: float,
        snr: Optional[Callable] = None,
        rescaler: Optional["SnrRescaler"] = None,
        invert_prob: Optional[float] = 0.5,
        reverse_prob: Optional[float] = 0.5,
        **polarizations: np.ndarray,
    ):

        super().__init__()

        glitch_prob = glitch_sampler.prob
        self.glitch_sampler = glitch_sampler
        self.downweight = downweight
        signal_prob = signal_prob / (1 - glitch_prob * (1 - downweight)) ** 2
        signal_prob = signal_prob / (
            1 - (swap_frac + mute_frac - (swap_frac * mute_frac))
        )
        self.signal_prob = signal_prob
        self.trigger_offset = int(trigger_distance * sample_rate)
        self.sample_rate = sample_rate

        self.muter = ChannelMuter(frac=mute_frac)
        self.swapper = ChannelSwapper(frac=swap_frac)
        self.inverter = SignalInverter(invert_prob)
        self.reverser = SignalReverser(reverse_prob)

        self.dec = dec
        self.psi = psi
        self.phi = phi
        self.snr = snr
        self.rescaler = rescaler

        # store ifo geometries
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        # make sure we have the same number of waveforms
        # for all the different polarizations
        num_waveforms = None
        self.polarizations = {}
        for polarization, tensor in polarizations.items():
            if num_waveforms is not None and len(tensor) != num_waveforms:
                raise ValueError(
                    "Polarization {} has {} waveforms "
                    "associated with it, expected {}".format(
                        polarization, len(tensor), num_waveforms
                    )
                )
            elif num_waveforms is None:
                num_waveforms, _ = tensor.shape

            # don't register these as buffers since they could
            # be large and we don't necessarily want them on
            # the same device as everything else
            self.polarizations[polarization] = torch.Tensor(tensor)
        self.num_waveforms = num_waveforms

    def sample_responses(self, N: int, kernel_size: int):

        dec, psi, phi = self.dec(N), self.psi(N), self.phi(N)
        dec, psi, phi = (
            dec.to(self.tensors.device),
            psi.to(self.tensors.device),
            phi.to(self.tensors.device),
        )

        idx = torch.randperm(self.num_waveforms)[:N]
        polarizations = {}
        for polarization, waveforms in self.polarizations.items():
            waveforms = waveforms[idx]
            polarizations[polarization] = waveforms.to(dec.device)

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
            target_snrs = self.snr(N).to(responses.device)
            responses, _ = self.rescaler(responses, target_snrs)

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
        X = self.inverter(X)
        X = self.reverser(X)

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

        responses, swap_indices = self.swapper(responses)
        responses, mute_indices = self.muter(responses)
        X[mask] += responses

        # set response augmentation labels to noise
        mask[mask][mute_indices] = False
        mask[mask][swap_indices] = False

        # set labels to 1 for injected signals
        y[mask] = -y[mask] + 1

        # curriculum learning step
        self.snr.step()

        return X, y
