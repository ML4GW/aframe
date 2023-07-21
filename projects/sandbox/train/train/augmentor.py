from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
from train.data_structures import (
    ChannelMuter,
    ChannelSwapper,
    SignalInverter,
    SignalReverser,
)

import ml4gw.gw as gw
from ml4gw.utils.slicing import sample_kernels

if TYPE_CHECKING:
    from train.data_structures import SnrRescaler


class AframeBatchAugmentor(torch.nn.Module):
    def __init__(
        self,
        ifos: List[str],
        sample_rate: float,
        signal_prob: float,
        dec: Callable,
        psi: Callable,
        phi: Callable,
        psd_estimator: Callable,
        whitener: Callable,
        trigger_distance: float,
        mute_frac: float = 0.0,
        swap_frac: float = 0.0,
        snr: Optional[Callable] = None,
        rescaler: Optional["SnrRescaler"] = None,
        invert_prob: float = 0.5,
        reverse_prob: float = 0.5,
        **polarizations: np.ndarray,
    ):

        super().__init__()
        signal_prob = signal_prob / (
            1 - (swap_frac + mute_frac - (swap_frac * mute_frac))
        )

        if not 0 < signal_prob <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1. "
                "Adjust the value(s) of waveform_prob, "
                "swap_frac, mute_frac, and/or downweight"
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
        self.psd_estimator = psd_estimator
        self.whitener = whitener

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

    def sample_responses(self, N: int, kernel_size: int, psds: torch.Tensor):
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
            responses, _ = self.rescaler(responses, psds**0.5, target_snrs)

        kernels = sample_kernels(
            responses,
            kernel_size=kernel_size,
            max_center_offset=self.trigger_offset,
            coincident=True,
        )
        return kernels

    @torch.no_grad()
    def forward(self, X, y):
        X, psds = self.psd_estimator(X)

        # apply inversion / flip augementations
        X = self.inverter(X)
        X = self.reverser(X)

        # calculate number of waveforms to generate
        probs = torch.ones_like(y) * self.signal_prob
        rvs = torch.rand(size=X.shape[:1], device=probs.device)
        mask = rvs < probs[:, 0]

        # sample waveforms and use them to compute
        # interferometer responses
        N = mask.sum().item()
        responses = self.sample_responses(N, X.shape[-1], psds[mask])
        responses.to(X.device)

        # perform swapping and muting augmentations
        # on those responses, and then inject them
        print(responses.shape)
        responses, swap_indices = self.swapper(responses)
        print(responses.shape)
        responses, mute_indices = self.muter(responses)
        print(X.shape)
        X[mask] += responses

        # now that injections have been made,
        # whiten _all_ the strain using the
        # background psds computed up top
        X = self.whitener(X, psds)

        # set response augmentation labels to noise
        idx = torch.where(mask)[0]
        mask[idx[mute_indices]] = 0
        mask[idx[swap_indices]] = 0

        # set labels to positive for injected signals
        y[mask] = -y[mask] + 1

        # curriculum learning step
        if self.snr is not None:
            self.snr.step()

        return X, y
