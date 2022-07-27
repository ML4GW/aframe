from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
from bilby.core.prior import Cosine, PriorDict, Uniform
from gwpy.timeseries import TimeSeries

from bbhnet.data.transforms.whitening import DEFAULT_FFTLENGTH
from bbhnet.data.utils import sample_kernels
from bbhnet.injection import project_raw_gw

PRIORS = PriorDict(
    {
        "ra": Uniform(minimum=0, maximum=2 * np.pi),
        "dec": Cosine(),
        "psi": Uniform(minimum=0, maximum=np.pi),
    }
)


@dataclass
class _DummyWaveformGenerator:
    sampling_frequency: float
    duration: float


class WaveformSampler:
    def __init__(
        self,
        dataset: str,
        sample_rate: float,
        min_snr: float,
        max_snr: float,
        t0: float = 1262588390,
        duration: float = 31536000,
        highpass: Optional[float] = 20,
        deterministic: bool = False,
        frac: Optional[float] = None,
    ):
        if max_snr <= min_snr:
            raise ValueError(
                f"max_snr {max_snr} must be greater than min_snr {min_snr}"
            )
        self.min_snr = min_snr
        self.max_snr = max_snr

        with h5py.File(dataset, "r") as f:
            # TODO: enforce that the sample rates match? Or resample?
            self.waveforms = f["signals"][:]

            if frac is not None:
                num_waveforms = int(frac * len(self.waveforms))
                if frac < 0:
                    self.waveforms = self.waveforms[num_waveforms:]
                else:
                    self.waveforms = self.waveforms[:num_waveforms]

            if deterministic:
                # load any sampled extrinsic parameters
                # associated with the waveform
                self.priors = {}
                for param in list(PRIORS.keys()):
                    try:
                        value = f[param][:]
                    except KeyError:
                        raise ValueError(
                            "Specified deterministic waveform sampling, "
                            "but waveform file {} is missing values for "
                            "extrinsic parameter {}".format(dataset, param)
                        )

                    if frac is not None and frac < 0:
                        value = value[num_waveforms:]
                    elif frac is not None:
                        value = value[:num_waveforms]
                    self.priors[param] = value

                try:
                    value = f["geocent_time"][:]
                except KeyError:
                    value = np.linspace(t0, t0 + duration, len(self.waveforms))
                self.priors["geocent_time"] = value

        if not deterministic:
            # if we're not sampling deterministically, set up
            # a dict of priors for our extrinsic parameters
            # to sample from at data loading time
            self.priors = PRIORS.copy()
            tf = t0 + duration
            self.priors["geocent_time"] = Uniform(minimum=t0, maximum=tf)

        self.deterministic = deterministic
        self.df = sample_rate / self.waveforms.shape[-1]
        self.sample_rate = sample_rate

        # compute a mask for deciding which frequency bins
        # ought to contribute to a signals SNR computation
        freqs = np.arange(0, sample_rate // 2 + self.df, self.df)
        highpass = highpass or 0
        self.mask = freqs >= highpass

        # initialize some attributes that need
        # to be fit to a particular background
        self.background_asd = self.ifos = None

    def fit(self, **backgrounds: torch.Tensor):
        """Provide a background against which to reweight sampled SNRs"""

        # ** is order preserving, so all this is deterministic
        asds, ifos = [], []
        for ifo, x in backgrounds.items():
            x = x.cpu().numpy()

            # TODO: do we know that they'll have the same sample rate?
            # Speaks to a larger question about the best way to create
            # ASDs from torch tensors (and possibly numpy arrays)
            ts = TimeSeries(x, dt=1 / self.sample_rate)
            asd = (
                ts.asd(
                    fftlength=DEFAULT_FFTLENGTH,
                    window="hanning",
                    method="median",
                )
                .interpolate(self.df)
                .value
            )

            # for now, make sure our background asds don't have 0 values
            if (asd == 0).any():
                raise ValueError(f"Found 0 values in asd for IFO {ifo}")

            asds.append(asd)
            ifos.append(ifo)

        self.background_asd = np.stack(asds)
        self.ifos = ifos

    def compute_snrs(self, signals: np.ndarray) -> np.ndarray:
        ffts = np.fft.rfft(signals, axis=-1) / self.sample_rate
        snrs = 2 * np.abs(ffts) / self.background_asd
        snrs = snrs**2 * self.df * self.mask
        return snrs.sum(axis=-1) ** 0.5

    def reweight_snrs(self, signals: np.ndarray) -> np.ndarray:
        snrs = self.compute_snrs(signals)
        snrs = (snrs**2).sum(axis=1) ** 0.5

        if self.deterministic:
            # in the deterministic case, map everything to
            # the geometric mean of the max and min snrs
            target_snrs = np.ones((len(signals),))
            target_snrs *= (self.max_snr * self.min_snr) ** 0.5
        else:
            # otherwise uniformly sample target snr values
            target_snrs = np.random.uniform(
                self.min_snr, self.max_snr, size=len(snrs)
            )

        weights = target_snrs / snrs
        signals = signals.transpose(1, 2, 0) * weights
        return signals.transpose(2, 0, 1)

    def sample(self, N: int, size: int, offset: int = 0) -> np.ndarray:
        if self.background_asd is None:
            raise RuntimeError(
                "Must fit WaveformGenerator to background asd before sampling"
            )

        if self.deterministic:
            # grab the first N waveforms and their extrinsic
            # parameters if we're sampling deterministically
            N = len(self.waveforms) if N == -1 else N
            idx = np.arange(N)

            sample_params = {}
            for param, values in self.priors.items():
                sample_params[param] = values[:N]

        else:
            # sample some waveform indices to inject as well
            # as sky localization parameters for computing
            # the antenna response in real-time
            idx = np.random.choice(len(self.waveforms), size=N, replace=False)
            sample_params = self.priors.sample(N)

        # initialize the output array and a dummy object
        # which has a couple attributes expected by the
        # argument passed to project_raw_gw
        # TODO: project_raw_gw should accept these arguments on their own
        signals = np.zeros((N, len(self.ifos), self.waveforms.shape[-1]))
        waveform_generator = _DummyWaveformGenerator(
            self.sample_rate, self.waveforms.shape[-1] // self.sample_rate
        )

        # for each one of the interferometers used in
        # the background asds passed to `.fit`, compute
        # its response to the waveform given the sky
        # localization parameters
        for i, ifo in enumerate(self.ifos):
            signal = project_raw_gw(
                self.waveforms[idx],
                sample_params,
                waveform_generator,
                ifo,
                get_snr=False,
            )
            signals[:, i] = signal

        # scale the amplitudes of the signals so that
        # their RMS SNR falls in an acceptable range
        signals = self.reweight_snrs(signals)

        if self.deterministic:
            # if we're sampling kernels deterministically,
            # just place them in the center of the kernel
            # plus some indicated offset
            center = int(signals.shape[-1] // 2)
            left = int(center + offset - size // 2)
            right = int(left + size)
            return signals[:, :, left:right]
        else:
            # otherwise randomly sample kernels from these signals
            return sample_kernels(signals, size, offset)
