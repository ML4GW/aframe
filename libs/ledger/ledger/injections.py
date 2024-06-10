from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, make_dataclass
from typing import Optional

import h5py
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from ledger.ledger import PATH, Ledger, metadata, parameter, waveform


@dataclass
class IntrinsicParameterSet(Ledger):
    """
    Easy to initialize with:
    params = prior.sample(N)
    params = IntrinsicParameterSet(**params)
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    redshift: np.ndarray = parameter()
    psi: np.ndarray = parameter()
    a_1: np.ndarray = parameter()
    a_2: np.ndarray = parameter()
    tilt_1: np.ndarray = parameter()
    tilt_2: np.ndarray = parameter()
    phi_12: np.ndarray = parameter()
    phi_jl: np.ndarray = parameter()


@dataclass
class InjectionMetadata(Ledger):
    sample_rate: np.ndarray = metadata()
    duration: np.ndarray = metadata()
    num_injections: int = metadata(default=0)
    coalescence_time: float = metadata()

    def __post_init__(self):
        # verify that all waveforms have the appropriate duration
        super().__post_init__()
        if self.num_injections < self._length:
            raise ValueError(
                "{} has fewer total injections {} than "
                "number of waveforms {}".format(
                    self.__class__.__name__, self.num_injections, self._length
                )
            )
        if self.sample_rate is None and self._length > 0:
            raise ValueError(
                "Must specify sample rate if not "
                "initializing {} as empty container ".format(
                    self.__class__.__name__
                )
            )
        elif self.sample_rate is None or not self._length:
            return

        for key in self.waveform_fields:
            value = getattr(self, key)
            duration = value.shape[-1] / self.sample_rate
            if duration != self.duration:
                raise ValueError(
                    "Specified waveform duration of {} but "
                    "waveform '{}' has duration {}".format(
                        self.duration, key, duration
                    )
                )

    @property
    def waveform_fields(self):
        fields = self.__dataclass_fields__.items()
        fields = filter(lambda x: x[1].metadata["kind"] == "waveform", fields)
        return [i[0] for i in fields]

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "num_injections":
            if ours is None:
                return theirs
            elif theirs is None:
                return ours
            return ours + theirs
        return Ledger.compare_metadata(key, ours, theirs)


@dataclass(frozen=True)
class _WaveformGenerator:
    """Thin wrapper so that we can potentially parallelize this"""

    gen: WaveformGenerator
    sample_rate: float
    waveform_duration: float

    def center(self, waveform):
        dt = self.waveform_duration / 2
        return np.roll(waveform, int(dt * self.sample_rate))

    def __call__(self, params):
        polarizations = self.gen.time_domain_strain(params)

        # could think about stacking then unstacking to
        # make this more efficient
        for key in polarizations.keys():
            polarizations[key] = self.center(polarizations[key])
        return polarizations


@dataclass
class IntrinsicWaveformSet(InjectionMetadata, IntrinsicParameterSet):
    cross: np.ndarray = waveform
    plus: np.ndarray = waveform

    @property
    def waveform_duration(self):
        return self.cross.shape[-1] / self.sample_rate

    def get_waveforms(self) -> np.ndarray:
        return np.stack([self.cross, self.plus])

    @classmethod
    def from_parameters(
        cls,
        params: IntrinsicParameterSet,
        minimum_frequency: float,
        reference_frequency: float,
        sample_rate: float,
        waveform_duration: float,
        waveform_approximant: str,
        ex: Optional[Executor] = None,
    ):
        gen = WaveformGenerator(
            duration=waveform_duration,
            sampling_frequency=sample_rate,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": waveform_approximant,
                "reference_frequency": reference_frequency,
                "minimum_frequency": minimum_frequency,
            },
        )
        waveform_generator = _WaveformGenerator(
            gen, waveform_duration, sample_rate
        )

        waveform_length = int(sample_rate * waveform_duration)
        polarizations = {
            "plus": np.zeros((len(params), waveform_length)),
            "cross": np.zeros((len(params), waveform_length)),
        }

        # give flexibility if we want to parallelize or not
        if ex is None:
            for i, polars in enumerate(map(waveform_generator, params)):
                for key, value in polars.items():
                    polarizations[key][i] = value
        else:
            futures = ex.map(waveform_generator, params)
            idx_map = {f: i for f, i in zip(futures, len(futures))}
            for f in as_completed(futures):
                i = idx_map.pop(f)
                polars = f.result()
                for key, value in polars.items():
                    polarizations[key][i] = value

        d = {k: getattr(params, k) for k in params.__dataclass_fields__}
        polarizations.update(d)
        polarizations["sample_rate"] = sample_rate
        polarizations["duration"] = waveform_duration
        return cls(**polarizations)


@dataclass
class EventParameterSet(Ledger):
    """
    Assume GPS times always correspond to un-shifted data
    """

    injection_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()  # 2D with shift values along 1th axis
    snr: np.ndarray = parameter()

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def get_times(
        self, start: Optional[float] = None, end: Optional[float] = None
    ):
        if start is None and end is None:
            raise ValueError("Must specify one of start or end")

        mask = True
        if start is not None:
            mask &= self.injection_time >= start
        if end is not None:
            mask &= self.injection_time < end
        return self[mask]


@dataclass
class SkyLocationParameterSet(Ledger):
    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    theta_jn: np.ndarray = parameter()
    phase: np.ndarray = parameter()
    redshift: np.ndarray = parameter()


@dataclass
class InjectionParameterSet(SkyLocationParameterSet, IntrinsicParameterSet):
    snr: np.ndarray = parameter()


@dataclass
class WaveformSet(InjectionMetadata, InjectionParameterSet):
    """
    A set of waveforms projected onto a set of interferometers
    """

    def __post_init__(self):
        InjectionMetadata.__post_init__(self)
        self._waveforms = None

    @property
    def waveforms(self) -> np.ndarray:
        if self._waveforms is None:
            fields = sorted(self.waveform_fields)
            waveforms = [getattr(self, i) for i in fields]
            waveforms = np.stack(waveforms, axis=1)
            self._waveforms = waveforms
        return self._waveforms

    def num_waveform_fields(self):
        return len(self.waveform_fields)


# TODO: rename this to InjectionCampaign


# note, dataclass inheritance goes from last to first,
# so the ordering of kwargs here would be:
# mass1, mass2, ..., ra, dec, psi, injection_time, shift, sample_rate, h1, l1
@dataclass
class InterferometerResponseSet(WaveformSet):
    """
    Represents a set of projected waveforms to be used in an injection campaign
    """

    injection_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()

    @classmethod
    def _raise_bad_shift_dim(cls, fname, dim1, dim2):
        raise ValueError(
            "Specified shifts with {} dimensions, but "
            "{} from file {} has {} dimensions".format(
                dim1, cls.__name__, fname, dim2
            )
        )

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def get_times(
        self, start: Optional[float] = None, end: Optional[float] = None
    ):
        if start is None and end is None:
            raise ValueError("Must specify one of start or end")

        mask = True
        if start is not None:
            mask &= self.injection_time >= start
        if end is not None:
            mask &= self.injection_time < end
        return self[mask]

    @classmethod
    def read(
        cls,
        fname: PATH,
        start: Optional[float] = None,
        end: Optional[float] = None,
        shifts: Optional[float] = None,
    ):
        """
        Similar wildcard behavior in loading. Additional
        kwargs to be able to load data for just a particular
        segment since the slice method below will make copies
        and you could start to run out of memory fast.
        """
        with h5py.File(fname, "r") as f:
            if all([i is None for i in [start, end, shifts]]):
                return cls._load_with_idx(f, None)

            coalescence_time = f.attrs["coalescence_time"]
            times = f["parameters"]["injection_time"][:]

            mask = True
            if start is not None:
                mask &= (times + coalescence_time) >= start
            if end is not None:
                mask &= (times - coalescence_time) <= end
            if shifts is not None:
                shifts = np.array(shifts)
                ndim = shifts.ndim

                fshifts = f["parameters"]["shift"][:]
                f_ndim = fshifts.ndim
                if f_ndim == 2:
                    if ndim == 1:
                        shifts = shifts[None]
                    elif ndim != 2:
                        cls._raise_bad_shift_dim(fname, ndim, f_ndim)
                elif f_ndim == 1:
                    if ndim != 1:
                        cls._raise_bad_shift_dim(fname, ndim, f_ndim)
                    fshifts = fshifts[:, None]
                    shifts = shifts[:, None]
                else:
                    cls._raise_bad_shift_dim(fname, ndim, f_ndim)

                if fshifts.shape[-1] != shifts.shape[-1]:
                    raise ValueError(
                        "Specified {} shifts when {} ifos "
                        "are present in {} {}".format(
                            shifts.shape[-1],
                            fshifts.shape[-1],
                            cls.__name__,
                            fname,
                        )
                    )

                shift_mask = False
                for shift in shifts:
                    shift_mask |= (fshifts == shift).all(axis=-1)
                mask &= shift_mask

            idx = np.where(mask)[0]
            return cls._load_with_idx(f, idx)

    def inject(self, x: np.ndarray, start: float):
        """
        Inject waveforms into background array with
        initial timestamp `start`
        """
        stop = start + x.shape[-1] / self.sample_rate
        mask = self.injection_time >= (start - self.coalescence_time)
        mask &= self.injection_time <= (stop + self.coalescence_time)

        if not mask.any():
            return x

        times = self.injection_time[mask]
        waveforms = self.waveforms[mask]

        # potentially pad x to inject waveforms
        # that fall over the boundaries of chunks
        pad = []
        earliest = (times - self.coalescence_time - start).min()
        if earliest < 0:
            num_early = int(-earliest * self.sample_rate)
            pad.append(num_early)
            start += earliest
        else:
            pad.append(0)

        latest = (times + self.coalescence_time - stop).max()
        if latest > 0:
            num_late = int(latest * self.sample_rate)
            pad.append(num_late)
        else:
            pad.append(0)

        if any(pad):
            x = np.pad(x, [(0, 0)] + [tuple(pad)])
        times = times - start

        # create matrix of indices of waveform_size for each waveform
        waveforms = waveforms.transpose((1, 0, 2))
        _, num_waveforms, waveform_size = waveforms.shape
        coalescence_time_idx = int(self.coalescence_time * self.sample_rate)

        idx = np.arange(waveform_size) - coalescence_time_idx
        idx = idx[None]
        idx = np.repeat(idx, num_waveforms, axis=0)

        # offset the indices of each waveform
        # according to their time offset
        idx_diffs = (times * self.sample_rate).astype("int64")
        idx += idx_diffs[:, None]

        # flatten these indices and the signals out
        # to 1D and then add them in-place all at once
        idx = idx.reshape(-1)
        waveforms = waveforms.reshape(len(self.waveform_fields), -1)
        x[:, idx] += waveforms
        if any(pad):
            start, stop = pad
            stop = -stop or None
            x = x[:, start:stop]
        return x


def waveform_class_factory(ifos: list[str], base_cls, cls_name: str):
    """
    Factory function for creating ledger
    dataclasses with arbitrary waveform fields

    Args:
        ifos:
            List of interferometers for which waveform
            fields will be populated
        base_cls:
            Base class the resulting dataclass will inherit from
        cls_name:
            Name of resulting dataclass

    """
    ifos = [ifo.lower() for ifo in ifos]
    fields = [(ifo, waveform()) for ifo in ifos]
    fields = [(name, field.type, field) for name, field in fields]
    cls = make_dataclass(cls_name, fields, bases=(base_cls,))
    return cls
