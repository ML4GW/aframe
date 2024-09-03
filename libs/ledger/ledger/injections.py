from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, make_dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
from pycbc.waveform import get_td_waveform

from ledger.ledger import PATH, Ledger, metadata, parameter, waveform
from utils.cosmology import DEFAULT_COSMOLOGY


def chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


def transpose(d: Dict[str, List]):
    """Turn a dict of lists into a list of dicts"""
    return [dict(zip(d, col)) for col in zip(*d.values())]


@dataclass
class IntrinsicParameterSet(Ledger):
    """
    Easy to initialize with:
    params = prior.sample(N)
    params = IntrinsicParameterSet(**params)
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    a_1: np.ndarray = parameter()
    a_2: np.ndarray = parameter()
    tilt_1: np.ndarray = parameter()
    tilt_2: np.ndarray = parameter()
    azimuth_1: np.ndarray = parameter()
    azimuth_2: np.ndarray = parameter()

    @property
    def chirp_mass(self):
        return chirp_mass(self.mass_1, self.mass_2)

    @property
    def total_mass(self):
        return self.mass_1 + self.mass_2

    @property
    def mass_ratio(self):
        return self.mass_2 / self.mass_1


@dataclass
class ExtrinsicParameterSet(Ledger):
    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    redshift: np.ndarray = parameter()
    psi: np.ndarray = parameter()
    inclination: np.ndarray = parameter()
    phase: np.ndarray = parameter()

    @property
    def mass_1_source(self):
        return self.mass_1 / (1 + self.redshift)

    @property
    def mass_2_source(self):
        return self.mass_2 / (1 + self.redshift)

    @property
    def luminosity_distance(self, cosmology=DEFAULT_COSMOLOGY):
        return cosmology.luminosity_distance(self.redshift).value

    @property
    def spin1x(self):
        return self.a_1 * np.sin(self.tilt_1) * np.cos(self.azimuth_1)

    @property
    def spin2x(self):
        return self.a_2 * np.sin(self.tilt_2) * np.cos(self.azimuth_2)

    @property
    def spin1y(self):
        return self.a_1 * np.sin(self.tilt_1) * np.sin(self.azimuth_1)

    @property
    def spin2y(self):
        return self.a_2 * np.sin(self.tilt_2) * np.sin(self.azimuth_2)

    @property
    def spin1z(self):
        return self.a_1 * np.cos(self.tilt_1)

    @property
    def spin2z(self):
        return self.a_2 * np.cos(self.tilt_2)


@dataclass
class PycbcParameterSet(ExtrinsicParameterSet, IntrinsicParameterSet):
    def waveform_generation_params(self):
        # For clarity, explicitly define the parameter dictionary
        # needed for waveform generation
        params = {
            "mass1": self.mass_1,
            "mass2": self.mass_2,
            "spin1x": self.spin1x,
            "spin1y": self.spin1y,
            "spin1z": self.spin1z,
            "spin2x": self.spin2x,
            "spin2y": self.spin2y,
            "spin2z": self.spin2z,
            "inclination": self.inclination,
            "distance": self.luminosity_distance,
            "coa_phase": self.phase,
        }
        return params


@dataclass
class InjectionMetadata(Ledger):
    sample_rate: np.ndarray = metadata()
    duration: np.ndarray = metadata()
    coalescence_time: float = metadata()
    num_injections: int = metadata(default=0)

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
        return super().compare_metadata(key, ours, theirs)


@dataclass(frozen=True)
class _WaveformGenerator:
    """Thin wrapper so that we can potentially parallelize this"""

    waveform_approximant: str
    sample_rate: float
    waveform_duration: float
    coalescence_time: float
    minimum_frequency: float
    reference_frequency: float

    def shift_coalescence(self, waveform, t_final):
        shift_time = t_final - (self.waveform_duration - self.coalescence_time)
        shift_idx = int(shift_time * self.sample_rate)
        return np.roll(waveform, shift_idx, axis=-1)

    def align_waveforms(self, waveforms, t_final):
        waveform_length = int(self.sample_rate * self.waveform_duration)
        if waveforms.shape[-1] < waveform_length:
            pad = waveform_length - waveforms.shape[-1]
            waveforms = np.pad(waveforms, ((0, 0), (pad, 0)))
            waveforms = self.shift_coalescence(waveforms, t_final)
        elif waveforms.shape[-1] >= waveform_length:
            waveforms = self.shift_coalescence(waveforms, t_final)
            waveforms = waveforms[:, -waveform_length:]
        return waveforms

    def __call__(self, params):
        hp, hc = get_td_waveform(
            approximant=self.waveform_approximant,
            f_lower=self.minimum_frequency,
            f_ref=self.reference_frequency,
            delta_t=1 / self.sample_rate,
            **params
        )

        t_final = max(hp.sample_times.data)
        stacked = np.stack([hp.data, hc.data])
        stacked = self.align_waveforms(stacked, t_final)

        unstacked = {k: v for (k, v) in zip(["plus", "cross"], stacked)}

        return unstacked


@dataclass
class WaveformPolarizationSet(InjectionMetadata, PycbcParameterSet):
    cross: np.ndarray = waveform()
    plus: np.ndarray = waveform()

    @property
    def waveform_duration(self):
        return self.cross.shape[-1] / self.sample_rate

    def get_waveforms(self) -> np.ndarray:
        return np.stack([self.cross, self.plus], axis=-2)

    @classmethod
    def from_parameters(
        cls,
        params: PycbcParameterSet,
        minimum_frequency: float,
        reference_frequency: float,
        sample_rate: float,
        waveform_duration: float,
        waveform_approximant: str,
        coalescence_time: float,
        ex: Optional[Executor] = None,
    ):
        waveform_generator = _WaveformGenerator(
            waveform_approximant=waveform_approximant,
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            coalescence_time=coalescence_time,
            minimum_frequency=minimum_frequency,
            reference_frequency=reference_frequency,
        )

        waveform_length = int(sample_rate * waveform_duration)
        polarizations = {
            "plus": np.zeros((len(params), waveform_length)),
            "cross": np.zeros((len(params), waveform_length)),
        }

        generation_params = params.waveform_generation_params()
        param_list = transpose(generation_params)
        # give flexibility if we want to parallelize or not
        if ex is None:
            for i, polars in enumerate(map(waveform_generator, param_list)):
                for key, value in polars.items():
                    polarizations[key][i] = value
        else:
            futures = ex.map(waveform_generator, param_list)
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
        polarizations["num_injections"] = len(params)
        polarizations["coalescence_time"] = coalescence_time
        return cls(**polarizations)


@dataclass
class InjectionParameterSet(ExtrinsicParameterSet, IntrinsicParameterSet):
    snr: np.ndarray = parameter()
    ifo_snrs: np.ndarray = parameter()
    ifos: list[str] = metadata(default_factory=list)

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "ifos":
            # cast as list since hdf5 will store as numpy array
            ours = list(ours)
            theirs = list(theirs)
            if not ours:
                return theirs
            elif not theirs:
                return ours
            elif ours != theirs:
                raise ValueError(
                    "Incompatible ifos {} and {}".format(ours, theirs)
                )
            return ours
        return super().compare_metadata(key, ours, theirs)


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
