from concurrent.futures import Executor, as_completed
from dataclasses import dataclass, make_dataclass

import h5py
import numpy as np
from astropy.cosmology import z_at_value
from astropy.cosmology import Cosmology
from astropy.units import Mpc
from lalsimulation import (
    SimInspiralTransformPrecessingNewInitialConditions,
    SimInspiralTransformPrecessingWvf2PE,
)
from pycbc.waveform import get_td_waveform

from ledger.ledger import PATH, Ledger, metadata, parameter, waveform
from utils.cosmology import DEFAULT_COSMOLOGY

# Solar mass in kg
MSUN = 1.988409902147041637325262574352366540e30


def chirp_mass(
    m1: float | np.ndarray, m2: float | np.ndarray
) -> float | np.ndarray:
    """Calculate chirp mass from component masses.

    Args:
        m1: Primary mass (scalar or array).
        m2: Secondary mass (scalar or array).

    Returns:
        Chirp mass with same shape as inputs.
    """
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


def transpose(d: dict[str, list]):
    """Convert dictionary of lists to list of dictionaries.

    Transposes data structure so each row becomes a dictionary entry
    keyed by the original dictionary keys.

    Args:
        d: Dictionary with string keys and list values of equal length.

    Returns:
        List of dictionaries, one per row of input data.
    """
    return [dict(zip(d, col)) for col in zip(*d.values())]


@dataclass
class IntrinsicParameterSet(Ledger):
    """A set of intrinsic parameters for a binary system.

    Stores the intrinsic parameters of a binary system that determine
    its waveform. Can be easily initialized from bilby prior samples.

    Attributes:
        mass_1: Primary component mass in solar masses.
        mass_2: Secondary component mass in solar masses.
        a_1: Primary spin magnitude.
        a_2: Secondary spin magnitude.
        tilt_1: Primary tilt angle (rad).
        tilt_2: Secondary tilt angle (rad).
        phi_12: Azimuthal angle between spins (rad).
        phi_jl: Azimuthal angle between total angular momentum and orbital
                angular momentum (rad).
    """

    mass_1: np.ndarray = parameter()
    mass_2: np.ndarray = parameter()
    a_1: np.ndarray = parameter()
    a_2: np.ndarray = parameter()
    tilt_1: np.ndarray = parameter()
    tilt_2: np.ndarray = parameter()
    phi_12: np.ndarray = parameter()
    phi_jl: np.ndarray = parameter()

    @property
    def chirp_mass(self):
        """Compute chirp mass from component masses.

        Returns:
            Chirp mass array.
        """
        return chirp_mass(self.mass_1, self.mass_2)

    @property
    def total_mass(self):
        """Compute total mass from component masses.

        Returns:
            Total mass array.
        """
        return self.mass_1 + self.mass_2

    @property
    def mass_ratio(self):
        """Compute mass ratio (mass_2 / mass_1).

        Returns:
            Mass ratio array.
        """
        return self.mass_2 / self.mass_1


@dataclass
class ExtrinsicParameterSet(Ledger):
    """A set of extrinsic parameters for a binary system.

    Stores the extrinsic parameters that determine the detector response
    but not the waveform shape.

    Attributes:
        ra: Right ascension sky coordinate (rad).
        dec: Declination sky coordinate (rad).
        redshift: Source redshift.
        psi: Polarization angle (rad).
        theta_jn: Inclination angle (rad).
        phase: Phase at reference frequency (rad).
    """

    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    redshift: np.ndarray = parameter()
    psi: np.ndarray = parameter()
    theta_jn: np.ndarray = parameter()
    phase: np.ndarray = parameter()

    @property
    def mass_1_source(self):
        """Convert mass_1 from detector frame to source frame.

        Returns:
            Source frame mass_1.
        """
        return self.mass_1 / (1 + self.redshift)

    @property
    def mass_2_source(self):
        """Convert mass_2 from detector frame to source frame.

        Returns:
            Source frame mass_2.
        """
        return self.mass_2 / (1 + self.redshift)

    @property
    def luminosity_distance(self, cosmology: Cosmology = DEFAULT_COSMOLOGY):
        """Calculate luminosity distance from redshift using cosmology.

        Args:
            cosmology: Astropy cosmology to use. Defaults to DEFAULT_COSMOLOGY.

        Returns:
            Luminosity distance in Mpc.
        """
        return cosmology.luminosity_distance(self.redshift).value


@dataclass
class BilbyParameterSet(ExtrinsicParameterSet, IntrinsicParameterSet):
    """Combined intrinsic and extrinsic parameters with Bilby convention.

    Stores both intrinsic (masses, spins) and extrinsic (sky location,
    phase) parameters in the format used by Bilby.
    """

    def convert_to_lal_param_set(self, reference_frequency: float):
        """Convert Bilby parameters to LAL convention.

        Converts spin parameters from angle/magnitude convention used by Bilby
        to Cartesian components used by LAL.

        Args:
            reference_frequency: Reference frequency for spin transformation.

        Returns:
            LALParameterSet with converted parameters.
        """
        mass_1_si = self.mass_1 * MSUN
        mass_2_si = self.mass_2 * MSUN
        inclination = np.zeros(len(self))
        spin1x = np.zeros(len(self))
        spin1y = np.zeros(len(self))
        spin1z = np.zeros(len(self))
        spin2x = np.zeros(len(self))
        spin2y = np.zeros(len(self))
        spin2z = np.zeros(len(self))

        for i in range(len(self)):
            (
                inclination[i],
                spin1x[i],
                spin1y[i],
                spin1z[i],
                spin2x[i],
                spin2y[i],
                spin2z[i],
            ) = SimInspiralTransformPrecessingNewInitialConditions(
                self.theta_jn[i],
                self.phi_jl[i],
                self.tilt_1[i],
                self.tilt_2[i],
                self.phi_12[i],
                self.a_1[i],
                self.a_2[i],
                mass_1_si[i],
                mass_2_si[i],
                reference_frequency,
                self.phase[i],
            )

        return LALParameterSet(
            mass1=self.mass_1,
            mass2=self.mass_2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=spin1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=spin2z,
            inclination=inclination,
            luminosity_distance=self.luminosity_distance,
            phase=self.phase,
            ra=self.ra,
            dec=self.dec,
            psi=self.psi,
        )


@dataclass
class LALParameterSet(Ledger):
    """Binary parameters with LAL convention.

    Stores parameters in the format expected by PyCBC's get_td_waveform
    function, with Cartesian spin components.

    Attributes:
        mass1: Primary component mass in solar masses.
        mass2: Secondary component mass in solar masses.
        spin1x, spin1y, spin1z: Primary spin Cartesian components.
        spin2x, spin2y, spin2z: Secondary spin Cartesian components.
        inclination: Orbital inclination angle (rad).
        luminosity_distance: Luminosity distance in Mpc.
        phase: Phase at reference frequency (rad).
        ra: Right ascension (rad).
        dec: Declination (rad).
        psi: Polarization angle (rad).
    """

    # No underscores in masses because of format PyCBC expects
    mass1: np.ndarray = parameter()
    mass2: np.ndarray = parameter()
    spin1x: np.ndarray = parameter()
    spin1y: np.ndarray = parameter()
    spin1z: np.ndarray = parameter()
    spin2x: np.ndarray = parameter()
    spin2y: np.ndarray = parameter()
    spin2z: np.ndarray = parameter()
    inclination: np.ndarray = parameter()
    luminosity_distance: np.ndarray = parameter()
    phase: np.ndarray = parameter()
    ra: np.ndarray = parameter()
    dec: np.ndarray = parameter()
    psi: np.ndarray = parameter()

    @property
    def redshift(self, cosmology: Cosmology = DEFAULT_COSMOLOGY):
        """Calculate redshift from luminosity distance using cosmology.

        Args:
            cosmology: Astropy cosmology to use. Defaults to DEFAULT_COSMOLOGY.

        Returns:
            Redshift array.
        """
        return z_at_value(
            cosmology.luminosity_distance, self.luminosity_distance * Mpc
        ).value

    @property
    def generation_params(self):
        """Format parameters for waveform generation with PyCBC.

        Returns:
            Dictionary with parameters in PyCBC format for get_td_waveform.
        """
        params = {
            "mass1": self.mass1,
            "mass2": self.mass2,
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

    def convert_to_bilby_param_set(self, reference_frequency: float):
        """Convert LAL parameters to Bilby format.

        Converts spin parameters from Cartesian components in LAL format
        to angle/magnitude convention used by Bilby.

        Args:
            reference_frequency: Reference frequency for spin transformation.

        Returns:
            BilbyParameterSet with converted parameters.
        """
        theta_jn = np.zeros(len(self))
        phi_jl = np.zeros(len(self))
        tilt_1 = np.zeros(len(self))
        tilt_2 = np.zeros(len(self))
        phi_12 = np.zeros(len(self))
        a_1 = np.zeros(len(self))
        a_2 = np.zeros(len(self))

        for i in range(len(self)):
            (
                theta_jn[i],
                phi_jl[i],
                tilt_1[i],
                tilt_2[i],
                phi_12[i],
                a_1[i],
                a_2[i],
            ) = SimInspiralTransformPrecessingWvf2PE(
                self.inclination[i],
                self.spin1x[i],
                self.spin1y[i],
                self.spin1z[i],
                self.spin2x[i],
                self.spin2y[i],
                self.spin2z[i],
                self.mass1[i],
                self.mass2[i],
                reference_frequency,
                self.phase[i],
            )

        # When the spin magnitude is 0, the conversion function sets
        # the tilts to pi/2. To me, 0 is a more sensible value
        tilt_1[a_1 == 0] = 0
        tilt_2[a_2 == 0] = 0

        return BilbyParameterSet(
            mass_1=self.mass1,
            mass_2=self.mass2,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
            phi_jl=phi_jl,
            ra=self.ra,
            dec=self.dec,
            redshift=self.redshift,
            psi=self.psi,
            theta_jn=theta_jn,
            phase=self.phase,
        )


@dataclass
class InjectionMetadata(Ledger):
    """Metadata for sets of generated waveforms.

    Stores metadata about a set of injected waveforms including sampling
    parameters and injection configuration.

    Attributes:
        sample_rate: Waveform sampling rate in Hz.
        duration: Waveform duration in seconds.
        right_pad: Time from coalescence to right edge in seconds.
        num_injections: Total number of injections generated.
    """

    sample_rate: np.ndarray = metadata()
    duration: np.ndarray = metadata()
    right_pad: float = metadata()
    num_injections: int = metadata(default=0)

    def __post_init__(self):
        """Validate metadata consistency for waveforms.

        Verifies that:
        - All waveforms have the specified duration
        - num_injections >= number of waveforms
        - Sample rate is specified if waveforms are present

        Raises:
            ValueError: If validation checks fail.
        """
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
        """Get list of field names that contain waveform data.

        Returns:
            List of waveform field names.
        """
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
    """Thin wrapper for parallelizable waveform generation.

    Provides callable interface for generating time-domain waveforms
    with specified parameters, designed to work with multiprocessing.

    Attributes:
        waveform_approximant: Waveform approximant name.
        sample_rate: Sampling rate in Hz.
        waveform_duration: Total waveform duration in seconds.
        right_pad: Padding from coalescence to right edge in seconds.
        minimum_frequency: Minimum frequency for waveform in Hz.
        reference_frequency: Reference frequency in Hz.
    """

    waveform_approximant: str
    sample_rate: float
    waveform_duration: float
    right_pad: float
    minimum_frequency: float
    reference_frequency: float

    def shift_coalescence(self, waveforms: np.ndarray, t_final: float):
        """Shift waveform so coalescence is right_pad seconds from edge.

        Rolls the waveform array so the coalescence point (originally at
        t=0 from PyCBC) is positioned right_pad seconds from the right edge.

        Args:
            waveforms: Stacked polarizations (2, num_samples) from PyCBC.
            t_final: Ending time of signal array relative to coalescence.

        Returns:
            Shifted waveform array with same shape as input.
        """
        shift_time = t_final - self.right_pad + 1 / self.sample_rate
        shift_idx = int(shift_time * self.sample_rate)
        return np.roll(waveforms, shift_idx, axis=-1)

    def align_waveforms(self, waveforms: np.ndarray, t_final: float):
        """Pad or crop waveform to desired length with correct merger time.

        Adjusts waveforms to the desired duration with coalescence point
        at the correct position. Waveforms from PyCBC have variable lengths
        depending on parameters, so this function pads or crops as needed.
        Shorter waveforms are zero-padded on the left then shifted, while
        longer waveforms are shifted then cropped from the right.

        Args:
            waveforms: Stacked polarizations (2, num_samples) from PyCBC.
            t_final: Ending time of signal array relative to coalescence.

        Returns:
            Waveform array with shape (2, waveform_duration*sample_rate).
        """
        waveform_length = int(self.sample_rate * self.waveform_duration)
        if waveforms.shape[-1] < waveform_length:
            pad = waveform_length - waveforms.shape[-1]
            waveforms = np.pad(waveforms, ((0, 0), (pad, 0)))
            waveforms = self.shift_coalescence(waveforms, t_final)
        elif waveforms.shape[-1] >= waveform_length:
            waveforms = self.shift_coalescence(waveforms, t_final)
            waveforms = waveforms[:, -waveform_length:]
        return waveforms

    def __call__(self, params: dict[str, float]):
        """Generate waveform for given parameters.

        Generates a time-domain waveform using PyCBC's get_td_waveform
        with the specified generator parameters and input parameters.
        The mimimum frequency is automatically adjusted if it exceeds
        the frequency limit for the given masses.

        Args:
            params: Dictionary of waveform parameters from generation_params.

        Returns:
            Dictionary with 'plus' and 'cross' polarization arrays.
        """
        # https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/transforms/detection/waveform.py?ref_type=heads#L112 # noqa
        freq_limit = 1899.0 / (params["mass1"] + params["mass2"])
        if self.minimum_frequency > freq_limit:
            minimum_frequency = freq_limit
            reference_frequency = freq_limit
        else:
            minimum_frequency = self.minimum_frequency
            reference_frequency = self.reference_frequency

        hp, hc = get_td_waveform(
            approximant=self.waveform_approximant,
            f_lower=minimum_frequency,
            f_ref=reference_frequency,
            delta_t=1 / self.sample_rate,
            **params,
        )

        t_final = hp.sample_times.data[-1]
        stacked = np.stack([hp.data, hc.data])
        stacked = self.align_waveforms(stacked, t_final)

        unstacked = dict(zip(["plus", "cross"], stacked))

        return unstacked


@dataclass
class WaveformPolarizationSet(InjectionMetadata, BilbyParameterSet):
    """A set of generated waveform polarizations.

    Stores the plus and cross polarizations of generated waveforms
    along with their parameters and metadata.

    Attributes:
        cross: Cross polarization waveform array.
        plus: Plus polarization waveform array.
    """

    cross: np.ndarray = waveform()
    plus: np.ndarray = waveform()

    @property
    def waveform_duration(self):
        """Calculate waveform duration from array length and sample rate."""
        return self.cross.shape[-1] / self.sample_rate

    def get_waveforms(self) -> np.ndarray:
        """Get stacked waveforms of shape (num_injections, 2, num_samples)."""
        return np.stack([self.cross, self.plus], axis=-2)

    @classmethod
    def from_parameters(
        cls,
        params: BilbyParameterSet,
        minimum_frequency: float,
        reference_frequency: float,
        sample_rate: float,
        waveform_duration: float,
        waveform_approximant: str,
        right_pad: float,
        ex: Executor | None = None,
    ):
        """Generate waveforms from parameters using PyCBC.

        Creates a WaveformPolarizationSet by generating time-domain waveforms
        for each set of parameters using PyCBC's get_td_waveform.

        Args:
            params: BilbyParameterSet with injection parameters.
            minimum_frequency: Minimum frequency in Hz for waveform generation.
            reference_frequency: Reference frequency in Hz.
            sample_rate: Sampling rate in Hz.
            waveform_duration: Total waveform duration in seconds.
            waveform_approximant: Waveform approximant name.
            right_pad: Padding from coalescence to right edge in seconds.
            ex: Optional Executor for parallel waveform generation.

        Returns:
            WaveformPolarizationSet with generated waveforms.

        Raises:
            ValueError: If right_pad >= waveform_duration.
        """
        if waveform_duration < right_pad:
            raise ValueError(
                "Right padding must be less than waveform duration; "
                f"got values of {right_pad} and {waveform_duration}"
            )

        waveform_generator = _WaveformGenerator(
            waveform_approximant=waveform_approximant,
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            right_pad=right_pad,
            minimum_frequency=minimum_frequency,
            reference_frequency=reference_frequency,
        )

        waveform_length = int(sample_rate * waveform_duration)
        polarizations = {
            "plus": np.zeros((len(params), waveform_length)),
            "cross": np.zeros((len(params), waveform_length)),
        }

        lal_params = params.convert_to_lal_param_set(reference_frequency)
        param_list = transpose(lal_params.generation_params)
        # give flexibility if we want to parallelize or not
        if ex is None:
            for i, polars in enumerate(map(waveform_generator, param_list)):
                for key, value in polars.items():
                    polarizations[key][i] = value
        else:
            futures = ex.map(waveform_generator, param_list)
            idx_map = dict(zip(futures, len(futures)))
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
        polarizations["right_pad"] = right_pad
        return cls(**polarizations)


@dataclass
class InjectionParameterSet(ExtrinsicParameterSet, IntrinsicParameterSet):
    """Intrinsic and extrinsic parameters for injection campaign.

    Combines extrinsic and intrinsic parameters along with SNR information
    for injections into detector data.

    Attributes:
        snr: Overall signal-to-noise ratio.
        ifo_snrs: SNR in each individual interferometer.
        ifos: List of interferometer names.
    """

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
    """A set of waveform polarizations along with their metadata.

    Stores both polarizations of gravitational wave strain for injections
    along with their parameters, SNRs, and metadata.
    """

    def __post_init__(self):
        InjectionMetadata.__post_init__(self)
        self._waveforms = None

    @property
    def waveforms(self) -> np.ndarray:
        """Get stacked waveform array (cached after first access).

        Returns:
            Array of shape (num_injections, num_polarizations, num_samples).
        """
        if self._waveforms is None:
            fields = sorted(self.waveform_fields)
            waveforms = [getattr(self, i) for i in fields]
            shape = (
                waveforms[0].shape[0],
                len(fields),
                waveforms[0].shape[-1],
            )
            self._waveforms = np.zeros(shape, dtype=np.float32)
            for i, field in enumerate(fields):
                self._waveforms[:, i, :] = getattr(self, field)
        return self._waveforms

    def num_waveform_fields(self):
        """Get number of waveform fields in this set.

        Returns:
            Number of waveform polarizations.
        """
        return len(self.waveform_fields)


# TODO: rename this to InjectionCampaign
@dataclass
class InterferometerResponseSet(WaveformSet):
    """Waveforms projected onto specific interferometers.

    Represents a set of projected waveforms to be used in an injection
    campaign, along with the times and shifts for each injection.

    Note:
        Dataclass inheritance order (last to first) determines field ordering:
        mass1, mass2, ..., ra, dec, psi, injection_time, shift, sample_rate,
        h1, l1
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
        """
        Get injections with specified shift.

        Args:
            shift:
                Shift value or array to filter injections by

        Returns:
            InjectionParameterSet with injections with the specified shift.
        """
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def get_times(self, start: float | None = None, end: float | None = None):
        """
        Get injections within specified time range.

        Args:
            start:
                Optional start time to filter injections from
            end:
                Optional end time to filter injections to

        Returns:
            InjectionParameterSet with injections that fall
            within the specified time range.
        """

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
        start: float | None = None,
        end: float | None = None,
        shifts: float | None = None,
    ):
        """
        Load InjectionCampaign from file with optional slicing.

        Args:
            fname:
                Path to file to load from
            start:
                Optional start time to load injections from
            end:
                Optional end time to load injections to
            shifts:
                Optional list of shifts to load injections for

        Returns:
            Loaded InjectionCampaign instance with injections that fall
            within the specified time range and/or shifts.
        """
        with h5py.File(fname, "r") as f:
            if all(i is None for i in [start, end, shifts]):
                return cls._load_with_idx(f, None)

            left_pad = f.attrs["duration"] - f.attrs["right_pad"]
            times = f["parameters"]["injection_time"][:]

            mask = True
            if start is not None:
                mask &= (times + left_pad) >= start
            if end is not None:
                mask &= (times - left_pad) <= end
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
        Inject waveforms into background timeseries

        Args:
            x:
                Array of shape (num_ifos, num_samples) to inject waveforms into
            start:
                GPS start time of x

        Returns:
            Background timeseries with waveforms injected
        """
        stop = start + x.shape[-1] / self.sample_rate
        left_pad = self.duration - self.right_pad

        mask = self.injection_time >= (start - left_pad)
        mask &= self.injection_time <= (stop + self.right_pad)

        if not mask.any():
            return x

        times = self.injection_time[mask]
        waveforms = self.waveforms[mask]

        # potentially pad x to inject waveforms
        # that fall over the boundaries of chunks
        pad = []
        earliest = (times - left_pad - start).min()
        if earliest < 0:
            # For consistency, we want to round down here
            # E.g., if earliest = -0.1 and sample_rate = 2048,
            # we want num_early = 205. The int function always
            # rounds towards 0, so we can't do int(-earliest * sample_rate)
            # or -int(earliest * sample_rate)
            num_early = -int((earliest * self.sample_rate) // 1)
            pad.append(num_early)
            start += earliest
        else:
            pad.append(0)

        latest = (times + self.right_pad - stop).max()
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

        idx = np.arange(waveform_size) - int(left_pad * self.sample_rate)
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

    Returns:
        Newly created dataclass with waveform fields for each ifo
    """
    ifos = [ifo.lower() for ifo in ifos]
    fields = [(ifo, waveform()) for ifo in ifos]
    fields = [(name, field.type, field) for name, field in fields]
    cls = make_dataclass(cls_name, fields, bases=(base_cls,))
    return cls
