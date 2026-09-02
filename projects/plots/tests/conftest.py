import numpy as np
import pytest
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet

SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25

T0 = 1240000000.0

IFOS = ["H1", "L1"]


def _intrinsic_params(rng, n):
    """Draw the parameter fields shared by injection and recovery ledgers."""
    redshift = rng.uniform(0.05, 2, size=n)
    # keep mass_1 >= mass_2, as the priors assume
    mass_1 = rng.uniform(20, 60, size=n) * (1 + redshift)
    mass_2 = rng.uniform(10, 20, size=n) * (1 + redshift)
    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": rng.uniform(0, 0.999, size=n),
        "a_2": rng.uniform(0, 0.999, size=n),
        "tilt_1": rng.uniform(0, np.pi, size=n),
        "tilt_2": rng.uniform(0, np.pi, size=n),
        "phi_12": rng.uniform(0, 2 * np.pi, size=n),
        "phi_jl": rng.uniform(0, 2 * np.pi, size=n),
        "ra": rng.uniform(0, 2 * np.pi, size=n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, size=n),
        "redshift": redshift,
        "psi": rng.uniform(0, np.pi, size=n),
        "theta_jn": rng.uniform(0, np.pi, size=n),
        "phase": rng.uniform(0, 2 * np.pi, size=n),
    }


@pytest.fixture
def make_background():
    """Build an `EventSet` of `n` noise events spanning `Tb` seconds.

    Detection statistics are drawn from an exponential, mimicking the
    distribution of events in a real search.
    """

    def factory(n=2000, Tb=0.25 * SECONDS_PER_YEAR, seed=1):
        rng = np.random.default_rng(seed)
        return EventSet(
            detection_statistic=rng.exponential(1.0, size=n) + 4.0,
            detection_time=T0 + np.sort(rng.uniform(0, Tb, size=n)),
            shift=np.stack(
                [np.zeros(n), rng.integers(0, 10, size=n) * 1.0], axis=-1
            ),
            Tb=Tb,
        )

    return factory


@pytest.fixture
def make_foreground():
    """Build a `RecoveredInjectionSet` of `n` recovered injections."""

    def factory(n=500, seed=2, detection_offset=0.01):
        rng = np.random.default_rng(seed)
        params = _intrinsic_params(rng, n)
        injection_time = T0 + np.sort(rng.uniform(0, 1e6, size=n))
        ifo_snrs = rng.uniform(4, 40, size=(n, len(IFOS)))
        return RecoveredInjectionSet(
            **params,
            snr=np.sqrt((ifo_snrs**2).sum(-1)),
            ifo_snrs=ifo_snrs,
            ifos=IFOS,
            sample_rate=2048,
            duration=8,
            right_pad=0.5,
            num_injections=n,
            injection_time=injection_time,
            shift=np.zeros((n, len(IFOS))),
            detection_statistic=rng.exponential(2.0, size=n) + 5.0,
            detection_time=injection_time + detection_offset,
            Tb=0,
        )

    return factory


@pytest.fixture
def make_rejected():
    """Build the `InjectionParameterSet` of rejected waveforms."""

    def factory(n=1500, seed=3):
        rng = np.random.default_rng(seed)
        params = _intrinsic_params(rng, n)
        ifo_snrs = rng.uniform(0, 4, size=(n, len(IFOS)))
        return InjectionParameterSet(
            **params,
            snr=np.sqrt((ifo_snrs**2).sum(-1)),
            ifo_snrs=ifo_snrs,
            ifos=IFOS,
        )

    return factory


@pytest.fixture
def analysis_files(tmp_path, make_background, make_foreground, make_rejected):
    """Write a synthetic background/foreground/rejected trio to disk.

    Returns the three paths in the order the `sensitive-volume` CLI
    takes them.
    """

    def factory(
        n_background=2000, n_foreground=500, n_rejected=1500, **kwargs
    ):
        background = make_background(n=n_background, **kwargs)
        foreground = make_foreground(n=n_foreground)
        rejected = make_rejected(n=n_rejected)

        paths = (
            tmp_path / "background.hdf5",
            tmp_path / "foreground.hdf5",
            tmp_path / "rejected_parameters.hdf5",
        )
        background.write(paths[0])
        foreground.write(paths[1])
        rejected.write(paths[2])
        return paths

    return factory
