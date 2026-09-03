import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from priors.priors import log_normal_masses
from utils.cosmology import DEFAULT_COSMOLOGY, get_astrophysical_volume

from plots.core import compute
from plots.core.constants import SECONDS_PER_YEAR
from plots.core.data import AnalysisData

if TYPE_CHECKING:
    from bilby.core.prior import PriorDict


def get_prob(prior, ledger):
    """Evaluate a source-frame mass prior on a ledger's injections."""
    sample = {
        "mass_1": ledger.mass_1_source,
        "mass_2": ledger.mass_2_source,
    }
    return prior.prob(sample, axis=0)


def _combo_key(combo) -> str:
    return "-".join(map(str, combo))


@dataclass
class SensitiveVolumeResult:
    """Sensitive volume vs. false alarm rate for each mass combination.

    Attributes:
        mass_combos: the `(m1, m2)` pairs the curves were computed for
        fars: false alarm rates, per year, ascending, shape `(T,)`
        thresholds: detection statistic at each FAR, descending, shape `(T,)`
        sv: sensitive volume in Gpc^3, shape `(n_combos, T)`
        err: standard error on `sv`, shape `(n_combos, T)`
    """

    mass_combos: list[tuple]
    fars: np.ndarray
    thresholds: np.ndarray
    sv: np.ndarray
    err: np.ndarray

    def write(self, path: Path) -> None:
        keys = [_combo_key(c) for c in self.mass_combos]
        path.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("thresholds", data=self.thresholds)
            f.create_dataset("fars", data=self.fars)
            # h5py iterates groups alphabetically, so the combo order has
            # to be recorded or `read` would permute the rows.
            f.attrs["mass_combos"] = keys
            for i, key in enumerate(keys):
                g = f.create_group(key)
                g.create_dataset("sv", data=self.sv[i])
                g.create_dataset("err", data=self.err[i])

    @classmethod
    def read(cls, path: Path) -> "SensitiveVolumeResult":
        """Read back a file written by `write`."""
        with h5py.File(path, "r") as f:
            keys = list(f.attrs.get("mass_combos"))
            return cls(
                mass_combos=[
                    tuple(float(m) for m in k.split("-")) for k in keys
                ],
                fars=f["fars"][:],
                thresholds=f["thresholds"][:],
                sv=np.stack([f[k]["sv"][:] for k in keys]),
                err=np.stack([f[k]["err"][:] for k in keys]),
            )


def _far_grid(background, max_far: float):
    """Build the FAR grid and the thresholds that produce it."""
    Tb = background.Tb / SECONDS_PER_YEAR
    max_events = min(int(max_far * Tb), len(background))
    if not max_events:
        return np.array([]), np.array([])
    fars = np.arange(1, max_events + 1) / Tb
    thresholds = np.sort(background.detection_statistic)[::-1][:max_events]
    return fars, thresholds


def _weights(
    data: AnalysisData,
    mass_combos: list[tuple],
    source_prior: "PriorDict",
    dt: float | None,
    sigma: float,
) -> np.ndarray:
    """Importance weights reweighting injections to each target prior."""
    logging.info("Computing data likelihood under source prior")
    source_probs = get_prob(source_prior, data.foreground)
    source_rejected_probs = get_prob(source_prior, data.rejected)

    weights = np.zeros((len(mass_combos), len(source_probs)))
    for i, combo in enumerate(mass_combos):
        logging.info(f"Computing likelihoods under {combo} log normal")
        prior, _ = log_normal_masses(
            *combo, sigma=sigma, cosmology=DEFAULT_COSMOLOGY
        )
        weight = get_prob(prior, data.foreground) / source_probs
        rejected_weights = (
            get_prob(prior, data.rejected) / source_rejected_probs
        )

        norm = weight.sum() + rejected_weights.sum()
        if norm > 0:
            weight /= norm

        # enforce the recovery time delta by zeroing the weight of
        # events recovered too far from their injection
        if dt is not None:
            logging.info(f"Enforcing recovery time delta of {dt} seconds")
            mask = (
                np.abs(
                    data.foreground.detection_time
                    - data.foreground.injection_time
                )
                <= dt
            )
            weight[~mask] = 0

        weights[i] = weight
    return weights


def _astrophysical_volume(source_prior: "PriorDict") -> float:
    """Total volume the injections were drawn from, in Gpc^3."""
    logging.info("Computing maximum astrophysical volume")
    zprior = source_prior["redshift"]
    try:
        decprior = source_prior["dec"]
    except KeyError:
        decrange = None
    else:
        decrange = (decprior.minimum, decprior.maximum)
    v0 = get_astrophysical_volume(
        zprior.minimum, zprior.maximum, DEFAULT_COSMOLOGY, decrange
    )
    return v0 / 10**9


def compute_sensitive_volume(
    data: AnalysisData,
    mass_combos: list[tuple],
    source_prior: "PriorDict",
    dt: float | None = None,
    max_far: float = 365,
    sigma: float = 0.1,
) -> SensitiveVolumeResult:
    """Compute sensitive volume vs. false alarm rate.

    Args:
        data: the background, foreground and rejected ledgers
        mass_combos: `(m1, m2)` pairs to compute curves for
        source_prior: the (already instantiated) prior the injections
            were drawn from
        dt: if given, discard injections recovered more than `dt`
            seconds from their injection time
        max_far: largest false alarm rate to compute out to, per year
        sigma: width of the log normal mass distributions
    """
    v0 = _astrophysical_volume(source_prior)
    fars, thresholds = _far_grid(data.background, max_far)
    weights = _weights(data, mass_combos, source_prior, dt, sigma)

    logging.info("Computing sensitive volume at thresholds")
    sv, err = compute.sensitive_volume(
        data.foreground.detection_statistic, weights, thresholds
    )
    return SensitiveVolumeResult(
        mass_combos=mass_combos,
        fars=fars,
        thresholds=thresholds,
        sv=sv * v0,
        err=err * v0,
    )
