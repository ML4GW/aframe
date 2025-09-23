import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from torch.multiprocessing import Queue

from .utils import subprocess_wrapper

import h5py
import pickle

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground
from p_astro.p_astro import Pastro

if TYPE_CHECKING:
    from online.utils.gdb import GraceDb

logger = logging.getLogger("pastro-process")

TIMEOUT = 10
TIMESTEP = 1e-3
MAX_RETRIES = int(TIMEOUT / TIMESTEP)


def fit_or_load_pastro(
    model_path: Path,
    background_path: Path,
    foreground_path: Path,
    rejected_path: Path,
    astro_event_rate: float,
) -> Pastro:
    """
    Load a pastro model from disk if it exists, otherwise fit a new one.

    Args:
        model_path:
            Path to the model file.
        background:
            Path to background event set.
        foreground:
            Path to foreground event set.
        rejected:
            Path to rejected injection set.
        astro_event_rate:
            Expected rate of astrophysical events

    Returns: Pastro model
    """

    if model_path.exists():
        logger.info("Loading p_astro model")
        with open(model_path, "rb") as f:
            pastro = pickle.load(f)
        logger.info("p_astro model loaded")
    else:
        logger.info(
            "Loading background, foreground, and "
            "rejected injections for pastro model"
        )
        background = EventSet.read(background_path)
        foreground = RecoveredInjectionSet.read(foreground_path)
        rejected = InjectionParameterSet.read(rejected_path)

        logger.info("Data loaded, fitting pastro model")
        background_model = KdeAndPolynomialBackground(background)
        foreground_model = KdeForeground(
            foreground, rejected, astro_event_rate
        )
        pastro = Pastro(foreground_model, background_model)
        logger.info("Fitting complete, saving pastro model")
        with open(model_path, "wb") as f:
            pickle.dump(pastro, f)
        logger.info("Model saved to disk")
    return pastro


@subprocess_wrapper
def pastro_subprocess(
    pastro_queue: Queue,
    background_path,
    foreground_path,
    rejected_path,
    astro_event_rate,
    gdb: "GraceDb",
    outdir: Path,
):
    logger.info("pastro subprocess initialized")

    # override with subprocesses logger
    gdb.logger = logger
    pastro_model = fit_or_load_pastro(
        outdir / "pastro.pkl",
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate=astro_event_rate,
    )
    while True:
        event = pastro_queue.get()
        logger.info("Calculating p_astro")
        pastro = float(pastro_model(event.detection_statistic))
        graceid = pastro_queue.get()

        event_dir = outdir / "events" / event.event_dir
        posterior_file = event_dir / "amplfi.posterior_samples.hdf5"

        retries = 0
        while True:
            try:
                with h5py.File(posterior_file, "r") as f:
                    samples = f["posterior_samples"][:]
                    m1 = samples["mass_1"]
                    m2 = samples["mass_2"]

                logger.info("Read posteriors from file")
                num_samples = len(m1)
                bns_frac = sum(m1 < 3) / num_samples
                bbh_frac = sum(m2 > 3) / num_samples
                nsbh_frac = 1 - bns_frac - bbh_frac

                probs = {
                    "BBH": pastro * bbh_frac,
                    "NSBH": pastro * nsbh_frac,
                    "BNS": pastro * bns_frac,
                    "Terrestrial": 1 - pastro,
                }

                break
            except Exception:
                time.sleep(TIMESTEP)
                retries += 1

            if retries >= MAX_RETRIES:
                logging.info(
                    f"Posterior file not found after {TIMEOUT} seconds, "
                    "assigning all probability to BBH"
                )
                probs = {
                    "BBH": pastro,
                    "NSBH": 0,
                    "BNS": 0,
                    "Terrestrial": 1 - pastro,
                }
                break

        logger.info(f"Submitting p_astro: {probs} for {graceid}")
        gdb.submit_pastro(probs, graceid, event.event_dir)
        logger.info(f"Submitted p_astro for {graceid}")
