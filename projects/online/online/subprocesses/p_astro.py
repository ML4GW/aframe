import logging
from pathlib import Path

from torch.multiprocessing import Queue

from online.utils.gdb import GdbServer, gracedb_factory

from .wrapper import subprocess_wrapper

import pickle

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground
from p_astro.p_astro import Pastro

logger = logging.getLogger("pastro-process")


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
        logger.info("Model loaded")
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
    server: GdbServer,
    outdir: Path,
):
    logger.info("pastro subprocess initialized")
    pastro_model = fit_or_load_pastro(
        outdir / "pastro.pkl",
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate=astro_event_rate,
    )
    gdb = gracedb_factory(server, outdir)
    while True:
        event = pastro_queue.get()
        logger.info("Calculating p_astro")
        pastro = pastro_model(event.detection_statistic)
        graceid = pastro_queue.get()

        logger.info(f"Submitting p_astro: {pastro}")
        gdb.submit_pastro(float(pastro), graceid, event.gpstime)
        logger.info("Submitted p_astro")
