import logging
import pickle
from pathlib import Path

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground
from p_astro.p_astro import Pastro


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
        logging.info("Loading p_astro model")
        with open(model_path, "rb") as f:
            pastro = pickle.load(f)
        logging.info("Model loaded")
    else:
        logging.info(
            "Loading background, foreground, and "
            "rejected injections for pastro model"
        )
        background = EventSet.read(background_path)
        foreground = RecoveredInjectionSet.read(foreground_path)
        rejected = InjectionParameterSet.read(rejected_path)

        logging.info("Data loaded, fitting pastro model")
        background_model = KdeAndPolynomialBackground(background)
        foreground_model = KdeForeground(
            foreground, rejected, astro_event_rate
        )
        pastro = Pastro(foreground_model, background_model)
        logging.info("Fitting complete, saving pastro model")
        with open(model_path, "wb") as f:
            pickle.dump(pastro, f)
        logging.info("Model saved to disk")
    return pastro
