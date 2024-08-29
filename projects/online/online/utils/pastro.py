import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground
from p_astro.p_astro import Pastro

if TYPE_CHECKING:
    from ledger.events import EventSet, RecoveredInjectionSet
    from ledger.injections import InjectionParameterSet


def fit_or_load_pastro(
    model_path: Path,
    background: "EventSet",
    foreground: "RecoveredInjectionSet",
    rejected: "InjectionParameterSet",
    astro_event_rate: float,
):
    if model_path.exists():
        logging.info(f"Loading pastro model from {model_path}")
        with open(model_path, "rb") as f:
            p_astro = pickle.load(f)
    else:
        background_model = KdeAndPolynomialBackground(background)
        foreground_model = KdeForeground(
            foreground, rejected, astro_event_rate
        )
        p_astro = Pastro(foreground_model, background_model)
        with open(model_path, "wb") as f:
            pickle.dump(p_astro, f)
    return p_astro
