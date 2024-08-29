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
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected: InjectionParameterSet,
    astro_event_rate: float,
) -> Pastro:
    """
    Load a pastro model from disk if it exists, otherwise fit a new one.

    Args:
        model_path:
            Path to the model file.
        background:
            Background event set.
        foreground:
            Foreground event set.
        rejected:
            Rejected injections.
        astro_event_rate:
            Expected rate of astrophysical events

    Returns: Pastro model
    """

    if model_path.exists():
        with open(model_path, "rb") as f:
            pastro = pickle.load(f)
    else:
        background_model = KdeAndPolynomialBackground(background)
        foreground_model = KdeForeground(
            foreground, rejected, astro_event_rate
        )
        pastro = Pastro(foreground_model, background_model)
        with open(model_path, "wb") as f:
            pickle.dump(pastro, f)
    return pastro
