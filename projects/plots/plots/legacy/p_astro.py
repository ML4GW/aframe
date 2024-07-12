import numpy as np
from ledger.events import EventSet, F, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet


def compute_p_astro(
    detection_statistic: F,
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected_params: InjectionParameterSet,
    injected_volume: float,
    astro_event_rate: float,
    min_det_stat: float = -np.inf,
) -> F:
    """
    Compute p_astro as the ratio of the expected signal rate
    to the sum of the expected signal rate and the expected
    background event rate for a given set of detection
    statistics

    Args:
        detection_statistic:
            A value or set of values for which to calculate the
            corresponding p_astro
        background:
            EventSet object corresponding to a background model
        foreground:
            RecoveredInjectionSet object corresponding to an
            injection campaign
        rejected_params:
            InjectionParameterSet object corresponding to signals
            that were simulated but rejected due to low SNR
        injected_volume:
            The astrophysical volume into which injections were
            made during the injection campaign
        astro_event_rate:
            The rate density of events for the relavent population.
            Should have the same spatial units as injected_volume
        min_det_stat:
            Then minimum detection statistic for which to compute
            p_astro. Detection statistics below this value will
            be assigned a p_astro of 0
    """
    p_astro = np.zeros_like(detection_statistic)
    mask = detection_statistic >= min_det_stat
    detection_statistic = detection_statistic[mask]

    n_inj = len(foreground) + len(rejected_params)
    events_above_statistic = foreground.nb(detection_statistic)
    sensitive_volume = injected_volume * events_above_statistic / n_inj
    foreground_rate = sensitive_volume * astro_event_rate
    background_rate = background.far(detection_statistic)

    p_astro[mask] += foreground_rate / (foreground_rate + background_rate)
    return p_astro
