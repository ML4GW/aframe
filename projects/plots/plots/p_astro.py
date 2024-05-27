from ledger.events import EventSet, F, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet


def compute_p_astro(
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected_params: InjectionParameterSet,
    detection_statistic: F,
    injected_volume: float,
    astro_event_rate: float,
) -> F:
    n_inj = len(foreground) + len(rejected_params)
    events_above_statistic = foreground.nb(detection_statistic)
    sensitive_volume = injected_volume * events_above_statistic / n_inj
    foreground_rate = sensitive_volume * astro_event_rate
    background_rate = background.far(detection_statistic)

    return foreground_rate / (foreground_rate + background_rate)
