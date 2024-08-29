import numpy as np
import pytest

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet


@pytest.fixture
def events():
    return EventSet(
        detection_time=np.arange(1000),
        detection_statistic=np.random.randn(1000),
        shift=np.zeros(1000),
        Tb=10,
    )


@pytest.fixture
def foreground():
    foreground = {}
    for field in RecoveredInjectionSet.__dataclass_fields__:
        foreground[field] = np.random.randn(1000)

    foreground["num_injections"] = 1000
    foreground["redshift"] = np.linspace(0, 1, 1000)
    return RecoveredInjectionSet(**foreground)


@pytest.fixture
def rejected():
    rejected = {}
    for field in InjectionParameterSet.__dataclass_fields__:
        rejected[field] = np.random.randn(1000)

    rejected["redshift"] = np.linspace(0, 1, 1000)
    return InjectionParameterSet(**rejected)


@pytest.fixture(params=[1, 2, 3])
def astro_event_rate(request):
    return request.param
