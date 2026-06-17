import time

import pytest
from infer.data import _throttle
from pytest import approx


@pytest.fixture
def fake_clock(monkeypatch):
    """Test _throttle against a controllable clock to guard against variance
    from the real clock and sleep."""
    clock = [0.0]
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", fake_sleep)
    return clock, slept


def test_throttle_spacing(fake_clock):
    # starting on time, every sleep is exactly one interval and the deadline
    # advances by one interval per call
    clock, slept = fake_clock
    interval = 0.02
    start = clock[0]
    deadline = start
    for _ in range(100):
        deadline = _throttle(deadline, interval)
    assert deadline == approx(start + 100 * interval)
    assert len(slept) == 99 and all(s == approx(interval) for s in slept)


def test_throttle_no_catchup_after_slow_iter(fake_clock):
    # A body slower than the interval should not cause the next
    # call to _throttle to sleep, and should base the next deadline
    # on the current time, not the previous deadline.
    clock, slept = fake_clock
    interval = 0.02
    deadline = _throttle(clock[0], interval)
    clock[0] += 0.05  # slow body overruns the interval by 2.5x

    # 1. the overdue call does not sleep
    n_sleeps = len(slept)
    next_deadline = _throttle(deadline, interval)
    assert len(slept) == n_sleeps

    # 2. the deadline is re-anchored to the current time
    assert next_deadline == approx(clock[0] + interval)

    # 3. the following call sleeps one interval
    _throttle(next_deadline, interval)
    assert slept[-1] == approx(interval)
