import numpy as np
import pytest
from ledger import events, injections


class TestEventSet:
    def test_append(self):
        det_stats = np.random.randn(10)
        times = np.arange(10)
        shifts = np.arange(10)
        obj1 = events.EventSet(det_stats, times, shifts, 100)
        obj2 = events.EventSet(-det_stats, times + 10, shifts * -1, 50)
        obj1.append(obj2)
        assert obj1.Tb == 150
        det_stats = np.split(obj1.detection_statistic, 2)
        assert (det_stats[0] == -det_stats[1]).all()
        times = np.split(obj1.detection_time, 2)
        assert (times[0] == times[1] - 10).all()
        shifts = np.split(obj1.shift, 2)
        assert (shifts[0] == shifts[1] * -1).all()

        # test with empty object
        obj1 = events.EventSet()
        det_stats, times, shifts = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        )
        obj2 = events.EventSet(det_stats, times, shifts, 100)
        obj2.append(obj1)
        assert obj2.Tb == 100
        assert (obj2.detection_statistic == det_stats).all()
        assert (obj2.detection_time == times).all()
        assert (obj2.shift == shifts).all()

    def test_nb(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        shifts = np.array([0] * 4 + [1] * 5 + [2])
        obj = events.EventSet(det_stats, times, shifts, 100)
        assert obj.nb(5) == 5
        assert obj.nb(5.5) == 4
        assert obj.nb(-1) == 10

        assert (obj.nb(np.array([5, 5.5])) == np.array([5, 4])).all()

    def test_far(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        shifts = np.array([0] * 4 + [1] * 5 + [2])
        Tb = 2 * events.SECONDS_IN_YEAR
        obj = events.EventSet(det_stats, times, shifts, Tb)

        assert obj.far(5) == 2.5
        assert (obj.far(np.array([5, 5.5])) == np.array([2.5, 2])).all()

    def test_get_shift(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        shifts = np.array([0] * 4 + [1] * 5 + [2])
        obj = events.EventSet(det_stats, times, shifts, 100)

        subobj = obj.get_shift(0)
        assert len(subobj) == 4
        assert (subobj.detection_statistic == np.arange(4)).all()
        assert (subobj.shift == 0).all()

        subobj = obj.get_shift(2)
        assert len(subobj) == 1
        assert (subobj.detection_statistic == 9).all()
        assert (subobj.shift == 2).all()

        shifts = np.array([[0, 0]] * 5 + [[0, 1]] * 2 + [[1, 1]] * 3)
        obj = events.EventSet(det_stats, times, shifts, 100)
        subobj = obj.get_shift(np.array([0, 1]))
        assert len(subobj) == 2
        assert (subobj.detection_statistic == np.arange(5, 7)).all()
        assert (subobj.shift == np.array([0, 1])).all()

    def test_apply_vetos(self):
        det_stats = np.arange(5)
        times = np.arange(5)
        shifts = np.array([[0, 1]] * 2 + [[0, 2]] * 2 + [[0, 3]])
        obj = events.EventSet(det_stats, times, shifts, 100)

        # apply vetoes to the first shift
        result = obj.apply_vetos(np.array([[0.5, 1.5], [3.5, 4.5]]), idx=0)
        expected = events.EventSet(
            detection_statistic=np.array([0, 2, 3]),
            detection_time=np.array([0, 2, 3]),
            Tb=100,
            shift=np.array([[0, 1]] + [[0, 2]] * 2),
        )
        assert all(result.detection_time == expected.detection_time)
        assert (result.shift == expected.shift).all()
        assert all(result.detection_statistic == expected.detection_statistic)
        assert result.Tb == expected.Tb

        # apply vetoes to the second shift
        result = obj.apply_vetos(np.array([[0.5, 1.5], [3.5, 4.5]]), idx=1)
        expected = events.EventSet(
            detection_statistic=np.array([1, 3, 4]),
            detection_time=np.array([1, 3, 4]),
            Tb=100,
            shift=np.array([[0, 1]] + [[0, 2]] + [[0, 3]]),
        )
        assert all(result.detection_time == expected.detection_time)
        assert (result.shift == expected.shift).all()
        assert all(result.detection_statistic == expected.detection_statistic)
        assert result.Tb == expected.Tb


class TestRecoveredInjectionSet:
    @pytest.fixture
    def event_set(self):
        det_stats = np.arange(5, 15)
        det_stats = np.concatenate((det_stats, det_stats))
        times = np.arange(10)
        times = np.concatenate((times, times))
        shifts = np.array([0] * 10 + [1] * 10)
        return events.EventSet(det_stats, times, shifts, 100)

    @pytest.fixture
    def response_set(self, response_set_cls):
        times = np.array([1.4, 8.6, 3.1])
        times = np.concatenate((times, times))
        shifts = np.array([0] * 3 + [1] * 3)

        params = {
            "injection_time": times,
            "shift": shifts,
            "sample_rate": 2048,
        }
        fields = response_set_cls.__dataclass_fields__
        for name, attr in fields.items():
            if name in ["injection_time", "sample_rate", "shift"]:
                continue
            if attr.metadata["kind"] == "parameter":
                params[name] = np.arange(6)

        return injections.InterferometerResponseSet(num_injections=6, **params)

    def test_recover(self, event_set, response_set):
        obj = events.RecoveredInjectionSet.recover(event_set, response_set)
        assert len(obj) == 6
        assert (
            obj.detection_statistic == np.array([6, 14, 8, 6, 14, 8])
        ).all()
        assert obj.Tb == 100
        assert np.all(obj.shift == np.array([0, 0, 0, 1, 1, 1]))
        assert np.all(obj.detection_time == np.array([1, 9, 3, 1, 9, 3]))
        assert obj.num_injections == 6
