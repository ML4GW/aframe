import numpy as np
import pytest

from aframe.analysis.ledger import events, injections


class TestTimeSlideEventSet:
    def test_append(self):
        det_stats = np.random.randn(10)
        times = np.arange(10)
        obj1 = events.TimeSlideEventSet(det_stats, times, 100)
        obj2 = events.TimeSlideEventSet(-det_stats, times + 10, 50)
        obj1.append(obj2)
        assert obj1.Tb == 150
        det_stats = np.split(obj1.detection_statistic, 2)
        assert (det_stats[0] == -det_stats[1]).all()
        times = np.split(obj1.time, 2)
        assert (times[0] == times[1] - 10).all()

    def test_nb(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        obj = events.TimeSlideEventSet(det_stats, times, 100)
        assert obj.nb(5) == 5
        assert obj.nb(5.5) == 4
        assert obj.nb(-1) == 10

        assert (obj.nb(np.array([5, 5.5])) == np.array([5, 4])).all()

    def test_far(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        Tb = 2 * events.SECONDS_IN_YEAR
        obj = events.TimeSlideEventSet(det_stats, times, Tb)

        assert obj.far(5) == 2.5
        assert (obj.far(np.array([5, 5.5])) == np.array([2.5, 2])).all()

    def test_apply_vetos(self):
        det_stats = np.array([1, 2, 3])
        times = np.array([0, 2, 4])
        set = events.TimeSlideEventSet(det_stats, times, 100)
        vetos = np.array([[0.5, 1.5], [3.5, 4.5]])

        result = set.apply_vetos(vetos)
        expected = events.TimeSlideEventSet(
            detection_statistic=np.array([1, 2]), time=np.array([0, 2])
        )

        assert np.array_equal(
            result.detection_statistic, expected.detection_statistic
        )
        assert np.array_equal(result.time, expected.time)
        assert result.Tb == 100

    # TODO: add a test for significance that doen'st
    # just replicate the logic of the function itself


class TestEventSet:
    def test_from_timeslide(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        obj = events.TimeSlideEventSet(det_stats, times, 100)

        shift = [0]
        supobj = events.EventSet.from_timeslide(obj, shift)
        assert len(supobj) == 10
        assert supobj.Tb == 100
        assert supobj.shift.ndim == 2
        assert (supobj.shift == 0).all()

        shift = [0, 1]
        supobj = events.EventSet.from_timeslide(obj, shift)
        assert len(supobj) == 10
        assert supobj.Tb == 100
        assert supobj.shift.ndim == 2
        assert (supobj.shift == np.array([0, 1])).all()

    def test_get_shift(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        shifts = np.array([0] * 4 + [1] * 5 + [2])
        obj = events.EventSet(det_stats, times, 100, shifts)

        subobj = obj.get_shift(0)
        assert len(subobj) == 4
        assert (subobj.detection_statistic == np.arange(4)).all()
        assert (subobj.shift == 0).all()

        subobj = obj.get_shift(2)
        assert len(subobj) == 1
        assert (subobj.detection_statistic == 9).all()
        assert (subobj.shift == 2).all()

        shifts = np.array([[0, 0]] * 5 + [[0, 1]] * 2 + [[1, 1]] * 3)
        obj = events.EventSet(det_stats, times, 100, shifts)
        subobj = obj.get_shift(np.array([0, 1]))
        assert len(subobj) == 2
        assert (subobj.detection_statistic == np.arange(5, 7)).all()
        assert (subobj.shift == np.array([0, 1])).all()

    def test_apply_vetos(self):
        det_stats = np.arange(5)
        times = np.arange(5)
        shifts = np.array([[0, 1]] * 2 + [[0, 2]] * 2 + [[0, 3]])
        obj = events.EventSet(det_stats, times, 100, shifts)

        # apply vetoes to the first shift
        result = obj.apply_vetos(np.array([[0.5, 1.5], [3.5, 4.5]]), idx=0)
        expected = events.EventSet(
            detection_statistic=np.array([0, 2, 3]),
            time=np.array([0, 2, 3]),
            Tb=100,
            shift=np.array([[0, 1]] + [[0, 2]] * 2),
        )
        assert all(result.time == expected.time)
        assert (result.shift == expected.shift).all()
        assert all(result.detection_statistic == expected.detection_statistic)
        assert result.Tb == expected.Tb

        # apply vetoes to the second shift
        result = obj.apply_vetos(np.array([[0.5, 1.5], [3.5, 4.5]]), idx=1)
        expected = events.EventSet(
            detection_statistic=np.array([1, 3, 4]),
            time=np.array([1, 3, 4]),
            Tb=100,
            shift=np.array([[0, 1]] + [[0, 2]] + [[0, 3]]),
        )
        assert all(result.time == expected.time)
        assert (result.shift == expected.shift).all()
        assert all(result.detection_statistic == expected.detection_statistic)
        assert result.Tb == expected.Tb


class TestRecoveredInjectionSet:
    @pytest.fixture
    def timeslide_event_set(self):
        det_stats = np.arange(5, 15)
        times = np.arange(10)
        return events.TimeSlideEventSet(det_stats, times, 100)

    @pytest.fixture
    def response_set(self):
        times = np.array([1.4, 8.6, 3.1])
        params = {"gps_time": times, "sample_rate": 2048}

        fields = injections.LigoResponseSet.__dataclass_fields__
        for name, attr in fields.items():
            if name == "gps_time" or name == "sample_rate":
                continue
            if attr.metadata["kind"] == "parameter":
                params[name] = np.arange(3)

        return injections.InterferometerResponseSet(num_injections=5, **params)

    def test_recover_single_shift(self, timeslide_event_set, response_set):

        obj = events.RecoveredInjectionSet.recover(
            timeslide_event_set, response_set
        )
        assert len(obj) == 3
        assert (obj.detection_statistic == np.array([6, 14, 8])).all()
        assert obj.Tb == 100
        assert obj.num_injections == 5
