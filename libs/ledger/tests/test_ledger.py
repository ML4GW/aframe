from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from ledger import ledger


class TestInjectionSet:
    def _test_read_write(self, obj, tmp_path):
        fname = tmp_path / "obj.h5"

        # test normal write then read
        obj.write(fname)
        new = obj.__class__.read(fname)
        for key, field in obj.__dataclass_fields__.items():
            value = getattr(new, key)
            old = getattr(obj, key)
            truth = value == old
            if field.metadata["kind"] == "metadata":
                assert truth
            else:
                assert truth.all()

        # make sure we catch the error when we try
        # to sample without replacement
        with pytest.raises(ValueError):
            obj.__class__.sample_from_file(fname, len(obj) + 1, replace=False)

        # now test sampling with np.random.choice
        # patched so that we know what to expect.
        # Include duplicate and out-of-order indices
        idx = np.array([1, 0, 2, 1])
        with patch("numpy.random.choice", return_value=idx):
            new = obj.__class__.sample_from_file(fname, 3)
        assert len(new) == 4
        for key, field in obj.__dataclass_fields__.items():
            value = getattr(new, key)
            old = getattr(obj, key)
            kind = field.metadata["kind"]

            if kind == "metadata":
                assert value == old
                continue

            for i in range(4):
                truth = value[i] == old[idx[i]]
                if field.metadata["kind"] == "parameter":
                    assert truth
                else:
                    assert truth.all()

    def test_just_metadata(self):
        @dataclass
        class MetadataClass(ledger.Ledger):
            foo: str = ledger.metadata()
            bar: str = ledger.metadata()

        obj = MetadataClass("hey", "you")
        assert len(obj) == 0

        # TODO: what's the desired slicing
        # behavior on metadata-only objects
        assert obj[1].foo == "hey"

    @pytest.fixture
    def parameter_set(self):
        @dataclass
        class Dummy(ledger.Ledger):
            ids: np.ndarray = ledger.parameter()
            age: np.ndarray = ledger.parameter()

        return Dummy

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        return tmp_path

    def test_parameter_set(self, parameter_set, tmp_dir):
        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        with pytest.raises(ValueError):
            parameter_set(ids, age[:2])

        obj = parameter_set(ids, age)
        assert len(obj) == 3
        assert next(iter(obj)) == {"ids": 1001, "age": 31}

        subobj = obj[1]
        assert len(subobj) == 1
        assert subobj.ids[0] == 1002
        assert subobj.age[0] == 35

        subobj = obj[[0, 2]]
        assert len(subobj) == 2
        assert subobj.ids[-1] == 1003
        assert subobj.age[-1] == 39

        with pytest.raises(TypeError):
            obj.append([])

        obj.append(obj)
        assert len(obj) == 6
        assert obj.ids[3] == 1001
        assert obj.age[3] == 31

        self._test_read_write(obj, tmp_dir)

    def test_parameter_set_with_metadata(self, parameter_set, tmp_dir):
        @dataclass
        class DummyMetadata(parameter_set):
            foo: str = ledger.metadata()

        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        obj = DummyMetadata(ids, age, "test")
        assert obj.foo == "test"

        subobj = obj[:2]
        assert len(subobj) == 2
        assert subobj.foo == "test"

        obj2 = DummyMetadata(ids, age, "bar")
        with pytest.raises(ValueError):
            obj.append(obj2)

        @dataclass
        class DummyMetadataCompare(DummyMetadata):
            fuz: str = ledger.metadata()

            def compare_metadata(self, key, ours, theirs):
                if key == "foo":
                    return ours + theirs
                return super().compare_metadata(key, ours, theirs)

        obj = DummyMetadataCompare(ids, age, "test", "qux")
        obj2 = DummyMetadataCompare(ids + 3, age - 2, "bar", "quz")
        with pytest.raises(ValueError):
            obj.append(obj2)
        obj2.fuz = "qux"
        obj.append(obj2)
        assert len(obj) == 6

        self._test_read_write(obj, tmp_dir)

    def test_waveform_set(self, parameter_set, tmp_dir):
        @dataclass
        class DummyWaveform(parameter_set):
            waves: np.ndarray = ledger.waveform()

        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        waves = np.random.randn(3, 10)

        obj = DummyWaveform(ids, age, waves)
        assert len(obj) == 3

        waves2 = np.random.randn(3, 10)
        obj2 = DummyWaveform(ids + 3, age - 2, waves2)
        obj.append(obj2)
        assert len(obj) == 6

        all_waves = np.concatenate([waves, waves2])
        assert (obj.waves == all_waves).all()
        assert (obj[2:4].waves == all_waves[2:4]).all()

        self._test_read_write(obj, tmp_dir)

    def test_sort(self, parameter_set):
        ids = np.array([1, 2, 3])
        age = np.array([3, 2, 1])
        obj = parameter_set(ids, age)

        assert obj.is_sorted_by("ids")
        assert not obj.is_sorted_by("age")

        obj = obj.sort_by("age")

        assert (obj.age == np.array([1, 2, 3])).all()
        assert (obj.ids == np.array([3, 2, 1])).all()
        assert obj.is_sorted_by("age")
        assert not obj.is_sorted_by("ids")
