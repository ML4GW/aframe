import logging
from unittest.mock import MagicMock

from utils.data import (
    segments_from_paths,
    get_num_shifts_from_Tb,
    get_num_shifts_from_num_signals,
    is_analyzeable_segment,
)


class TestSegmentsFromPaths:
    """Test suite for segments_from_paths function."""

    def test_single_segment_parsing(self):
        paths = [MagicMock(path="background-1234567890-1000.hdf5")]
        result = segments_from_paths(paths)
        assert len(result) == 1
        assert result[0] == [1234567890.0, 1234568890.0]

    def test_multiple_segments(self):
        paths = [
            MagicMock(path="background-1000000000-1000.hdf5"),
            MagicMock(path="background-2000000000-2000.hdf5"),
        ]
        result = segments_from_paths(paths)
        assert len(result) == 2
        assert result[0] == [1000000000.0, 1000001000.0]
        assert result[1] == [2000000000.0, 2000002000.0]

    def test_fractional_times(self):
        paths = [MagicMock(path="background-1000000000.5-50.25.hdf5")]
        result = segments_from_paths(paths)
        assert result[0] == [1000000000.5, 1000000050.75]

    def test_invalid_filename_warning(self, caplog):
        paths = [MagicMock(path="invalid_format")]
        with caplog.at_level(logging.WARNING):
            result = segments_from_paths(paths)
        assert len(result) == 0
        assert "Couldn't parse file" in caplog.text

    def test_mixed_valid_invalid(self, caplog):
        paths = [
            MagicMock(path="background-1000000000-1000.hdf5"),
            MagicMock(path="invalid_format"),
            MagicMock(path="background-2000000000-2000.hdf5"),
        ]
        with caplog.at_level(logging.WARNING):
            result = segments_from_paths(paths)
        assert len(result) == 2
        assert "Couldn't parse file" in caplog.text


class TestGetNumShiftsFromTb:
    """Test suite for get_num_shifts_from_Tb function."""

    def test_zero_target_background(self):
        segments = [(0, 1000), (2000, 3000)]
        assert get_num_shifts_from_Tb(segments, 0, 1, 64) == 0

    def test_non_zero_target_background(self):
        segments = [(0, 1000)]
        result = get_num_shifts_from_Tb(segments, 1, 1, 64)
        assert result == 1

    def test_multiple_shifts_needed(self):
        segments = [(0, 200)]
        result = get_num_shifts_from_Tb(segments, 1000, 1, 64)
        assert result == 8

    def test_multiple_segments(self):
        segments = [(0, 600), (1000, 1600)]
        result = get_num_shifts_from_Tb(segments, 2000, 1, 64)
        assert result == 2


class TestGetNumShiftsFromNumSignals:
    """Test suite for get_num_shifts_from_num_signals function."""

    def test_zero_signals(self):
        segments = [(0, 10000)]
        result = get_num_shifts_from_num_signals(segments, 0, 8, 16, 1, 8)
        assert result == 0

    def test_single_signal(self):
        segments = [(0, 10000)]
        result = get_num_shifts_from_num_signals(segments, 1, 8, 16, 1, 8)
        assert result == 1

    def test_multiple_segments(self):
        segments = [(0, 5000), (10000, 15000)]
        result = get_num_shifts_from_num_signals(segments, 1000, 8, 16, 1, 8)
        assert result == 3


class TestIsAnalyzeableSegment:
    """Test suite for is_analyzeable_segment function."""

    def test_analyzeable_segment(self):
        assert is_analyzeable_segment(0, 1000, [0, 1], 100) is True

    def test_unanalyzeable_segment_short(self):
        assert is_analyzeable_segment(0, 100, [0, 1], 100) is False

    def test_unanalyzeable_segment_exact(self):
        assert is_analyzeable_segment(0, 100, [0, 1], 99) is False

    def test_single_shift(self):
        assert is_analyzeable_segment(0, 1000, [0], 100) is True
