import pytest
from data.waveforms.injection import WaveformGenerator


@pytest.fixture
def waveform_generator():
    return WaveformGenerator(waveform_duration=1.0, sample_rate=4096)


def test_waveform_size(waveform_generator):
    assert waveform_generator.waveform_size == 4096
