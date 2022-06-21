import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hermes.stillwater.process import PipelineProcess
from hermes.stillwater.utils import Package
from infer import main as infer

from bbhnet.io.h5 import read_timeseries, write_timeseries


class DummyInferInput:
    def __init__(self, stream_size: int):
        self.name = "stream"
        self.shape = (2, stream_size)

    def set_data_from_numpy(self, x):
        self.x = x


def get_async_stream_infer(obj):
    def f(*args, **kwargs):
        obj.out_q.put(
            {
                "prob": Package(
                    x=kwargs["inputs"][0].x[0, 0, -1] + 1,
                    t0=time.time(),
                    sequence_id=kwargs["sequence_id"],
                    sequence_end=kwargs["sequence_end"],
                )
            }
        )

    return f


@pytest.fixture(params=[2, 4])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture
def new_init(sample_rate, inference_sampling_rate):
    stream_size = int(sample_rate // inference_sampling_rate)
    infer_input = DummyInferInput(stream_size)

    def __init__(
        obj,
        url,
        model_name,
        model_version,
        name,
    ):
        PipelineProcess.__init__(obj, name)
        obj.client = MagicMock()
        obj.client.async_stream_infer = get_async_stream_infer(obj)

        obj.inputs = []
        obj.states = [(infer_input, {"": (1, 2, stream_size)})]
        obj.model_name = model_name
        obj.model_version = model_version
        obj.message_start_times = {}
        obj.profile = False

    return __init__


@pytest.fixture
def tmpdir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def data_dir(tmpdir, sample_rate):
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    for dt in ["0.0", "0.5", "1.0"]:
        write_dir = data_dir / f"dt-{dt}" / "raw"
        write_dir.mkdir(parents=True)

        x = np.arange(0, 1024, 1 / sample_rate)
        t = 1234567890 + x

        # create two segments of length 3 * 1024
        # and 2 * 1024 seconds respectively for
        # each timeslide
        for i in range(3):
            write_timeseries(
                write_dir,
                prefix="raw",
                hanford=x + i * 1024,
                livingston=-x,
                t=t + i * 1024,
            )
        for i in range(2):
            write_timeseries(
                write_dir,
                prefix="raw",
                hanford=x + i * 1024,
                livingston=-x,
                t=t + (i + 4) * 1024,
            )

    return data_dir


def fake_init(obj, *args, **kwargs):
    obj._instance = MagicMock()
    obj._thread = MagicMock()
    obj._response_queue = MagicMock()


@patch("tritonserve.SingularityInstance.__init__", new=fake_init)
@patch("tritonserve.SingularityInstance.run")
@patch("tritonserve.SingularityInstance.name", return_value="FAKE")
@patch(
    "tritonclient.grpc.InferenceServerClient.is_server_live", return_value=True
)
def test_infer(
    init_mock,
    run_mock,
    name_mock,
    data_dir,
    sample_rate,
    inference_sampling_rate,
    new_init,
):
    with patch("hermes.stillwater.InferenceClient.__init__", new=new_init):
        infer(
            "",
            "",
            data_dir=data_dir,
            field="raw",
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            num_workers=2,
        )

    step_size = int(sample_rate // inference_sampling_rate)
    for dt in ["0.0", "0.5", "1.0"]:
        out_dir = data_dir / f"dt-{dt}" / "out"
        assert out_dir.exists()

        for i, fname in enumerate(out_dir.iterdir()):
            length = int(fname.stem.split("-")[-1])
            y, t = read_timeseries(fname, "out")
            assert len(y) == (length * inference_sampling_rate)

            expected = np.arange(0, length, 1 / sample_rate)
            expected = expected[::step_size] + expected[step_size - 1] + 1
            assert (expected == y).all()
        assert i == 1
