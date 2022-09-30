import logging
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from infer.main import main as infer

from bbhnet.io.h5 import read_timeseries, write_timeseries


class DummyInferInput:
    def __init__(self, stream_size: int):
        self.name = "stream"
        self.shape = (1, 2, stream_size)

    def set_data_from_numpy(self, x):
        self.x = x


def get_async_stream_infer(obj, step_size):
    def f(*args, **kwargs):
        x = kwargs["inputs"][0].x
        y = x[0, 0, ::step_size][:, None] + 1

        request_id = int(kwargs["request_id"].split("_")[0])
        response = obj.callback(y, request_id, kwargs["sequence_id"])
        obj.callback_q.put(response)

    return f


@pytest.fixture(params=[2, 4])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[1, 8])
def batch_size(request):
    return request.param


@pytest.fixture
def streams_per_gpu():
    return 2


@pytest.fixture
def new_init(batch_size, sample_rate, inference_sampling_rate):
    stride_size = int(sample_rate // inference_sampling_rate)
    stream_size = batch_size * stride_size
    infer_input = DummyInferInput(stream_size)

    def __init__(
        obj,
        url,
        model_name,
        model_version,
    ):
        obj.client = MagicMock()
        obj.client.async_stream_infer = get_async_stream_infer(
            obj, stride_size
        )

        obj.inputs = []
        obj.states = [(infer_input, {"": (1, 2, stream_size)})]
        obj.num_states = 1
        obj.model_name = model_name
        obj.model_version = model_version
        obj.message_start_times = {}
        obj.clock = None
        obj._sequences = {}
        obj.callback_q = Queue()

    return __init__


@pytest.fixture
def tmpdir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    logging.shutdown()


@pytest.fixture(params=[["raw", "injection"]])
def fields(request):
    return request.param


@pytest.fixture
def data_dir(tmpdir, sample_rate, fields):
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    for field in fields:
        for dt in ["0.0", "0.5", "1.0"]:
            write_dir = data_dir / f"dt-{dt}" / field
            write_dir.mkdir(parents=True)

            x = np.arange(0, 1024, 1 / sample_rate)
            t = 1234567890 + x

            # create two segments of length 3 * 1024
            # and 2 * 1024 seconds respectively for
            # each timeslide
            for i in range(3):
                write_timeseries(
                    write_dir,
                    prefix=field,
                    H1=x + i * 1024,
                    L1=-x,
                    t=t + i * 1024,
                )
            for i in range(2):
                write_timeseries(
                    write_dir,
                    prefix=field,
                    H1=x + i * 1024,
                    L1=-x,
                    t=t + (i + 4) * 1024,
                )
    return data_dir


def fake_init(self, *args, **kwargs):
    self.name = "dummy-instance"


@patch("spython.main.Client.execute")
@patch("spython.instance.Instance.__init__", new=fake_init)
@patch("spython.instance.Instance.stop")
@patch(
    "tritonclient.grpc.InferenceServerClient.is_server_live", return_value=True
)
@patch("hermes.stillwater.monitor.ServerMonitor.__init__", return_value=None)
@patch("hermes.stillwater.monitor.ServerMonitor.__enter__")
@patch("hermes.stillwater.monitor.ServerMonitor.__exit__")
def test_infer(
    exec_mock,
    stop_mock,
    is_live_mock,
    monitor_init_mock,
    monitor_enter_mock,
    monitor_exit_mock,
    data_dir,
    tmpdir,
    fields,
    sample_rate,
    inference_sampling_rate,
    batch_size,
    streams_per_gpu,
    new_init,
):
    with patch("hermes.aeriel.client.InferenceClient.__init__", new=new_init):
        infer(
            "",
            "",
            data_dir=data_dir,
            write_dir=data_dir,
            fields=fields,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            streams_per_gpu=streams_per_gpu,
            log_file=tmpdir / "infer.log",
            inference_rate=5000,
            num_workers=2,
        )

    stride_size = int(sample_rate // inference_sampling_rate)
    step_size = batch_size * stride_size
    for field in fields:
        for dt in ["0.0", "0.5", "1.0"]:
            out_dir = data_dir / f"dt-{dt}" / f"{field}-out"
            assert out_dir.exists()

            for i, fname in enumerate(out_dir.iterdir()):
                length = int(fname.stem.split("-")[-1])
                num_steps = int(length * sample_rate) // step_size
                size = step_size * num_steps

                y, t = read_timeseries(fname, "out")
                assert len(y) == (num_steps * batch_size)

                expected = np.arange(0, length, 1 / sample_rate)
                expected = expected[:size:stride_size] + 1
                assert (expected == y).all()
            assert i == 1
