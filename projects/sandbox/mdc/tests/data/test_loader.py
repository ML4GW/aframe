import h5py
import numpy as np
import pytest
from infer.data import loader


@pytest.fixture(params=[0, 1, 2])
def shift(request):
    return request.param


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture
def data_dir(shift, sample_rate, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = np.arange(1024)

    for i in range(4):
        start = 1234567890 + i * 16
        duration = 8
        fname = tmp_path / f"tmp-{start}-{duration}.h5"
        with h5py.File(fname, "w") as f:
            f["H1"] = x
            f["L1"] = x - int(shift * sample_rate)
    return tmp_path


@pytest.fixture(params=[1, 2.5, 4])
def chunk_length(request):
    return request.param


@pytest.fixture
def validate_fname_gen(shift, chunk_length, sample_rate):
    chunk_size = int(chunk_length * sample_rate)
    num_chunks, rem = divmod(8 - shift, chunk_length)
    if rem:
        num_chunks += 1

    def func(it):
        outputs = []
        for i, x in enumerate(it):
            assert x.shape[0] == 2

            if (i + 1) < num_chunks:
                assert x.shape[-1] == chunk_size
            elif (i + 1) == num_chunks and not rem:
                assert x.shape[-1] == chunk_size
            elif (i + 1) == num_chunks:
                assert x.shape[-1] == int(rem * sample_rate)

            outputs.append(x)
        assert (i + 1) == num_chunks

        output = np.concatenate(outputs, axis=-1)
        expected_length = int(sample_rate * (8 - shift))
        assert output.shape == (2, expected_length)
        assert (output == np.arange(expected_length)).all()

    return func


def validate_start_stop(start, stop, shift):
    duration = stop - start
    assert duration == (8 - shift)

    div, rem = divmod(start - 1234567890, 16)
    assert not rem
    assert div in range(4)


def test_load_fname(
    data_dir, shift, chunk_length, sample_rate, validate_fname_gen
):
    fname = next(data_dir.iterdir())
    shifts = [0, int(shift * sample_rate)]
    chunk_size = int(chunk_length * sample_rate)

    it = loader.load_fname(fname, ["H1", "L1"], shifts, chunk_size)
    validate_fname_gen(it)


def test_crawl_through_directory(data_dir, shift, chunk_length, sample_rate):
    crawler = loader.crawl_through_directory(
        data_dir,
        ["H1", "L1"],
        chunk_length=chunk_length,
        sample_rate=sample_rate,
        shifts=[0, shift],
    )

    num_chunks, rem = divmod(8 - shift, chunk_length)
    if rem:
        num_chunks += 1

    outputs = []
    for i, x in enumerate(crawler):
        if isinstance(x, tuple):
            assert not outputs
            start, stop = x
            validate_start_stop(start, stop, shift)
        elif x is None and outputs:
            output = np.concatenate(outputs, axis=-1)
            expected_length = int(sample_rate * (8 - shift))
            assert output.shape == (2, expected_length)
            assert (output == np.arange(expected_length)).all()
            outputs = []
        elif x is None:
            assert not outputs
            # all of the chunks, plus a (start, stop)
            # for each segment, plus a None for each segment
            assert i == (num_chunks * 4) + 8
        else:
            outputs.append(x)

    # confirm that the loop broke after
    # the back-to-back None
    assert i == (num_chunks * 4) + 8


def test_chunked_segment_loader(
    data_dir, sample_rate, chunk_length, shift, validate_fname_gen
):
    dataloader = loader.ChunkedSegmentLoader(
        data_dir, ["H1", "L1"], chunk_length, sample_rate, [0, shift]
    )

    with dataloader as it:
        for (start, stop), gen in it:
            validate_start_stop(start, stop, shift)
            validate_fname_gen(gen)
