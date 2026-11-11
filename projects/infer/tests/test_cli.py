from infer.cli import _batches


def test_batches_start_a_new_batch_when_full_or_at_a_file_change():
    a, b = {"fname": "a.hdf5"}, {"fname": "b.hdf5"}
    branches = {"0": a, "1": a, "2": a, "3": b}

    batches = _batches(branches, shifts_per_batch=2)

    assert batches == [[("0", a), ("1", a)], [("2", a)], [("3", b)]]


def test_batch_size_counts_from_each_file_change():
    a, b = {"fname": "a.hdf5"}, {"fname": "b.hdf5"}
    branches = {"0": a, "1": b, "2": b, "3": b}

    batches = _batches(branches, shifts_per_batch=2)

    # Cutting every 2 branches of the group would give [b], [b, b]
    assert batches == [[("0", a)], [("1", b), ("2", b)], [("3", b)]]
