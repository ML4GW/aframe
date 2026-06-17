from infer.server import _activity_since


def test_activity_since(tmp_path):
    csv = tmp_path / "stats.csv"
    # header + two windows with no inferences (count is 0)
    csv.write_text(
        "timestamp,ip,model,count,queue\n"
        "1.0,localhost,aframe-stream,0,5\n"
        "2.0,localhost,aframe-stream,0,5\n"
    )

    pos, active = _activity_since(csv, 0)
    assert not active

    # a window with count > 0 appended since the last offset
    with open(csv, "a") as f:
        f.write("3.0,localhost,aframe-stream,7,5\n")
    pos2, active2 = _activity_since(csv, pos)
    assert active2
    assert pos2 > pos

    # missing file: no activity, offset unchanged
    assert _activity_since(tmp_path / "missing.csv", 0) == (0, False)
