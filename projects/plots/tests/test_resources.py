from plots.legacy.main import GATE_PATHS, VETO_DEFINER_FILE

EXPECTED_IFOS = ["H1", "L1"]


def test_veto_definer_file_exists():
    assert VETO_DEFINER_FILE.exists(), VETO_DEFINER_FILE
    assert VETO_DEFINER_FILE.stat().st_size > 0
    assert VETO_DEFINER_FILE.suffix == ".xml"


def test_gate_paths_exist():
    assert sorted(GATE_PATHS) == EXPECTED_IFOS

    for ifo, path in GATE_PATHS.items():
        assert path.exists(), path
        assert path.stat().st_size > 0
        assert path.name.startswith(f"{ifo}-"), path.name
