from pathlib import Path

from ppfive import File


def test_real_pp_file_exposes_variables():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"
    assert path.exists(), "Expected fixture tests/data/test2.pp"

    with File(str(path)) as f:
        names = list(f)
        assert names, "Expected at least one parsed variable name"

        first = names[0]
        assert first.startswith("m")
        v = f[first]
        assert len(v.shape) == 4
