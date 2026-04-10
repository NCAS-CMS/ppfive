from pathlib import Path

import cf
import numpy as np

from ppfive import File


def test_real_pp_file_exposes_variables():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"
    assert path.exists(), "Expected fixture tests/data/test2.pp"

    with File(str(path)) as f:
        names = list(f)
        assert names, "Expected at least one parsed variable name"

        first = names[0]
        assert first == "m01s15i201"
        v = f[first]
        assert len(v.shape) == 4
        assert v.shape == (3, 5, 110, 106)

        arr = v[:]
        assert arr.dtype == np.dtype("float32")
        assert arr.shape == (3, 5, 110, 106)

    baseline = cf.read(str(path))[0].array
    assert baseline.shape == arr.shape
    assert baseline.dtype == arr.dtype
    assert np.allclose(arr, baseline, rtol=1e-6, atol=1e-6)
