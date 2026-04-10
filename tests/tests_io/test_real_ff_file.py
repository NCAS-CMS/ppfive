from pathlib import Path

import cf
import numpy as np

from ppfive import File


def test_real_fields_file_data_parity_with_cf():
    path = Path(__file__).resolve().parents[1] / "data" / "cl_umfile"
    assert path.exists(), "Expected fixture tests/data/cl_umfile"

    with File(str(path)) as f:
        assert f.fmt == "FF"
        names = list(f)
        assert names

        first = names[0]
        assert first == "m01s00i001"

        arr = f[first][:]
        assert arr.shape == (12, 1, 37, 48)

    baseline = cf.read(str(path))[0].array
    assert baseline.shape == np.squeeze(arr).shape
    assert baseline.dtype == arr.dtype
    assert np.allclose(np.squeeze(arr), baseline, rtol=1e-6, atol=1e-6)
