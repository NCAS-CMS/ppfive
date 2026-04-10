from pathlib import Path

import cf
import numpy as np
import pytest

from ppfive import File


@pytest.mark.parametrize(
    ("filename", "expected_name", "expected_shape"),
    [
        ("aaaaoa.pmh8dec.pp", "m01s03i245", (1, 1, 30, 24)),
        ("extra_data.pp", "m01s03i236", (1, 1, 100, 3)),
        ("test2.pp", "m01s15i201", (3, 5, 110, 106)),
        ("umfile.pp", "m01s00i001", (3, 1, 73, 96)),
        ("wgdos_packed.pp", "m01s30i201", (1, 1, 145, 192)),
    ],
)
def test_real_pp_file_exposes_variables(filename, expected_name, expected_shape):
    path = Path(__file__).resolve().parents[1] / "data" / filename
    assert path.exists(), f"Expected fixture tests/data/{filename}"

    with File(str(path)) as f:
        names = list(f)
        assert names, "Expected at least one parsed variable name"

        first = names[0]
        assert first == expected_name
        v = f[first]
        assert len(v.shape) == 4
        assert v.shape == expected_shape

        arr = v[:]
        assert arr.dtype == np.dtype("float32")
        assert arr.shape == expected_shape

    baseline = cf.read(str(path))[0].array
    assert baseline.shape == np.squeeze(arr).shape
    assert baseline.dtype == arr.dtype
    assert np.allclose(np.squeeze(arr), baseline, rtol=1e-6, atol=1e-6)
