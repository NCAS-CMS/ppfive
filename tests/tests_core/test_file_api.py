import logging

import numpy as np

from ppfive import File


def test_file_iterates_variable_names():
    f = File(
        __file__,
        variable_index={
            "temp": {
                "shape": (2, 2),
                "dtype": "f8",
                "chunk_shape": (1, 2),
                "attrs": {"units": "K"},
                "data_loader": lambda: np.array([[1.0, 2.0], [3.0, 4.0]]),
            }
        },
    )

    assert list(f) == ["temp"]
    v = f["temp"]
    assert v.shape == (2, 2)
    assert v.dtype == "f8"
    assert v.chunk_shape == (1, 2)
    assert v.chunks == (1, 2)
    assert v.attrs["units"] == "K"
    assert np.all(v[:] == np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert len(v) == 2
    assert v.len() == 2
    assert v.value.shape == (2, 2)
    assert v.parent is f
    assert v.file is f
    assert "shape (2, 2)" in repr(v)


def test_variable_dataset_like_helpers():
    f = File(
        __file__,
        variable_index={
            "temp": {
                "shape": (2, 2),
                "dtype": "f8",
                "chunk_shape": (1, 1),
                "data_loader": lambda: np.array([[1.25, 2.25], [3.25, 4.25]]),
            }
        },
    )

    v = f["temp"]
    target = np.empty((2, 2), dtype="f8")
    v.read_direct(target)
    assert np.allclose(target, np.array([[1.25, 2.25], [3.25, 4.25]]))

    with v.astype("f4"):
        cast = v[:]

    assert cast.dtype == np.dtype("float32")
    assert v.id.shape == (2, 2)
    assert v.id.dtype == np.dtype("float64")
    assert list(v.iter_chunks()) == [
        (slice(0, 1, 1), slice(0, 1, 1)),
        (slice(0, 1, 1), slice(1, 2, 1)),
        (slice(1, 2, 1), slice(0, 1, 1)),
        (slice(1, 2, 1), slice(1, 2, 1)),
    ]
    assert v.dims is None


def test_get_lazy_view_falls_back_with_log(caplog):
    f = File(__file__, variable_index={"x": {"data_loader": lambda: np.array([1, 2])}})

    with caplog.at_level(logging.INFO):
        view = f.get_lazy_view("x")

    assert view is f["x"]
    assert "get_lazy_view is not supported" in caplog.text
