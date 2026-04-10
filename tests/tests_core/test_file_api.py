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
    assert v.attrs["units"] == "K"
    assert np.all(v[:] == np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_get_lazy_view_falls_back_with_log(caplog):
    f = File(__file__, variable_index={"x": {"data_loader": lambda: np.array([1, 2])}})

    with caplog.at_level(logging.INFO):
        view = f.get_lazy_view("x")

    assert view is f["x"]
    assert "get_lazy_view is not supported" in caplog.text
