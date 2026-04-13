import pytest

from ppfive import File


def test_ppfive_file_registers_as_pyfive_file():
    pyfive = pytest.importorskip("pyfive")

    with File("tests/data/test2.pp") as f:
        assert isinstance(f, pyfive.File)
