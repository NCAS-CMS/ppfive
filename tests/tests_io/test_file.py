from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

import umfive

# Important: set linewidth=120 to ensure that dump outputs match those
# in reference files (see generate_dump_outputs.py)
np.set_printoptions(linewidth=120)


@pytest.mark.parametrize(
    "dataset, suffix",
    [
        ("cl_umfile", ""),
        ("test2", ".pp"),
        ("extra_data", ".pp"),
        ("umfile", ".pp"),
        ("wgdos_packed", ".pp"),
    ],
)
def test_file(dataset, suffix):
    """Test umfive.File with various input files."""
    # Build the file paths dynamically based on parameters
    input_file = f"tests/data/{dataset}{suffix}"
    dump_file = f"tests/data/{dataset}_dump.txt"

    with umfive.File(input_file) as f:
        with open(dump_file, "r") as d:
            dump_contents = d.read()
            assert (
                f.dump(display=False, data=True).rstrip()
                == dump_contents.rstrip()
            )


def test_File_with_builtin_open_as_input():
    """Test umfive.File with open file handle 'filename' argument."""
    with open("tests/data/test2.pp", "rb") as fh:
        f = umfive.File(fh)
        assert (
            repr(f)
            == "tests/data/test2.pp: <umfive.File: 1 data variable, 9 metadata variables>"
        )


def test_File_with_bytesio_as_input():
    """Test umfive.File with BytesIO 'filename' argument."""
    path = Path("tests/data/test2.pp")
    raw = BytesIO(path.read_bytes())
    f = umfive.File(raw)
    assert (
        repr(f)
        == "<file-like>: <umfive.File: 1 data variable, 9 metadata variables>"
    )


@pytest.mark.parametrize(
    "filename",
    [[], {}, (), None, 0, 3.14, True],
)
def test_File_with_invalid_input(filename):
    """Test umfive.File with bad 'filename' type."""
    with pytest.raises(ValueError):
        umfive.File(filename)


def test_File_with_directory_input():
    """Test umfive.File with 'filename' pointing to a directory."""
    with pytest.raises(IsADirectoryError):
        umfive.File("tests/data")
