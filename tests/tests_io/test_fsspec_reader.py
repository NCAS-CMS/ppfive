from pathlib import Path

import fsspec
import numpy as np

from ppfive import File
from ppfive.io.fsspec_reader import FsspecReader


def test_fsspec_reader_read_at(tmp_path: Path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"abcdefghij")

    fs = fsspec.filesystem("file")
    with FsspecReader(fs, str(p)) as reader:
        assert reader.read_at(2, 4) == b"cdef"


def test_file_can_parse_via_fsspec_reader():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"
    fs = fsspec.filesystem("file")

    with FsspecReader(fs, str(path)) as reader:
        f = File(str(path), reader=reader)
        names = list(f)
        assert names
        arr = f[names[0]][:]

    assert arr.shape == (3, 5, 110, 106)
    assert arr.dtype == np.dtype("float32")
