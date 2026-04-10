from pathlib import Path

import pytest

from ppfive import File
from ppfive.io.local import LocalPosixReader


def test_local_reader_read_at(tmp_path: Path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"abcdefghij")

    with LocalPosixReader(p) as reader:
        assert reader.read_at(2, 4) == b"cdef"


def test_file_accepts_local_reader_as_first_argument():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"

    with LocalPosixReader(path) as reader:
        f = File(reader)
        names = list(f)
        arr = f[names[0]][:]

    assert names
    assert arr.shape == (3, 5, 110, 106)


def test_file_rejects_reader_conflict():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"

    with LocalPosixReader(path) as reader:
        with pytest.raises(ValueError):
            File(reader, reader=reader)
