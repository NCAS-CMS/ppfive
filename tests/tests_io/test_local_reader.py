from pathlib import Path

from ppfive.io.local import LocalPosixReader


def test_local_reader_read_at(tmp_path: Path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"abcdefghij")

    with LocalPosixReader(p) as reader:
        assert reader.read_at(2, 4) == b"cdef"
