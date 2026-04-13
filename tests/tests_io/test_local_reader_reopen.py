from pathlib import Path

from ppfive.io.local import LocalPosixReader


def test_local_reader_reopens_after_close(tmp_path: Path):
    p = tmp_path / "sample.bin"
    p.write_bytes(b"abcdef")

    reader = LocalPosixReader(p)
    assert reader.read_at(1, 3) == b"bcd"
    reader.close()

    # Should transparently reopen and still serve absolute reads.
    assert reader.read_at(2, 2) == b"cd"
    reader.close()
