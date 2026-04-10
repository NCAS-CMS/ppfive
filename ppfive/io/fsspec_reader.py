from __future__ import annotations

from .base import ByteReader


class FsspecReader(ByteReader):
    """fsspec-backed byte reader with absolute reads."""

    def __init__(self, filesystem, path: str):
        self.filesystem = filesystem
        self.path = path
        self._fh = self.filesystem.open(self.path, "rb")

    def read_at(self, offset: int, nbytes: int) -> bytes:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if nbytes < 0:
            raise ValueError("nbytes must be >= 0")

        self._fh.seek(offset)
        return self._fh.read(nbytes)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "FsspecReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
