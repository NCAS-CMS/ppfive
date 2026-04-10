from __future__ import annotations

import os
from pathlib import Path

from .base import ByteReader


class LocalPosixReader(ByteReader):
    """POSIX file reader using pread-style absolute reads."""

    def __init__(self, path: str | os.PathLike[str], disable_os_cache: bool = False):
        self.path = str(Path(path))
        self._fd = os.open(self.path, os.O_RDONLY)
        self._disable_os_cache = disable_os_cache
        self._set_cache_policy()

    def _set_cache_policy(self) -> None:
        # Best effort hint for benchmarking without page cache on macOS.
        if not self._disable_os_cache:
            return

        if hasattr(os, "F_NOCACHE"):
            try:
                import fcntl

                fcntl.fcntl(self._fd, os.F_NOCACHE, 1)
            except OSError:
                # Cache hint is optional; do not fail reads if unsupported.
                pass

    def read_at(self, offset: int, nbytes: int) -> bytes:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if nbytes < 0:
            raise ValueError("nbytes must be >= 0")

        return os.pread(self._fd, nbytes, offset)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "LocalPosixReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
