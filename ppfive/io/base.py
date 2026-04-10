from __future__ import annotations

from abc import ABC, abstractmethod


class ByteReader(ABC):
    """Minimal transport boundary for random-access reads."""

    @abstractmethod
    def read_at(self, offset: int, nbytes: int) -> bytes:
        """Read ``nbytes`` from absolute byte ``offset``."""

    def close(self) -> None:
        """Close underlying resources if needed."""
