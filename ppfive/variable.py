from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Variable:
    """Minimal pyfive-like variable surface for PP/Fields data."""

    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] = field(default_factory=tuple)
    dtype: Any = None
    chunk_shape: tuple[int, ...] | None = None
    data_loader: Callable[[], Any] | None = None
    file: Any = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        if not self.shape:
            return 0
        return int(np.prod(self.shape))

    def _materialize(self) -> Any:
        if self.data_loader is None:
            return None
        return self.data_loader()

    def __getitem__(self, key):
        data = self._materialize()
        if data is None:
            return None
        return data[key]

    def __array__(self):
        data = self._materialize()
        if data is None:
            raise TypeError("Variable has no data loader configured")
        return np.asarray(data)

    # Dataset-like attributes that are not currently meaningful for PP/Fields.
    @property
    def chunks(self):
        return None

    @property
    def compression(self):
        return None

    @property
    def compression_opts(self):
        return None

    @property
    def shuffle(self):
        return None

    @property
    def fletcher32(self):
        return None

    @property
    def maxshape(self):
        return None

    @property
    def fillvalue(self):
        return None

    @property
    def scaleoffset(self):
        return None

    @property
    def external(self):
        return None

    @property
    def is_virtual(self):
        return None
