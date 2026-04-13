from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
import posixpath
from typing import Any

import numpy as np

from .core import detect_file_type, scan_ff_headers, scan_pp_headers
from .core.variables import build_variable_index
from .io.base import ByteReader
from .io.fileobj import FileObjReader
from .io.local import LocalPosixReader
from .variable import Variable

logger = logging.getLogger(__name__)


class _DimensionScale:
    """Internal pyfive-like dimension-scale dataset for cfdm bridging."""

    def __init__(self, name: str, size: int, file_obj: "File"):
        self.name = name
        self.file = file_obj
        self.shape = (int(size),)
        self.dtype = np.dtype("int32")
        self.maxshape = self.shape
        self.chunks = None
        self.attrs = {
            "CLASS": b"DIMENSION_SCALE",
            "NAME": b"This is a netCDF dimension but not a netCDF variable",
            "_Netcdf4Dimid": 0,
        }

    def __getitem__(self, key):
        return np.arange(self.shape[0], dtype=self.dtype)[key]


class File(Mapping[str, Variable]):
    """A pyfive-style file handle exposing variables as a Mapping."""

    def __init__(
        self,
        filename: str | ByteReader | Any,
        mode: str = "r",
        metadata_buffer_size: int = 1,
        disable_os_cache: bool = False,
        *,
        reader: ByteReader | None = None,
        variable_index: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if mode != "r":
            raise ValueError("ppfive.File currently supports read-only mode='r'")

        if isinstance(filename, ByteReader):
            if reader is not None:
                raise ValueError("Do not provide both filename as ByteReader and reader=")
            reader = filename
            filename = getattr(reader, "path", "<byte-reader>")
        elif reader is None and hasattr(filename, "read") and hasattr(filename, "seek"):
            reader = FileObjReader(filename)
            filename = getattr(filename, "name", "<fileobj>")

        self.filename = str(Path(filename))
        self.mode = mode
        self.metadata_buffer_size = metadata_buffer_size
        self.disable_os_cache = bool(disable_os_cache)
        self._owns_reader = reader is None
        self._reader = reader or LocalPosixReader(
            self.filename,
            disable_os_cache=self.disable_os_cache,
        )
        self._records = []
        self._thread_count = 0
        self._cat_range_allowed = True
        self.parent = None
        self.name = "/"
        self.path = "/"
        self.attrs: dict[str, Any] = {}
        self.groups: dict[str, Any] = {}
        self.dimensions: dict[str, Any] = {}
        self._pyfive_dimension_scales: dict[str, _DimensionScale] = {}

        if variable_index is None:
            file_type = detect_file_type(self._reader)
            self.fmt = file_type.fmt
            self.byte_ordering = file_type.byte_ordering
            self.word_size = file_type.word_size
            if file_type.fmt == "PP":
                self._records = scan_pp_headers(self._reader, file_type)
            else:
                self._records = scan_ff_headers(self._reader, file_type)
            
            if not self._records:
                raise ValueError(
                    f"No valid records found in {self.fmt} file {self.filename}. "
                    f"The file may be corrupted or empty."
                )
            
            variable_index = build_variable_index(
                self._records,
                self._reader,
                self.word_size,
                self.byte_ordering,
                parallel_config={
                    "thread_count": self._thread_count,
                    "cat_range_allowed": self._cat_range_allowed,
                },
            )
        else:
            self.fmt = None
            self.byte_ordering = None
            self.word_size = None

        self._variables = self._build_variables(variable_index or {})
        self.variables = self._variables

    def _build_variables(self, variable_index: dict[str, dict[str, Any]]) -> dict[str, Variable]:
        def _dim_names(shape: tuple[int, ...]) -> tuple[str, ...]:
            return tuple(f"dim_{axis}_{size}" for axis, size in enumerate(shape))

        variables: dict[str, Variable] = {}
        for name, meta in variable_index.items():
            shape = tuple(meta.get("shape", ()))
            dim_names = _dim_names(shape)
            for dim_name, dim_size in zip(dim_names, shape):
                if dim_name not in self._pyfive_dimension_scales:
                    self._pyfive_dimension_scales[dim_name] = _DimensionScale(
                        dim_name, dim_size, self
                    )

            attrs = dict(meta.get("attrs", {}))
            if dim_names:
                # Mirrors the structure expected by cfdm's p5netcdf adapter.
                attrs.setdefault(
                    "DIMENSION_LIST",
                    tuple((dim_name,) for dim_name in dim_names),
                )

            variables[name] = Variable(
                name=name,
                attrs=attrs,
                shape=shape,
                dtype=meta.get("dtype"),
                chunk_shape=meta.get("chunk_shape"),
                data_loader=meta.get("data_loader"),
                file=self,
                parent=self,
                chunk_records=list(meta.get("chunk_records", [])),
            )
        return variables

    @property
    def userblock_size(self) -> int:
        return 0

    @property
    def consolidated_metadata(self) -> bool | None:
        return None

    def get_lazy_view(self, key: str) -> Variable:
        # UM guidance says this cannot be fully implemented yet.
        logger.info("get_lazy_view is not supported; returning normal variable view")
        return self[key]

    def close(self) -> None:
        if self._owns_reader and self._reader is not None:
            self._reader.close()
            self._reader = None

    def set_parallelism(self, thread_count: int = 5, cat_range_allowed: bool = True):
        """Configure experimental chunk/record read parallelism."""
        if thread_count is None:
            thread_count = 0
        thread_count = int(thread_count)
        if thread_count < 0:
            raise ValueError("thread_count must be >= 0")

        self._thread_count = thread_count
        self._cat_range_allowed = bool(cat_range_allowed)

        if self._records:
            variable_index = build_variable_index(
                self._records,
                self._reader,
                self.word_size,
                self.byte_ordering,
                parallel_config={
                    "thread_count": self._thread_count,
                    "cat_range_allowed": self._cat_range_allowed,
                },
            )
            self._variables = self._build_variables(variable_index)

    def __getitem__(self, key: str) -> Variable:
        if not isinstance(key, str):
            raise TypeError("Variable key must be a string")

        path = posixpath.normpath(key)
        if path == ".":
            raise KeyError("'.' does not reference a variable")
        if path.startswith("/"):
            path = path[1:]
        if path.startswith("./"):
            path = path[2:]

        if "/" in path:
            raise KeyError(f"Nested paths are not supported: {key!r}")

        if path in self._pyfive_dimension_scales:
            return self._pyfive_dimension_scales[path]

        return self._variables[path]

    def items(self):
        for name, variable in self._variables.items():
            yield name, variable
        for name, dim in self._pyfive_dimension_scales.items():
            yield name, dim

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def __enter__(self) -> "File":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<PP file "{self.filename}" ({len(self)} variables)>'

    def to_reference_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "path": self.filename,
            "variables": {
                name: variable.to_reference_dict()
                for name, variable in self._variables.items()
            },
        }
