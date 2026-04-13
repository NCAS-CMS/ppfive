from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
import posixpath
from typing import Any

import numpy as np

from .core.constants import INT_MISSING_DATA
from .core import detect_file_type, scan_ff_headers, scan_pp_headers
from .core.variables import build_variable_index
from .io.base import ByteReader
from .io.fileobj import FileObjReader
from .io.local import LocalPosixReader
from .variable import Variable

logger = logging.getLogger(__name__)


class _PyfiveAttrs(dict):
    """Attribute mapping tuned for cfdm/p5netcdf compatibility.

    Keep normal Python `str` values for direct user access, but expose those
    strings as byte scalars when iterating `.items()` so cfdm's p5netcdf
    adapter formats them as scalar text instead of character arrays.
    """

    @staticmethod
    def _coerce_for_items(value: Any) -> Any:
        if isinstance(value, str):
            return np.bytes_(value)

        return value

    def items(self):
        for key, value in super().items():
            yield key, self._coerce_for_items(value)


class _DimensionScale:
    """Internal pyfive-like dimension-scale dataset for cfdm bridging."""

    def __init__(
        self,
        name: str,
        size: int,
        file_obj: "File",
        *,
        standard_name: str | None = None,
        units: str | None = None,
    ):
        self.name = name
        self.file = file_obj
        self.shape = (int(size),)
        self.dtype = np.dtype("int32")
        self.maxshape = self.shape
        self.chunks = None
        self.attrs = {
            "CLASS": b"DIMENSION_SCALE",
            "NAME": b"netCDF dimension coordinate variable",
            "_Netcdf4Dimid": 0,
        }
        if standard_name:
            self.attrs["standard_name"] = np.bytes_(standard_name)
        if units:
            self.attrs["units"] = np.bytes_(units)

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
        self._refresh_variable_views()

    def _refresh_variable_views(self) -> None:
        all_variables: dict[str, Any] = {}
        all_variables.update(self._variables)
        all_variables.update(self._pyfive_dimension_scales)
        self.variables = all_variables

    def _build_variables(self, variable_index: dict[str, dict[str, Any]]) -> dict[str, Variable]:
        def _vertical_dim_name(lbvc: int) -> str:
            if lbvc == 8:
                return "air_pressure"
            return "model_level_number"

        def _semantic_dim_names(shape: tuple[int, ...], attrs: Mapping[str, Any]) -> tuple[str, ...]:
            if len(shape) != 4:
                return tuple(f"dim_{axis}_{size}" for axis, size in enumerate(shape))

            lbvc = int(attrs.get("lbvc", 0) or 0)
            lbuser5 = int(attrs.get("lbuser5", 0) or 0)
            has_pseudo = lbuser5 not in (0, INT_MISSING_DATA)
            z_name = "pseudo_level" if has_pseudo else _vertical_dim_name(lbvc)

            # Mirrors build_variable_index ordering for pseudo-level fields.
            z_first = has_pseudo and shape[0] > 1 and shape[1] > 1
            if z_first:
                return (z_name, "time", "grid_latitude", "grid_longitude")

            return ("time", z_name, "grid_latitude", "grid_longitude")

        def _dim_units(name: str) -> str | None:
            if name == "air_pressure":
                return "Pa"
            if name in ("grid_latitude", "grid_longitude"):
                return "degrees"
            return None

        def _dim_standard_name(name: str) -> str | None:
            if name.startswith("dim_"):
                return None
            return name

        def _resolve_dim_name(base_name: str, dim_size: int) -> str:
            existing = self._pyfive_dimension_scales.get(base_name)
            if existing is None:
                return base_name
            if existing.shape == (int(dim_size),):
                return base_name

            return f"{base_name}_{dim_size}"

        variables: dict[str, Variable] = {}
        for name, meta in variable_index.items():
            shape = tuple(meta.get("shape", ()))
            attrs = _PyfiveAttrs(dict(meta.get("attrs", {})))

            raw_dim_names = _semantic_dim_names(shape, attrs)
            dim_names = tuple(
                _resolve_dim_name(dim_name, dim_size)
                for dim_name, dim_size in zip(raw_dim_names, shape)
            )
            for dim_name, dim_size in zip(dim_names, shape):
                if dim_name not in self._pyfive_dimension_scales:
                    self._pyfive_dimension_scales[dim_name] = _DimensionScale(
                        dim_name,
                        dim_size,
                        self,
                        standard_name=(dim_name if "dim_" not in dim_name else None),
                        units=_dim_units(dim_name),
                    )

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
            self._refresh_variable_views()

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

        return self.variables[path]

    def items(self):
        return self.variables.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

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
