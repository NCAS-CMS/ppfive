from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
import posixpath
from typing import Any

from .core import detect_file_type, scan_ff_headers, scan_pp_headers
from .core.variables import build_variable_index
from .io.base import ByteReader
from .io.local import LocalPosixReader
from .variable import Variable

logger = logging.getLogger(__name__)


class File(Mapping[str, Variable]):
    """A pyfive-style file handle exposing variables as a Mapping."""

    def __init__(
        self,
        filename: str | ByteReader,
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

        if variable_index is None:
            file_type = detect_file_type(self._reader)
            self.fmt = file_type.fmt
            self.byte_ordering = file_type.byte_ordering
            self.word_size = file_type.word_size
            if file_type.fmt == "PP":
                self._records = scan_pp_headers(self._reader, file_type)
            else:
                self._records = scan_ff_headers(self._reader, file_type)
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

    def _build_variables(self, variable_index: dict[str, dict[str, Any]]) -> dict[str, Variable]:
        variables: dict[str, Variable] = {}
        for name, meta in variable_index.items():
            variables[name] = Variable(
                name=name,
                attrs=dict(meta.get("attrs", {})),
                shape=tuple(meta.get("shape", ())),
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

        return self._variables[path]

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
