from __future__ import annotations

import numpy as np

from ppfive.io.base import ByteReader

from .constants import INDEX_LBPACK
from .interpret import get_type_and_num_words
from .models import RecordInfo


def _endian_prefix(byte_ordering: str) -> str:
    if byte_ordering == "little_endian":
        return "<"
    if byte_ordering == "big_endian":
        return ">"
    raise ValueError(f"Unsupported byte_ordering: {byte_ordering!r}")


def _dtype_for_record(rec: RecordInfo, word_size: int, byte_ordering: str) -> np.dtype:
    data_type, _ = get_type_and_num_words(rec.int_hdr, word_size)
    prefix = _endian_prefix(byte_ordering)
    if data_type == "integer":
        return np.dtype(f"{prefix}i{word_size}")
    return np.dtype(f"{prefix}f{word_size}")


def read_record_array(reader: ByteReader, rec: RecordInfo, word_size: int, byte_ordering: str) -> np.ndarray:
    pack = int(rec.int_hdr[INDEX_LBPACK]) % 10
    if pack != 0:
        raise NotImplementedError(f"Packed data mode {pack} is not implemented yet")

    _, nwords = get_type_and_num_words(rec.int_hdr, word_size)
    raw = reader.read_at(rec.data_offset, rec.disk_length)
    need = nwords * word_size
    if len(raw) < need:
        raise ValueError("Short read while loading record data")

    dtype = _dtype_for_record(rec, word_size, byte_ordering)
    return np.frombuffer(raw[:need], dtype=dtype, count=nwords).copy()
