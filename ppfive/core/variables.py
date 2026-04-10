from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import (
    INDEX_BDX,
    INDEX_BDY,
    INDEX_BGOR,
    INDEX_BHLEV,
    INDEX_BLEV,
    INDEX_BPLAT,
    INDEX_BPLON,
    INDEX_BZX,
    INDEX_BZY,
    INDEX_LBCODE,
    INDEX_LBDAT,
    INDEX_LBDATD,
    INDEX_LBDAY,
    INDEX_LBDAYD,
    INDEX_LBFT,
    INDEX_LBHEM,
    INDEX_LBHR,
    INDEX_LBHRD,
    INDEX_LBLREC,
    INDEX_LBLEV,
    INDEX_LBMIN,
    INDEX_LBMIND,
    INDEX_LBMON,
    INDEX_LBMOND,
    INDEX_LBNPT,
    INDEX_LBPACK,
    INDEX_LBPROC,
    INDEX_LBROW,
    INDEX_LBTIM,
    INDEX_LBUSER4,
    INDEX_LBUSER7,
    INDEX_LBVC,
    INDEX_LBYR,
    INDEX_LBYRD,
    INT_MISSING_DATA,
)
from .interpret import get_type
from .models import RecordInfo


def _float_key(val: float) -> float:
    return round(float(val), 9)


def _between_var_key(rec: RecordInfo) -> tuple[Any, ...]:
    ih = rec.int_hdr
    rh = rec.real_hdr
    return (
        int(ih[INDEX_LBUSER4]),
        int(ih[INDEX_LBUSER7]),
        int(ih[INDEX_LBCODE]),
        int(ih[INDEX_LBVC]),
        int(ih[INDEX_LBTIM]),
        int(ih[INDEX_LBPROC]),
        _float_key(rh[INDEX_BPLAT]),
        _float_key(rh[INDEX_BPLON]),
        int(ih[INDEX_LBHEM]),
        int(ih[INDEX_LBROW]),
        int(ih[INDEX_LBNPT]),
        _float_key(rh[INDEX_BGOR]),
        _float_key(rh[INDEX_BZY]),
        _float_key(rh[INDEX_BDY]),
        _float_key(rh[INDEX_BZX]),
        _float_key(rh[INDEX_BDX]),
    )


def _within_var_key(rec: RecordInfo) -> tuple[Any, ...]:
    ih = rec.int_hdr
    rh = rec.real_hdr
    lblev = int(ih[INDEX_LBLEV])
    lblev_rank = -1 if lblev == 9999 else lblev
    return (
        int(ih[INDEX_LBFT]),
        int(ih[INDEX_LBYR]),
        int(ih[INDEX_LBMON]),
        int(ih[INDEX_LBDAT]),
        int(ih[INDEX_LBDAY]),
        int(ih[INDEX_LBHR]),
        int(ih[INDEX_LBMIN]),
        int(ih[INDEX_LBYRD]),
        int(ih[INDEX_LBMOND]),
        int(ih[INDEX_LBDATD]),
        int(ih[INDEX_LBDAYD]),
        int(ih[INDEX_LBHRD]),
        int(ih[INDEX_LBMIND]),
        lblev_rank,
        _float_key(rh[INDEX_BLEV]),
        _float_key(rh[INDEX_BHLEV]),
    )


def _record_is_skippable(rec: RecordInfo) -> bool:
    ih = rec.int_hdr

    # Mirrors key skip logic in process_vars.c
    if int(ih[INDEX_LBNPT]) == INT_MISSING_DATA:
        return True
    if int(ih[INDEX_LBROW]) == INT_MISSING_DATA:
        return True

    compression = (int(ih[INDEX_LBPACK]) // 10) % 10
    if compression == 1:
        return True

    return False


def _stash_name(rec: RecordInfo) -> str:
    ih = rec.int_hdr
    model = int(ih[INDEX_LBUSER7])
    user4 = int(ih[INDEX_LBUSER4])
    section = user4 // 1000
    item = user4 % 1000
    return f"m{model:02d}s{section:02d}i{item:03d}"


def _dtype_name(first: RecordInfo, word_size: int) -> str:
    kind = get_type(first.int_hdr)
    if kind == "integer":
        return "int32" if word_size == 4 else "int64"
    return "float32" if word_size == 4 else "float64"


def build_variable_index(records: list[RecordInfo], word_size: int) -> dict[str, dict[str, Any]]:
    filtered = [r for r in records if not _record_is_skippable(r)]
    ordered = sorted(filtered, key=lambda r: (_between_var_key(r), _within_var_key(r)))

    grouped: dict[tuple[Any, ...], list[RecordInfo]] = defaultdict(list)
    for rec in ordered:
        grouped[_between_var_key(rec)].append(rec)

    variable_index: dict[str, dict[str, Any]] = {}
    name_counts: dict[str, int] = defaultdict(int)

    for _, recs in grouped.items():
        first = recs[0]
        base = _stash_name(first)
        name_counts[base] += 1
        name = base if name_counts[base] == 1 else f"{base}_{name_counts[base]}"

        z_levels = {
            (int(r.int_hdr[INDEX_LBLEV]), _float_key(r.real_hdr[INDEX_BLEV]), _float_key(r.real_hdr[INDEX_BHLEV]))
            for r in recs
        }
        t_steps = {
            (
                int(r.int_hdr[INDEX_LBFT]),
                int(r.int_hdr[INDEX_LBYR]),
                int(r.int_hdr[INDEX_LBMON]),
                int(r.int_hdr[INDEX_LBDAT]),
                int(r.int_hdr[INDEX_LBHR]),
                int(r.int_hdr[INDEX_LBMIN]),
                int(r.int_hdr[INDEX_LBYRD]),
                int(r.int_hdr[INDEX_LBMOND]),
                int(r.int_hdr[INDEX_LBDATD]),
                int(r.int_hdr[INDEX_LBHRD]),
                int(r.int_hdr[INDEX_LBMIND]),
            )
            for r in recs
        }

        ny = int(first.int_hdr[INDEX_LBROW])
        nx = int(first.int_hdr[INDEX_LBNPT])
        nz = len(z_levels)
        nt = len(t_steps)

        variable_index[name] = {
            "attrs": {
                "stash_model": int(first.int_hdr[INDEX_LBUSER7]),
                "stash_code": int(first.int_hdr[INDEX_LBUSER4]),
            },
            "shape": (nt, nz, ny, nx),
            "dtype": _dtype_name(first, word_size),
            "chunk_shape": (1, 1, ny, nx),
            "records": recs,
        }

    return variable_index
