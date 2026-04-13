from __future__ import annotations

import csv
import re
from functools import lru_cache
from importlib.resources import files

# Matches numeric value and optional exponent; mirrors cf loader behavior.
_NUMBER_REGEX = r"([-+]?\d*\.?\d+(e[-+]?\d+)?)"

# Column indices in STASH_to_CF.txt
_MODEL = 0
_STASH = 1
_NAME = 2
_UNITS = 3
_VALID_FROM = 4
_VALID_TO = 5
_STANDARD_NAME = 6
_CF_EXTRA = 7
_PP_EXTRA = 8


def _parse_version(value: str):
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_cf_extra(value: str) -> dict[str, object]:
    out: dict[str, object] = {}
    if not value:
        return out

    for token in value.split():
        if token.startswith("height="):
            out["height"] = re.split(_NUMBER_REGEX, token, re.IGNORECASE)[1:4:2]
        elif token.startswith("below_"):
            out["below"] = re.split(_NUMBER_REGEX, token, re.IGNORECASE)[1:4:2]
        elif token.startswith("where_"):
            out["where"] = token.replace("where_", "where ", 1)
        elif token.startswith("over_"):
            out["over"] = token.replace("over_", "over ", 1)

    return out


@lru_cache(maxsize=1)
def load_stash_table() -> dict[tuple[int, int], tuple[tuple[object, ...], ...]]:
    table_path = files("ppfive").joinpath("data/STASH_to_CF.txt")
    stash2sn: dict[tuple[int, int], tuple[tuple[object, ...], ...]] = {}

    with table_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="!", skipinitialspace=True))

    for row in rows:
        if not row or row[0].startswith("#"):
            continue

        # Normalize to expected width.
        if len(row) < 9:
            row = row + [""] * (9 - len(row))

        key = (int(row[_MODEL]), int(row[_STASH]))
        name = row[_NAME]
        units = row[_UNITS] or None
        valid_from = _parse_version(row[_VALID_FROM])
        valid_to = _parse_version(row[_VALID_TO])
        standard_name = row[_STANDARD_NAME] or None
        cf_info = _parse_cf_extra(row[_CF_EXTRA])
        pp_extra = row[_PP_EXTRA].rstrip()

        entry = (
            name,
            units,
            valid_from,
            valid_to,
            standard_name,
            cf_info,
            pp_extra,
        )

        if key in stash2sn:
            stash2sn[key] += (entry,)
        else:
            stash2sn[key] = (entry,)

    return stash2sn


def stash_records(submodel: int, stash_code: int):
    return load_stash_table().get((int(submodel), int(stash_code)), ())
