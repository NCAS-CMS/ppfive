from pathlib import Path

import numpy as np

import ppfive.variable as variable_module
from ppfive import File


def _first_data_variable_name(f: File) -> str:
    return next(
        name
        for name, variable in f.variables.items()
        if variable.attrs.get("CLASS") not in (b"DIMENSION_SCALE", b"AUXILIARY_COORDINATE")
        and "grid_mapping_name" not in variable.attrs
    )


def test_subselection_reads_only_intersecting_chunks(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"

    with File(str(path)) as f:
        name = _first_data_variable_name(f)
        v = f[name]

        calls = []
        original = variable_module.read_record_array

        def _counting_read(reader, rec, word_size, byte_ordering):
            calls.append(int(rec.data_offset))
            return original(reader, rec, word_size, byte_ordering)

        monkeypatch.setattr(variable_module, "read_record_array", _counting_read)

        sub_one = np.asarray(v[0, 0, :, :])
        assert sub_one.shape == (110, 106)
        assert len(calls) == 1

        calls.clear()
        sub_five = np.asarray(v[0, :, :, :])
        assert sub_five.shape == (5, 110, 106)
        assert len(calls) == 4

        # Re-reading the same slice should hit in-memory chunk cache.
        calls.clear()
        sub_five_again = np.asarray(v[0, :, :, :])
        assert sub_five_again.shape == (5, 110, 106)
        assert len(calls) == 0

        # Reading a subset of previously-read chunks should also hit cache.
        calls.clear()
        sub_cached = np.asarray(v[0, 1, :, :])
        assert sub_cached.shape == (110, 106)
        assert len(calls) == 0

        # A full read may be satisfied outside this hook, but it should populate full-data cache.
        calls.clear()
        full = np.asarray(v[:])
        assert full.shape == v.shape
        assert v._data_cache is not None

        # Once full-data cache exists, any slice should avoid additional chunk reads.
        calls.clear()
        sub_from_full_cache = np.asarray(v[0, 0, :, :])
        assert sub_from_full_cache.shape == (110, 106)
        assert len(calls) == 0
