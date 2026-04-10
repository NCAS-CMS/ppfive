from pathlib import Path

import fsspec
import numpy as np

from ppfive import File
from ppfive.io.fsspec_reader import FsspecReader


def test_local_parallel_matches_serial():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"

    with File(str(path)) as f_serial:
        name = list(f_serial)[0]
        serial = f_serial[name][:]

    with File(str(path)) as f_parallel:
        f_parallel.set_parallelism(thread_count=4)
        name = list(f_parallel)[0]
        parallel = f_parallel[name][:]

    assert np.allclose(parallel, serial, rtol=1e-6, atol=1e-6)


def test_fsspec_bulk_range_matches_serial_for_unpacked_data():
    path = Path(__file__).resolve().parents[1] / "data" / "test2.pp"

    with File(str(path)) as f_serial:
        name = list(f_serial)[0]
        serial = f_serial[name][:]

    fs = fsspec.filesystem("file")
    with FsspecReader(fs, str(path)) as reader:
        f = File(str(path), reader=reader)
        f.set_parallelism(thread_count=4, cat_range_allowed=True)
        name = list(f)[0]
        parallel = f[name][:]

    assert np.allclose(parallel, serial, rtol=1e-6, atol=1e-6)


def test_fsspec_bulk_range_matches_serial_for_wgdos_packed_data():
    path = Path(__file__).resolve().parents[1] / "data" / "wgdos_packed.pp"

    with File(str(path)) as f_serial:
        name = list(f_serial)[0]
        serial = f_serial[name][:]

    fs = fsspec.filesystem("file")
    with FsspecReader(fs, str(path)) as reader:
        f = File(str(path), reader=reader)
        f.set_parallelism(thread_count=4, cat_range_allowed=True)
        name = list(f)[0]
        parallel = f[name][:]

    assert np.allclose(parallel, serial, rtol=1e-6, atol=1e-6)
