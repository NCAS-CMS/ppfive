import ppfive
from pathlib import Path
import fsspec
from ppfive.io import FsspecReader
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FILENAME = "xjanpa.pa19910301"
EXAMPLE_FILE = str(Path.home()/"data"/FILENAME)
HTTP = f"https://gws-access.jasmin.ac.uk/public/hiresgw/{FILENAME}.pp" 


fs = fsspec.filesystem("https")
reader = FsspecReader(fs, HTTP)
ff_ctx = ppfive.File(HTTP, reader=reader)

with ff_ctx as ff:
    for y in ff:
        if y.startswith('m0'):
            variable = ff[y]
            try:
                name = variable.attrs['standard_name']
            except KeyError:
                name = variable.attrs.get('long_name', 'unknown')
                print('Doing ', y, '(', name, ')')
            wgdos = {True:'T', False:'F'}[bool(variable.attrs.get('is_wgdos_packed', False))]
            s = variable.shape
            ndim = len(s)
            indices = tuple([0] * (ndim - 1) + [slice(None)])

            p0 = time.perf_counter()
            x = variable[indices]
            p1 = time.perf_counter()
            var_elapsed = p1 - p0
            print(y, f'{x.shape} {x.dtype} {indices} elapsed={var_elapsed:.6f}s W={wgdos}')
            

