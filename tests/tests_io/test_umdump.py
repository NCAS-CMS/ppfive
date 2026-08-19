from __future__ import annotations

import re
from contextlib import redirect_stdout
from io import StringIO

from umfive.umdump import main as umdump_main


def _run_umdump(path: str) -> str:
    buf = StringIO()
    with redirect_stdout(buf):
        rc = umdump_main([path])

    assert rc == 0
    return buf.getvalue()


def _variable_names(dump_text: str) -> list[str]:
    names = []
    in_variables = False
    for line in dump_text.splitlines():
        stripped = line.strip()
        if stripped == "variables:":
            in_variables = True
            continue

        if in_variables and stripped.startswith("//"):
            break

        if not in_variables or not stripped or ":" in stripped:
            continue

        m = re.match(r"^(?:\w+\d*|char)\s+(\w+)\s*(?:\(|;)", stripped)
        if m:
            names.append(m.group(1))

    return names


def test_umdump_return_codes():
    assert umdump_main(["bad_arg"]) == 3
    assert umdump_main(["tests/data/test2.pp"]) == 0
