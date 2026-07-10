"""Optional long-run regression for the ill-conditioned quadratic case."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

from experiments.benchmarks.run_regular import _worker_mc_regular


@pytest.mark.slow
def test_quadbad_disgrem_trace_remains_finite() -> None:
    protocol = {
        "Nagent": 10,
        "p_edge": 0.5,
        "maxIt": 1500,
        "tol": 1e-8,
        "tolType": "relF",
        "verbose": False,
        "showPlots": False,
        "far": False,
        "useWorst": False,
        "nStart": 1,
        "d_override": 10,
        "info": 2,
        "NC": 3,
        "NC_schedule": "log",
        "log_p": 3.0,
        "log_c_mix": 2.0,
        "NC_max": 10,
        "countComm": True,
    }
    _, logs, _ = _worker_mc_regular(("quadbad", 0, protocol))
    trace = np.asarray(logs["DisGrem"].get("ValueF", []), dtype=float)

    assert trace.size > 0
    assert np.all(np.isfinite(trace))
