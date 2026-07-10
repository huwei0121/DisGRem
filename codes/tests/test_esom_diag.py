"""Optional ESOM sensitivity diagnostic with executable assertions."""

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
def test_esom_produces_a_finite_trace_on_paper_ridge_case() -> None:
    protocol = {
        "Nagent": 10,
        "p_edge": 0.5,
        "maxIt": 50,
        "tol": 1e-12,
        "tolType": "combo",
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
    _, logs, _ = _worker_mc_regular(("ridge", 0, protocol))
    trace = np.asarray(logs["ESOM"].get("ValueF", []), dtype=float)

    assert trace.size > 0
    assert np.all(np.isfinite(trace))
