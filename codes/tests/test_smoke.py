"""Fast executable contracts for the experiment and solver registry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

from experiments.benchmarks.run_regular import _worker_mc_regular
from utils.alg.alg_bank import get_alg_bank
from utils.helper.graph import generate_fully_connected_graph, spectral_gap
from utils.data.load_dataset import load_dataset


def _small_protocol() -> dict[str, object]:
    return {
        "Nagent": 4,
        "p_edge": 0.8,
        "maxIt": 4,
        "iteration_budget_override": 4,
        "tol": 1e-12,
        "tolType": "combo",
        "verbose": False,
        "showPlots": False,
        "far": False,
        "useWorst": False,
        "nStart": 1,
        "d_override": 3,
        "info": 2,
        "NC": 1,
        "NC_schedule": "fixed",
        "log_p": 3.0,
        "log_c_mix": 2.0,
        "NC_max": 2,
        "countComm": True,
    }


def test_regular_worker_executes_every_main_algorithm() -> None:
    index, logs, f0 = _worker_mc_regular(("ridge", 0, _small_protocol()))

    assert index == 0
    assert np.isfinite(f0)
    assert set(logs) == {name for name, _ in get_alg_bank("MainComp")}
    proposed = {"DisGrem", "CeDisGrem", "AdaDisGrem", "CeAdaDisGrem"}
    for name, record in logs.items():
        values = np.asarray(record.get("ValueF", []), dtype=float)
        assert values.size > 0, name
        if name in proposed:
            assert not record.get("fail"), f"{name}: {record.get('failReason')}"
            assert np.all(np.isfinite(values)), name
        elif not np.all(np.isfinite(values)):
            assert record.get("fail") or record.get("__failed__"), name
            assert record.get("failReason"), name


def test_regular_worker_is_seed_reproducible() -> None:
    first = _worker_mc_regular(("ridge", 2, _small_protocol()))
    second = _worker_mc_regular(("ridge", 2, _small_protocol()))

    assert first[2] == pytest.approx(second[2], rel=0.0, abs=0.0)
    for name in first[1]:
        np.testing.assert_allclose(
            first[1][name]["ValueF"],
            second[1][name]["ValueF"],
            rtol=0.0,
            atol=0.0,
        )


def test_complete_graph_has_exact_consensus_matrix() -> None:
    adjacency, weights = generate_fully_connected_graph(5)

    np.testing.assert_allclose(adjacency, np.ones((5, 5)) - np.eye(5))
    np.testing.assert_allclose(weights.sum(axis=0), np.ones(5))
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(5))
    assert spectral_gap(weights) == pytest.approx(1.0)


def test_dataset_loader_reuses_immutable_arrays() -> None:
    load_dataset.cache_clear()
    first = load_dataset("svmguide3", standardize="zscore", label_style="pm1")
    second = load_dataset("svmguide3", standardize="zscore", label_style="pm1")

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert not first[0].flags.writeable
    assert not first[1].flags.writeable
    assert load_dataset.cache_info().hits == 1
