"""Tests for auditable experiment manifests and tuning isolation."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

from experiments import protocol
from experiments.benchmarks import run_fair_tuning as tuning_module
from experiments.benchmarks.run_fair_tuning import _run_task
from scripts.freeze_artifact import freeze, verify


def _double_task(value: int) -> int:
    return value * 2


def _fail_task(value: int) -> int:
    raise AssertionError(f"cache miss for {value}")


def test_source_hash_excludes_result_files(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    dataset = tmp_path / "sample.libsvm"
    dataset.write_text("1 1:0.5\n", encoding="utf-8")
    result = tmp_path / "results" / "raw.json"
    result.parent.mkdir()
    result.write_text("first\n", encoding="utf-8")
    before = protocol.source_tree_sha256(tmp_path)

    result.write_text("second\n", encoding="utf-8")

    assert protocol.source_tree_sha256(tmp_path) == before
    dataset.write_text("1 1:0.75\n", encoding="utf-8")
    assert protocol.source_tree_sha256(tmp_path) != before


def test_gzip_writer_emits_strict_json(tmp_path: Path) -> None:
    destination = tmp_path / "raw.json.gz"
    protocol.write_json_gz(
        destination,
        {
            "schema_version": 1,
            "values": [np.nan, np.inf, -np.inf, np.float64(2.0)],
        },
    )

    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        raw = handle.read()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw)["values"] == [None, None, None, 2.0]


def test_iteration_budget_override_is_explicit_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {"maxIt": 800}
    assert protocol.resolve_iteration_budget(policy, {"maxIt": 500}) == 800
    assert protocol.resolve_iteration_budget(
        policy,
        {"maxIt": 500, "iteration_budget_override": 7},
    ) == 7

    monkeypatch.setenv("LOG_SCHEDULE_MAXIT", "7")
    run_protocol = protocol.mode_protocol("regular")
    assert set(run_protocol["per_objective_iteration_budgets"].values()) == {7}
    assert protocol.paper_grade_protocol_errors("regular", run_protocol)


def test_policy_budgets_match_frozen_paper_protocol() -> None:
    from problems.init_policy import init_policy

    _, policies, _ = init_policy("regular")
    actual = {
        objective: int(policies[objective]["maxIt"])
        for objective in protocol.PRIMARY_OBJECTIVES
    }
    assert actual == protocol.PAPER_OBJECTIVE_ITERATION_BUDGETS


def test_paper_grade_scale_protocol_survives_json_round_trip() -> None:
    serialized = json.loads(json.dumps(protocol.mode_protocol("scale")))

    assert protocol.paper_grade_protocol_errors("scale", serialized) == []


def test_nanmean_columns_preserves_empty_columns_as_nan() -> None:
    values = np.array([[1.0, np.nan], [3.0, np.nan]])
    result = protocol.nanmean_columns(values)
    assert result[0] == pytest.approx(2.0)
    assert np.isnan(result[1])


def test_cached_tasks_preserve_order_and_recover_from_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(protocol, "source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setenv("DISGREM_USE_CACHE", "0")

    first = protocol.execute_cached_tasks(
        [3, 1, 2],
        _double_task,
        cache_root=tmp_path,
        namespace="unit",
        max_workers=2,
        progress_every=0,
    )
    assert first == [6, 2, 4]

    monkeypatch.setenv("DISGREM_USE_CACHE", "1")
    second = protocol.execute_cached_tasks(
        [3, 1, 2],
        _double_task,
        cache_root=tmp_path,
        namespace="unit",
        max_workers=2,
        progress_every=0,
    )
    assert second == first

    cache_files = sorted(tmp_path.rglob("*.pkl"))
    assert len(cache_files) == 3
    cache_files[0].write_bytes(b"not a pickle")

    recovered = protocol.execute_cached_tasks(
        [3, 1, 2],
        _double_task,
        cache_root=tmp_path,
        namespace="unit",
        max_workers=2,
        progress_every=0,
    )
    assert recovered == first


def test_cached_task_signature_binds_worker_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(protocol, "source_tree_sha256", lambda: "b" * 64)
    monkeypatch.setenv("DISGREM_USE_CACHE", "0")
    protocol.execute_cached_tasks(
        [1],
        _double_task,
        cache_root=tmp_path,
        namespace="unit",
        max_workers=1,
        progress_every=0,
    )

    monkeypatch.setenv("DISGREM_USE_CACHE", "1")
    with pytest.raises(AssertionError, match="cache miss"):
        protocol.execute_cached_tasks(
            [1],
            _fail_task,
            cache_root=tmp_path,
            namespace="unit",
            max_workers=1,
            progress_every=0,
        )


def test_tracked_run_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(protocol, "_MANIFEST_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="intentional"):
        with protocol.tracked_run("regular", ["main.py", "regular"]):
            raise RuntimeError("intentional")

    manifest = json.loads((tmp_path / "regular.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["source_tree_sha256_at_start"]


def test_fair_tuning_worker_is_reproducible() -> None:
    task = {
        "phase": "tuning",
        "objective": "ridge",
        "algorithm": "DisGrem",
        "factor": 1.0,
        "replicates": 1,
        "seed_base": 12345,
        "agents": 4,
        "connection_radius": 0.8,
        "dimension": 3,
        "max_iterations": 3,
        "iteration_limit_override": 3,
    }

    first = _run_task(task)
    second = _run_task(task)
    first_run = first["runs"][0]
    second_run = second["runs"][0]

    assert first_run["parameter"] == "M"
    assert first_run["graph_edges"] == second_run["graph_edges"]
    assert first_run["graph_rho"] == pytest.approx(second_run["graph_rho"], abs=0.0)
    np.testing.assert_allclose(
        first_run["metrics"]["ValueF"],
        second_run["metrics"]["ValueF"],
        rtol=0.0,
        atol=0.0,
    )


def test_reduced_fair_tuning_pipeline_writes_complete_ledgers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tuning_module, "PRIMARY_OBJECTIVES", ("ridge",))
    monkeypatch.setattr(tuning_module, "PRIMARY_ALGORITHMS", ("DisGrem",))
    monkeypatch.setattr(tuning_module, "_FACTORS", (1.0,))
    monkeypatch.setattr(tuning_module, "_TUNING_REPLICATES", 1)
    monkeypatch.setattr(tuning_module, "_EVALUATION_REPLICATES", 1)
    monkeypatch.setattr(tuning_module, "_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(tuning_module, "configured_iteration_override", lambda: 2)
    monkeypatch.setattr(
        tuning_module,
        "execute_cached_tasks",
        lambda tasks, worker, **_: [worker(task) for task in tasks],
    )

    tuning_module.run_fair_tuning()

    data_log = tmp_path / "data_log"
    assert (data_log / "selection_ledger.csv").is_file()
    assert (data_log / "held_out_evaluation.csv").is_file()
    assert (data_log / "paired_selected_vs_default.csv").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    with gzip.open(data_log / "raw_evaluation_ridge.json.gz", "rt", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    assert len(evaluation["results"]) == 2
    assert all(len(result["runs"]) == 1 for result in evaluation["results"])


def test_artifact_freezer_fails_closed_on_empty_results(tmp_path: Path) -> None:
    audit = freeze(tmp_path, require_clean_git=False)

    assert audit["ok"] is False
    assert (tmp_path / "ARTIFACT_REPORT.md").is_file()
    assert not (tmp_path / "ARTIFACT_MANIFEST.json").exists()
    assert verify(tmp_path)["ok"] is False


def test_failed_refreeze_removes_stale_pass_files(tmp_path: Path) -> None:
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text("stale\n", encoding="utf-8")

    audit = freeze(tmp_path, require_clean_git=False)

    assert audit["ok"] is False
    assert not (tmp_path / "ARTIFACT_MANIFEST.json").exists()
    assert not (tmp_path / "SHA256SUMS").exists()
