"""Auditable protocol and run manifests for the DisGRem paper experiments."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np


PROTOCOL_VERSION = "1.0.0"
_CODES_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_ROOT = _CODES_ROOT / "results"
_MANIFEST_ROOT = _RESULTS_ROOT / "run_manifests"
_SOURCE_SUFFIXES = {".py", ".md", ".txt", ".ini", ".cff", ".csv", ".libsvm"}
_SOURCE_EXCLUDES = {"results", "_run_cache", "__pycache__", ".pytest_cache"}

PRIMARY_OBJECTIVES = (
    "ridge",
    "quadbad",
    "logsumexp",
    "huber",
    "linlog",
    "logreg_real",
    "rosenbrock",
    "styblinski_tang",
    "logreg_ncvr",
)
PRIMARY_ALGORITHMS = (
    "DisGrem",
    "CeDisGrem",
    "AdaDisGrem",
    "CeAdaDisGrem",
    "EXTRA",
    "DIGing",
    "DQM",
    "ESOM",
    "SONATA",
    "NetworkGIANT",
)
PAPER_OBJECTIVE_ITERATION_BUDGETS = {
    "ridge": 200,
    "quadbad": 1500,
    "logsumexp": 400,
    "huber": 800,
    "linlog": 1500,
    "logreg_real": 600,
    "rosenbrock": 300,
    "styblinski_tang": 100,
    "logreg_ncvr": 1000,
}
MODE_REQUIRED_GLOBS: dict[str, tuple[tuple[str, int], ...]] = {
    "regular": (
        ("main/data_log/raw_*.json.gz", 9),
        ("main/summary_*.txt", 9),
    ),
    "robust": (
        ("robust/data_log/raw_starting_point_robustness.json.gz", 1),
        ("robust/data_log/raw_parameter_sensitivity.json.gz", 1),
        ("robust/data_log/success_rate.csv", 1),
    ),
    "comm": (
        ("comm/data_log/raw_ce_benefit.json.gz", 1),
        ("comm/data_log/raw_klazy_sweep.json.gz", 1),
        ("comm/data_log/raw_compression_sweep.json.gz", 1),
        ("comm/data_log/ce_benefit.csv", 1),
    ),
    "ada": (
        ("ada/data_log/raw_m_trajectory.json.gz", 1),
        ("ada/data_log/raw_ada_vs_fixed_m.json.gz", 1),
        ("ada/data_log/raw_initial_m_robustness.json.gz", 1),
    ),
    "tuning": (
        ("tuning/data_log/raw_tuning_*.json.gz", 9),
        ("tuning/data_log/raw_evaluation_*.json.gz", 9),
        ("tuning/data_log/selection_ledger.csv", 1),
        ("tuning/data_log/held_out_evaluation.csv", 1),
        ("tuning/data_log/paired_selected_vs_default.csv", 1),
        ("tuning/REPORT.md", 1),
    ),
    "scale": (("scale/data_log/raw_*.json.gz", 9),),
}
PAPER_GRADE_PROTOCOL: dict[str, dict[str, Any]] = {
    "regular": {
        "objectives": list(PRIMARY_OBJECTIVES),
        "algorithms": list(PRIMARY_ALGORITHMS),
        "monte_carlo_replicates": 20,
    },
    "robust": {
        "objectives": list(PRIMARY_OBJECTIVES),
        "algorithms": list(PRIMARY_ALGORITHMS),
        "starting_point_replicates": 100,
        "sensitivity_replicates": 5,
    },
    "comm": {
        "objectives": ["ridge", "logsumexp", "huber", "logreg_real"],
        "replicates": 5,
    },
    "ada": {
        "objectives": ["ridge", "logsumexp", "logreg_real", "logreg_ncvr"],
        "replicates": 5,
    },
    "tuning": {
        "objectives": list(PRIMARY_OBJECTIVES),
        "algorithms": list(PRIMARY_ALGORITHMS),
        "candidate_factors": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0],
        "tuning_replicates": 5,
        "evaluation_replicates": 20,
    },
    "scale": {
        "objectives": ["ridge", "logsumexp", "rosenbrock"],
        "dimensions": [30, 100, 200],
        "algorithms": ["DisGrem", "AdaDisGrem", "EXTRA", "SONATA"],
        "replicates": 5,
        "iteration_budgets": {
            "ridge": {30: 200, 100: 400, 200: 600},
            "logsumexp": {30: 400, 100: 800, 200: 1200},
            "rosenbrock": {30: 300, 100: 600, 200: 900},
        },
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set | tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    """Convert scientific objects to strict JSON, mapping nonfinite values to null."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return value


def _json_round_trip(value: Any) -> Any:
    """Normalize values exactly as a manifest JSON write/read cycle does."""
    return json.loads(json.dumps(value, default=_json_default))


def stable_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination


def write_json_gz(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def source_tree_sha256(root: str | Path = _CODES_ROOT) -> str:
    source_root = Path(root).resolve()
    digest = hashlib.sha256()
    files = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source_root)
        if any(part in _SOURCE_EXCLUDES or part.startswith(".venv") for part in relative.parts):
            continue
        files.append((relative, path))
    for relative, path in sorted(files, key=lambda item: item[0].as_posix()):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def required_outputs_present(
    mode: str,
    results_root: str | Path = _RESULTS_ROOT,
) -> bool:
    root = Path(results_root)
    requirements = MODE_REQUIRED_GLOBS.get(mode)
    if requirements is None:
        raise ValueError(f"unknown experiment mode: {mode}")
    return all(
        len([path for path in root.glob(pattern) if path.is_file()]) == expected
        for pattern, expected in requirements
    )


def nanmean_columns(values: Any) -> np.ndarray:
    """Compute a finite-value column mean without empty-slice warnings."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    counts = finite.sum(axis=0)
    totals = np.where(finite, array, 0.0).sum(axis=0)
    output = np.full(np.shape(totals), np.nan, dtype=float)
    np.divide(totals, counts, out=output, where=counts > 0)
    return output


def execute_cached_tasks(
    tasks: list[Any],
    worker,
    *,
    cache_root: str | Path,
    namespace: str,
    max_workers: int,
    progress_every: int = 20,
) -> list[Any]:
    """Execute picklable tasks with source- and configuration-matched checkpoints."""
    root = Path(cache_root) / namespace
    source_hash = source_tree_sha256()
    worker_identity = f"{worker.__module__}.{worker.__qualname__}"
    use_cache = os.environ.get("DISGREM_USE_CACHE", "0") == "1"
    outputs: list[Any | None] = [None] * len(tasks)
    pending = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for index, task in enumerate(tasks):
            signature = stable_payload_sha256(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "source_tree_sha256": source_hash,
                    "namespace": namespace,
                    "worker": worker_identity,
                    "task": task,
                }
            )
            cache_path = root / signature[:2] / f"{signature}.pkl"
            if use_cache and cache_path.is_file():
                try:
                    with cache_path.open("rb") as handle:
                        payload = pickle.load(handle)
                except (OSError, EOFError, pickle.UnpicklingError):
                    payload = {}
                if payload.get("signature") == signature:
                    outputs[index] = payload.get("result")
                    completed_count += 1
                    continue
            future = pool.submit(worker, task)
            pending[future] = (index, signature, cache_path)

        for future in as_completed(pending):
            index, signature, cache_path = pending[future]
            result = future.result()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(
                    {"signature": signature, "result": result},
                    handle,
                    protocol=4,
                )
            temporary.replace(cache_path)
            outputs[index] = result
            completed_count += 1
            if progress_every > 0 and completed_count % progress_every == 0:
                print(
                    f"  [{namespace}] {completed_count}/{len(tasks)} tasks complete",
                    flush=True,
                )

    if any(output is None for output in outputs):
        raise RuntimeError(f"missing cached-task output in namespace {namespace}")
    return outputs


def _git_state() -> dict[str, Any]:
    repository = _CODES_ROOT.parent

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "changed_paths": [line[3:] for line in status.splitlines() if len(line) >= 4],
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "scipy", "matplotlib", "openpyxl", "pytest")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _integer_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def configured_iteration_override() -> int | None:
    value = os.environ.get("LOG_SCHEDULE_MAXIT")
    return int(value) if value is not None else None


def resolve_iteration_budget(policy: dict[str, Any], configuration: dict[str, Any]) -> int:
    override = configuration.get("iteration_budget_override")
    if override is not None:
        return int(override)
    return int(policy.get("maxIt", configuration.get("maxIt", 500)))


def _objective_iteration_budgets(override: int | None = None) -> dict[str, int]:
    from problems.init_policy import init_policy

    _, policies, _ = init_policy("regular")
    return {
        objective: int(override if override is not None else policies[objective]["maxIt"])
        for objective in PRIMARY_OBJECTIVES
    }


def mode_protocol(mode: str) -> dict[str, Any]:
    iteration_override = configured_iteration_override()
    objective_budgets = _objective_iteration_budgets(iteration_override)
    scale_budgets = {
        "ridge": {30: 200, 100: 400, 200: 600},
        "logsumexp": {30: 400, 100: 800, 200: 1200},
        "rosenbrock": {30: 300, 100: 600, 200: 900},
    }
    if iteration_override is not None:
        scale_budgets = {
            objective: {dimension: iteration_override for dimension in dimensions}
            for objective, dimensions in scale_budgets.items()
        }
    shared = {
        "agents": 10,
        "graph": {
            "model": "connected random geometric graph",
            "node_positions": "independent uniform draws in the two-dimensional unit square",
            "connection_radius": 0.5,
            "mixing": "W = I - L / d_max",
            "regeneration": "one graph per Monte Carlo replicate after the starting-point draw",
        },
        "vector_mixing": {
            "schedule": "logarithmic with finite experimental cap",
            "p": 3.0,
            "c_mix": 2.0,
            "maximum_rounds": 10,
        },
        "stopping_tolerance": 1e-12,
        "iteration_budget_override": iteration_override,
        "per_objective_iteration_budgets": objective_budgets,
        "source_tree_sha256": source_tree_sha256(),
    }
    configurations: dict[str, dict[str, Any]] = {
        "regular": {
            "objectives": PRIMARY_OBJECTIVES,
            "algorithms": PRIMARY_ALGORITHMS,
            "monte_carlo_replicates": _integer_env("LOG_SCHEDULE_NSTART", 20),
            "seed_rule": "100 + mc_index; the starting point and graph consume one shared RNG stream",
            "synthetic_dimension": 30,
        },
        "robust": {
            "objectives": PRIMARY_OBJECTIVES,
            "algorithms": PRIMARY_ALGORITHMS,
            "starting_point_replicates": _integer_env("LOG_SCHEDULE_NSTART_PART1", 100),
            "sensitivity_replicates": _integer_env("LOG_SCHEDULE_NSTART", 5),
            "seed_rules": {
                "center": 42,
                "starting_point": "1000 + mc_index",
                "graph": "2000 + mc_index",
                "sensitivity_shared_stream": "500 + mc_index",
            },
        },
        "comm": {
            "objectives": ("ridge", "logsumexp", "huber", "logreg_real"),
            "replicates": _integer_env("LOG_SCHEDULE_NSTART", 5),
            "seed_rule": "300 + mc_index; matched across variants",
        },
        "ada": {
            "objectives": ("ridge", "logsumexp", "logreg_real", "logreg_ncvr"),
            "replicates": _integer_env("LOG_SCHEDULE_NSTART", 5),
            "seed_rule": "700 + mc_index; matched across variants",
        },
        "scale": {
            "objectives": ("ridge", "logsumexp", "rosenbrock"),
            "dimensions": (30, 100, 200),
            "algorithms": ("DisGrem", "AdaDisGrem", "EXTRA", "SONATA"),
            "replicates": _integer_env("LOG_SCHEDULE_NSTART", 5),
            "seed_rule": "200 + mc_index; matched within objective and dimension",
            "iteration_budgets": scale_budgets,
        },
        "tuning": {
            "objectives": PRIMARY_OBJECTIVES,
            "algorithms": PRIMARY_ALGORITHMS,
            "candidate_factors": (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0),
            "tuning_replicates": _integer_env("DISGREM_TUNING_NSTART", 5),
            "evaluation_replicates": _integer_env("DISGREM_EVAL_NSTART", 20),
            "seed_rules": {
                "tuning": "10000 + mc_index",
                "held_out_evaluation": "20000 + mc_index",
            },
            "selection": "failures, median penalized log10 relF, successes, distance from factor one",
        },
    }
    if mode not in configurations:
        raise ValueError(f"unknown experiment mode: {mode}")
    return {**shared, **configurations[mode]}


def paper_grade_protocol_errors(mode: str, protocol: dict[str, Any]) -> list[str]:
    if mode not in PAPER_GRADE_PROTOCOL:
        raise ValueError(f"unknown experiment mode: {mode}")
    errors = []
    if protocol.get("agents") != 10:
        errors.append(f"paper-grade agent count mismatch: {mode}")
    if protocol.get("stopping_tolerance") != 1e-12:
        errors.append(f"paper-grade stopping tolerance mismatch: {mode}")
    if protocol.get("iteration_budget_override") is not None:
        errors.append(f"paper-grade iteration budget was overridden: {mode}")
    if (
        protocol.get("per_objective_iteration_budgets")
        != PAPER_OBJECTIVE_ITERATION_BUDGETS
    ):
        errors.append(f"paper-grade objective iteration budgets mismatch: {mode}")
    if protocol.get("graph", {}).get("connection_radius") != 0.5:
        errors.append(f"paper-grade graph radius mismatch: {mode}")
    if protocol.get("vector_mixing", {}).get("maximum_rounds") != 10:
        errors.append(f"paper-grade vector-mixing cap mismatch: {mode}")
    for key, expected in PAPER_GRADE_PROTOCOL[mode].items():
        actual = protocol.get(key)
        if _json_round_trip(actual) != _json_round_trip(expected):
            errors.append(
                f"paper-grade protocol mismatch: {mode}.{key}="
                f"{actual!r}, expected {expected!r}"
            )
    return errors


def theory_implementation_contract() -> list[dict[str, str]]:
    return [
        {
            "component": "vector consensus depth",
            "theory": "uncapped logarithmic schedule",
            "experiment": "same formula capped at 10 rounds",
            "claim_boundary": "the capped implementation is a heuristic, not a certified theorem instance",
        },
        {
            "component": "Hessian consensus depth",
            "theory": "same logarithmic depth as the vector trackers",
            "experiment": "at most 3 full-matrix rounds, or 2 rounds for communication-efficient variants",
            "claim_boundary": "no finite-tolerance bridge has been proved for the matrix cap",
        },
        {
            "component": "regularization constant",
            "theory": "M is at least the Hessian-Lipschitz constant L2",
            "experiment": "M is a tuned multiple of the initial maximum Hessian norm",
            "claim_boundary": "the experimental scaling is not a certificate that M >= L2",
        },
        {
            "component": "trajectory boundedness",
            "theory": "assumed explicitly",
            "experiment": "finite sampled traces are checked for overflow and divergence",
            "claim_boundary": "finite benchmark traces do not prove the global assumption",
        },
    ]


def manifest_path(mode: str) -> Path:
    return _MANIFEST_ROOT / f"{mode}.json"


@contextmanager
def tracked_run(mode: str, argv: list[str] | None = None) -> Iterator[Path]:
    path = manifest_path(mode)
    command = list(argv if argv is not None else sys.argv)
    full_mode_scope = len(command) <= 2 or (len(command) > 1 and command[1] == "all")
    started_source_hash = source_tree_sha256()
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "status": "running",
        "started_at": _utc_now(),
        "command": command,
        "full_mode_scope": full_mode_scope,
        "protocol": mode_protocol(mode),
        "theory_implementation_contract": theory_implementation_contract(),
        "environment": _runtime_environment(),
        "git": _git_state(),
        "source_tree_sha256_at_start": started_source_hash,
    }
    write_json(path, manifest)
    try:
        yield path
    except BaseException as exc:
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "finished_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "source_tree_sha256_at_finish": source_tree_sha256(),
            }
        )
        write_json(path, manifest)
        raise
    else:
        finished_source_hash = source_tree_sha256()
        outputs_complete = (
            required_outputs_present(mode) if full_mode_scope else True
        )
        source_unchanged = finished_source_hash == started_source_hash
        manifest.update(
            {
                "status": "complete" if source_unchanged and outputs_complete else "invalid",
                "finished_at": _utc_now(),
                "source_tree_sha256_at_finish": finished_source_hash,
                "source_unchanged_during_run": source_unchanged,
                "required_outputs_complete": outputs_complete,
            }
        )
        write_json(path, manifest)
        if not source_unchanged:
            raise RuntimeError(f"source tree changed during {mode} run")
        if not outputs_complete:
            raise RuntimeError(f"required outputs are incomplete for {mode} run")
