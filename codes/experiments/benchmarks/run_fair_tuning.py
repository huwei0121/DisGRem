"""Nested, seed-separated hyperparameter tuning and held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.protocol import (
    PRIMARY_ALGORITHMS,
    PRIMARY_OBJECTIVES,
    configured_iteration_override,
    execute_cached_tasks,
    write_json_gz,
)
from problems.init_policy import init_policy
from problems.obj_factory import obj_factory
from utils.alg.alg_bank import get_alg_bank
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged


_FACTORS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0)
_TUNING_REPLICATES = int(os.environ.get("DISGREM_TUNING_NSTART", "5"))
_EVALUATION_REPLICATES = int(os.environ.get("DISGREM_EVAL_NSTART", "20"))
_TUNING_SEED_BASE = 10_000
_EVALUATION_SEED_BASE = 20_000
_SUCCESS_TOLERANCE = 1e-6
_N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))
_RESULTS_ROOT = _ROOT / "results" / "tuning"

_PARAMETER_SPECS: dict[str, tuple[str, float]] = {
    "DisGrem": ("M", 1.0),
    "CeDisGrem": ("M", 1.0),
    "AdaDisGrem": ("initial_M", 1.0),
    "CeAdaDisGrem": ("initial_M", 1.0),
    "EXTRA": ("alpha_extra", 1.0),
    "DIGing": ("alpha_diging", 2.5),
    "DQM": ("c", 1.0),
    "ESOM": ("esom_penalty", 1.0),
    "SONATA": ("sonata_tau", 1.0),
    "NetworkGIANT": ("alpha_step", 1.0),
}


def _algorithm_callable(name: str):
    algorithms = dict(get_alg_bank("All"))
    if name not in algorithms:
        raise ValueError(f"unknown algorithm: {name}")
    return algorithms[name]


def _objective_data(objective: str, agents: int, dimension: int):
    parameter_bank, policy_bank, start_bank = init_policy("regular")
    arguments = parameter_bank.get(objective, [dimension])
    if arguments and isinstance(arguments[0], (int, float)):
        arguments = [dimension, *arguments[1:]]
    data = obj_factory(objective, agents, *arguments)
    policy = policy_bank[objective]
    start = start_bank.get(objective, lambda d, far: np.random.randn(d))
    return data, policy, start


def _base_parameters(task: dict[str, Any], data, policy, weights) -> dict[str, Any]:
    fun_list, dimension, lipschitz, x_opt, f_opt, _, name, fparam = data
    return {
        "Nagent": task["agents"],
        "p_edge": task["connection_radius"],
        "maxIt": task.get("iteration_limit_override", task["max_iterations"]),
        "tol": 1e-12,
        "tolType": policy.get("tolType", "combo"),
        "verbose": False,
        "NC": 3,
        "NC_schedule": "log",
        "log_p": 3.0,
        "log_c_mix": 2.0,
        "NC_max": 10,
        "info": 2,
        "countComm": True,
        "f": fun_list,
        "fname": name,
        "fparam": fparam,
        "dim": dimension,
        "M": policy["M_factor"] * float(lipschitz.max()),
        "alpha": policy["alpha"] / float(lipschitz.max()),
        "decay_alpha": policy.get("decay", False),
        "x_opt": x_opt[0] if x_opt else None,
        "f_opt": float(np.mean(f_opt)),
        "W": weights,
        "esom_penalty": 1.0,
    }


def _apply_factor(parameters: dict[str, Any], algorithm: str, factor: float) -> str:
    parameter_name, _ = _PARAMETER_SPECS[algorithm]
    if parameter_name in {"M", "initial_M"}:
        parameters["M"] *= factor
    elif parameter_name == "alpha_extra":
        parameters["alpha_extra"] = parameters["alpha"] * factor
    elif parameter_name == "alpha_diging":
        parameters["alpha_diging"] = parameters["alpha"] * factor
    elif parameter_name == "c":
        parameters["c"] = 0.5 * factor
    elif parameter_name == "esom_penalty":
        parameters["esom_penalty"] = factor
    elif parameter_name == "sonata_tau":
        parameters["sonata_tau"] = 1e-4 * factor
    elif parameter_name == "alpha_step":
        parameters["alpha_step"] = factor
    else:
        raise ValueError(f"unsupported parameter: {parameter_name}")
    return parameter_name


def _finite_values(record: dict[str, Any], key: str) -> np.ndarray:
    values = np.asarray(record.get(key, []), dtype=float).ravel()
    return values[np.isfinite(values)]


def _summarize_trace(record: dict[str, Any], failed: bool) -> dict[str, Any]:
    rel_f = _finite_values(record, "relF")
    comm = np.asarray(record.get("commCost", []), dtype=float).ravel()
    best_rel_f = float(np.min(rel_f)) if rel_f.size else float("inf")
    hits = np.where(np.asarray(record.get("relF", []), dtype=float) < _SUCCESS_TOLERANCE)[0]
    hit_iteration = int(hits[0] + 1) if hits.size else None
    communication_at_hit = None
    if hits.size and hits[0] < comm.size and np.isfinite(comm[hits[0]]):
        communication_at_hit = float(comm[hits[0]])
    return {
        "failed": bool(failed),
        "best_relF": best_rel_f,
        "success": bool(not failed and best_rel_f < _SUCCESS_TOLERANCE),
        "hit_iteration": hit_iteration,
        "communication_at_hit_mb": communication_at_hit,
    }


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    algorithm = task["algorithm"]
    algorithm_function = _algorithm_callable(algorithm)
    data, policy, start_generator = _objective_data(
        task["objective"], task["agents"], task["dimension"]
    )
    fun_list, dimension, _, _, _, _, _, _ = data
    runs = []
    for replicate in range(task["replicates"]):
        seed = task["seed_base"] + replicate
        rng = np.random.RandomState(seed)
        np.random.set_state(rng.get_state())
        starting_point = start_generator(dimension, False)
        adjacency, weights = generate_random_graph(
            task["agents"], task["connection_radius"]
        )
        parameters = _base_parameters(task, data, policy, weights)
        parameter_name = _apply_factor(parameters, algorithm, task["factor"])
        initial_objective = float(np.mean([function(starting_point) for function in fun_list]))
        error = None
        try:
            _, output = algorithm_function(starting_point.copy(), dict(parameters))
            failed = bool(output.get("fail") or detect_diverged(output, initial_objective))
        except Exception as exc:  # noqa: BLE001 - exception is part of the raw artifact
            output = {"fail": True, "failReason": str(exc)}
            failed = True
            error = {"type": type(exc).__name__, "message": str(exc)}
        metrics = {
            key: output.get(key, [])
            for key in (
                "ValueF",
                "relF",
                "relX",
                "gradNrm",
                "cons",
                "combo",
                "commCost",
                "timeCost",
                "Mavg",
                "NC",
            )
            if key in output
        }
        runs.append(
            {
                "replicate": replicate,
                "seed": seed,
                "parameter": parameter_name,
                "factor": task["factor"],
                "initial_objective": initial_objective,
                "graph_edges": int(adjacency.sum() / 2),
                "graph_rho": float(
                    np.linalg.norm(
                        weights - np.ones_like(weights) / task["agents"],
                        2,
                    )
                ),
                "summary": _summarize_trace(output, failed),
                "metrics": metrics,
                "failure_reason": output.get("failReason"),
                "exception": error,
            }
        )
    return {"task": task, "runs": runs}


def _execute_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase = tasks[0]["phase"]
    objective = tasks[0]["objective"]
    completed = execute_cached_tasks(
        tasks,
        _run_task,
        cache_root=_ROOT / "_run_cache",
        namespace=f"fair_tuning_{phase}_{objective}",
        max_workers=_N_WORKERS,
        progress_every=10,
    )
    return sorted(
        completed,
        key=lambda result: (
            result["task"]["algorithm"],
            result["task"].get("variant", ""),
            result["task"]["factor"],
        ),
    )


def _candidate_summary(result: dict[str, Any]) -> dict[str, Any]:
    runs = result["runs"]
    penalized = []
    successes = 0
    failures = 0
    for run in runs:
        summary = run["summary"]
        failures += int(summary["failed"])
        successes += int(summary["success"])
        if summary["failed"] or not np.isfinite(summary["best_relF"]):
            penalized.append(6.0)
        else:
            penalized.append(float(np.log10(np.clip(summary["best_relF"], 1e-16, 1e6))))
    task = result["task"]
    factor = float(task["factor"])
    return {
        "objective": task["objective"],
        "algorithm": task["algorithm"],
        "parameter": _PARAMETER_SPECS[task["algorithm"]][0],
        "factor": factor,
        "replicates": len(runs),
        "failures": failures,
        "successes": successes,
        "median_penalized_log10_relF": float(np.median(penalized)),
        "rank": (
            failures,
            float(np.median(penalized)),
            -successes,
            abs(float(np.log(factor))),
            factor,
        ),
    }


def _select_candidates(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = [_candidate_summary(result) for result in results]
    selected: dict[str, float] = {}
    for algorithm in PRIMARY_ALGORITHMS:
        candidates = [row for row in rows if row["algorithm"] == algorithm]
        winner = min(candidates, key=lambda row: row["rank"])
        selected[algorithm] = float(winner["factor"])
        for row in candidates:
            row["selected"] = row is winner
            row.pop("rank")
    return rows, selected


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return float(center - half_width), float(center + half_width)


def _bootstrap_median_interval(values: list[float], key: str) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(finite, size=(4000, finite.size), replace=True)
    medians = np.median(samples, axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def _evaluation_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        task = result["task"]
        summaries = [run["summary"] for run in result["runs"]]
        best = [float(item["best_relF"]) for item in summaries]
        successes = sum(bool(item["success"]) for item in summaries)
        failures = sum(bool(item["failed"]) for item in summaries)
        success_low, success_high = _wilson_interval(successes, len(summaries))
        median_low, median_high = _bootstrap_median_interval(
            best,
            f"{task['objective']}:{task['algorithm']}:{task['variant']}",
        )
        rows.append(
            {
                "objective": task["objective"],
                "algorithm": task["algorithm"],
                "variant": task["variant"],
                "factor": task["factor"],
                "replicates": len(summaries),
                "failures": failures,
                "successes": successes,
                "success_rate": successes / len(summaries),
                "success_rate_ci95_low": success_low,
                "success_rate_ci95_high": success_high,
                "median_best_relF": float(np.median(best)),
                "median_best_relF_ci95_low": median_low,
                "median_best_relF_ci95_high": median_high,
            }
        )
    return rows


def _paired_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {
        (result["task"]["objective"], result["task"]["algorithm"], result["task"]["variant"]): result
        for result in results
    }
    rows = []
    for objective in PRIMARY_OBJECTIVES:
        for algorithm in PRIMARY_ALGORITHMS:
            selected = grouped[(objective, algorithm, "selected")]
            default = grouped[(objective, algorithm, "default")]
            selected_values = []
            default_values = []
            for selected_run, default_run in zip(selected["runs"], default["runs"], strict=True):
                selected_values.append(
                    np.log10(np.clip(selected_run["summary"]["best_relF"], 1e-16, 1e6))
                    if not selected_run["summary"]["failed"]
                    else 6.0
                )
                default_values.append(
                    np.log10(np.clip(default_run["summary"]["best_relF"], 1e-16, 1e6))
                    if not default_run["summary"]["failed"]
                    else 6.0
                )
            differences = np.asarray(selected_values) - np.asarray(default_values)
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(differences, alternative="two-sided", zero_method="zsplit")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "objective": objective,
                    "algorithm": algorithm,
                    "paired_replicates": len(differences),
                    "median_log10_relF_difference_selected_minus_default": float(
                        np.median(differences)
                    ),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value_unadjusted": p_value,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    selection_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> None:
    selected = [row for row in selection_rows if row["selected"]]
    evaluation = {
        (row["objective"], row["algorithm"], row["variant"]): row
        for row in evaluation_rows
    }
    lines = [
        "# Seed-Separated Fair-Tuning Report",
        "",
        "## Protocol",
        "",
        f"- Tuning replicates: {_TUNING_REPLICATES} (seeds {_TUNING_SEED_BASE}+index).",
        f"- Held-out replicates: {_EVALUATION_REPLICATES} (seeds {_EVALUATION_SEED_BASE}+index).",
        f"- Success threshold: `relF < {_SUCCESS_TOLERANCE:.0e}`.",
        "- Each algorithm exposes one declared principal hyperparameter; all other settings remain fixed.",
        "- Starting points and graphs are paired by objective and replicate across algorithms and candidates.",
        "- Selection minimizes failures first, then median penalized log10 relative error, then uses deterministic tie-breaks.",
        "- Held-out Wilcoxon p-values are descriptive and unadjusted; they are not used for selection.",
        "",
        "## Selected Factors",
        "",
        "| Objective | Algorithm | Parameter | Selected | Default | Selected Success | Default Success |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        key = (row["objective"], row["algorithm"])
        selected_eval = evaluation[(*key, "selected")]
        default_eval = evaluation[(*key, "default")]
        default_factor = _PARAMETER_SPECS[row["algorithm"]][1]
        lines.append(
            f"| {row['objective']} | {row['algorithm']} | {row['parameter']} | "
            f"{row['factor']:g} | {default_factor:g} | "
            f"{selected_eval['success_rate']:.2f} | {default_eval['success_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This experiment audits sensitivity and selection fairness within the declared grids. "
            "It does not prove that any grid is globally optimal, that one principal parameter fully "
            "represents every implementation, or that nonconvex outcomes generalize beyond these instances.",
            "",
        ]
    )
    (_RESULTS_ROOT / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def run_fair_tuning() -> None:
    """Run nested tuning on five seeds and evaluate on twenty disjoint seeds."""
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    data_log = _RESULTS_ROOT / "data_log"
    data_log.mkdir(parents=True, exist_ok=True)
    _, policies, _ = init_policy("regular")
    iteration_override = configured_iteration_override()
    iteration_budgets = {
        objective: int(
            iteration_override
            if iteration_override is not None
            else policies[objective]["maxIt"]
        )
        for objective in PRIMARY_OBJECTIVES
    }
    selection_rows: list[dict[str, Any]] = []
    selected_factors: dict[tuple[str, str], float] = {}

    for objective in PRIMARY_OBJECTIVES:
        print(f"\n{'=' * 70}\n  Tuning: {objective}\n{'=' * 70}")
        tasks = [
            {
                "phase": "tuning",
                "objective": objective,
                "algorithm": algorithm,
                "factor": factor,
                "replicates": _TUNING_REPLICATES,
                "seed_base": _TUNING_SEED_BASE,
                "agents": 10,
                "connection_radius": 0.5,
                "dimension": 30,
                "max_iterations": iteration_budgets[objective],
            }
            for algorithm in PRIMARY_ALGORITHMS
            for factor in _FACTORS
        ]
        results = _execute_tasks(tasks)
        write_json_gz(
            data_log / f"raw_tuning_{objective}.json.gz",
            {"schema_version": 1, "phase": "tuning", "results": results},
        )
        rows, selected = _select_candidates(results)
        selection_rows.extend(rows)
        selected_factors.update(
            {(objective, algorithm): factor for algorithm, factor in selected.items()}
        )

    _write_csv(data_log / "selection_ledger.csv", selection_rows)

    evaluation_results: list[dict[str, Any]] = []
    for objective in PRIMARY_OBJECTIVES:
        print(f"\n{'=' * 70}\n  Held-out evaluation: {objective}\n{'=' * 70}")
        tasks = []
        for algorithm in PRIMARY_ALGORITHMS:
            for variant, factor in (
                ("selected", selected_factors[(objective, algorithm)]),
                ("default", _PARAMETER_SPECS[algorithm][1]),
            ):
                tasks.append(
                    {
                        "phase": "evaluation",
                        "variant": variant,
                        "objective": objective,
                        "algorithm": algorithm,
                        "factor": factor,
                        "replicates": _EVALUATION_REPLICATES,
                        "seed_base": _EVALUATION_SEED_BASE,
                        "agents": 10,
                        "connection_radius": 0.5,
                        "dimension": 30,
                        "max_iterations": iteration_budgets[objective],
                    }
                )
        results = _execute_tasks(tasks)
        evaluation_results.extend(results)
        write_json_gz(
            data_log / f"raw_evaluation_{objective}.json.gz",
            {"schema_version": 1, "phase": "evaluation", "results": results},
        )

    evaluation_rows = _evaluation_rows(evaluation_results)
    paired_rows = _paired_rows(evaluation_results)
    _write_csv(data_log / "held_out_evaluation.csv", evaluation_rows)
    _write_csv(data_log / "paired_selected_vs_default.csv", paired_rows)
    _write_report(selection_rows, evaluation_rows)
    print(f"\nFair-tuning results written to {_RESULTS_ROOT}")


if __name__ == "__main__":
    run_fair_tuning()
