"""Regenerate paper figures from auditable raw experiment outputs."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from problems.init_policy import init_policy
from problems.obj_factory import obj_factory
from utils.alg.alg_bank import get_alg_bank
from utils.export.log_export import merge_logs
from utils.export.plot_utils import (
    fig_ce_benefit,
    fig_perf_profiles_comm_panel,
    fig_perf_profiles_tol_panel,
    fig_plot_multiobj_custom,
    fig_plot_multiobj_repr,
)


def _load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported raw schema: {path}")
    return payload


def _objective_optimum(objective: str, configuration: dict) -> float:
    parameter_bank, _, _ = init_policy("regular")
    dimension = int(configuration.get("d_override", 30))
    arguments = parameter_bank.get(objective, [dimension])
    if arguments and isinstance(arguments[0], (int, float)):
        arguments = [dimension, *arguments[1:]]
    _, _, _, _, optimum_values, _, _, _ = obj_factory(
        objective,
        int(configuration.get("Nagent", 10)),
        *arguments,
    )
    return float(np.mean(optimum_values))


def _restore_algorithm_record(record: dict) -> dict:
    return {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in record.items()
    }


def replot_regular() -> None:
    """Rebuild aggregate regular-benchmark figures from raw per-run traces."""
    results_dir = _ROOT / "results" / "main"
    raw_files = sorted((results_dir / "data_log").glob("raw_*.json.gz"))
    if not raw_files:
        print("[skip] No regular raw outputs found.")
        return

    all_logs = {}
    all_individual = {}
    for path in raw_files:
        payload = _load_json_gz(path)
        objective = payload["objective"]
        runs = sorted(payload["runs"], key=lambda run: run["mc_index"])
        logs = [
            {
                algorithm: _restore_algorithm_record(record)
                for algorithm, record in run["algorithms"].items()
            }
            for run in runs
        ]
        initial_values = payload.get("initial_objectives", [])
        if len(initial_values) != len(logs):
            raise ValueError(f"initial-objective coverage mismatch: {path}")
        initial_value = float(initial_values[-1])
        optimum = _objective_optimum(objective, payload["configuration"])
        all_logs[objective] = merge_logs(
            logs,
            False,
            initial_value,
            optimum,
            use_median=True,
        )
        all_individual[objective] = logs
        print(f"  Loaded {objective}: {len(logs)} runs")

    if len(all_logs) == 9:
        alg_bank = get_alg_bank("MainComp")
        for x_key, y_key in (
            ("steps", "relF"),
            ("steps", "combo"),
            ("timeCost", "relF"),
            ("timeCost", "combo"),
            ("commCost", "relF"),
            ("commCost", "combo"),
        ):
            fig_plot_multiobj_custom(
                all_logs,
                alg_bank,
                x_key,
                y_key,
                "semilogy",
                str(results_dir),
                n_cols=3,
            )
        fig_plot_multiobj_repr(
            all_logs,
            alg_bank,
            "steps",
            "relF",
            "semilogy",
            str(results_dir),
        )
        fig_perf_profiles_tol_panel(
            all_logs,
            alg_bank,
            str(results_dir),
            tol_levels=[1e-3, 1e-6, 1e-9],
            all_individual=all_individual,
        )
        fig_perf_profiles_comm_panel(
            all_logs,
            alg_bank,
            str(results_dir),
            tol_levels=[1e-3, 1e-6, 1e-9],
            all_individual=all_individual,
        )
        print("[done] Regular figures regenerated from raw JSON.\n")
    else:
        print(
            f"[skip] Aggregate regular figures require 9 objectives; found {len(all_logs)}.\n"
        )


def replot_ce_benefit() -> None:
    """Regenerate the communication-benefit figure from its CSV ledger."""
    results_dir = _ROOT / "results" / "comm"
    csv_path = results_dir / "data_log" / "ce_benefit.csv"
    if not csv_path.is_file():
        print("[skip] No ce_benefit.csv found.")
        return

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        tolerances = [float(column.split("=")[1]) for column in header[2:]]
        all_data = {}
        for row in reader:
            values = []
            for value in row[2:]:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(float("nan"))
            all_data.setdefault(row[0], {})[row[1]] = values

    pairs = [("DisGrem", "CeDisGrem"), ("AdaDisGrem", "CeAdaDisGrem")]
    fig_ce_benefit(all_data, pairs, tolerances, str(results_dir))
    print("[done] Communication-benefit figure regenerated.\n")


def main() -> None:
    print("=" * 60)
    print("  Regenerating figures from raw experiment outputs")
    print("=" * 60)
    replot_regular()
    replot_ce_benefit()
    print("All requested figures regenerated.")


if __name__ == "__main__":
    main()
