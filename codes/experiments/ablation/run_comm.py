"""
run_comm.py -Communication-cost study (two parts).

Part 1 -Ce benefit
    Compare DisGrem vs CeDisGrem and AdaDisGrem vs CeAdaDisGrem to show
    communication savings from compression.  Output: precision-vs-comm figure
    and savings table.

Part 2 -Klazy and compression ablation
    2a) Klazy sweep   -CeDisGrem with Klazy in {1, 5, 10, 20, 40, 80}
    2b) Compress sweep -CeDisGrem with full / topk(5,10,20,50%) / lowrank(1,2,d//5,d//2)
"""

from __future__ import annotations
import os
import sys
import csv
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged
from utils.export.plot_utils import fig_ce_benefit, fig_comm_ablation
from experiments.protocol import (
    configured_iteration_override,
    execute_cached_tasks,
    nanmean_columns,
    resolve_iteration_budget,
    write_json_gz,
)


#  Constants
_COMM_FUNCS = ["ridge", "logsumexp", "huber", "logreg_real"]
_TOL_LEVELS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
_NSTART     = int(os.environ.get("LOG_SCHEDULE_NSTART", "5"))
_DIM        = 30
_N_WORKERS  = min(8, max(1, (os.cpu_count() or 4) - 2))

# Part 1 algorithms
_CE_PAIRS = [
    ("DisGrem",    "CeDisGrem"),
    ("AdaDisGrem", "CeAdaDisGrem"),
]

# Part 2a Klazy values
_KLAZY_VALUES = [1, 5, 10, 20, 40, 80]

# Part 2b compression configs: (label, compressH, compressor, compressParam_func)
def _compress_configs(d):
    d2 = d * d
    return [
        ("Full matrix",    False, "vector",  None),
        ("TopK 5%",        True,  "topk",    max(1, int(0.05 * d2))),
        ("TopK 10%",       True,  "topk",    max(1, int(0.10 * d2))),
        ("TopK 20%",       True,  "topk",    max(1, int(0.20 * d2))),
        ("TopK 50%",       True,  "topk",    max(1, int(0.50 * d2))),
        ("LowRank r=1",    True,  "lowrank", 1),
        ("LowRank r=2",    True,  "lowrank", 2),
        (f"LowRank r={d//5}", True, "lowrank", max(1, d // 5)),
        (f"LowRank r={d//2}", True, "lowrank", max(1, d // 2)),
    ]


#  Shared helpers

def _set_blas_single():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def _build_prm(P, policy, fun_list, d, L_vec,
               x_opt_list, f_opt_list, fname, fparam, W):
    prm = dict(P)
    prm["maxIt"] = resolve_iteration_budget(policy, P)
    prm["verbose"] = False
    if "tolType" in policy:
        prm["tolType"] = policy["tolType"]
    prm.update({
        "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
        "M": policy["M_factor"] * float(L_vec.max()),
        "alpha": policy["alpha"] / float(L_vec.max()),
        "decay_alpha": policy.get("decay", False),
        "x_opt": x_opt_list[0] if x_opt_list else None,
        "f_opt": float(np.mean(f_opt_list)) if len(f_opt_list) else 0.0,
        "W": W, "esom_penalty": 1.0,
    })
    return prm


def _comm_at_tol(out, tol_levels):
    """Return comm-cost to first reach each tol threshold, or NaN."""
    relF = np.asarray(out.get("relF", []), dtype=float)
    comm = np.asarray(out.get("commCost", []), dtype=float)
    results = []
    for tol in tol_levels:
        hit = np.where(relF < tol)[0]
        if len(hit) == 0:
            results.append(np.nan)
        else:
            idx = int(hit[0])
            results.append(float(comm[min(idx, len(comm) - 1)]))
    return results


def _run_one_alg(obj_name, alg_name, mc_idx, P_base, prm_overrides=None):
    """Run one configuration and retain failed traces instead of dropping them."""
    _set_blas_single()

    param_bank, M_alpha_policy, x0_generator = init_policy("regular")
    full_bank = get_alg_bank("All")
    alg_func = None
    for n, f in full_bank:
        if n == alg_name:
            alg_func = f
            break
    if alg_func is None:
        raise ValueError(f"unknown algorithm: {alg_name}")

    obj_args = param_bank.get(obj_name, [_DIM])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [_DIM] + list(obj_args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_base["Nagent"], *obj_args)

    policy = M_alpha_policy.get(
        obj_name,
        {"M_factor": 1.0, "alpha": 0.1, "decay": False, "maxIt": 500})

    x0_gen = x0_generator.get(obj_name, lambda d_, _: np.random.randn(d_))
    rng_s = np.random.RandomState(300 + mc_idx)
    np.random.set_state(rng_s.get_state())
    x0 = x0_gen(d, False)
    _, W_s = generate_random_graph(P_base["Nagent"], P_base["p_edge"])

    prm = _build_prm(P_base, policy, fun_list, d, L_vec,
                      x_opt_list, f_opt_list, fname, fparam, W_s)
    if prm_overrides:
        prm.update(prm_overrides)

    f0_val = float(np.mean([fi(x0) for fi in fun_list]))
    error = None
    try:
        _, out = alg_func(x0.copy(), dict(prm))
    except Exception as exc:
        out = {"fail": True, "failReason": str(exc)}
        error = {"type": type(exc).__name__, "message": str(exc)}

    diverged = detect_diverged(out, f0_val)
    failed = bool(out.get("fail") or diverged)
    failure_reason = out.get("failReason")
    if diverged and not failure_reason:
        failure_reason = "detect_diverged"

    return {
        "failed": failed,
        "failure_reason": failure_reason,
        "exception": error,
        "relF": np.asarray(out.get("relF", []), dtype=float),
        "commCost": np.asarray(out.get("commCost", []), dtype=float),
    }


#  Part 1 workers and runner

def _worker_part1(args):
    """Worker for Part 1 with explicit failure metadata."""
    obj_name, alg_name, mc_idx, P_base = args
    result = _run_one_alg(obj_name, alg_name, mc_idx, P_base)
    comm_values = (
        [np.nan] * len(_TOL_LEVELS)
        if result["failed"]
        else _comm_at_tol(result, _TOL_LEVELS)
    )
    return (
        obj_name,
        alg_name,
        mc_idx,
        comm_values,
        result["failed"],
        result["failure_reason"],
        result["exception"],
    )


def _run_part1(results_dir):
    print("\n" + "=" * 60)
    print("  Part 1 -Ce benefit (compression vs full)")
    print("=" * 60)

    P_base = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True,
    }

    alg_names = []
    for no_ce, with_ce in _CE_PAIRS:
        alg_names.extend([no_ce, with_ce])

    tasks = []
    for obj_name in _COMM_FUNCS:
        for alg_name in alg_names:
            for s in range(_NSTART):
                tasks.append((obj_name, alg_name, s, dict(P_base)))

    print(f"  {len(tasks)} tasks on {_N_WORKERS} workers")

    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_part1,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="comm_ce_benefit",
        max_workers=_N_WORKERS,
    )
    for obj_name, alg_name, mc_idx, comm_vals, failed, reason, error in completed:
        raw.setdefault((obj_name, alg_name), []).append(
            (mc_idx, comm_vals, failed, reason, error)
        )

    all_data = {}
    csv_rows = [["Function", "Algorithm"] +
                [f"tol={t:.0e}" for t in _TOL_LEVELS]]

    for obj_name in _COMM_FUNCS:
        all_data[obj_name] = {}
        for alg_name in alg_names:
            indexed_runs = sorted(raw.get((obj_name, alg_name), []))
            runs = [values for _, values, _, _, _ in indexed_runs]
            if runs:
                comm_mean = nanmean_columns(runs)
            else:
                comm_mean = np.full(len(_TOL_LEVELS), np.nan)
            all_data[obj_name][alg_name] = comm_mean.tolist()
            csv_rows.append([obj_name, alg_name] +
                            [f"{v:.2f}" if np.isfinite(v) else "NaN"
                             for v in comm_mean])

    csv_path = os.path.join(results_dir, "data_log", "ce_benefit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"  [Saved] {csv_path}")

    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_ce_benefit.json.gz"),
        {
            "schema_version": 1,
            "mode": "comm",
            "study": "ce_benefit",
            "tolerances": _TOL_LEVELS,
            "runs": [
                {
                    "objective": objective,
                    "algorithm": algorithm,
                    "mc_index": mc_index,
                    "seed": 300 + mc_index,
                    "communication_at_tolerance_mb": values,
                    "failed": failed,
                    "failure_reason": reason,
                    "exception": error,
                }
                for (objective, algorithm), indexed in sorted(raw.items())
                for mc_index, values, failed, reason, error in sorted(indexed)
            ],
        },
    )

    fig_ce_benefit(all_data, _CE_PAIRS, _TOL_LEVELS, results_dir)


#  Part 2 workers and runners

def _worker_part2(args):
    """Worker for Part 2 with trace and failure metadata."""
    obj_name, config_label, mc_idx, P_base, prm_overrides = args
    result = _run_one_alg(obj_name, "CeDisGrem", mc_idx, P_base, prm_overrides)
    return (
        obj_name,
        config_label,
        mc_idx,
        result["relF"].tolist(),
        result["commCost"].tolist(),
        result["failed"],
        result["failure_reason"],
        result["exception"],
    )


def _run_part2a(results_dir):
    """Part 2a: Klazy sweep."""
    print("\n" + "-" * 60)
    print("  Part 2a -Klazy sweep")
    print("-" * 60)

    P_base = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True,
    }

    tasks = []
    for obj_name in _COMM_FUNCS:
        for kl in _KLAZY_VALUES:
            for s in range(_NSTART):
                overrides = {"Klazy": kl}
                label = f"Klazy={kl}"
                tasks.append((obj_name, label, s, dict(P_base), overrides))

    print(f"  {len(tasks)} tasks on {_N_WORKERS} workers")

    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_part2,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="comm_klazy_sweep",
        max_workers=_N_WORKERS,
    )
    for obj_name, label, mc_idx, relF, commCost, failed, reason, error in completed:
        raw.setdefault((obj_name, label), []).append(
            (mc_idx, np.array(relF), np.array(commCost), failed, reason, error)
        )

    curves_by_func = {}
    klazy_labels = [f"Klazy={kl}" for kl in _KLAZY_VALUES]
    for obj_name in _COMM_FUNCS:
        curves = []
        for label in klazy_labels:
            runs = [
                run
                for run in raw.get((obj_name, label), [])
                if not run[3] and len(run[1]) > 0
            ]
            if not runs:
                curves.append((label, None, None))
                continue
            max_len = max(len(run[1]) for run in runs)
            relF_mat = np.full((_NSTART, max_len), np.nan)
            comm_mat = np.full((_NSTART, max_len), np.nan)
            for si, (_, rF, cC, _, _, _) in enumerate(sorted(runs)):
                relF_mat[si, :len(rF)] = rF
                comm_mat[si, :len(cC)] = cC
            curves.append((label,
                           nanmean_columns(relF_mat),
                           nanmean_columns(comm_mat)))
        curves_by_func[obj_name] = curves

    fig_comm_ablation(curves_by_func, _COMM_FUNCS,
                      "Klazy Sweep (CeDisGrem)", "klazy_sweep",
                      results_dir, cmap_name="Blues")
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_klazy_sweep.json.gz"),
        {
            "schema_version": 1,
            "mode": "comm",
            "study": "klazy_sweep",
            "runs": [
                {
                    "objective": objective,
                    "configuration": label,
                    "mc_index": mc_index,
                    "seed": 300 + mc_index,
                    "relF": rel_f,
                    "commCost": comm_cost,
                    "failed": failed,
                    "failure_reason": reason,
                    "exception": error,
                }
                for (objective, label), indexed in sorted(raw.items())
                for mc_index, rel_f, comm_cost, failed, reason, error in sorted(indexed)
            ],
        },
    )
    print("  Part 2a done.")


def _run_part2b(results_dir):
    """Part 2b: Compression method/param sweep."""
    print("\n" + "-" * 60)
    print("  Part 2b -Compression sweep")
    print("-" * 60)

    P_base = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True,
    }

    configs = _compress_configs(_DIM)
    tasks = []
    for obj_name in _COMM_FUNCS:
        for label, compH, compressor, cparam in configs:
            for s in range(_NSTART):
                overrides = {
                    "compressH": compH,
                    "compressor": compressor,
                }
                if cparam is not None:
                    overrides["compressParam"] = cparam
                tasks.append((obj_name, label, s, dict(P_base), overrides))

    print(f"  {len(tasks)} tasks on {_N_WORKERS} workers")

    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_part2,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="comm_compression_sweep",
        max_workers=_N_WORKERS,
    )
    for obj_name, label, mc_idx, relF, commCost, failed, reason, error in completed:
        raw.setdefault((obj_name, label), []).append(
            (mc_idx, np.array(relF), np.array(commCost), failed, reason, error)
        )

    config_labels = [c[0] for c in configs]
    curves_by_func = {}
    for obj_name in _COMM_FUNCS:
        curves = []
        for label in config_labels:
            runs = [
                run
                for run in raw.get((obj_name, label), [])
                if not run[3] and len(run[1]) > 0
            ]
            if not runs:
                curves.append((label, None, None))
                continue
            max_len = max(len(run[1]) for run in runs)
            relF_mat = np.full((_NSTART, max_len), np.nan)
            comm_mat = np.full((_NSTART, max_len), np.nan)
            for si, (_, rF, cC, _, _, _) in enumerate(sorted(runs)):
                relF_mat[si, :len(rF)] = rF
                comm_mat[si, :len(cC)] = cC
            curves.append((label,
                           nanmean_columns(relF_mat),
                           nanmean_columns(comm_mat)))
        curves_by_func[obj_name] = curves

    fig_comm_ablation(curves_by_func, _COMM_FUNCS,
                      "Compression Sweep (CeDisGrem)", "compress_sweep",
                      results_dir, cmap_name="RdYlGn")
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_compression_sweep.json.gz"),
        {
            "schema_version": 1,
            "mode": "comm",
            "study": "compression_sweep",
            "runs": [
                {
                    "objective": objective,
                    "configuration": label,
                    "mc_index": mc_index,
                    "seed": 300 + mc_index,
                    "relF": rel_f,
                    "commCost": comm_cost,
                    "failed": failed,
                    "failure_reason": reason,
                    "exception": error,
                }
                for (objective, label), indexed in sorted(raw.items())
                for mc_index, rel_f, comm_cost, failed, reason, error in sorted(indexed)
            ],
        },
    )
    print("  Part 2b done.")


#  Entry point

def run_comm(part: str = "all") -> None:
    """
    Communication cost study.

    Parameters
    ----------
    part : 'all' | 'ce' | 'ablation'
    """
    results_dir = os.path.join(_root, "results", "comm")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data_log"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "paper", "comm_profile"), exist_ok=True)

    p = part.lower()
    if p in ("all", "ce"):
        _run_part1(results_dir)
    if p in ("all", "ablation"):
        _run_part2a(results_dir)
        _run_part2b(results_dir)

    print("\n[run_comm] All done.")
