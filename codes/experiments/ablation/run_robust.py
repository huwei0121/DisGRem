"""
run_robust.py - Robustness analysis (two parts), with parallel acceleration.

Part 1 - Starting-point robustness
  100 Monte-Carlo runs x {near, far} x all 9 functions.
  Starting points sampled uniformly in a d-ball of radius r_near / r_far
  centred at each function's default starting point.
  Uses ProcessPoolExecutor for roughly 10x speedup on multi-core machines.
  Output: 2 success-rate heatmap tables (one per scenario) + CSV.

Part 2 - Parameter sensitivity
  Sweep alpha / M with gradient-coloured convergence curves.
  3 algorithm groups x 4 functions:
    DisGrem (no Ada) : DisGrem, CeDisGrem - sweep alpha and M
    First-order      : EXTRA, DIGing - sweep alpha
    Second-order     : DQM, ESOM, SONATA, NetworkGIANT - sweep one declared
                       algorithm-specific damping or step parameter
  Output: one combined gradient-colour figure per algorithm group.
"""

from __future__ import annotations
import os
import sys
import csv
import time
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged
from utils.export.plot_utils import (fig_success_rate_table,
                                      fig_param_sweep_combined)
from experiments.protocol import (
    configured_iteration_override,
    execute_cached_tasks,
    nanmean_columns,
    resolve_iteration_budget,
    write_json_gz,
)


#  Constants
_NSTART_PART1 = int(os.environ.get("LOG_SCHEDULE_NSTART_PART1", "100"))
_TOL          = 1e-6
_R_NEAR       = 1.0
_R_FAR        = 3.0

_SWEEP_FUNCS   = ["ridge", "logsumexp", "logreg_real", "logreg_ncvr"]
_SWEEP_FACTORS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
_NSTART_SWEEP  = int(os.environ.get("LOG_SCHEDULE_NSTART", "5"))

_ALL_FUNCS = ["ridge", "quadbad", "logsumexp", "huber", "linlog",
              "logreg_real", "rosenbrock", "styblinski_tang", "logreg_ncvr"]

_N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))


#  Helpers

def _sample_in_ball(center: np.ndarray, radius: float,
                    rng: np.random.RandomState) -> np.ndarray:
    """Uniform sample inside a d-ball of given radius, centred at *center*."""
    d = len(center)
    z = rng.randn(d)
    z /= np.linalg.norm(z) + 1e-30
    u = rng.rand() ** (1.0 / d)
    return center + radius * u * z


def _run_summary(out: dict, f0_val: float, error: dict | None = None) -> dict:
    """Retain convergence and failure evidence for one robustness run."""
    diverged = detect_diverged(out, f0_val)
    failed = bool(out.get("fail") or out.get("__failed__") or diverged)
    relF = np.asarray(out.get("relF", [np.nan]), dtype=float)
    combo = np.asarray(out.get("combo", [np.nan]), dtype=float)
    success = bool(not failed and (np.any(relF < _TOL) or np.any(combo < 1e-10)))
    finite_relF = relF[np.isfinite(relF)]
    finite_combo = combo[np.isfinite(combo)]
    failure_reason = out.get("failReason")
    if diverged and not failure_reason:
        failure_reason = "detect_diverged"
    return {
        "success": success,
        "failed": failed,
        "failure_reason": failure_reason,
        "exception": error,
        "best_relF": float(np.min(finite_relF)) if finite_relF.size else np.nan,
        "best_combo": float(np.min(finite_combo)) if finite_combo.size else np.nan,
        "iterations": int(max(relF.size, combo.size)),
    }


def _build_prm(P: dict, policy: dict, fun_list, d, L_vec,
               x_opt_list, f_opt_list, fname, fparam, W) -> dict:
    """Assemble the algorithm parameter dictionary from policy + global P."""
    maxIt_obj = resolve_iteration_budget(policy, P)
    prm = dict(P)
    prm["maxIt"] = maxIt_obj
    if "tolType" in policy:
        prm["tolType"] = policy["tolType"]
    prm.update({
        "f":            fun_list,
        "fname":        fname,
        "fparam":       fparam,
        "dim":          d,
        "M":            policy["M_factor"] * float(L_vec.max()),
        "alpha":        policy["alpha"] / float(L_vec.max()),
        "decay_alpha":  policy.get("decay", False),
        "objName":      "",
        "x_opt":        x_opt_list[0] if x_opt_list else None,
        "f_opt":        float(np.mean(f_opt_list)) if len(f_opt_list) else 0.0,
        "W":            W,
        "esom_penalty": 1.0,
    })
    return prm


def _set_single_thread_blas():
    """Set env vars to force single-threaded BLAS (must be called before numpy import in worker)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


#  Part 1 - Parallel worker

def _worker_mc_part1(args):
    """
    Worker process: run all algorithms for one MC sample on one function.
    Returns dict {alg_name: bool(success)}.
    """
    obj_name, mc_idx, radius, P_dict = args
    _set_single_thread_blas()

    alg_bank = get_alg_bank("MainComp")
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    obj_args = param_bank.get(obj_name, [P_dict["d_override"]])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [P_dict["d_override"]] + list(obj_args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_dict["Nagent"], *obj_args)

    policy = M_alpha_policy.get(
        obj_name,
        {"M_factor": 1.0, "alpha": 0.1, "decay": False,
         "maxIt": P_dict["maxIt"]})

    x0_gen = x0_generator.get(obj_name, lambda d_, _: np.random.randn(d_))
    rng_center = np.random.RandomState(42)
    np.random.set_state(rng_center.get_state())
    center = x0_gen(d, False)

    rng_s = np.random.RandomState(1000 + mc_idx)
    x0 = _sample_in_ball(center, radius, rng_s)

    np.random.seed(2000 + mc_idx)
    _, W_s = generate_random_graph(P_dict["Nagent"], P_dict["p_edge"])

    prm = _build_prm(P_dict, policy, fun_list, d, L_vec,
                      x_opt_list, f_opt_list, fname, fparam, W_s)
    f0_val = float(np.mean([fi(x0) for fi in fun_list]))

    results = {}
    for alg_name, alg_func in alg_bank:
        error = None
        try:
            _, out = alg_func(x0.copy(), dict(prm))
        except Exception as exc:
            out = {"fail": True, "failReason": str(exc)}
            error = {"type": type(exc).__name__, "message": str(exc)}
        results[alg_name] = _run_summary(out, f0_val, error)

    return mc_idx, results


#  Part 1 - Starting-point robustness (parallel)

def _run_part1(results_dir: str) -> None:
    print("\n" + "=" * 72)
    print("  Part 1 - Starting-point robustness")
    print(f"  ({_NSTART_PART1} MC x {len(_ALL_FUNCS)} functions x 2 scenarios,"
          f" {_N_WORKERS} workers)")
    print("=" * 72)

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True, "d_override": 30,
    }

    alg_names = [an for an, _ in get_alg_bank("MainComp")]
    scenarios = [("near", _R_NEAR), ("far", _R_FAR)]
    csv_rows = [["Scenario", "Function", "Algorithm", "SuccessRate%"]]
    raw_runs = []

    # Force single-threaded BLAS in spawned workers
    _orig_env = {}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _orig_env[k] = os.environ.get(k)
        os.environ[k] = "1"

    try:
        for sc_name, radius in scenarios:
            print(f"\n{'-'*60}")
            print(f"  Scenario: {sc_name}  (radius={radius})")
            print(f"{'-'*60}")

            success_data = {}
            for obj_name in _ALL_FUNCS:
                t0 = time.perf_counter()
                print(f"\n  [{sc_name}] {obj_name}  ", end="", flush=True)
                tasks = [
                    (obj_name, sample, radius, P)
                    for sample in range(_NSTART_PART1)
                ]
                completed = execute_cached_tasks(
                    tasks,
                    _worker_mc_part1,
                    cache_root=os.path.join(_root, "_run_cache"),
                    namespace=f"robust_start_{sc_name}_{obj_name}",
                    max_workers=_N_WORKERS,
                    progress_every=25,
                )
                counts = {algorithm: 0 for algorithm in alg_names}
                for mc_idx, result in completed:
                    for algorithm, summary in result.items():
                        counts[algorithm] += int(summary["success"])
                    raw_runs.append(
                        {
                            "scenario": sc_name,
                            "radius": radius,
                            "objective": obj_name,
                            "mc_index": mc_idx,
                            "starting_point_seed": 1000 + mc_idx,
                            "graph_seed": 2000 + mc_idx,
                            "algorithms": result,
                        }
                    )

                elapsed = time.perf_counter() - t0
                print(f"  ({elapsed:.1f}s)")
                obj_sr = {}
                for algorithm in alg_names:
                    rate = 100.0 * counts[algorithm] / _NSTART_PART1
                    obj_sr[algorithm] = rate
                    csv_rows.append([sc_name, obj_name, algorithm, f"{rate:.1f}"])
                success_data[obj_name] = obj_sr
                best_algorithm = max(alg_names, key=lambda name: obj_sr[name])
                print(f"    Best: {best_algorithm} ({obj_sr[best_algorithm]:.0f}%)")

            fig_success_rate_table(
                success_data,
                alg_names,
                _ALL_FUNCS,
                sc_name,
                results_dir,
            )

    finally:
        for k, v in _orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    csv_path = os.path.join(results_dir, "data_log", "success_rate.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"\n[Saved] {csv_path}")
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_starting_point_robustness.json.gz"),
        {
            "schema_version": 1,
            "mode": "robust",
            "study": "starting_point_robustness",
            "success_tolerance": _TOL,
            "runs": sorted(
                raw_runs,
                key=lambda row: (row["scenario"], row["objective"], row["mc_index"]),
            ),
        },
    )


#  Part 2 - Parameter sensitivity (parallel per-factor)

_SWEEP_GROUPS = [
    {
        "label": "disgrem_family",
        "alg_mode": "DisGremNoAda",
        "params": ["M"],
        "cmap": "Blues",
    },
    {
        "label": "first-order",
        "alg_mode": "FirstOrder",
        "params": ["alpha"],
        "cmap": "Oranges",
    },
    {
        "label": "second-order",
        "alg_mode": "SecondOrder",
        "params": ["alg_specific"],
        "cmap": "Reds",
    },
]

# Each second-order baseline has its own key tuning parameter.
# DQM / ESOM: ADMM penalty coefficient (analogous to step size).
# SONATA: Hessian regularisation tau (controls Newton damping).
# NetworkGIANT: Newton step size alpha_step.
_SECOND_ORDER_PARAM = {
    "DQM":          ("c",             0.5),
    "ESOM":         ("esom_penalty",  1.0),
    "SONATA":       ("sonata_tau",    0.01),
    "NetworkGIANT": ("alpha_step",    1.0),
}
_SECOND_ORDER_LABEL = {
    "DQM":          r"$c$",
    "ESOM":         r"$\rho$",
    "SONATA":       r"$\tau$",
    "NetworkGIANT": r"$\alpha$",
}


def _worker_sweep_factor(args):
    """
    Worker: run one algorithm at one factor value for _NSTART_SWEEP MC runs.
    Returns (factor, relF_mean, is_diverged).
    """
    (obj_name, alg_mode, alg_idx, pname, factor, P_dict) = args
    _set_single_thread_blas()

    alg_bank_g = get_alg_bank(alg_mode)
    alg_name, alg_func = alg_bank_g[alg_idx]
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    obj_args = param_bank.get(obj_name, [P_dict["d_override"]])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [P_dict["d_override"]] + list(obj_args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_dict["Nagent"], *obj_args)

    policy = M_alpha_policy.get(
        obj_name,
        {"M_factor": 1.0, "alpha": 0.1, "decay": False,
         "maxIt": P_dict["maxIt"]})

    base_alpha = policy["alpha"] / float(L_vec.max())
    base_M = policy["M_factor"] * float(L_vec.max())
    x0_gen = x0_generator.get(obj_name, lambda d_, _: np.random.randn(d_))

    relF_all = []
    n_div = 0
    raw_runs = []

    for s in range(_NSTART_SWEEP):
        rng_s = np.random.RandomState(500 + s)
        np.random.set_state(rng_s.get_state())
        x0 = x0_gen(d, False)
        _, W_s = generate_random_graph(P_dict["Nagent"], P_dict["p_edge"])
        prm = _build_prm(P_dict, policy, fun_list, d, L_vec,
                          x_opt_list, f_opt_list, fname, fparam, W_s)

        if pname == "alpha":
            prm["alpha"] = base_alpha * factor
        elif pname == "alg_specific" and alg_name in _SECOND_ORDER_PARAM:
            key, base_val = _SECOND_ORDER_PARAM[alg_name]
            prm[key] = base_val * factor
        else:
            prm["M"] = base_M * factor

        f0_val = float(np.mean([fi(x0) for fi in fun_list]))
        diverged = False
        failure_reason = None
        error = None
        try:
            _, out = alg_func(x0.copy(), dict(prm))
            relF = np.asarray(out.get("relF", [np.nan]), dtype=float)
            if out.get("fail") or detect_diverged(out, f0_val):
                n_div += 1
                diverged = True
                failure_reason = out.get("failReason") or "detect_diverged"
        except Exception as exc:
            relF = np.array([1.0])
            n_div += 1
            diverged = True
            failure_reason = str(exc)
            error = {"type": type(exc).__name__, "message": str(exc)}
        relF_all.append(relF)
        raw_runs.append(
            {
                "mc_index": s,
                "seed": 500 + s,
                "diverged": diverged,
                "failure_reason": failure_reason,
                "exception": error,
                "relF": relF.tolist(),
            }
        )

    max_len = max(len(r) for r in relF_all)
    padded = np.full((_NSTART_SWEEP, max_len), np.nan)
    for si, r in enumerate(relF_all):
        padded[si, :len(r)] = r
    relF_mean = nanmean_columns(padded)

    is_div = n_div > _NSTART_SWEEP // 2
    return (factor, relF_mean.tolist(), is_div, alg_idx, pname, raw_runs)


def _run_part2(results_dir: str) -> None:
    print("\n" + "=" * 72)
    print("  Part 2 - Parameter sensitivity")
    print(f"  ({_N_WORKERS} workers)")
    print("=" * 72)

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True, "d_override": 30,
    }

    _orig_env = {}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _orig_env[k] = os.environ.get(k)
        os.environ[k] = "1"

    raw_output = []
    try:
        for grp in _SWEEP_GROUPS:
                group_label = grp["label"]
                alg_mode = grp["alg_mode"]
                alg_bank_g = get_alg_bank(alg_mode)
                alg_names_g = [an for an, _ in alg_bank_g]
                params_to_sweep = grp["params"]
                cmap_name = grp["cmap"]

                all_entries_by_func = {}

                for obj_name in _SWEEP_FUNCS:
                    t0 = time.perf_counter()
                    print(f"\n  [{group_label}] {obj_name}  ", end="", flush=True)

                    task_list = []
                    for alg_idx in range(len(alg_bank_g)):
                        for pname in params_to_sweep:
                            for fac in _SWEEP_FACTORS:
                                task_list.append((
                                    obj_name, alg_mode, alg_idx,
                                    pname, fac, P))

                    raw = {}
                    completed = execute_cached_tasks(
                        task_list,
                        _worker_sweep_factor,
                        cache_root=os.path.join(_root, "_run_cache"),
                        namespace=f"robust_sweep_{group_label}_{obj_name}",
                        max_workers=_N_WORKERS,
                        progress_every=10,
                    )
                    for result in completed:
                        fac, relF_list, is_div, ai, pn, raw_runs = result
                        raw.setdefault((ai, pn), []).append(
                            (fac, np.array(relF_list), is_div)
                        )
                        raw_output.extend(
                            {
                                "group": group_label,
                                "objective": obj_name,
                                "algorithm": alg_names_g[ai],
                                "parameter": pn,
                                "factor": fac,
                                **run,
                            }
                            for run in raw_runs
                        )

                    entries = []
                    for alg_idx, alg_name in enumerate(alg_names_g):
                        for pname in params_to_sweep:
                            key = (alg_idx, pname)
                            if key not in raw:
                                continue
                            curves_raw = sorted(raw[key], key=lambda x: x[0])
                            curves = [(fac, relF, div)
                                      for fac, relF, div in curves_raw]
                            if pname == "alpha":
                                p_label = r"$\alpha$"
                            elif pname == "alg_specific":
                                p_label = _SECOND_ORDER_LABEL.get(
                                    alg_name, r"param")
                            else:
                                p_label = r"$M$"
                            entries.append({
                                "alg_name": alg_name,
                                "param_label": p_label,
                                "factors": [c[0] for c in curves],
                                "curves": curves,
                            })
                            n_div = sum(1 for c in curves if c[2])
                            status = ("OK" if n_div == 0
                                      else f"{n_div}/{len(curves)} div")
                            print(f"\n    {alg_name} {pname}: {status}",
                                  end="", flush=True)

                    elapsed = time.perf_counter() - t0
                    print(f"  ({elapsed:.1f}s)")
                    all_entries_by_func[obj_name] = entries

                # Combined big figure: all functions for this group
                fig_param_sweep_combined(
                    all_entries_by_func, group_label, _SWEEP_FUNCS,
                    results_dir, cmap_name)

    finally:
        for k, v in _orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_parameter_sensitivity.json.gz"),
        {
            "schema_version": 1,
            "mode": "robust",
            "study": "parameter_sensitivity",
            "factors": _SWEEP_FACTORS,
            "runs": sorted(
                raw_output,
                key=lambda row: (
                    row["group"],
                    row["objective"],
                    row["algorithm"],
                    row["parameter"],
                    row["factor"],
                    row["mc_index"],
                ),
            ),
        },
    )


#  Entry point

def run_robust(part: str = "all") -> None:
    """
    Run robustness analysis.

    Parameters
    ----------
    part : 'all' | 'start' | 'param'
        'start' - Part 1 only (starting-point robustness)
        'param' - Part 2 only (parameter sensitivity)
        'all'   - both parts
    """
    results_dir = os.path.join(_root, "results", "robust")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "paper"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data_log"), exist_ok=True)

    if part in ("all", "start"):
        _run_part1(results_dir)
    if part in ("all", "param"):
        _run_part2(results_dir)

    print("\n[run_robust] All done.")
