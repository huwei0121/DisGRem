"""
run_ada.py -Adaptive-mechanism study for AdaDisGrem.

Three experiments on 4 representative functions:

  1. M trajectory  -plot M(t) for AdaDisGrem / CeAdaDisGrem vs DisGrem's fixed M.
  2. Ada vs Fixed-M - compare relF convergence of Ada (auto M) against DisGrem
     run at five manually chosen fixed M values.
  3. Initial-M robustness  -run Ada from 5 different initial M values and show
     that M trajectories converge to a similar operating point.

Functions: ridge, logsumexp, logreg_real, logreg_ncvr
Output: results/ada/paper/ada_mechanism/
"""

from __future__ import annotations
import os
import sys
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.helper.run_utils import detect_diverged
from utils.export.plot_utils import (fig_ada_m_trajectory,
                                      fig_ada_vs_fixed_m,
                                      fig_ada_init_m_robust)
from experiments.protocol import (
    configured_iteration_override,
    execute_cached_tasks,
    nanmean_columns,
    resolve_iteration_budget,
    write_json_gz,
)


_ADA_FUNCS = ["ridge", "logsumexp", "logreg_real", "logreg_ncvr"]
_NSTART    = int(os.environ.get("LOG_SCHEDULE_NSTART", "5"))

_FIXED_M_FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]
_INIT_M_FACTORS  = [0.1, 0.5, 1.0, 3.0, 10.0]
_N_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))


#  Helpers

def _make_prm(P, policy, fun_list, d, L_vec,
              x_opt_list, f_opt_list, fname, fparam, W):
    maxIt_obj = resolve_iteration_budget(policy, P)
    prm = dict(P)
    prm["maxIt"] = maxIt_obj
    if "tolType" in policy:
        prm["tolType"] = policy["tolType"]
    prm.update({
        "f": fun_list, "fname": fname, "fparam": fparam, "dim": d,
        "M":            policy["M_factor"] * float(L_vec.max()),
        "alpha":        policy["alpha"] / float(L_vec.max()),
        "decay_alpha":  policy.get("decay", False),
        "x_opt":        x_opt_list[0] if x_opt_list else None,
        "f_opt":        float(np.mean(f_opt_list)) if len(f_opt_list) else 0.0,
        "W":            W,
        "esom_penalty": 1.0,
    })
    return prm


def _worker_ada_mc(args):
    """Worker: run one (obj_name, alg_name, mc_idx) with optional M override."""
    obj_name, alg_name, mc_idx, P_dict, M_override = args

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"

    param_bank, M_alpha_policy, x0_generator = init_policy("regular")
    full_bank = get_alg_bank("All")
    alg_func = None
    for n, f in full_bank:
        if n == alg_name:
            alg_func = f
            break
    if alg_func is None:
        return obj_name, alg_name, mc_idx, M_override, {"fail": True}

    obj_args = param_bank.get(obj_name, [P_dict["d_override"]])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [P_dict["d_override"]] + list(obj_args[1:])
    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_dict["Nagent"], *obj_args)

    policy = M_alpha_policy.get(
        obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False,
                   "maxIt": P_dict["maxIt"]})

    x0_gen = x0_generator.get(obj_name, lambda d_, _: np.random.randn(d_))

    rng_s = np.random.RandomState(700 + mc_idx)
    np.random.set_state(rng_s.get_state())
    x0 = x0_gen(d, False)
    _, W_s = generate_random_graph(P_dict["Nagent"], P_dict["p_edge"])

    prm = _make_prm(P_dict, policy, fun_list, d, L_vec,
                     x_opt_list, f_opt_list, fname, fparam, W_s)
    prm["verbose"] = False
    if M_override is not None:
        prm["M"] = M_override

    f0_val = float(np.mean([fi(x0) for fi in fun_list]))
    try:
        _, out = alg_func(x0.copy(), dict(prm))
        if detect_diverged(out, f0_val):
            out["__failed__"] = True
    except Exception:
        out = {"fail": True}

    fields = ["relF", "Mavg", "combo", "gradNrm", "cons", "commCost"]
    result = {}
    for fld in fields:
        arr = out.get(fld, [])
        if arr is not None and len(arr) > 0:
            result[fld] = list(np.asarray(arr, dtype=float))
    if out.get("fail") or out.get("__failed__"):
        result["fail"] = True

    return obj_name, alg_name, mc_idx, M_override, result


def _average_outs(results_list, nstart):
    """Average numeric fields across MC runs."""
    fields = ["relF", "Mavg", "combo", "gradNrm", "cons", "commCost"]
    all_runs = results_list
    if not all_runs:
        return {"fail": True}
    max_len = max(len(np.asarray(o.get("relF", []))) for o in all_runs)
    if max_len == 0:
        return {"fail": True}
    avg = {}
    for f_name in fields:
        mat = np.full((len(all_runs), max_len), np.nan)
        for si, o in enumerate(all_runs):
            arr = np.asarray(o.get(f_name, []), dtype=float)
            n = min(len(arr), max_len)
            mat[si, :n] = arr[:n]
        avg[f_name] = nanmean_columns(mat).tolist()
    return avg


#  Experiment 1 -M trajectory (parallelized)

def _exp1_trajectory(results_dir, P, param_bank, M_alpha_policy, x0_generator):
    print("\n  Exp 1 -M trajectory (DisGrem / AdaDisGrem / CeAdaDisGrem)")
    alg_list = ["DisGrem", "AdaDisGrem", "CeAdaDisGrem"]

    tasks = []
    for obj_name in _ADA_FUNCS:
        for an in alg_list:
            for s in range(_NSTART):
                tasks.append((obj_name, an, s, dict(P), None))

    print(f"    {len(tasks)} tasks on {_N_WORKERS} workers")
    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_ada_mc,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="ada_m_trajectory",
        max_workers=_N_WORKERS,
    )
    for obj_name, alg_name, mc_idx, _, result in completed:
        raw.setdefault((obj_name, alg_name), []).append((mc_idx, result))

    logs = {}
    for obj_name in _ADA_FUNCS:
        obj_log = {}
        for an in alg_list:
            runs = [result for _, result in sorted(raw.get((obj_name, an), []))]
            obj_log[an] = _average_outs(runs, _NSTART)
        logs[obj_name] = obj_log
        print(f"    {obj_name} done")

    fig_ada_m_trajectory(logs, _ADA_FUNCS, results_dir)
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_m_trajectory.json.gz"),
        {
            "schema_version": 1,
            "mode": "ada",
            "study": "m_trajectory",
            "runs": [
                {
                    "objective": objective,
                    "algorithm": algorithm,
                    "mc_index": mc_index,
                    "seed": 700 + mc_index,
                    "metrics": result,
                }
                for (objective, algorithm), indexed in sorted(raw.items())
                for mc_index, result in sorted(indexed)
            ],
        },
    )
    return logs


#  Experiment 2 -Ada vs Fixed-M (parallelized)

def _exp2_fixed_m(results_dir, P, param_bank, M_alpha_policy, x0_generator,
                   ada_logs_from_exp1):
    print("\n  Exp 2 -Ada vs Fixed-M DisGrem")

    tasks = []
    base_M_cache = {}
    for obj_name in _ADA_FUNCS:
        obj_args = param_bank.get(obj_name, [P["d_override"]])
        if obj_args and isinstance(obj_args[0], (int, float)):
            obj_args = [P["d_override"]] + list(obj_args[1:])
        _, _, L_vec, _, _, _, _, _ = \
            obj_factory(obj_name, P["Nagent"], *obj_args)
        policy = M_alpha_policy.get(
            obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False,
                       "maxIt": P["maxIt"]})
        base_M = policy["M_factor"] * float(L_vec.max())
        base_M_cache[obj_name] = base_M

        for mf in _FIXED_M_FACTORS:
            M_val = base_M * mf
            for s in range(_NSTART):
                tasks.append((obj_name, "DisGrem", s, dict(P), M_val))

    print(f"    {len(tasks)} tasks on {_N_WORKERS} workers")
    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_ada_mc,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="ada_fixed_m",
        max_workers=_N_WORKERS,
    )
    for obj_name, _, mc_idx, M_over, result in completed:
        raw.setdefault((obj_name, M_over), []).append((mc_idx, result))

    ada_log = {}
    fixed_m_logs = {}
    for obj_name in _ADA_FUNCS:
        ada_log[obj_name] = ada_logs_from_exp1.get(obj_name, {}).get(
            "AdaDisGrem", {})
        base_M = base_M_cache[obj_name]
        fm = {}
        for mf in _FIXED_M_FACTORS:
            M_val = base_M * mf
            runs = [result for _, result in sorted(raw.get((obj_name, M_val), []))]
            fm[mf] = _average_outs(runs, _NSTART)
        fixed_m_logs[obj_name] = fm
        print(f"    {obj_name} done")

    fig_ada_vs_fixed_m(ada_log, fixed_m_logs, _FIXED_M_FACTORS,
                        _ADA_FUNCS, results_dir)
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_ada_vs_fixed_m.json.gz"),
        {
            "schema_version": 1,
            "mode": "ada",
            "study": "ada_vs_fixed_m",
            "runs": [
                {
                    "objective": objective,
                    "M": m_value,
                    "mc_index": mc_index,
                    "seed": 700 + mc_index,
                    "metrics": result,
                }
                for (objective, m_value), indexed in sorted(raw.items())
                for mc_index, result in sorted(indexed)
            ],
        },
    )


#  Experiment 3 -Initial-M robustness (parallelized)

def _exp3_init_m(results_dir, P, param_bank, M_alpha_policy, x0_generator):
    print("\n  Exp 3 -Initial-M robustness")

    tasks = []
    base_M_cache = {}
    for obj_name in _ADA_FUNCS:
        obj_args = param_bank.get(obj_name, [P["d_override"]])
        if obj_args and isinstance(obj_args[0], (int, float)):
            obj_args = [P["d_override"]] + list(obj_args[1:])
        _, _, L_vec, _, _, _, _, _ = \
            obj_factory(obj_name, P["Nagent"], *obj_args)
        policy = M_alpha_policy.get(
            obj_name, {"M_factor": 1.0, "alpha": 0.1, "decay": False,
                       "maxIt": P["maxIt"]})
        base_M = policy["M_factor"] * float(L_vec.max())
        base_M_cache[obj_name] = base_M

        for mf in _INIT_M_FACTORS:
            M_val = base_M * mf
            for s in range(_NSTART):
                tasks.append((obj_name, "AdaDisGrem", s, dict(P), M_val))

    print(f"    {len(tasks)} tasks on {_N_WORKERS} workers")
    raw = {}
    completed = execute_cached_tasks(
        tasks,
        _worker_ada_mc,
        cache_root=os.path.join(_root, "_run_cache"),
        namespace="ada_initial_m",
        max_workers=_N_WORKERS,
    )
    for obj_name, _, mc_idx, M_over, result in completed:
        raw.setdefault((obj_name, M_over), []).append((mc_idx, result))

    init_m_logs = {}
    for obj_name in _ADA_FUNCS:
        base_M = base_M_cache[obj_name]
        im = {}
        for mf in _INIT_M_FACTORS:
            M_val = base_M * mf
            runs = [result for _, result in sorted(raw.get((obj_name, M_val), []))]
            im[mf] = _average_outs(runs, _NSTART)
        init_m_logs[obj_name] = im
        print(f"    {obj_name} done")

    fig_ada_init_m_robust(init_m_logs, _INIT_M_FACTORS,
                           _ADA_FUNCS, results_dir)
    write_json_gz(
        os.path.join(results_dir, "data_log", "raw_initial_m_robustness.json.gz"),
        {
            "schema_version": 1,
            "mode": "ada",
            "study": "initial_m_robustness",
            "runs": [
                {
                    "objective": objective,
                    "initial_M": m_value,
                    "mc_index": mc_index,
                    "seed": 700 + mc_index,
                    "metrics": result,
                }
                for (objective, m_value), indexed in sorted(raw.items())
                for mc_index, result in sorted(indexed)
            ],
        },
    )


#  Entry point

def run_ada() -> None:
    """Run the full adaptive-mechanism study (3 experiments)."""
    results_dir = os.path.join(_root, "results", "ada")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "paper", "ada_mechanism"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data_log"), exist_ok=True)

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo",
        "verbose": False, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10, "info": 2,
        "countComm": True, "d_override": 30,
    }
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    print("=" * 60)
    print("  Adaptive Mechanism Study")
    print("=" * 60)

    # Exp 1: M trajectory
    exp1_logs = _exp1_trajectory(results_dir, P, param_bank,
                                  M_alpha_policy, x0_generator)

    # Exp 2: Ada vs Fixed-M comparison
    _exp2_fixed_m(results_dir, P, param_bank, M_alpha_policy,
                   x0_generator, exp1_logs)

    # Exp 3: Initial-M robustness
    _exp3_init_m(results_dir, P, param_bank, M_alpha_policy, x0_generator)

    print("\n[run_ada] All done.")
