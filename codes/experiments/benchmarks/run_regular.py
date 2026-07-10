"""
run_regular.py -Standard benchmark with parallel MC acceleration.
Ported from MATLAB: run_regular.m
"""

from __future__ import annotations
import os
import sys
import shutil
import time
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.export.log_export import merge_logs, write_txt_summary, write_log_to_excel
from utils.export.plot_utils import (fig_plot_custom, fig_plot_multiobj_custom,
                                      fig_plot_multiobj_repr,
                                      fig_perf_profiles_tol_panel,
                                      fig_perf_profiles_comm_panel)
from utils.helper.run_utils import detect_diverged
from experiments.protocol import (
    configured_iteration_override,
    execute_cached_tasks,
    resolve_iteration_budget,
    write_json_gz,
)

_N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))


# -- Results directory management --------------------------------------------
def clear_results(results_dir: str) -> None:
    """Delete and recreate the results directory tree (call before each full run)."""
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        print(f"[clear_results] Removed old results: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    for sub in ("fig_single", "fig_multiobj", "data_log", "param_study"):
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)
    print(f"[clear_results] Created fresh directory tree under {results_dir}")


# -- Main benchmark set (9 functions) ---------------------------------------
_CONVEX_BASES    = ["logsumexp", "huber", "logreg_real"]
_NONCONVEX_BASES = ["linlog", "rosenbrock", "styblinski_tang", "logreg_ncvr"]


#  Parallel worker

def _worker_mc_regular(args):
    """
    Worker process: run all algorithms for one MC sample on one function.
    Returns (mc_idx, log_dict, f0_val).
    """
    obj_name, mc_idx, P_dict = args

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"

    alg_bank = get_alg_bank("MainComp")
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    obj_args = param_bank.get(obj_name, [P_dict["d_override"]])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [P_dict["d_override"]] + list(obj_args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_dict["Nagent"], *obj_args)

    policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1,
                                            "decay": False, "maxIt": P_dict["maxIt"]})
    maxIt_obj = resolve_iteration_budget(policy, P_dict)
    M_val = policy["M_factor"] * float(L_vec.max())
    alp_val = policy["alpha"] / float(L_vec.max())

    prm = dict(P_dict)
    prm["maxIt"] = maxIt_obj
    prm["verbose"] = False
    if "tolType" in policy:
        prm["tolType"] = policy["tolType"]
    prm.update({
        "f": fun_list, "fname": fname, "fparam": fparam,
        "dim": d, "M": M_val, "alpha": alp_val,
        "decay_alpha": policy["decay"], "objName": obj_name,
        "x_opt": x_opt_list[0] if x_opt_list else None,
        "f_opt": float(np.mean(f_opt_list)),
        "esom_penalty": 1.0,
    })

    x0_gen = x0_generator.get(obj_name, lambda d_, far: np.random.randn(d_))

    rng_s = np.random.RandomState(100 + mc_idx)
    np.random.set_state(rng_s.get_state())
    x0 = x0_gen(d, P_dict.get("far", False))
    _, W_s = generate_random_graph(P_dict["Nagent"], P_dict["p_edge"])
    prm["W"] = W_s

    f0_val = float(np.mean([fi(x0) for fi in fun_list]))

    log_s = {}
    for alg_name, alg_func in alg_bank:
        try:
            _, out = alg_func(x0.copy(), dict(prm))
        except Exception as e:
            out = {"fail": True, "failReason": str(e)}
        if detect_diverged(out, f0_val):
            out["__failed__"] = True
        log_s[alg_name] = out

    return mc_idx, log_s, f0_val


#  Main entry

def run_regular(func_group: str = "all") -> None:
    """
    Run the standard benchmark for the given function group.

    Parameters
    ----------
    func_group : 'all' (default, 9-function main set) |
                 'convexset' | 'nonconvexset' |
                 'ridge' | 'quadbad' | any atomic function name
    """
    results_dir = os.path.join(_root, "results", "main")
    if func_group == "all":
        clear_results(results_dir)

    P = {
        "Nagent":      10,
        "p_edge":      0.5,
        "maxIt":       int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol":         1e-12,
        "tolType":     "combo",
        "verbose":     True,
        "showPlots":   True,
        "far":         False,
        "useWorst":    False,
        "nStart":      int(os.environ.get("LOG_SCHEDULE_NSTART", "20")),
        "d_override":  30,
        "info":        2,
        "NC":          3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10,
        "countComm":   True,
    }

    alg_bank = get_alg_bank("MainComp")
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    fg = func_group.lower()
    if fg == "ridge":
        obj_list = ["ridge"]
    elif fg == "quadbad":
        obj_list = ["quadbad"]
    elif fg == "convexset":
        obj_list = ["ridge", "quadbad"] + _CONVEX_BASES
    elif fg == "nonconvexset":
        obj_list = _NONCONVEX_BASES
    elif fg == "all":
        obj_list = ["ridge", "quadbad"] + _CONVEX_BASES + _NONCONVEX_BASES
    else:
        obj_list = [fg]

    all_logs = {}
    all_individual = {}

    # Force single-threaded BLAS in spawned workers
    _orig_env = {}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _orig_env[k] = os.environ.get(k)
        os.environ[k] = "1"

    try:
        for obj_name in obj_list:
            print(f"\n{'='*60}")
            print(f" Objective: {obj_name}  ({_N_WORKERS} workers x "
                  f"{P['nStart']} MC)")
            print(f"{'='*60}")

            t0_func = time.perf_counter()
            tasks = [(obj_name, sample, P) for sample in range(P["nStart"])]
            completed = execute_cached_tasks(
                tasks,
                _worker_mc_regular,
                cache_root=os.path.join(_root, "_run_cache"),
                namespace=f"regular_{obj_name}",
                max_workers=_N_WORKERS,
                progress_every=5,
            )

            logs_each = [None] * P["nStart"]
            f0_vals = [0.0] * P["nStart"]
            for idx, log_s, f0_val in completed:
                logs_each[idx] = log_s
                f0_vals[idx] = f0_val
                n_div = sum(1 for value in log_s.values() if value.get("__failed__"))
                tag = f" ({n_div} div)" if n_div else ""
                print(f"  MC #{idx+1:2d} ready{tag}", flush=True)

            elapsed_func = time.perf_counter() - t0_func
            print(f"  [{obj_name}] all {P['nStart']} MC ready in "
                  f"{elapsed_func:.1f}s")

            # -- merge & export -----------------------------------
            f0_val = f0_vals[-1] if f0_vals else np.nan

            # Rebuild prm for summary (need obj metadata)
            obj_args = param_bank.get(obj_name, [P["d_override"]])
            if obj_args and isinstance(obj_args[0], (int, float)):
                obj_args = [P["d_override"]] + list(obj_args[1:])
            fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
                obj_factory(obj_name, P["Nagent"], *obj_args)
            policy = M_alpha_policy.get(
                obj_name, {"M_factor": 1.0, "alpha": 0.1,
                           "decay": False, "maxIt": P["maxIt"]})
            prm = dict(P)
            prm["maxIt"] = resolve_iteration_budget(policy, P)
            if "tolType" in policy:
                prm["tolType"] = policy["tolType"]
            prm.update({
                "f": fun_list, "fname": fname, "fparam": fparam,
                "dim": d,
                "M": policy["M_factor"] * float(L_vec.max()),
                "alpha": policy["alpha"] / float(L_vec.max()),
                "decay_alpha": policy["decay"], "objName": obj_name,
                "x_opt": x_opt_list[0] if x_opt_list else None,
                "f_opt": float(np.mean(f_opt_list)),
                "esom_penalty": 1.0,
            })

            log_merged = merge_logs(logs_each, P["useWorst"],
                                     f0_val, prm["f_opt"], use_median=True)
            all_logs[obj_name] = log_merged
            all_individual[obj_name] = logs_each

            write_json_gz(
                os.path.join(results_dir, "data_log", f"raw_{obj_name}.json.gz"),
                {
                    "schema_version": 1,
                    "mode": "regular",
                    "objective": obj_name,
                    "configuration": {
                        key: value
                        for key, value in P.items()
                        if key not in {"f", "W"}
                    },
                    "initial_objectives": f0_vals,
                    "runs": [
                        {
                            "mc_index": index,
                            "seed": 100 + index,
                            "algorithms": run,
                        }
                        for index, run in enumerate(logs_each)
                    ],
                },
            )

            if P["showPlots"]:
                for x_key, y_key in [("steps",    "combo"),
                                      ("timeCost", "combo"),
                                      ("steps",    "relF"),
                                      ("timeCost", "relF"),
                                      ("commCost", "relF"),
                                      ("commCost", "combo")]:
                    fig_plot_custom(log_merged, alg_bank, x_key, y_key,
                                    "semilogy", results_dir, obj_name)

            write_txt_summary(results_dir, log_merged, alg_bank,
                              obj_name, "regular", prm, f0_val,
                              prm["f_opt"])

            excel_path = os.path.join(results_dir, "data_log",
                                      "exp_full_record.xlsx")
            write_log_to_excel(logs_each, alg_bank, obj_name, prm,
                               f0_val, prm["f_opt"], excel_path)

    finally:
        for k, v in _orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    # -- multi-objective paper figures -----------------------------------------
    if P["showPlots"] and len(all_logs) > 1:
        for x_key, y_key in [
            ("steps",    "combo"),
            ("timeCost", "combo"),
            ("steps",    "relF"),
            ("timeCost", "relF"),
            ("commCost", "relF"),
            ("commCost", "combo"),
        ]:
            fig_plot_multiobj_custom(all_logs, alg_bank, x_key, y_key,
                                     "semilogy", results_dir, n_cols=3)

        fig_plot_multiobj_repr(all_logs, alg_bank, "steps", "relF",
                               "semilogy", results_dir)

        fig_perf_profiles_tol_panel(all_logs, alg_bank, results_dir,
                                    tol_levels=[1e-3, 1e-6, 1e-9],
                                    all_individual=all_individual)

        fig_perf_profiles_comm_panel(all_logs, alg_bank, results_dir,
                                     tol_levels=[1e-3, 1e-6, 1e-9],
                                     all_individual=all_individual)

    print("\n[run_regular] All done.")
