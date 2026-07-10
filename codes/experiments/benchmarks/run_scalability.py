"""
run_scalability.py -Dimension-scalability study.

Runs 3 representative functions x 3 dimensions x 4 algorithms x 5 MC.
Produces a 3x3 grid figure (rows = functions, cols = dimensions) of
relF vs iteration, suitable for a half-page scalability subsection.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.alg.alg_bank import get_alg_bank, get_alg_style
from problems.obj_factory import obj_factory
from problems.init_policy import init_policy
from utils.helper.graph import generate_random_graph
from utils.export.log_export import merge_logs
from utils.helper.run_utils import detect_diverged
from experiments.protocol import (
    configured_iteration_override,
    execute_cached_tasks,
    resolve_iteration_budget,
    write_json_gz,
)

_N_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))

_FUNCTIONS = ["ridge", "logsumexp", "rosenbrock"]
_DIMS = [30, 100, 200]
_N_MC = int(os.environ.get("LOG_SCHEDULE_NSTART", "5"))
_ALG_MODE = "ScaleStudy"

_MAXITS = {
    "ridge":      {30: 200,  100: 400,  200: 600},
    "logsumexp":  {30: 400,  100: 800,  200: 1200},
    "rosenbrock": {30: 300,  100: 600,  200: 900},
}


def _worker(args):
    obj_name, d_val, mc_idx, P_dict = args

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"

    alg_bank = get_alg_bank(_ALG_MODE)
    param_bank, M_alpha_policy, x0_generator = init_policy("regular")

    obj_args = param_bank.get(obj_name, [d_val])
    if obj_args and isinstance(obj_args[0], (int, float)):
        obj_args = [d_val] + list(obj_args[1:])

    fun_list, d, L_vec, x_opt_list, f_opt_list, _, fname, fparam = \
        obj_factory(obj_name, P_dict["Nagent"], *obj_args)

    policy = M_alpha_policy.get(obj_name, {"M_factor": 1.0, "alpha": 0.1,
                                            "decay": False, "maxIt": 500})
    dimension_policy = {
        "maxIt": _MAXITS.get(obj_name, {}).get(d_val, policy.get("maxIt", 500))
    }
    maxIt_obj = resolve_iteration_budget(dimension_policy, P_dict)
    M_val = policy["M_factor"] * float(L_vec.max())
    alp_val = policy["alpha"] / float(L_vec.max())

    prm = dict(P_dict)
    prm["maxIt"] = maxIt_obj
    prm["verbose"] = False
    prm["d_override"] = d_val
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
    rng_s = np.random.RandomState(200 + mc_idx)
    np.random.set_state(rng_s.get_state())
    x0 = x0_gen(d, False)
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


def run_scalability() -> dict:
    results_dir = os.path.join(_root, "results", "scale")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data_log"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "paper"), exist_ok=True)

    P = {
        "Nagent": 10, "p_edge": 0.5, "maxIt": int(os.environ.get("LOG_SCHEDULE_MAXIT", "500")),
        "iteration_budget_override": configured_iteration_override(),
        "tol": 1e-12, "tolType": "combo", "verbose": False,
        "showPlots": False, "far": False, "useWorst": False,
        "nStart": _N_MC, "d_override": 30, "info": 2, "NC": 3, "NC_schedule": "log", "log_p": 3.0, "log_c_mix": 2.0, "NC_max": 10,
        "countComm": True,
    }

    param_bank, M_alpha_policy, _ = init_policy("regular")

    _orig_env = {}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _orig_env[k] = os.environ.get(k)
        os.environ[k] = "1"

    all_results = {}

    try:
        for obj_name in _FUNCTIONS:
            for d_val in _DIMS:
                key = f"{obj_name}_d{d_val}"
                print(f"\n{'='*50}")
                print(f"  {obj_name}  d={d_val}  ({_N_MC} MC, "
                      f"{_N_WORKERS} workers)")
                print(f"{'='*50}")

                t0 = time.perf_counter()
                P_run = dict(P)
                P_run["d_override"] = d_val
                tasks = [(obj_name, d_val, sample, P_run) for sample in range(_N_MC)]
                completed = execute_cached_tasks(
                    tasks,
                    _worker,
                    cache_root=os.path.join(_root, "_run_cache"),
                    namespace=f"scale_{key}",
                    max_workers=_N_WORKERS,
                    progress_every=1,
                )

                logs_each = [None] * _N_MC
                f0_vals = [0.0] * _N_MC
                for idx, log_s, f0_val in completed:
                    logs_each[idx] = log_s
                    f0_vals[idx] = f0_val
                    print(f"  MC #{idx+1} ready", flush=True)

                elapsed = time.perf_counter() - t0
                print(f"  [{key}] ready in {elapsed:.1f}s")

                obj_args = param_bank.get(obj_name, [d_val])
                if obj_args and isinstance(obj_args[0], (int, float)):
                    obj_args = [d_val] + list(obj_args[1:])
                fun_list, d, L_vec, x_opt_list, f_opt_list, _, _, _ = \
                    obj_factory(obj_name, P["Nagent"], *obj_args)

                f0_val = f0_vals[-1] if f0_vals else np.nan
                f_star = float(np.mean(f_opt_list))
                log_merged = merge_logs(logs_each, False, f0_val, f_star, use_median=True)
                all_results[key] = log_merged

                raw_payload = {
                    "schema_version": 1,
                    "mode": "scale",
                    "objective": obj_name,
                    "dimension": d_val,
                    "runs": [
                        {
                            "mc_index": index,
                            "seed": 200 + index,
                            "initial_objective": f0_vals[index],
                            "algorithms": run,
                        }
                        for index, run in enumerate(logs_each)
                    ],
                }
                write_json_gz(
                    os.path.join(results_dir, "data_log", f"raw_{key}.json.gz"),
                    raw_payload,
                )
    finally:
        for k, v in _orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    _plot_scalability_grid(all_results, results_dir)
    print(f"\nScalability study complete. Results in {results_dir}/")
    return all_results


def _plot_scalability_grid(all_results: dict, results_dir: str) -> None:
    """3x3 grid: rows = functions, cols = dimensions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plot.")
        return

    alg_bank = get_alg_bank(_ALG_MODE)
    alg_names = [a[0] for a in alg_bank]

    _COL2_W = 7.16
    fig, axes = plt.subplots(len(_FUNCTIONS), len(_DIMS),
                             figsize=(_COL2_W,
                                      1.8 * len(_FUNCTIONS) + 0.5),
                             squeeze=False)

    for row, obj_name in enumerate(_FUNCTIONS):
        for col, d_val in enumerate(_DIMS):
            ax = axes[row][col]
            key = f"{obj_name}_d{d_val}"
            log_merged = all_results.get(key, {})

            for alg_name in alg_names:
                if alg_name not in log_merged:
                    continue
                rec = log_merged[alg_name]
                if rec.get("__failed__") or rec.get("fail"):
                    continue

                y = rec.get("relF")
                if y is None:
                    continue
                y = np.array(y, dtype=float)
                x = np.arange(1, len(y) + 1)

                label, color, ls = get_alg_style(alg_name)
                ax.semilogy(x, np.maximum(y, 1e-16),
                            color=color, linestyle=ls,
                            linewidth=1.4, label=label)

            ax.set_ylim(1e-14, 1e1)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"$d = {d_val}$", fontsize=9)
            if col == 0:
                nice = {"ridge": "Ridge", "logsumexp": "LogSumExp",
                        "rosenbrock": "Rosenbrock"}
                ax.set_ylabel(nice.get(obj_name, obj_name), fontsize=8)
            if row == len(_FUNCTIONS) - 1:
                ax.set_xlabel("Iteration", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(alg_names), fontsize=7,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = os.path.join(results_dir, "paper")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"scalability_grid.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved scalability_grid.pdf/png → {out_dir}")


if __name__ == "__main__":
    run_scalability()
