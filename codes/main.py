"""
main.py - Unified entry point for DisGrem experiment suite.

USAGE
-----
    python main.py [mode] [arg]

    mode : regular | robust | comm | ada | scale | tune | all (default)
    arg  : mode-specific sub-argument (see below)

MODES
-----
    regular  Main 9-function benchmark (d=30, 20 MC, MainComp algorithms).
             arg = all | ridge | quadbad | convexset | nonconvexset  (default: all)
    robust   Robustness: Part1 starting-point (100MC) + Part2 param sweep.
             arg = all | start | param  (default: all)
    comm     Comm-cost study: Part 1 Ce benefit + Part 2 Klazy/compression ablation.
             arg = all | ce | ablation  (default: all)
    ada      Adaptive mechanism: M trajectory + Ada-vs-fixed-M + init-M.
             (no sub-arguments)
    scale    Dimension scalability study (3 functions x 3 dims x 4 algs x 5 MC).
             (no sub-arguments)
    tune     Seed-separated nested tuning and held-out evaluation.
             (no sub-arguments)
    all      Run all six experiments sequentially (default).

EXAMPLES
--------
    python main.py                  # run all 6 experiments
    python main.py regular          # regular benchmark only (all 9 functions)
    python main.py regular ridge    # regular benchmark, ridge only
    python main.py robust start     # robustness Part 1 only
    python main.py comm             # comm study (Ce benefit + ablation)
    python main.py comm ce          # Part 1 only (Ce benefit)
    python main.py comm ablation    # Part 2 only (Klazy + compression)
    python main.py ada              # adaptive mechanism study
    python main.py scale            # dimension scalability study
    python main.py tune             # fair tuning + held-out evaluation

OUTPUT
------
    results/main/       regular benchmark
    results/robust/     robustness study
    results/comm/       communication cost study
    results/ada/        adaptive mechanism study
    results/scale/      dimension scalability study
    results/tuning/     nested tuning and held-out evaluation
"""

import sys
import os
import shutil
import numpy as np

from experiments.protocol import (
    manifest_path,
    paper_grade_protocol_errors,
    required_outputs_present,
    source_tree_sha256,
    tracked_run,
)

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

np.random.seed(42)

# Result directories for each mode
_RESULT_DIRS = {
    "regular": os.path.join(_root, "results", "main"),
    "robust":  os.path.join(_root, "results", "robust"),
    "comm":    os.path.join(_root, "results", "comm"),
    "ada":     os.path.join(_root, "results", "ada"),
    "scale":   os.path.join(_root, "results", "scale"),
    "tuning":  os.path.join(_root, "results", "tuning"),
}


def _banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _clear_results(mode: str) -> None:
    """Remove one mode plus stale completion and frozen-artifact records."""
    d = _RESULT_DIRS.get(mode)
    if d and os.path.exists(d):
        shutil.rmtree(d)
        print(f"[clean] Removed old results: {d}")
    if d:
        os.makedirs(d, exist_ok=True)
    mode_manifest = manifest_path(mode)
    if mode_manifest.is_file():
        mode_manifest.unlink()
    results_root = os.path.join(_root, "results")
    for name in ("ARTIFACT_MANIFEST.json", "ARTIFACT_REPORT.md", "SHA256SUMS"):
        path = os.path.join(results_root, name)
        if os.path.isfile(path):
            os.remove(path)


def _clear_all() -> None:
    """Remove all generated results, manifests, and task checkpoints."""
    for mode in _RESULT_DIRS:
        _clear_results(mode)
    cache_root = os.path.join(_root, "_run_cache")
    if os.path.isdir(cache_root):
        shutil.rmtree(cache_root)
        print(f"[clean] Removed experiment caches: {cache_root}")


def _existing_manifest(mode: str) -> dict:
    path = manifest_path(mode)
    if not path.is_file():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _prepare_results(mode: str) -> bool:
    """Preserve only a source-matched interrupted run; otherwise start fresh."""
    manifest = _existing_manifest(mode)
    resumable = (
        manifest.get("status") in {"running", "failed", "interrupted"}
        and manifest.get("source_tree_sha256_at_start") == source_tree_sha256()
        and os.path.isdir(_RESULT_DIRS[mode])
    )
    if resumable:
        os.environ["DISGREM_USE_CACHE"] = "1"
        print(f"[resume] Continuing source-matched {mode} results.")
    else:
        os.environ["DISGREM_USE_CACHE"] = "0"
        _clear_results(mode)
    return resumable


def _completed_current_run(mode: str) -> bool:
    manifest = _existing_manifest(mode)
    return (
        manifest.get("status") == "complete"
        and manifest.get("full_mode_scope") is True
        and manifest.get("source_tree_sha256_at_finish") == source_tree_sha256()
        and not paper_grade_protocol_errors(mode, manifest.get("protocol", {}))
        and required_outputs_present(mode)
    )


def _run_tracked(mode: str, runner) -> None:
    """Run one experiment mode and persist its auditable lifecycle manifest."""
    with tracked_run(mode, list(sys.argv)):
        runner()


def _run_all_stage(mode: str, runner) -> None:
    if _completed_current_run(mode):
        print(f"[resume] Skipping completed source-matched {mode} mode.")
        return
    _prepare_results(mode)
    _run_tracked(mode, runner)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode in {"-h", "--help", "help"}:
        print(__doc__)

    elif mode == "regular":
        _prepare_results("regular")
        func_group = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.benchmarks.run_regular import run_regular
        _run_tracked("regular", lambda: run_regular(func_group))

    elif mode == "robust":
        _prepare_results("robust")
        part = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.ablation.run_robust import run_robust
        _run_tracked("robust", lambda: run_robust(part))

    elif mode == "comm":
        _prepare_results("comm")
        part = sys.argv[2] if len(sys.argv) > 2 else "all"
        from experiments.ablation.run_comm import run_comm
        _run_tracked("comm", lambda: run_comm(part))

    elif mode == "ada":
        _prepare_results("ada")
        from experiments.ablation.run_ada import run_ada
        _run_tracked("ada", run_ada)

    elif mode == "scale":
        _prepare_results("scale")
        from experiments.benchmarks.run_scalability import run_scalability
        _run_tracked("scale", run_scalability)

    elif mode == "tune":
        _prepare_results("tuning")
        from experiments.benchmarks.run_fair_tuning import run_fair_tuning
        _run_tracked("tuning", run_fair_tuning)

    elif mode == "all":
        _run_all()

    elif mode == "clean":
        _clear_all()
        cache_root = os.path.join(_root, "_run_cache")
        if os.path.exists(cache_root):
            shutil.rmtree(cache_root)
            print(f"[clean] Removed resume cache: {cache_root}")
        print("[clean] All result directories cleared.")

    else:
        print(
            f"Unknown mode '{mode}'.\n"
            "Available: regular | robust | comm | ada | scale | tune | all | clean"
        )
        sys.exit(1)


def _run_all() -> None:
    """Run all six experiments sequentially with fresh result directories.

    Order: regular, communication, adaptive, robustness, tuning, scalability.
    """
    _banner("Step 1/6 - Regular benchmark (9 functions, d=30, 20 MC)")
    from experiments.benchmarks.run_regular import run_regular
    _run_all_stage("regular", lambda: run_regular("all"))

    _banner("Step 2/6 - Communication cost study (Ce benefit + ablation)")
    from experiments.ablation.run_comm import run_comm
    _run_all_stage("comm", lambda: run_comm("all"))

    _banner("Step 3/6 - Adaptive mechanism study (M trajectory + comparisons)")
    from experiments.ablation.run_ada import run_ada
    _run_all_stage("ada", run_ada)

    _banner("Step 4/6 - Robustness study (100 MC + param sweep)")
    from experiments.ablation.run_robust import run_robust
    _run_all_stage("robust", lambda: run_robust("all"))

    _banner("Step 5/6 - Seed-separated fair tuning and held-out evaluation")
    from experiments.benchmarks.run_fair_tuning import run_fair_tuning
    _run_all_stage("tuning", run_fair_tuning)

    _banner("Step 6/6 - Dimension scalability study")
    from experiments.benchmarks.run_scalability import run_scalability
    _run_all_stage("scale", run_scalability)

    _banner("All experiments complete.")
    print(
        "\nOutput directories:\n"
        "  results/main/       regular benchmark\n"
        "  results/comm/       communication cost study\n"
        "  results/ada/        adaptive mechanism study\n"
        "  results/robust/     robustness study\n"
        "  results/scale/      dimension scalability study\n"
        "  results/tuning/     nested tuning and held-out evaluation\n"
    )


if __name__ == "__main__":
    main()
