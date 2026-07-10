"""Freeze or verify a complete DisGRem numerical-results artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

from experiments.protocol import (
    MODE_REQUIRED_GLOBS,
    PRIMARY_ALGORITHMS,
    PROTOCOL_VERSION,
    paper_grade_protocol_errors,
    source_tree_sha256,
    write_json,
)


_MODES = ("regular", "robust", "comm", "ada", "tuning", "scale")
_GENERATED = {"ARTIFACT_MANIFEST.json", "SHA256SUMS", "ARTIFACT_REPORT.md"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _load_gzip_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid gzip JSON {path.name}: {exc}"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None, f"unsupported raw schema in {path.name}"
    return payload, None


def _raw_count_error(relative: str, payload: dict[str, Any]) -> str | None:
    runs = payload.get("runs")
    expected_runs = None
    if relative.startswith("main/data_log/raw_"):
        expected_runs = 20
        if isinstance(runs, list) and any(
            set(run.get("algorithms", {})) != set(PRIMARY_ALGORITHMS)
            for run in runs
        ):
            return f"regular algorithm coverage mismatch in {relative}"
    elif relative.startswith("scale/data_log/raw_"):
        expected_runs = 5
        expected_algorithms = {"DisGrem", "AdaDisGrem", "EXTRA", "SONATA"}
        if isinstance(runs, list) and any(
            set(run.get("algorithms", {})) != expected_algorithms
            for run in runs
        ):
            return f"scalability algorithm coverage mismatch in {relative}"
    elif relative.endswith("raw_starting_point_robustness.json.gz"):
        expected_runs = 1800
        if isinstance(runs, list) and any(
            set(run.get("algorithms", {})) != set(PRIMARY_ALGORITHMS)
            for run in runs
        ):
            return f"robustness algorithm coverage mismatch in {relative}"
    elif relative.endswith("raw_parameter_sensitivity.json.gz"):
        expected_runs = 1600
    elif relative.endswith("raw_ce_benefit.json.gz"):
        expected_runs = 80
    elif relative.endswith("raw_klazy_sweep.json.gz"):
        expected_runs = 120
    elif relative.endswith("raw_compression_sweep.json.gz"):
        expected_runs = 180
    elif relative.endswith("raw_m_trajectory.json.gz"):
        expected_runs = 60
    elif relative.endswith("raw_ada_vs_fixed_m.json.gz"):
        expected_runs = 100
    elif relative.endswith("raw_initial_m_robustness.json.gz"):
        expected_runs = 100

    if expected_runs is not None:
        actual = len(runs) if isinstance(runs, list) else -1
        if actual != expected_runs:
            return f"raw run count mismatch for {relative}: expected {expected_runs}, found {actual}"

    if "/raw_tuning_" in f"/{relative}" or "/raw_evaluation_" in f"/{relative}":
        results = payload.get("results")
        tuning = "raw_tuning_" in relative
        expected_tasks = 110 if tuning else 20
        expected_replicates = 5 if tuning else 20
        actual_tasks = len(results) if isinstance(results, list) else -1
        if actual_tasks != expected_tasks:
            return f"tuning task count mismatch for {relative}: expected {expected_tasks}, found {actual_tasks}"
        if any(len(result.get("runs", [])) != expected_replicates for result in results):
            return f"tuning replicate count mismatch in {relative}"
    return None


def audit_results(
    results_root: str | Path,
    *,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    root = Path(results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    manifests: dict[str, Any] = {}
    current_source_hash = source_tree_sha256()
    commits: set[str] = set()
    run_source_hashes: set[str] = set()

    for mode in _MODES:
        path = root / "run_manifests" / f"{mode}.json"
        if not path.is_file():
            errors.append(f"missing run manifest: {mode}")
            continue
        try:
            manifest = _read_json(path)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid run manifest {mode}: {exc}")
            continue
        manifests[mode] = manifest
        if manifest.get("status") != "complete":
            errors.append(f"run is not complete: {mode}={manifest.get('status')}")
        if not manifest.get("full_mode_scope"):
            errors.append(f"run covered only a partial mode: {mode}")
        if not manifest.get("source_unchanged_during_run"):
            errors.append(f"source changed during run: {mode}")
        start_hash = manifest.get("source_tree_sha256_at_start")
        finish_hash = manifest.get("source_tree_sha256_at_finish")
        if not start_hash or start_hash != finish_hash:
            errors.append(f"source hash mismatch within run: {mode}")
        elif start_hash != current_source_hash:
            errors.append(f"run source is stale relative to current source: {mode}")
        else:
            run_source_hashes.add(start_hash)
        git = manifest.get("git", {})
        commit = git.get("commit")
        if commit:
            commits.add(commit)
        else:
            errors.append(f"missing Git commit in run manifest: {mode}")
        if require_clean_git and git.get("dirty"):
            errors.append(f"run began from a dirty Git tree: {mode}")
        protocol = manifest.get("protocol", {})
        if manifest.get("protocol_version") != PROTOCOL_VERSION:
            errors.append(f"protocol version mismatch: {mode}")
        if protocol.get("graph", {}).get("model") != "connected random geometric graph":
            errors.append(f"unexpected graph model: {mode}")
        errors.extend(paper_grade_protocol_errors(mode, protocol))

    if len(commits) > 1:
        errors.append(f"multiple source commits across runs: {sorted(commits)}")
    if len(run_source_hashes) > 1:
        errors.append("multiple source-tree hashes across runs")

    required_matches: dict[str, list[str]] = {}
    for mode, requirements in MODE_REQUIRED_GLOBS.items():
        for pattern, expected_count in requirements:
            matches = sorted(path for path in root.glob(pattern) if path.is_file())
            required_matches[pattern] = [path.relative_to(root).as_posix() for path in matches]
            if len(matches) != expected_count:
                errors.append(
                    f"required output count mismatch for {pattern}: "
                    f"expected {expected_count}, found {len(matches)}"
                )

    raw_files = sorted(root.glob("**/*.json.gz"))
    for path in raw_files:
        payload, error = _load_gzip_json(path)
        if error:
            errors.append(error)
            continue
        relative = path.relative_to(root).as_posix()
        count_error = _raw_count_error(relative, payload or {})
        if count_error:
            errors.append(count_error)

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in _GENERATED:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "audited_at": _utc_now(),
        "ok": not errors,
        "errors": errors,
        "source_tree_sha256": current_source_hash,
        "source_commit": next(iter(commits)) if len(commits) == 1 else None,
        "modes": {mode: manifests.get(mode, {}).get("status", "missing") for mode in _MODES},
        "required_outputs": required_matches,
        "file_count": len(files),
        "total_bytes": sum(file["bytes"] for file in files),
        "files": files,
    }


def _write_report(root: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# DisGRem Numerical Artifact Report",
        "",
        f"- Status: `{'PASS' if audit['ok'] else 'FAIL'}`",
        f"- Protocol: `{audit['protocol_version']}`",
        f"- Source commit: `{audit['source_commit'] or 'unresolved'}`",
        f"- Source tree SHA-256: `{audit['source_tree_sha256']}`",
        f"- Files: {audit['file_count']}",
        f"- Bytes: {audit['total_bytes']}",
        "",
        "## Mode Gates",
        "",
    ]
    lines.extend(f"- `{mode}`: `{status}`" for mode, status in audit["modes"].items())
    lines.extend(["", "## Errors", ""])
    if audit["errors"]:
        lines.extend(f"- {error}" for error in audit["errors"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This gate verifies provenance, completeness, parseability, and byte-level integrity. "
            "It does not convert empirical observations into theorem verification or human proof review.",
            "",
        ]
    )
    (root / "ARTIFACT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def freeze(results_root: str | Path, *, require_clean_git: bool = True) -> dict[str, Any]:
    root = Path(results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ARTIFACT_MANIFEST.json", "SHA256SUMS"):
        stale = root / name
        if stale.is_file():
            stale.unlink()
    audit = audit_results(root, require_clean_git=require_clean_git)
    _write_report(root, audit)
    if not audit["ok"]:
        return audit
    write_json(root / "ARTIFACT_MANIFEST.json", audit)
    checksum_lines = [f"{item['sha256']}  {item['path']}" for item in audit["files"]]
    (root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def verify(results_root: str | Path) -> dict[str, Any]:
    root = Path(results_root).resolve()
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": ["missing ARTIFACT_MANIFEST.json"]}
    try:
        manifest = _read_json(manifest_path)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid artifact manifest: {exc}"]}
    errors = []
    if manifest.get("ok") is not True:
        errors.append("artifact manifest does not record a passing freeze")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("artifact protocol version mismatch")
    if manifest.get("source_tree_sha256") != source_tree_sha256():
        errors.append("artifact source tree is stale relative to current source")

    items = manifest.get("files", [])
    listed_paths = []
    for item in items:
        relative = Path(item["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"unsafe frozen path: {item['path']}")
            continue
        listed_paths.append(relative.as_posix())
        if not path.is_file():
            errors.append(f"missing frozen file: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        elif _sha256(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")

    current_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in _GENERATED
    )
    if sorted(listed_paths) != current_paths:
        errors.append("frozen file inventory differs from artifact manifest")

    checksum_path = root / "SHA256SUMS"
    expected_checksums = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in items
    )
    if not checksum_path.is_file():
        errors.append("missing SHA256SUMS")
    elif checksum_path.read_text(encoding="utf-8") != expected_checksums:
        errors.append("SHA256SUMS differs from artifact manifest")
    return {
        "ok": not errors,
        "errors": errors,
        "checked_files": len(items),
        "source_commit": manifest.get("source_commit"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--results", default=str(_CODES_ROOT / "results"))
    parser.add_argument(
        "--allow-dirty-run",
        action="store_true",
        help="Allow manifests that began from a dirty Git tree (not for release artifacts).",
    )
    arguments = parser.parse_args()
    if arguments.action == "freeze":
        result = freeze(
            arguments.results,
            require_clean_git=not arguments.allow_dirty_run,
        )
    else:
        result = verify(arguments.results)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
