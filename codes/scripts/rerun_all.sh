#!/usr/bin/env bash
# Run from any directory: bash codes/scripts/rerun_all.sh [full|quick]
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-full}"
case "$mode" in
  full)
    python main.py all
    python scripts/replot.py
    python scripts/freeze_artifact.py freeze
    python scripts/freeze_artifact.py verify
    ;;
  quick)
    python main.py regular ridge
    python main.py comm ce
    ;;
  *)
    echo "Usage: bash codes/scripts/rerun_all.sh [full|quick]" >&2
    exit 2
    ;;
esac
