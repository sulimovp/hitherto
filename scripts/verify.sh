#!/usr/bin/env bash
# Mandatory verification gate for casefile changes.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== casefile verify: pytest ==="
pytest -v "$@"

echo "=== casefile verify: report contracts (if reports present) ==="
if compgen -G "reports/*.md" > /dev/null; then
  python scripts/validate_reports.py
else
  echo "SKIP validate_reports.py (no reports/*.md — run run_sample_assessments.sh for live samples)"
fi

echo "=== casefile verify: OK ==="
