#!/usr/bin/env bash
# Regenerate sample reports (requires casefile/.env with CASEFILE_GITHUB_TOKEN).
# Scenarios align with eval/sample_cases.yaml (primary four presets).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports

run() {
  echo "=== $1 ==="
  shift
  casefile assess "$@"
  sleep 25
}

run masked \
  -q "Is reviving torch.masked / MaskedTensor worth an upstream contribution?" \
  -r pytorch/pytorch -p torch/masked -e pytorch --tier 2 -o reports/torch-masked.md

run nested \
  -q "Is contributing to torch.nested worth it for variable-length sequences?" \
  -r pytorch/pytorch -p torch/nested -e pytorch --tier 2 -o reports/torch-nested.md

run numpy \
  -q "Is improving numpy.ma __array_function__ support worth an upstream contribution?" \
  -r numpy/numpy -e numpy --tier 2 -o reports/numpy-ma.md

run sklearn \
  -q "Is contributing to sklearn metadata routing (SLEP006) still worth it?" \
  -r scikit-learn/scikit-learn -e sklearn --tier 2 -o reports/sklearn-metadata-routing.md

echo "Done. Run: pytest -v && python scripts/validate_reports.py"
