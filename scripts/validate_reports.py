#!/usr/bin/env python3
"""Validate sample report markdown against content/format contracts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.report_contract import (  # noqa: E402
    GOLDEN_PYTORCH_MASKED,
    assert_markdown_report_contract,
)

REPORTS = {
    "torch-masked.md": {
        "expect_summary": True,
        "needles": [*(f"/issues/{n}" for n in GOLDEN_PYTORCH_MASKED), "/332", "dev-discuss"],
    },
    "torch-nested.md": {
        "expect_summary": True,
        "needles": ["/issues/112398", "torch/nested", "NestedTensor"],
    },
    "numpy-ma.md": {
        "expect_summary": True,
        "needles": ["/issues/22338", "numpy.ma", "numpy/numpy"],
    },
    "sklearn-metadata-routing.md": {
        "expect_summary": True,
        "needles": ["/issues/22893", "scikit-learn", "metadata"],
    },
}


def main() -> int:
    reports_dir = ROOT / "reports"
    failed = False
    for name, spec in REPORTS.items():
        path = reports_dir / name
        if not path.is_file():
            print(f"SKIP {name}: file missing (run scripts/run_sample_assessments.sh)")
            failed = True
            continue
        md = path.read_text(encoding="utf-8")
        try:
            assert_markdown_report_contract(md, expect_summary=spec["expect_summary"])
            for needle in spec["needles"]:
                if needle.lower() not in md.lower():
                    raise AssertionError(f"missing {needle!r}")
            print(f"OK {name}")
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
