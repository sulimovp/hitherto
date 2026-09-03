"""Agreement metrics for two Block-6 extractor runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from casefile.predict.extract_run import load_jsonl

_CATEGORICAL_FIELDS = (
    "intent",
    "proposed_solution_present",
    "patch_offered",
    "maintainer_stance",
    "scope",
    "names_alternative",
    "names_alternative_text",
)
_ORDINAL_FIELDS = ("specificity", "blocking_severity", "affect")


def cohen_kappa(left: list[object], right: list[object]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[value] * right_counts[value]
        for value in left_counts.keys() | right_counts.keys()
    ) / (len(left) ** 2)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def ordinal_alpha(left: list[int], right: list[int]) -> float | None:
    """Krippendorff alpha with ordinal distances from pooled category frequencies."""
    if len(left) != len(right) or not left:
        return None
    counts = Counter([*left, *right])
    categories = sorted(counts)

    def distance(a: int, b: int) -> float:
        lo, hi = sorted((a, b))
        mass = sum(counts[value] for value in categories if lo <= value <= hi)
        return (mass - (counts[lo] + counts[hi]) / 2) ** 2

    observed = sum(distance(a, b) for a, b in zip(left, right, strict=True)) / len(left)
    total = len(left) + len(right)
    expected_numerator = sum(
        counts[a] * counts[b] * distance(a, b) for a in categories for b in categories
    )
    expected = expected_numerator / (total * (total - 1))
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1 - observed / expected


def _rows_by_id(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            item_id = str(row.get("item_id") or "")
            if item_id:
                rows[item_id] = row
    return rows


def _valid_fields(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("parse_error") or row.get("span_errors"):
        return None
    fields = row.get("fields")
    return fields if isinstance(fields, dict) else None


def agreement_report(
    primary_paths: Iterable[Path],
    secondary_paths: Iterable[Path],
) -> dict[str, Any]:
    primary = _rows_by_id(primary_paths)
    secondary = _rows_by_id(secondary_paths)
    item_ids = sorted(primary.keys() & secondary.keys())
    valid_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item_id in item_ids:
        left = _valid_fields(primary[item_id])
        right = _valid_fields(secondary[item_id])
        if left is not None and right is not None:
            valid_pairs.append((left, right))

    fields: dict[str, dict[str, Any]] = {}
    for field in _CATEGORICAL_FIELDS:
        pairs = [
            (left[field], right[field])
            for left, right in valid_pairs
            if left.get(field) is not None and right.get(field) is not None
        ]
        fields[field] = {
            "metric": "cohen_kappa",
            "n": len(pairs),
            "agreement": (
                sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            ),
            "value": cohen_kappa(
                [a for a, _ in pairs],
                [b for _, b in pairs],
            ),
        }
    for field in _ORDINAL_FIELDS:
        pairs = [
            (int(left[field]), int(right[field]))
            for left, right in valid_pairs
            if left.get(field) is not None and right.get(field) is not None
        ]
        fields[field] = {
            "metric": "krippendorff_alpha_ordinal",
            "n": len(pairs),
            "agreement": (
                sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            ),
            "value": ordinal_alpha(
                [a for a, _ in pairs],
                [b for _, b in pairs],
            ),
        }

    primary_versions = sorted(
        {str(primary[item_id].get("extractor_version")) for item_id in item_ids}
    )
    secondary_versions = sorted(
        {str(secondary[item_id].get("extractor_version")) for item_id in item_ids}
    )
    return {
        "n_overlap": len(item_ids),
        "n_valid_rows": len(valid_pairs),
        "primary_versions": primary_versions,
        "secondary_versions": secondary_versions,
        "fields": fields,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two Block-6 extractor runs.")
    parser.add_argument("--primary", type=Path, nargs="+", required=True)
    parser.add_argument("--secondary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = agreement_report(args.primary, args.secondary)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"agreement: overlap={report['n_overlap']} valid={report['n_valid_rows']} "
        f"wrote={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
