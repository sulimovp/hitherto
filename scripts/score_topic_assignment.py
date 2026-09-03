#!/usr/bin/env python3
"""Score item→topic assignment against a hand-labelled YAML set.

Usage:
  python scripts/score_topic_assignment.py eval/topic_assignment/pytorch.yaml

Prints precision, recall, n, and a Wilson interval on precision — paste into the
profile's assignment_precision block. Offline: pass --fixture with a pre-built
predictions YAML when GitHub is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from casefile.predict.assignment_score import score_assignment


def load_labelled(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected mapping in {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labelled", type=Path, help="eval/topic_assignment/{profile}.yaml")
    parser.add_argument(
        "--path",
        default=None,
        help="topic path the assigner claims (default: labelled YAML `path`, else torch/masked)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Fit a train-split logistic and print weights (does not change the decision rule)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="optional YAML with pre-filled title/body per item (offline)",
    )
    args = parser.parse_args()

    data = load_labelled(args.labelled)
    items = list(data.get("items") or [])
    if args.fixture is not None:
        fixture = load_labelled(args.fixture)
        by_key = {
            (str(i["repo"]), int(i["number"])): i for i in (fixture.get("items") or [])
        }
        for item in items:
            extra = by_key.get((str(item["repo"]), int(item["number"])))
            if extra:
                item.setdefault("title", extra.get("title", ""))
                item.setdefault("body", extra.get("body", ""))

    if not items:
        print(
            f"{args.labelled}: no labelled items yet — fill eval/topic_assignment "
            "and re-run. Scaffold only.",
            file=sys.stderr,
        )
        return 1

    path = args.path or data.get("path") or "torch/masked"
    synonyms = tuple(data.get("synonyms") or ())
    result = score_assignment(items, path=path, synonyms=synonyms)
    lo, hi = result["wilson_precision_95"]
    print(f"profile: {data.get('profile')}")
    print(f"path: {path}")
    print(f"n: {result['n']}")
    print(f"precision: {result['precision']:.4f}")
    print(f"recall: {result['recall']:.4f}")
    print(f"wilson_precision_95: [{lo:.4f}, {hi:.4f}]")
    print(f"tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']}")
    if args.tune:
        from casefile.predict.assignment_score import fit_assignment_logistic

        train = [i for i in items if int(i["number"]) % 3 != 0]
        hold = [i for i in items if int(i["number"]) % 3 == 0]
        weights, bias = fit_assignment_logistic(train, path=args.path)
        print()
        print("train-split logistic (issue number % 3 != 0):")
        print("  weights", {k: round(v, 3) for k, v in weights.items()})
        print("  bias", round(bias, 3))
        print(f"  holdout n={len(hold)} (decision rule is feature combination, not this logistic)")
    print()
    print("Paste into profiles/{id}.yaml:")
    print("assignment_precision:")
    print(f"  measured_at: {data.get('labelled_at') or 'YYYY-MM-DD'}")
    print(f"  n: {result['n']}")
    print(f"  precision: {result['precision']:.2f}")
    print(f"  recall: {result['recall']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
