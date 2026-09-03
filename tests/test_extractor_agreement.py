import json
from pathlib import Path

import pytest

from casefile.predict.agreement import agreement_report, cohen_kappa, ordinal_alpha


def test_cohen_kappa_perfect_and_chance_adjusted():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_ordinal_alpha():
    assert ordinal_alpha([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert ordinal_alpha([], []) is None
    assert ordinal_alpha([0, 0, 3, 3], [3, 3, 0, 0]) < 0


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_agreement_report_excludes_invalid_rows(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    good = {
        "parse_error": None,
        "span_errors": [],
        "extractor_version": "v1",
        "fields": {
            "intent": "bug",
            "specificity": 2,
            "proposed_solution_present": True,
            "patch_offered": False,
            "blocking_severity": 1,
            "affect": 0,
            "maintainer_stance": "acknowledged",
            "scope": "contained",
            "names_alternative": False,
            "names_alternative_text": None,
        },
    }
    _write(
        primary,
        [
            {"item_id": "a", **good},
            {"item_id": "b", **{**good, "span_errors": ["bad"]}},
        ],
    )
    _write(
        secondary,
        [
            {"item_id": "a", **{**good, "extractor_version": "v2"}},
            {"item_id": "b", **{**good, "extractor_version": "v2"}},
        ],
    )
    report = agreement_report([primary], [secondary])
    assert report["n_overlap"] == 2
    assert report["n_valid_rows"] == 1
    assert report["fields"]["intent"]["value"] == pytest.approx(1.0)
    assert report["fields"]["specificity"]["value"] == pytest.approx(1.0)
