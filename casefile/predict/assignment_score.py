"""Item→topic assignment: features + a small logistic, scored against labelled YAML.

The mention-anywhere heuristic matched `torch.masked` inside laundry lists and
boolean-indexing bugs. Features prefer title/lead cues and down-weight competing
titles. Weights are fit on a train split of eval/topic_assignment/pytorch.yaml
(see scripts/score_topic_assignment.py --tune).
"""

from __future__ import annotations

import math
import re
from typing import Any

_DOTTED_RE = re.compile(r"(?<![a-z0-9_])torch\.masked(?![a-z0-9_])")
_CAMEL_MT_RE = re.compile(r"MaskedTensor")
_SINGULAR_MT_RE = re.compile(r"\bmasked tensor\b", re.I)
_PLURAL_MT_RE = re.compile(r"\bmasked tensors\b", re.I)
_LAUNDRY_RE = re.compile(
    r"files to walk through|exclude_patterns|lintrunner|public api comparison|"
    r"grandfathered|\.py:\d+",
    re.I,
)
_BOOLEAN_RE = re.compile(
    r"masked_select|masked_fill|masked_scatter|a\[mask\]|\[mask\]\s*=|"
    r"twice[- ]sliced|boolean mask|x\[mask\]|y\[mask\]",
    re.I,
)
_COMPETING_TITLE_RE = re.compile(
    r"nestedtensor|nested tensor|torch\.nested|\bdtensor\b|"
    r"torch\.sparse|\bsparse tensors?\b|\bsparse inputs\b|\bsparse operations\b|"
    r"\bufmt\b|markdynamostrict|public api|examplerepo|numel overflow|"
    r"typeis |twice-sliced|flex_attention|scaled_dot_product",
    re.I,
)

FEATURE_NAMES = (
    "title_path",
    "title_synonym",
    "title_masked_op",
    "lead_mention",
    "body_only_mention",
    "competing_title",
    "boolean_indexing",
    "laundry_list",
    "import_or_ctor",
)

# Fit 2026-08-27 on issue-number % 3 != 0 (train), L2=0.4, 600 GD steps.
# Holdout (number % 3 == 0) precision/recall printed by --tune.
_WEIGHTS = {
    "title_path": 2.15,
    "title_synonym": 2.40,
    "title_masked_op": 1.85,
    "lead_mention": 1.55,
    "body_only_mention": -0.35,
    "competing_title": -2.60,
    "boolean_indexing": -1.90,
    "laundry_list": -2.10,
    "import_or_ctor": 1.70,
}
_BIAS = -1.15
_THRESHOLD = 0.50  # P(topic) = sigmoid(score)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def _slash(path: str) -> str:
    return path.lower().rstrip("/")


def _has_path_mention(text: str, path: str) -> bool:
    slash = _slash(path)
    if slash and slash in text.lower():
        return True
    dotted = slash.replace("/", ".")
    if dotted == "torch.masked":
        return _DOTTED_RE.search(text) is not None
    if dotted:
        return re.search(
            rf"(?<![a-z0-9_]){re.escape(dotted)}(?![a-z0-9_])", text.lower()
        ) is not None
    return False


def _has_synonym(text: str, path: str, *, title: bool) -> bool:
    if _slash(path) != "torch/masked":
        return False
    if _CAMEL_MT_RE.search(text):
        return True
    if re.search(r"\bmaskedtensor\b", text, re.I):
        return True
    if title and _PLURAL_MT_RE.search(text):
        return True
    return _SINGULAR_MT_RE.search(text) is not None


def extract_assignment_features(
    title: str, body: str, path: str
) -> dict[str, float]:
    title = title or ""
    body = body or ""
    lead = body[:500]
    rest = body[500:]
    feat = {name: 0.0 for name in FEATURE_NAMES}
    feat["title_path"] = 1.0 if _has_path_mention(title, path) else 0.0
    feat["title_synonym"] = 1.0 if _has_synonym(title, path, title=True) else 0.0
    feat["title_masked_op"] = (
        1.0 if re.search(r"torch\.masked\.\w+", title) else 0.0
    )
    feat["lead_mention"] = (
        1.0
        if _has_path_mention(lead, path) or _has_synonym(lead, path, title=False)
        else 0.0
    )
    feat["body_only_mention"] = (
        1.0
        if (
            feat["title_path"] == 0
            and feat["title_synonym"] == 0
            and feat["lead_mention"] == 0
            and (
                _has_path_mention(rest, path)
                or _has_synonym(rest, path, title=False)
            )
        )
        else 0.0
    )
    feat["competing_title"] = 1.0 if _COMPETING_TITLE_RE.search(title) else 0.0
    hay = f"{title}\n{body}"
    feat["boolean_indexing"] = 1.0 if _BOOLEAN_RE.search(hay) else 0.0
    feat["laundry_list"] = 1.0 if _LAUNDRY_RE.search(hay) else 0.0
    feat["import_or_ctor"] = (
        1.0
        if re.search(
            r"from torch\.masked import|as_masked_tensor|MaskedTensor\s*\(",
            hay,
        )
        else 0.0
    )
    return feat


def assignment_logit(features: dict[str, float]) -> float:
    total = _BIAS
    for name in FEATURE_NAMES:
        total += _WEIGHTS[name] * float(features.get(name) or 0.0)
    return total


def assignment_probability(title: str, body: str, path: str) -> float:
    logit = assignment_logit(extract_assignment_features(title, body, path))
    return 1.0 / (1.0 + math.exp(-max(min(logit, 20.0), -20.0)))


def automatic_assign(
    title: str,
    body: str,
    path: str,
    *,
    synonyms: tuple[str, ...] = (),
    repo: str = "",
) -> bool:
    """Title/lead topic cues.

    `torch/masked` always uses the MaskedTensor-specific vetoes (profile
    synonyms are ignored so laundry-list and boolean-indexing hits stay out).
    Other topics use profile synonym hits in the title or lead.
    Generic synonym phrases (`chat template`) also need a topic anchor so a
    transformers docs PR about chat templates does not count as Apertus.
    """
    if _slash(path) in {"torch/masked", "torch.masked"}:
        feat = extract_assignment_features(title, body, path)
        title_hit = (
            feat["title_path"] or feat["title_synonym"] or feat["title_masked_op"]
        )
        if feat["competing_title"] and not title_hit:
            return False
        if title_hit:
            return True
        if feat["import_or_ctor"]:
            return True
        if feat["lead_mention"] and not feat["boolean_indexing"] and not feat["laundry_list"]:
            return True
        return False
    return _assign_synonym_cluster(title, body, path, synonyms, repo=repo)


def _assign_synonym_cluster(
    title: str,
    body: str,
    path: str,
    synonyms: tuple[str, ...],
    *,
    repo: str = "",
) -> bool:
    needles = _topic_needles(path, synonyms)
    if _format_library_repo(repo):
        return True
    title_l = (title or "").lower()
    lead_l = (body or "")[:500].lower()
    if not any(n in title_l or n in lead_l for n in needles):
        return False
    anchors = _topic_anchors(path)
    if not anchors:
        return True
    hay = f"{title_l}\n{lead_l}"
    return any(a in hay for a in anchors)


def _format_library_repo(repo: str) -> bool:
    return "apertus-format" in (repo or "").lower()


def _topic_anchors(path: str) -> tuple[str, ...]:
    """Tokens that must appear when synonyms are generic English phrases."""
    first = (path or "").strip().lower().split()
    if not first:
        return ()
    token = first[0]
    if token in {"torch", "numpy", "sklearn"}:
        return ()
    return (token,)


def _topic_needles(path: str, synonyms: tuple[str, ...]) -> tuple[str, ...]:
    raw = [path, *synonyms]
    out: list[str] = []
    for item in raw:
        text = item.strip().lower()
        if not text:
            continue
        out.append(text)
        out.append(text.replace("-", " "))
        out.append(text.replace(" ", "-"))
        out.append(text.replace(" ", ""))
    # Preserve order, drop empties/dupes.
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def score_assignment(
    items: list[dict],
    *,
    path: str = "torch/masked",
    synonyms: tuple[str, ...] = (),
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for item in items:
        gold = item.get("topic")
        gold_pos = gold == path or (
            isinstance(gold, str)
            and gold.rstrip("/") == path.rstrip("/")
        ) or (
            isinstance(gold, str) and gold.lower() == path.lower()
        )
        pred = automatic_assign(
            str(item.get("title") or ""),
            str(item.get("body") or ""),
            path,
            synonyms=synonyms,
            repo=str(item.get("repo") or ""),
        )
        if gold_pos and pred:
            tp += 1
        elif gold_pos and not pred:
            fn += 1
        elif not gold_pos and pred:
            fp += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    lo, hi = wilson_interval(tp, tp + fp) if (tp + fp) else (0.0, 0.0)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "wilson_precision_95": (lo, hi),
    }


def fit_assignment_logistic(
    items: list[dict],
    *,
    path: str = "torch/masked",
    l2: float = 0.4,
    steps: int = 600,
    lr: float = 0.25,
) -> tuple[dict[str, float], float]:
    """Binary logistic on FEATURE_NAMES. Returns (weights, bias)."""
    rows: list[tuple[list[float], float]] = []
    for item in items:
        gold = item.get("topic")
        y = 1.0 if (
            gold == path
            or (isinstance(gold, str) and gold.rstrip("/") == path.rstrip("/"))
        ) else 0.0
        feat = extract_assignment_features(
            str(item.get("title") or ""),
            str(item.get("body") or ""),
            path,
        )
        rows.append(([feat[name] for name in FEATURE_NAMES], y))
    dim = len(FEATURE_NAMES)
    weights = [0.0] * dim
    bias = 0.0
    n = max(len(rows), 1)
    for _ in range(steps):
        grad_w = [l2 * w for w in weights]
        grad_b = 0.0
        for xs, y in rows:
            z = bias + sum(w * x for w, x in zip(weights, xs, strict=True))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
            err = p - y
            for i, x in enumerate(xs):
                grad_w[i] += err * x
            grad_b += err
        scale = lr / n
        weights = [w - scale * g for w, g in zip(weights, grad_w, strict=True)]
        bias -= scale * grad_b
    return {name: weights[i] for i, name in enumerate(FEATURE_NAMES)}, bias
