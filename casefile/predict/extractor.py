"""Schema-constrained LLM feature extraction for topic-hazard Block 6.

The LLM never issues the verdict. Fields are ordinal/categorical only.
extractor_version is a stored column; changing model, prompt, or schema forces
full re-extraction. Router aliases like `:fastest` are not a pin.

block6-v3 splits evidence: a ≤15-word quote for fields a contiguous span can
support, and a closed rubric cell for judgements about the thread. v2 required
a quote for every non-null field and the model correctly nulled the ordinals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

Intent = Literal[
    "bug",
    "feature_request",
    "support",
    "docs",
    "design_proposal",
    "integration_report",
    "other",
]
MaintainerStance = Literal[
    "none",
    "acknowledged",
    "planned",
    "deferred",
    "declined",
    "needs_info",
]
Scope = Literal[
    "one_line_fix",
    "contained",
    "cross_cutting",
    "requires_design",
]

# Prompt text is hashed into extractor_version. Bump SCHEMA_VERSION on field changes.
_SCHEMA_VERSION = "block6-v3"

# Captured 2026-08-30. Hub revision + groq provider; see docs/hf_snapshot/extractor_pin.json.
# CASEFILE_LLM_MODEL must be PINNED_ROUTER_MODEL_ID — the router does not take @sha.
PINNED_ROUTER_MODEL_ID = "openai/gpt-oss-120b:groq"
PINNED_EXTRACTOR_MODEL_ID = (
    "openai/gpt-oss-120b:groq@b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
)

_QUOTE_FIELDS = (
    "intent",
    "proposed_solution_present",
    "patch_offered",
    "maintainer_stance",
    "names_alternative",
    "names_alternative_text",
)
_RUBRIC_FIELDS = (
    "specificity",
    "blocking_severity",
    "affect",
    "scope",
)
_FIELD_NAMES = _QUOTE_FIELDS + _RUBRIC_FIELDS

# Closed cells. Validator checks (field, value) → anchor membership; it does not
# re-score the thread. Keep the lists short so the model cannot hide a judgement
# behind a novel string.
_RUBRIC: dict[str, dict[Any, tuple[str, ...]]] = {
    "specificity": {
        0: ("title_only", "question_only", "no_repro_no_code"),
        1: ("partial_steps", "environment_or_version_only", "code_without_expected"),
        2: ("steps_or_code", "expected_xor_actual"),
        3: ("expected_and_actual", "minimal_reproducer"),
    },
    "blocking_severity": {
        0: ("curiosity", "nice_to_have", "docs_or_test_only"),
        1: ("workaround_exists", "degraded_but_usable"),
        2: ("blocks_a_workflow", "wrong_results"),
        3: ("blocks_production", "data_loss_or_crash_in_prod"),
    },
    "affect": {
        0: ("neutral_report", "matter_of_fact"),
        1: ("frustrated_or_urgent", "repeated_asks"),
        2: ("hostile_or_ultimatum",),
    },
    "scope": {
        "one_line_fix": ("typo_or_link", "single_function"),
        "contained": ("one_module", "local_api"),
        "cross_cutting": ("multiple_modules", "dispatch_or_device"),
        "requires_design": ("new_public_api", "rfc_or_semver"),
    },
}


def _format_rubric() -> str:
    lines: list[str] = []
    for field, levels in _RUBRIC.items():
        cells = "; ".join(
            f"{level}: {', '.join(anchors)}" for level, anchors in levels.items()
        )
        lines.append(f"- {field}: {cells}")
    return "\n".join(lines)


_PROMPT = f"""Extract schema-constrained fields from this issue/discussion thread.
Use ONLY text with timestamps at or before the snapshot time T.
Return JSON only, matching the supplied JSON schema exactly.
Categorical values MUST use one of the schema's enum strings, never free text.

Two evidence rules:
- Quotable fields ({", ".join(_QUOTE_FIELDS)}): evidence_span[field] is an exact
  contiguous quote from the input of at most 15 words. Copy it; do not paraphrase.
  Use null when no direct quote supports the value. Absence is not a quote: do not
  emit false or "none" just because the thread never mentions the thing.
- Judged fields ({", ".join(_RUBRIC_FIELDS)}): evidence_anchor[field] is one rubric
  cell from the list below that matches the value. Do not invent a quote for these
  fields. Null the field if no cell fits.

Rubric (value → allowed anchors):
{_format_rubric()}
"""


def _span_properties() -> dict[str, Any]:
    return {name: {"type": ["string", "null"]} for name in _QUOTE_FIELDS}


def _anchor_properties() -> dict[str, Any]:
    props: dict[str, Any] = {}
    for field, levels in _RUBRIC.items():
        enum = sorted({anchor for anchors in levels.values() for anchor in anchors})
        props[field] = {"type": ["string", "null"], "enum": [*enum, None]}
    return props


EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "block6_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {
                    "type": ["string", "null"],
                    "enum": [
                        "bug",
                        "feature_request",
                        "support",
                        "docs",
                        "design_proposal",
                        "integration_report",
                        "other",
                        None,
                    ],
                },
                "specificity": {"type": ["integer", "null"], "minimum": 0, "maximum": 3},
                "proposed_solution_present": {"type": ["boolean", "null"]},
                "patch_offered": {"type": ["boolean", "null"]},
                "blocking_severity": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 3,
                },
                "affect": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
                "maintainer_stance": {
                    "type": ["string", "null"],
                    "enum": [
                        "none",
                        "acknowledged",
                        "planned",
                        "deferred",
                        "declined",
                        "needs_info",
                        None,
                    ],
                },
                "scope": {
                    "type": ["string", "null"],
                    "enum": [
                        "one_line_fix",
                        "contained",
                        "cross_cutting",
                        "requires_design",
                        None,
                    ],
                },
                "names_alternative": {"type": ["boolean", "null"]},
                "names_alternative_text": {"type": ["string", "null"]},
                "evidence_span": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": _span_properties(),
                    "required": list(_QUOTE_FIELDS),
                },
                "evidence_anchor": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": _anchor_properties(),
                    "required": list(_RUBRIC_FIELDS),
                },
            },
            "required": [*_FIELD_NAMES, "evidence_span", "evidence_anchor"],
        },
    },
}


@dataclass
class ExtractedFields:
    intent: Intent | None = None
    specificity: int | None = None  # 0–3
    proposed_solution_present: bool | None = None
    patch_offered: bool | None = None
    blocking_severity: int | None = None  # 0–3
    affect: int | None = None  # 0–2; expect low importance in ablations
    maintainer_stance: MaintainerStance | None = None
    scope: Scope | None = None
    names_alternative: bool | None = None
    names_alternative_text: str | None = None
    evidence_span: dict[str, str] | None = None
    evidence_anchor: dict[str, str] | None = None
    extractor_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pin_extractor_version(*, model_id: str) -> str:
    """Stable pin: schema + prompt hash + concrete model id (no floating aliases)."""
    if not model_id or model_id.endswith(":fastest") or ":latest" in model_id:
        raise ValueError(
            f"extractor model_id must be a concrete pin, not a floating alias: {model_id!r}"
        )
    prompt_hash = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()[:12]
    return f"{_SCHEMA_VERSION}|{prompt_hash}|{model_id}"


def extraction_prompt() -> str:
    return _PROMPT


def quote_is_required(key: str, value: object) -> bool:
    """Presence needs a quote. Absence (false / stance none) is the missing quote."""
    if value is None or key not in _QUOTE_FIELDS:
        return False
    if value is False:
        return False
    if key == "maintainer_stance" and value == "none":
        return False
    return True


def parse_extracted_json(raw: str, *, extractor_version: str) -> ExtractedFields:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("extractor output must be a JSON object")
    spans = data.get("evidence_span")
    if spans is not None and not isinstance(spans, dict):
        raise ValueError("evidence_span must be an object")
    anchors = data.get("evidence_anchor")
    if anchors is not None and not isinstance(anchors, dict):
        raise ValueError("evidence_anchor must be an object")
    return ExtractedFields(
        intent=_opt_literal(data.get("intent"), "intent", Intent.__args__),  # type: ignore[attr-defined]
        specificity=_opt_int(data.get("specificity"), 0, 3),
        proposed_solution_present=_opt_bool(data.get("proposed_solution_present")),
        patch_offered=_opt_bool(data.get("patch_offered")),
        blocking_severity=_opt_int(data.get("blocking_severity"), 0, 3),
        affect=_opt_int(data.get("affect"), 0, 2),
        maintainer_stance=_opt_literal(
            data.get("maintainer_stance"),
            "maintainer_stance",
            MaintainerStance.__args__,  # type: ignore[attr-defined]
        ),
        scope=_opt_literal(data.get("scope"), "scope", Scope.__args__),  # type: ignore[attr-defined]
        names_alternative=_opt_bool(data.get("names_alternative")),
        names_alternative_text=_opt_str(data.get("names_alternative_text")),
        evidence_span={
            str(k): str(v) for k, v in (spans or {}).items() if v is not None
        },
        evidence_anchor={
            str(k): str(v) for k, v in (anchors or {}).items() if v is not None
        },
        extractor_version=extractor_version,
    )


def validate_evidence_spans(fields: ExtractedFields, haystack: str) -> list[str]:
    """Quote check for quotable fields; rubric membership for judged fields."""
    errors: list[str] = []
    spans = fields.evidence_span or {}
    anchors = fields.evidence_anchor or {}
    hay = " ".join(haystack.lower().split())
    values = fields.to_dict()
    for key in _QUOTE_FIELDS:
        if quote_is_required(key, values[key]) and not spans.get(key, "").strip():
            errors.append(f"evidence_span[{key}] missing for non-null field")
    for key in _RUBRIC_FIELDS:
        if values[key] is None:
            continue
        anchor = str(anchors.get(key) or "").strip()
        if not anchor:
            errors.append(f"evidence_anchor[{key}] missing for non-null field")
            continue
        allowed = _RUBRIC[key].get(values[key], ())
        if anchor not in allowed:
            errors.append(
                f"evidence_anchor[{key}]={anchor!r} is not valid for {key}={values[key]!r}"
            )
    for key, quote in spans.items():
        if key in _RUBRIC_FIELDS:
            continue
        if key not in _QUOTE_FIELDS:
            errors.append(f"evidence_span[{key}] is not a schema field")
            continue
        q = " ".join(quote.lower().split())
        if not q:
            continue
        if len(q.split()) > 15:
            errors.append(f"evidence_span[{key}] exceeds 15 words")
        if q not in hay:
            errors.append(f"evidence_span[{key}] quote not found in thread text ≤ T")
    for key in anchors:
        if key not in _RUBRIC_FIELDS:
            errors.append(f"evidence_anchor[{key}] is not a judged field")
    return errors


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_literal(value: object, field: str, allowed: tuple[str, ...]) -> str | None:
    text = _opt_str(value)
    if text is None:
        return None
    if text not in allowed:
        raise ValueError(f"{field} must be one of {allowed}, got {text!r}")
    return text


def _opt_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _opt_int(value: object, lo: int, hi: int) -> int | None:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n < lo or n > hi:
        return None
    return n
