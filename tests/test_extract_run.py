"""Block-6 extraction runner: resume, span errors kept, YAML gold → JSONL."""

import json
from pathlib import Path

import httpx
import pytest

from casefile.clients.llm import LlmClient
from casefile.config import Settings
from casefile.predict.extract_run import (
    _DEFAULT_KINDS,
    collect_items,
    filter_by_kind,
    gold_yaml_to_items,
    item_kind,
    item_stratum,
    main,
    parse_kinds,
    row_is_complete,
    revalidate_output,
    run_extract,
    stratified_sample,
    strip_json_fence,
    write_merged_output,
)
from casefile.predict.extractor import PINNED_EXTRACTOR_MODEL_ID, PINNED_ROUTER_MODEL_ID
from casefile.predict.extractor import pin_extractor_version


def test_pinned_extractor_id_is_concrete():
    version = pin_extractor_version(model_id=PINNED_EXTRACTOR_MODEL_ID)
    assert ":fastest" not in version
    assert ":latest" not in version
    assert "b5c939de8f754692c1647ca79fbf85e8c1e70f8a" in version
    assert PINNED_ROUTER_MODEL_ID == "openai/gpt-oss-120b:groq"


def test_pinned_id_matches_snapshot():
    pin_path = (
        Path(__file__).resolve().parents[1] / "docs" / "hf_snapshot" / "extractor_pin.json"
    )
    data = json.loads(pin_path.read_text(encoding="utf-8"))
    assert data["extractor_model_id"] == PINNED_EXTRACTOR_MODEL_ID
    assert data["router_id"] == PINNED_ROUTER_MODEL_ID


def test_extract_json_object_from_reasoning():
    from casefile.predict.extract_run import extract_json_object

    raw = 'thinking first\n{"intent": "bug", "specificity": 1}\nand more'
    assert json.loads(extract_json_object(raw))["intent"] == "bug"


def test_gold_yaml_to_items(tmp_path: Path):
    path = tmp_path / "gold.yaml"
    path.write_text(
        "labelled_at: '2026-08-28'\n"
        "items:\n"
        "  - repo: pytorch/pytorch\n"
        "    number: 1\n"
        "    title: Hello\n"
        "    body: add masked fill support\n"
        "    is_pull_request: true\n",
        encoding="utf-8",
    )
    items = gold_yaml_to_items(path)
    assert items[0]["item_id"] == "github:pytorch/pytorch#1"
    assert items[0]["url"].endswith("/pull/1")
    assert items[0]["snapshot_T"] == "2026-08-28"
    assert items[0]["is_pull_request"] is True
    assert items[0]["source"] == "github"


@pytest.mark.asyncio
async def test_extract_run_records_span_errors_and_resumes(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    item = {
        "repo": "pytorch/pytorch",
        "item_id": "github:pytorch/pytorch#1",
        "url": "https://github.com/pytorch/pytorch/issues/1",
        "title": "add masked fill support",
        "body": "Please add masked fill support.",
        "created_at": "2024-01-01T00:00:00Z",
        "snapshot_T": "2026-08-28",
    }
    payload = {
        "intent": "feature_request",
        "specificity": 2,
        "proposed_solution_present": True,
        "patch_offered": False,
        "blocking_severity": 1,
        "affect": 0,
        "maintainer_stance": "none",
        "scope": "contained",
        "names_alternative": False,
        "names_alternative_text": None,
        "evidence_span": {"intent": "this quote is not in the thread at all"},
        "evidence_anchor": {
            "specificity": "steps_or_code",
            "blocking_severity": "workaround_exists",
            "affect": "neutral_report",
            "scope": "one_module",
        },
    }
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    out = tmp_path / "out.jsonl"
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract([item], out, settings=settings, llm=llm)
    assert stats["wrote"] == 1
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["extractor_version"] == pin_extractor_version(model_id=PINNED_EXTRACTOR_MODEL_ID)
    assert row["span_errors"]
    assert row["repair_attempted"] is True
    assert row["repair_attempts"] == 2
    assert row["fields"]["intent"] == "feature_request"
    assert row["input_hash"]

    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats2 = await run_extract([item], out, settings=settings, llm=llm)
    assert stats2["wrote"] == 0
    assert stats2["skipped"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_collect_items_jsonl(tmp_path: Path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps({"item_id": "a", "title": "t", "body": "b", "repo": "x/y"}) + "\n",
        encoding="utf-8",
    )
    items = collect_items([path], from_yaml=False)
    assert items[0]["item_id"] == "a"


def test_row_is_complete_requires_repair_metadata():
    version = pin_extractor_version(model_id=PINNED_EXTRACTOR_MODEL_ID)
    stale = {"item_id": "github:pytorch/pytorch#1", "extractor_version": version}
    fresh = {**stale, "repair_attempts": 0}
    failed = {**fresh, "parse_error": "llm_error: 400"}
    assert not row_is_complete(stale, version)
    assert row_is_complete(fresh, version)
    assert not row_is_complete(failed, version)


@pytest.mark.asyncio
async def test_extract_run_reruns_stale_rows_and_compacts(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    item = {
        "repo": "pytorch/pytorch",
        "item_id": "github:pytorch/pytorch#1",
        "url": "https://github.com/pytorch/pytorch/issues/1",
        "title": "add masked fill support",
        "body": "Please add masked fill support.",
        "created_at": "2024-01-01T00:00:00Z",
        "snapshot_T": "2026-08-28",
    }
    version = pin_extractor_version(model_id=PINNED_EXTRACTOR_MODEL_ID)
    payload = {
        "intent": "feature_request",
        "specificity": None,
        "proposed_solution_present": True,
        "patch_offered": False,
        "blocking_severity": None,
        "affect": None,
        "maintainer_stance": None,
        "scope": None,
        "names_alternative": False,
        "names_alternative_text": None,
        "evidence_span": {
            "intent": "add masked fill support",
            "proposed_solution_present": "Please add masked fill support.",
            "patch_offered": "Please add masked fill support.",
            "names_alternative": "Please add masked fill support.",
        },
        "evidence_anchor": {},
    }
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    out = tmp_path / "out.jsonl"
    stale_row = {
        "repo": item["repo"],
        "item_id": item["item_id"],
        "url": item["url"],
        "created_at": item["created_at"],
        "snapshot_T": item["snapshot_T"],
        "input_hash": "old",
        "extractor_version": version,
        "raw": "{}",
        "parse_error": None,
        "fields": payload,
        "span_errors": [],
    }
    out.write_text(json.dumps(stale_row) + "\n" + json.dumps(stale_row) + "\n", encoding="utf-8")
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract([item], out, settings=settings, llm=llm)
    assert stats["skipped"] == 0
    assert stats["wrote"] == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["repair_attempts"] == 0


def test_write_merged_output_dedupes_by_item_order(tmp_path: Path):
    items = [
        {"item_id": "a", "title": "first"},
        {"item_id": "b", "title": "second"},
    ]
    merged = {
        "a": {"item_id": "a", "fields": {"intent": "bug"}},
        "b": {"item_id": "b", "fields": {"intent": "feature_request"}},
    }
    out = tmp_path / "out.jsonl"
    write_merged_output(out, items, merged)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["item_id"] for row in rows] == ["a", "b"]


def test_write_merged_output_keeps_rows_not_in_this_draw(tmp_path: Path):
    items = [{"item_id": "d"}]
    merged = {
        "a": {"item_id": "a"},
        "b": {"item_id": "b"},
        "c": {"item_id": "c"},
        "d": {"item_id": "d"},
    }
    out = tmp_path / "out.jsonl"
    write_merged_output(out, items, merged)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["item_id"] for row in rows] == ["d", "a", "b", "c"]


def test_revalidate_output_clears_quote_on_false(tmp_path: Path):
    item = {
        "item_id": "github:pytorch/pytorch#1",
        "title": "bug in masked fill",
        "body": "masked_fill_ is wrong on cpu",
    }
    version = pin_extractor_version(model_id=PINNED_EXTRACTOR_MODEL_ID)
    row = {
        "item_id": item["item_id"],
        "extractor_version": version,
        "parse_error": None,
        "span_errors": ["evidence_span[patch_offered] missing for non-null field"],
        "fields": {
            "intent": "bug",
            "patch_offered": False,
            "proposed_solution_present": False,
            "maintainer_stance": "none",
            "names_alternative": False,
            "evidence_span": {"intent": "bug in masked fill"},
            "evidence_anchor": {},
        },
    }
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps(row) + "\n", encoding="utf-8")
    stats = revalidate_output([item], out)
    assert stats["rows"] == 1
    assert stats["updated"] == 1
    saved = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert saved["span_errors"] == []
    items = [{"item_id": "d"}]
    merged = {
        "a": {"item_id": "a"},
        "b": {"item_id": "b"},
        "c": {"item_id": "c"},
        "d": {"item_id": "d"},
    }
    out = tmp_path / "out.jsonl"
    write_merged_output(out, items, merged)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["item_id"] for row in rows] == ["d", "a", "b", "c"]


_OK_FIELDS = {
    "intent": "feature_request",
    "specificity": None,
    "proposed_solution_present": True,
    "patch_offered": False,
    "blocking_severity": None,
    "affect": None,
    "maintainer_stance": None,
    "scope": None,
    "names_alternative": False,
    "names_alternative_text": None,
    "evidence_span": {
        "intent": "add masked fill support",
        "proposed_solution_present": "Please add masked fill support.",
        "patch_offered": "Please add masked fill support.",
        "names_alternative": "Please add masked fill support.",
    },
    "evidence_anchor": {},
}


@pytest.mark.asyncio
async def test_extract_run_keeps_rows_not_in_this_draw(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    item = {
        "repo": "pytorch/pytorch",
        "item_id": "github:pytorch/pytorch#9",
        "url": "https://github.com/pytorch/pytorch/issues/9",
        "title": "add masked fill support",
        "body": "Please add masked fill support.",
        "created_at": "2024-01-01T00:00:00Z",
        "snapshot_T": "2026-08-28",
    }
    out = tmp_path / "out.jsonl"
    extras = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    out.write_text(
        "".join(json.dumps(row) + "\n" for row in extras),
        encoding="utf-8",
    )
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(_OK_FIELDS)}}]},
    )
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract([item], out, settings=settings, llm=llm)
    assert stats["wrote"] == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["item_id"] for row in rows] == [
        "github:pytorch/pytorch#9",
        "a",
        "b",
        "c",
    ]


def test_stratified_sample_covers_origin_and_kind():
    holdout = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment"
    items = collect_items(
        [holdout / "pytorch_holdout.yaml", holdout / "apertus.yaml"],
        from_yaml=True,
    )
    assert item_stratum(items[0]) == "pytorch:pr"
    prefix = items[:10]
    assert {item_stratum(item) for item in prefix} == {"pytorch:pr"}
    sample = stratified_sample(items, 20, seed=0)
    assert len(sample) == 20
    strata = {item_stratum(item) for item in sample}
    assert strata == {
        "pytorch:pr",
        "pytorch:issue",
        "other:hub",
        "other:issue",
        "other:pr",
    }
    again = stratified_sample(items, 20, seed=0)
    assert [item["item_id"] for item in again] == [item["item_id"] for item in sample]


def test_filter_by_kind_drops_pull_requests():
    holdout = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment"
    items = collect_items(
        [holdout / "pytorch_holdout.yaml", holdout / "apertus.yaml"],
        from_yaml=True,
    )
    kept = filter_by_kind(items, parse_kinds(_DEFAULT_KINDS))
    assert kept
    assert {item_kind(item) for item in kept} == {"issue", "hub"}
    assert _DEFAULT_KINDS == "issue,hub"
    assert parse_kinds("issue,hub") == frozenset({"issue", "hub"})


def test_main_default_kinds_drops_pull_requests(tmp_path: Path, monkeypatch):
    captured: dict = {}

    async def fake_run(items, output_path, **kwargs):
        captured["item_ids"] = [item["item_id"] for item in items]
        return {"total": len(items), "skipped": 0, "wrote": 0, "stopped": None}

    monkeypatch.setattr("casefile.predict.extract_run.run_extract", fake_run)
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "labelled_at: '2026-08-28'\n"
        "items:\n"
        "  - repo: pytorch/pytorch\n"
        "    number: 1\n"
        "    title: pr\n"
        "    is_pull_request: true\n"
        "  - repo: pytorch/pytorch\n"
        "    number: 2\n"
        "    title: issue\n"
        "    is_pull_request: false\n",
        encoding="utf-8",
    )
    assert main(["-i", str(gold), "-o", str(tmp_path / "out.jsonl")]) == 0
    assert captured["item_ids"] == ["github:pytorch/pytorch#2"]


@pytest.mark.asyncio
async def test_extract_run_records_failed_generation(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    item = {
        "repo": "pytorch/pytorch",
        "item_id": "github:pytorch/pytorch#1",
        "url": "https://github.com/pytorch/pytorch/issues/1",
        "title": "add masked fill support",
        "body": "Please add masked fill support.",
        "created_at": "2024-01-01T00:00:00Z",
        "snapshot_T": "2026-08-28",
    }
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        status_code=400,
        json={
            "error": {
                "message": "Generated JSON does not match the expected schema.",
                "code": "json_validate_failed",
                "failed_generation": '{"scope": 2, "intent": "bug"}',
            }
        },
    )
    out = tmp_path / "out.jsonl"
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract([item], out, settings=settings, llm=llm)
    assert stats["wrote"] == 1
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row.get("llm_http_status") == 400
    assert row.get("llm_failed_generation") == '{"scope": 2, "intent": "bug"}'
    assert "failed_generation" in (row.get("parse_error") or "")


@pytest.mark.asyncio
async def test_extract_run_stops_after_payment_required(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    items = [
        {
            "repo": "pytorch/pytorch",
            "item_id": f"github:pytorch/pytorch#{n}",
            "url": f"https://github.com/pytorch/pytorch/issues/{n}",
            "title": "add masked fill support",
            "body": "Please add masked fill support.",
            "created_at": "2024-01-01T00:00:00Z",
            "snapshot_T": "2026-08-28",
        }
        for n in (1, 2)
    ]
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        status_code=402,
        json={"error": {"message": "Payment Required"}},
    )
    out = tmp_path / "out.jsonl"
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract(items, out, settings=settings, llm=llm)
    assert stats["wrote"] == 1
    assert stats["stopped"] == "llm_auth_or_payment"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0].get("llm_http_status") == 402


@pytest.mark.asyncio
async def test_extract_run_does_not_stop_on_402_in_error_body(tmp_path: Path, httpx_mock):
    settings = Settings(
        llm_provider="huggingface",
        llm_model="openai/gpt-oss-120b:fastest",
        hf_token="hf_test",
    )
    payload = {
        "intent": "feature_request",
        "specificity": None,
        "proposed_solution_present": True,
        "patch_offered": False,
        "blocking_severity": None,
        "affect": None,
        "maintainer_stance": None,
        "scope": None,
        "names_alternative": False,
        "names_alternative_text": None,
        "evidence_span": {
            "intent": "add masked fill support",
            "proposed_solution_present": "Please add masked fill support.",
            "patch_offered": "Please add masked fill support.",
            "names_alternative": "Please add masked fill support.",
        },
        "evidence_anchor": {},
    }
    items = [
        {
            "repo": "pytorch/pytorch",
            "item_id": f"github:pytorch/pytorch#{n}",
            "url": f"https://github.com/pytorch/pytorch/issues/{n}",
            "title": "add masked fill support",
            "body": "Please add masked fill support.",
            "created_at": "2024-01-01T00:00:00Z",
            "snapshot_T": "2026-08-28",
        }
        for n in (1, 2)
    ]
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        status_code=400,
        json={"error": {"message": "see RFC 402 for payment"}},
    )
    httpx_mock.add_response(
        url="https://router.huggingface.co/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    out = tmp_path / "out.jsonl"
    async with httpx.AsyncClient() as client:
        llm = LlmClient(settings, client)
        stats = await run_extract(items, out, settings=settings, llm=llm)
    assert stats["wrote"] == 2
    assert stats["stopped"] is None
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0].get("llm_http_status") == 400
    assert rows[1].get("parse_error") is None
