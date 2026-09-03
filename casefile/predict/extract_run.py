"""Block-6 extraction runner: JSONL in, JSONL out, resumable, no verdicts.

Input rows: repo, item_id, url, title, body, created_at, snapshot_T.
Gold YAML from eval/topic_assignment/ is accepted via --from-yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

from casefile.clients.llm import LlmClient
from casefile.config import Settings, get_settings
from casefile.predict.extractor import (
    EXTRACTION_RESPONSE_FORMAT,
    PINNED_EXTRACTOR_MODEL_ID,
    PINNED_ROUTER_MODEL_ID,
    extraction_prompt,
    parse_extracted_json,
    pin_extractor_version,
    validate_evidence_spans,
)

_EXTRACT_MAX_TOKENS = 2200


def input_hash(*, title: str, body: str, snapshot_t: str) -> str:
    payload = f"{snapshot_t}\n{title}\n{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def thread_text(*, title: str, body: str) -> str:
    return f"{title}\n\n{body}".strip()


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_object(raw: str) -> str:
    """Last JSON object in the text. gpt-oss often reasons first, then emits JSON."""
    text = strip_json_fence(raw)
    decoder = json.JSONDecoder()
    last: str | None = None
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            last = text[start:end]
        idx = end
    if last is None:
        raise json.JSONDecodeError("no JSON object in model output", text, 0)
    return last


def gold_yaml_to_items(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    labelled_at = str(data.get("labelled_at") or "")
    items: list[dict[str, Any]] = []
    for raw in data.get("items") or []:
        repo = str(raw["repo"])
        number = raw["number"]
        source = str(raw.get("source") or "github")
        url = raw.get("url")
        if not url:
            kind = "pull" if raw.get("is_pull_request") else "issues"
            url = f"https://github.com/{repo}/{kind}/{number}"
        if "is_pull_request" in raw:
            is_pr = bool(raw.get("is_pull_request"))
        else:
            is_pr = "/pull/" in str(url)
        item_id = f"{source}:{repo}#{number}"
        items.append(
            {
                "repo": repo,
                "item_id": item_id,
                "url": url,
                "title": str(raw.get("title") or ""),
                "body": str(raw.get("body") or ""),
                "created_at": str(raw.get("created_at") or ""),
                "snapshot_T": labelled_at,
                "source": source,
                "is_pull_request": is_pr,
            }
        )
    return items


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_is_complete(row: dict[str, Any], extractor_version: str) -> bool:
    if row.get("extractor_version") != extractor_version:
        return False
    if not row.get("item_id"):
        return False
    if "repair_attempts" not in row:
        return False
    # Span errors are kept; transport/parse failures are retried.
    return not row.get("parse_error")


def _llm_http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _llm_error_details(exc: BaseException) -> tuple[str, str | None]:
    """Human error string plus router failed_generation when present."""
    parts = [str(exc)]
    failed: str | None = None
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            text = (response.text or "").strip()
        except Exception:  # noqa: BLE001 — best-effort body for the JSONL row
            text = ""
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                    if msg:
                        parts.append(str(msg)[:400])
                    gen = err.get("failed_generation")
                    if gen is not None:
                        failed = str(gen)[:2000]
                        parts.append("failed_generation: " + failed[:800])
                else:
                    parts.append(text[:500])
            else:
                parts.append(text[:500])
    return "llm_error: " + " | ".join(parts), failed


def _format_llm_error(exc: BaseException) -> str:
    message, _failed = _llm_error_details(exc)
    return message


def pick_best_row(rows: list[dict[str, Any]], extractor_version: str) -> dict[str, Any]:
    complete = [row for row in rows if row_is_complete(row, extractor_version)]
    if complete:
        return complete[-1]
    return rows[-1]


def load_merged_rows(output_path: Path, extractor_version: str) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(output_path):
        item_id = row.get("item_id")
        if not item_id:
            continue
        grouped.setdefault(str(item_id), []).append(row)
    return {key: pick_best_row(rows, extractor_version) for key, rows in grouped.items()}


_ALLOWED_KINDS = frozenset({"issue", "pr", "hub"})
_DEFAULT_KINDS = "issue,hub"


def item_kind(item: dict[str, Any]) -> str:
    iid = str(item.get("item_id") or "")
    source = str(item.get("source") or (iid.split(":", 1)[0] if ":" in iid else "github"))
    url = str(item.get("url") or "")
    is_pr = item.get("is_pull_request")
    if is_pr is None:
        is_pr = "/pull/" in url
    if source == "hub":
        return "hub"
    if is_pr:
        return "pr"
    return "issue"


def item_stratum(item: dict[str, Any]) -> str:
    """Origin × kind. `--limit` over pytorch_holdout.yaml is all PRs for N ≤ 45."""
    repo = str(item.get("repo") or "")
    origin = "pytorch" if repo.startswith("pytorch/") else "other"
    return f"{origin}:{item_kind(item)}"


def parse_kinds(raw: str) -> frozenset[str]:
    kinds = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = kinds - _ALLOWED_KINDS
    if not kinds:
        raise ValueError("kinds must be a comma-separated subset of issue, pr, hub")
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}; use issue, pr, hub")
    return kinds


def filter_by_kind(items: list[dict[str, Any]], kinds: frozenset[str]) -> list[dict[str, Any]]:
    return [item for item in items if item_kind(item) in kinds]


def stratified_sample(
    items: list[dict[str, Any]], n: int, *, seed: int = 0
) -> list[dict[str, Any]]:
    """Round-robin across origin×kind so a smoke cannot be ten PRs from file order."""
    if n <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item_stratum(item), []).append(item)
    rng = random.Random(seed)
    keys = sorted(grouped)
    buckets = {key: list(grouped[key]) for key in keys}
    for key in keys:
        rng.shuffle(buckets[key])
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(picked) < n:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            while bucket:
                item = bucket.pop(0)
                item_id = str(item.get("item_id"))
                if item_id in seen:
                    continue
                seen.add(item_id)
                picked.append(item)
                progressed = True
                break
            if len(picked) >= n:
                break
        if not progressed:
            break
    return picked


def revalidate_output(items: list[dict[str, Any]], output_path: Path) -> dict[str, int]:
    """Recompute span_errors from stored fields. No LLM calls."""
    by_id = {str(item.get("item_id")): item for item in items}
    rows = load_jsonl(output_path)
    updated = 0
    for row in rows:
        item = by_id.get(str(row.get("item_id") or ""))
        fields = row.get("fields")
        if item is None or not isinstance(fields, dict) or row.get("parse_error"):
            continue
        parsed = parse_extracted_json(
            json.dumps(fields),
            extractor_version=str(row.get("extractor_version") or ""),
        )
        hay = thread_text(
            title=str(item.get("title") or ""),
            body=str(item.get("body") or ""),
        )
        new_errors = validate_evidence_spans(parsed, hay)
        if new_errors != (row.get("span_errors") or []):
            updated += 1
        row["span_errors"] = new_errors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "updated": updated}


def already_done(output_path: Path, extractor_version: str) -> set[str]:
    merged = load_merged_rows(output_path, extractor_version)
    return {
        item_id
        for item_id, row in merged.items()
        if row_is_complete(row, extractor_version)
    }


def write_merged_output(
    output_path: Path,
    items: list[dict[str, Any]],
    merged: dict[str, dict[str, Any]],
) -> None:
    """Draw order first, then any other rows already in the file. Never drop extras."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("item_id"))
        row = merged.get(item_id)
        if row is not None and item_id not in seen:
            ordered.append(row)
            seen.add(item_id)
    for item_id, row in merged.items():
        if item_id not in seen:
            ordered.append(row)
            seen.add(item_id)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(
    item: dict[str, Any],
    *,
    extractor_version: str,
    raw: str | None,
    parse_error: str | None,
    fields: dict[str, Any] | None,
    span_errors: list[str],
    repair_attempted: bool = False,
    repair_attempts: int = 0,
    repair_raw: str | None = None,
    repair_error: str | None = None,
    llm_http_status: int | None = None,
    llm_failed_generation: str | None = None,
) -> dict[str, Any]:
    title = str(item.get("title") or "")
    body = str(item.get("body") or "")
    snapshot_t = str(item.get("snapshot_T") or "")
    row = {
        "repo": item.get("repo"),
        "item_id": item.get("item_id"),
        "url": item.get("url"),
        "source": item.get("source"),
        "is_pull_request": item.get("is_pull_request"),
        "created_at": item.get("created_at"),
        "snapshot_T": snapshot_t,
        "input_hash": input_hash(title=title, body=body, snapshot_t=snapshot_t),
        "extractor_version": extractor_version,
        "raw": raw,
        "parse_error": parse_error,
        "fields": fields,
        "span_errors": span_errors,
        "repair_attempted": repair_attempted,
        "repair_attempts": repair_attempts,
        "repair_raw": repair_raw,
        "repair_error": repair_error,
    }
    if llm_http_status is not None:
        row["llm_http_status"] = llm_http_status
    if llm_failed_generation is not None:
        row["llm_failed_generation"] = llm_failed_generation
    return row


async def extract_item(
    llm: LlmClient,
    item: dict[str, Any],
    *,
    extractor_version: str,
) -> dict[str, Any]:
    hay = thread_text(title=str(item.get("title") or ""), body=str(item.get("body") or ""))
    snapshot_t = str(item.get("snapshot_T") or "")
    user = (
        f"Snapshot time T: {snapshot_t or '(unspecified; use only the text below)'}\n"
        f"Thread text:\n{hay}\n\n"
        "Reply with a single JSON object that matches the schema. No analysis."
    )
    try:
        raw = await llm.complete(
            extraction_prompt(),
            user,
            max_tokens=_EXTRACT_MAX_TOKENS,
            response_format=EXTRACTION_RESPONSE_FORMAT,
        )
    except Exception as exc:  # noqa: BLE001 — record and continue the batch
        parse_error, failed_generation = _llm_error_details(exc)
        return build_row(
            item,
            extractor_version=extractor_version,
            raw=None,
            parse_error=parse_error,
            fields=None,
            span_errors=[],
            llm_http_status=_llm_http_status(exc),
            llm_failed_generation=failed_generation,
        )
    try:
        parsed = parse_extracted_json(
            extract_json_object(raw), extractor_version=extractor_version
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return build_row(
            item,
            extractor_version=extractor_version,
            raw=raw,
            parse_error=str(exc),
            fields=None,
            span_errors=[],
        )
    span_errors = validate_evidence_spans(parsed, hay)
    repair_raw: str | None = None
    repair_error: str | None = None
    repair_attempts = 0
    current_json = extract_json_object(raw)
    while span_errors and repair_attempts < 2:
        repair_user = (
            f"Snapshot time T: {snapshot_t or '(unspecified)'}\n"
            f"Thread text:\n{hay}\n\n"
            f"Invalid JSON extraction:\n{current_json}\n\n"
            f"Validation errors:\n- " + "\n- ".join(span_errors) + "\n\n"
            "Return a corrected JSON object. Copy every evidence_span quote exactly "
            "from the thread. For judged fields, set evidence_anchor to a rubric cell "
            "that matches the value. If a quote or rubric cell does not support the "
            "field, set that field and its evidence to null. No analysis."
        )
        repair_attempts += 1
        try:
            repair_raw = await llm.complete(
                extraction_prompt(),
                repair_user,
                max_tokens=_EXTRACT_MAX_TOKENS,
                response_format=EXTRACTION_RESPONSE_FORMAT,
            )
            repaired = parse_extracted_json(
                extract_json_object(repair_raw), extractor_version=extractor_version
            )
            repaired_errors = validate_evidence_spans(repaired, hay)
            if len(repaired_errors) <= len(span_errors):
                parsed = repaired
                span_errors = repaired_errors
                current_json = extract_json_object(repair_raw)
        except Exception as exc:  # noqa: BLE001 — keep the original validated row
            repair_error = str(exc)
            break
    return build_row(
        item,
        extractor_version=extractor_version,
        raw=raw,
        parse_error=None,
        fields=parsed.to_dict(),
        span_errors=span_errors,
        repair_attempted=repair_attempts > 0,
        repair_attempts=repair_attempts,
        repair_raw=repair_raw,
        repair_error=repair_error,
    )


def collect_items(inputs: Iterable[Path], *, from_yaml: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in inputs:
        if from_yaml or path.suffix.lower() in {".yaml", ".yml"}:
            items.extend(gold_yaml_to_items(path))
        else:
            items.extend(load_jsonl(path))
    return items


async def run_extract(
    items: list[dict[str, Any]],
    output_path: Path,
    *,
    settings: Settings | None = None,
    llm: LlmClient | None = None,
    router_model_id: str = PINNED_ROUTER_MODEL_ID,
    extractor_model_id: str = PINNED_EXTRACTOR_MODEL_ID,
) -> dict[str, int]:
    extractor_version = pin_extractor_version(model_id=extractor_model_id)
    merged = load_merged_rows(output_path, extractor_version)
    done = already_done(output_path, extractor_version)
    pending = [item for item in items if str(item.get("item_id")) not in done]
    stats = {
        "total": len(items),
        "skipped": len(items) - len(pending),
        "wrote": 0,
        "stopped": None,
    }

    existing_lines = len(load_jsonl(output_path)) if output_path.exists() else 0
    if pending:
        own_client = llm is None
        client: httpx.AsyncClient | None = None
        if llm is None:
            cfg = settings or get_settings()
            cfg = cfg.model_copy(update={"llm_model": router_model_id})
            client = httpx.AsyncClient(timeout=max(cfg.http_timeout, 120.0))
            llm = LlmClient(cfg, client)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("a", encoding="utf-8") as handle:
                for item in pending:
                    row = await extract_item(llm, item, extractor_version=extractor_version)
                    merged[str(item.get("item_id"))] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    stats["wrote"] += 1
                    if row.get("llm_http_status") in {401, 402}:
                        stats["stopped"] = "llm_auth_or_payment"
                        break
        finally:
            if own_client and client is not None:
                await client.aclose()

    if items and (stats["wrote"] > 0 or existing_lines != len(items)):
        write_merged_output(output_path, items, merged)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Block-6 extraction over JSONL or gold YAML.")
    parser.add_argument("--input", "-i", type=Path, nargs="+", required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--router-model", default=PINNED_ROUTER_MODEL_ID)
    parser.add_argument("--extractor-model", default=PINNED_EXTRACTOR_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--stratify",
        type=int,
        default=None,
        metavar="N",
        help="Round-robin N items across origin×kind (pytorch/other × pr/issue/hub). "
        "Use this for smokes; --limit takes file order and is all PRs on the holdout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for --stratify.")
    parser.add_argument(
        "--kinds",
        default=_DEFAULT_KINDS,
        help="Comma-separated item kinds to keep: issue, pr, hub. "
        "Default issue,hub — demand-side fields are not defined on pull requests. "
        "Pass issue,pr,hub for a contrast smoke that still includes PRs.",
    )
    parser.add_argument(
        "--from-yaml",
        action="store_true",
        help="Treat inputs as eval/topic_assignment gold YAML even if the suffix is not .yaml",
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Recompute span_errors on --output from stored fields. No LLM calls.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.stratify is not None:
        parser.error("use --limit or --stratify, not both")
    try:
        kinds = parse_kinds(args.kinds)
    except ValueError as exc:
        parser.error(str(exc))
    items = filter_by_kind(collect_items(args.input, from_yaml=args.from_yaml), kinds)
    if args.revalidate:
        stats = revalidate_output(items, args.output)
        print(f"revalidate: rows={stats['rows']} updated={stats['updated']}")
        return 0
    if args.stratify is not None:
        items = stratified_sample(items, args.stratify, seed=args.seed)
    elif args.limit is not None:
        items = items[: args.limit]
    stats = asyncio.run(
        run_extract(
            items,
            args.output,
            router_model_id=args.router_model,
            extractor_model_id=args.extractor_model,
        )
    )
    print(
        f"extract-run: total={stats['total']} skipped={stats['skipped']} "
        f"wrote={stats['wrote']}"
        + (f" stopped={stats['stopped']}" if stats.get("stopped") else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
