import asyncio
import time
from datetime import UTC, datetime

from casefile.clients import ClientBundle, build_clients
from casefile.config import Settings, get_settings
from casefile.engine.citation_checker import check_citations
from casefile.engine.planner import build_plan
from casefile.engine.synthesizer import synthesize
from casefile.engine.validator import validate_evidence
from casefile.models.assessment import AssessmentReport, AssessmentRequest
from casefile.models.evidence import EvidenceBundle, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.predict.time import MaintainerSet
from casefile.predict.topic_hazard import (
    assess_topic_hazard,
    build_topic_provenance,
    topic_hazard_to_dict,
)
from casefile.predict.topic_history import TopicHistory, compute_topic_history
from casefile.retrievers import ALL_RETRIEVERS
from casefile.retrievers.base import RetrievalPlan


class AssessmentEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def run(
        self,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        clients: ClientBundle | None = None,
    ) -> AssessmentReport:
        start = time.monotonic()
        own_client = clients is None
        if clients is None:
            import httpx

            httpx_client = httpx.AsyncClient(timeout=self._settings.http_timeout)
            clients = build_clients(self._settings, httpx_client)
        else:
            httpx_client = None

        try:
            plan = build_plan(request, profile)
            bundle = await self._retrieve(request, profile, plan, clients)
            kept, excluded = validate_evidence(bundle.items, profile, request)
            bundle.items = kept
            bundle.excluded = excluded
            bundle.dedupe_by_url()
            bundle.items = bundle.items[: request.max_evidence]
            bundle.freshness = _bundle_freshness(bundle, profile)
            bundle.open_questions.extend(
                _heuristic_open_questions(bundle, profile, plan.resolved_path)
            )
            bundle.retrieval_stats["duration_sec"] = round(time.monotonic() - start, 2)
            bundle.retrieval_stats["excluded_count"] = len(bundle.excluded)

            _record_vital_fetch_errors(bundle)
            summary: str | None = None
            citation_map: dict[int, str] = {}
            validation_errors: list[str] = []
            activity_forecast = None
            now = datetime.now(UTC)
            history = await _path_topic_history(
                clients, request.repo, plan.resolved_path, now, bundle, profile
            )
            topic_forecast = _topic_forecast(
                bundle=bundle,
                profile=profile,
                path=plan.resolved_path,
                now=now,
                history=history,
            )

            if request.synthesize and clients.llm.available and bundle.items:
                summary, citation_map, llm_questions, id_by_num = await synthesize(
                    request, bundle.items, clients.llm
                )
                for q in llm_questions:
                    if q not in bundle.open_questions:
                        bundle.open_questions.append(q)
                items_by_id = {item.id: item for item in bundle.items}
                validation_errors = check_citations(
                    summary,
                    citation_map,
                    bundle.evidence_ids(),
                    id_by_num=id_by_num,
                    items_by_id=items_by_id,
                )
                if validation_errors:
                    summary = None

            return AssessmentReport(
                request=request,
                evidence=bundle,
                summary=summary,
                citation_map=citation_map,
                validation_errors=validation_errors,
                activity_forecast=activity_forecast,
                topic_forecast=topic_forecast,
            )
        finally:
            if own_client and httpx_client is not None:
                await httpx_client.aclose()

    async def _retrieve(
        self,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        plan: RetrievalPlan,
        clients: ClientBundle,
    ) -> EvidenceBundle:
        bundle = EvidenceBundle()
        active = [r for r in ALL_RETRIEVERS if r.tier <= request.tier]

        async def run_one(retriever) -> list:
            spec = retriever.plan(request, profile, plan)
            if spec is None:
                return []
            try:
                return await retriever.fetch(spec, request, profile, clients)
            except Exception as exc:  # noqa: BLE001 — collect per-retriever failures
                bundle.open_questions.append(f"{retriever.name} failed: {exc}")
                return []

        results = await asyncio.gather(*(run_one(r) for r in active))
        for chunk in results:
            bundle.items.extend(chunk)
        return bundle


async def _path_topic_history(
    clients: ClientBundle,
    repo: str,
    path: str | None,
    now: datetime,
    bundle: EvidenceBundle,
    profile: EcosystemProfile | None,
) -> TopicHistory | None:
    """Fetch rollup inputs only when path-scoped — S2 refuses otherwise."""
    if path is None:
        return None
    maintainers = MaintainerSet(logins=frozenset(), as_of=now)
    synonyms: tuple[str, ...] = ()
    if profile is not None:
        extra: list[str] = []
        for group in profile.synonyms.values():
            extra.extend(group)
        synonyms = tuple(extra)
    try:
        history = await compute_topic_history(
            clients.github,
            repo=repo,
            path=path,
            topic_paths=(path,),
            now=now,
            maintainers=maintainers,
            synonyms=synonyms,
        )
    except Exception as exc:  # noqa: BLE001 — history failure must not abort assess
        note = f"topic_history failed: {exc}"
        if note not in bundle.open_questions:
            bundle.open_questions.append(note)
        return None
    for err in history.fetch_errors:
        note = f"topic_history: {err}"
        if note not in bundle.open_questions:
            bundle.open_questions.append(note)
    return history


def _topic_forecast(
    *,
    bundle: EvidenceBundle,
    profile: EcosystemProfile | None,
    path: str | None,
    now: datetime,
    history: TopicHistory | None,
) -> dict:
    measured = False
    precision_refusal: str | None = (
        "item→topic assignment precision unmeasured for this profile"
    )
    if profile is not None:
        measured, status_reason = profile.assignment_precision_status(as_of=now)
        precision_refusal = None if measured else status_reason

    result = assess_topic_hazard(
        bundle=bundle,
        profile=profile,
        path=path,
        now=now,
        topic_first_seen=history.topic_first_seen if history else None,
        assignment_precision_measured=measured,
        assignment_precision_refusal=precision_refusal,
        n_resolved_items=history.n_resolved_items if history else None,
        demand_recent_per_month=(
            history.demand_recent_per_month if history else None
        ),
        demand_baseline_per_month=(
            history.demand_baseline_per_month if history else None
        ),
        realized_r1_rate_recent=(
            history.realized_r1_rate_recent if history else None
        ),
        realized_r1_rate_baseline=(
            history.realized_r1_rate_baseline if history else None
        ),
        inflow_recent_total=history.inflow_recent_total if history else None,
        inflow_baseline_total=history.inflow_baseline_total if history else None,
        r1_sampled_n_recent=history.r1_sampled_n_recent if history else None,
        r1_sampled_n_baseline=history.r1_sampled_n_baseline if history else None,
        # No trained artifact; C3 keeps score refused.
        trained_artifact_available=False,
        observation_window_days=None,
    )

    ap = profile.assignment_precision if profile is not None else None
    provenance = build_topic_provenance(
        demand_recent_per_month=(
            history.demand_recent_per_month if history else None
        ),
        demand_baseline_per_month=(
            history.demand_baseline_per_month if history else None
        ),
        realized_r1_rate_recent=(
            history.realized_r1_rate_recent if history else None
        ),
        realized_r1_rate_baseline=(
            history.realized_r1_rate_baseline if history else None
        ),
        r1_sampled_n_recent=history.r1_sampled_n_recent if history else 0,
        r1_sampled_n_baseline=history.r1_sampled_n_baseline if history else 0,
        r1_excluded_n=history.r1_excluded_n if history else 0,
        assignment_precision=ap.precision if ap else None,
        assignment_precision_n=ap.n if ap else None,
        assignment_precision_measured_at=(
            ap.measured_at.isoformat() if ap else None
        ),
    )
    return topic_hazard_to_dict(result, provenance=provenance)


def _record_vital_fetch_errors(bundle: EvidenceBundle) -> None:
    for item in bundle.items:
        if item.kind != EvidenceKind.VITAL_SIGNS:
            continue
        for err in item.metadata.get("fetch_errors") or []:
            note = f"vital_signs: {err}"
            if note not in bundle.open_questions:
                bundle.open_questions.append(note)


def _bundle_freshness(bundle: EvidenceBundle, profile: EcosystemProfile | None) -> datetime:
    timestamps = [item.retrieved_at for item in bundle.items]
    if profile is not None:
        timestamps.append(
            datetime.combine(profile.last_verified, datetime.min.time(), tzinfo=UTC)
        )
    if not timestamps:
        return datetime.now(UTC)
    return min(timestamps)


def _heuristic_open_questions(
    bundle: EvidenceBundle,
    profile: EcosystemProfile | None,
    path: str | None = None,
) -> list[str]:
    questions: list[str] = []
    adjacent_kept = [
        item for item in bundle.items if item.kind == EvidenceKind.ADJACENT_PROJECT
    ]
    adjacent_dropped = [
        item for item in bundle.excluded if item.kind == EvidenceKind.ADJACENT_PROJECT
    ]
    if adjacent_kept:
        names = [item.title for item in adjacent_kept]
        questions.append(
            "Whether adjacent libraries or APIs already cover this use case better than extending "
            f"the target feature (see evidence: {', '.join(names[:3])}). Confirm with maintainers before investing."
        )
    for item in adjacent_dropped:
        reason = item.metadata.get("exclusion_reason", "excluded")
        questions.append(
            f"Adjacent project {item.title} was retrieved but excluded ({reason}): {item.url}"
        )
    if (
        profile is not None
        and profile.adjacent_projects
        and not adjacent_kept
        and not adjacent_dropped
    ):
        questions.append(
            "No adjacent-project pages were retrieved — alternatives may exist that this "
            "report does not show."
        )
    pinned_ids = {item.id for item in bundle.items}
    if profile is not None:
        for number in profile.pinned_issues_for_path(path):
            expected = f"issue-{number}"
            if expected not in pinned_ids:
                questions.append(
                    f"Pinned tracking issue #{number} was not retrieved — verify manually on GitHub."
                )
    has_discourse = any(item.kind == EvidenceKind.DISCOURSE_THREAD for item in bundle.items)
    if profile is not None and profile.discourse:
        base = str(profile.discourse.base_url)
        if not has_discourse:
            questions.append(
                f"No dev-discuss threads were retrieved — search {base} manually "
                "(pinned URLs and search may have failed)."
            )
        elif not any(
            item.metadata.get("discourse_search") for item in bundle.items if item.kind == EvidenceKind.DISCOURSE_THREAD
        ):
            questions.append(
                f"Discourse search on {base} returned no extra threads — review search terms or add pinned_threads."
            )
    has_hf = any(item.kind == EvidenceKind.HF_DISCUSSION for item in bundle.items)
    if profile is not None and profile.huggingface and not has_hf:
        repos = ", ".join(r.repo_id for r in profile.huggingface.hub_repos[:3]) or "configured Hub repos"
        questions.append(
            f"No Hugging Face discussions were retrieved for {repos} — "
            "confirm discussions are enabled and CASEFILE_HF_TOKEN if the model is gated."
        )
    return questions
