import pytest

from casefile.engine.citation_checker import check_citations
from casefile.engine.planner import _issue_search_queries, build_plan
from casefile.engine.synthesizer import _parse_synthesis
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceBundle, EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.profiles import load_profile
from casefile.render.markdown import render_markdown


@pytest.fixture
def pytorch_profile(profiles_dir):
    return load_profile(profiles_dir, "pytorch", allow_stale=True)


@pytest.fixture
def profiles_dir():
    from casefile.config import get_settings

    return get_settings().resolved_profiles_dir()


def test_build_plan_torch_masked(pytorch_profile):
    request = AssessmentRequest(
        question="Is reviving torch.masked / MaskedTensor worth an upstream contribution?",
        repo="pytorch/pytorch",
        ecosystem="pytorch",
        tier=2,
    )
    plan = build_plan(request, pytorch_profile)
    assert plan.resolved_path == "torch/masked"
    retriever_names = {s.retriever for s in plan.specs}
    assert "github_issues" in retriever_names
    assert "adjacent_projects" in retriever_names
    assert "github_prs" in retriever_names
    issue_spec = next(s for s in plan.specs if s.retriever == "github_issues")
    assert "repo:pytorch/pytorch" in issue_spec.queries[0]
    assert len(issue_spec.queries) >= 2
    joined = " ".join(issue_spec.queries)
    assert "MaskedTensor" in joined or "masked" in joined
    assert "reviving" not in joined
    assert "worth" not in joined


def test_issue_search_queries_use_profile_terms(pytorch_profile):
    queries = _issue_search_queries(
        "pytorch/pytorch",
        "Is reviving torch.masked / MaskedTensor worth an upstream contribution?",
        pytorch_profile,
        "torch/masked",
    )
    assert len(queries) <= 8
    assert any("89734" in q for q in queries)
    assert any("label:" in q for q in queries)
    assert any("MaskedTensor" in q or "torch.masked" in q for q in queries)
    assert all("is:issue" in q for q in queries)
    assert any(q.endswith("is:issue") for q in queries)


def test_citation_checker_valid():
    errors = check_citations(
        'MaskedTensor is "prototype" [1].',
        {1: "issue-89734"},
        {"issue-89734"},
        id_by_num={1: "issue-89734"},
        items_by_id={
            "issue-89734": EvidenceItem(
                id="issue-89734",
                kind=EvidenceKind.ISSUE,
                title="MaskedTensor docs",
                url="https://github.com/pytorch/pytorch/issues/89734",
                snippet="The module remains a prototype.",
                source_retriever="test",
            )
        },
    )
    assert errors == []


def test_parse_synthesis_normalizes_numeric_issue_ids():
    raw = '{"paragraphs": "Docs gap [1].", "citations": {"1": "89734"}, "open_questions": []}'
    paragraphs, cites, _ = _parse_synthesis(raw, {1: "issue-89734"})
    assert paragraphs == "Docs gap [1]."
    assert cites == {1: "issue-89734"}


def test_parse_synthesis_does_not_fill_from_bare_brackets():
    """Prose-only fallback must leave citations empty so the checker can reject unsourced [n]."""
    paragraphs, cites, _ = _parse_synthesis("Native tool use [1][2].", {1: "a", 2: "b"})
    assert paragraphs == "Native tool use [1][2]."
    assert cites == {}


def test_parse_synthesis_keeps_disagreeing_map_entry():
    raw = (
        '{"paragraphs": "claim [1].", "citations": {"1": "issue-wrong"}, '
        '"open_questions": []}'
    )
    _, cites, _ = _parse_synthesis(raw, {1: "issue-right"})
    assert cites == {1: "issue-wrong"}


def test_parse_synthesis_prose_plus_json():
    raw = (
        "Summary with cite [1].\n\n"
        '{"paragraphs": "Clean summary [1].", "citations": {"1": "issue-1"}, '
        '"open_questions": ["Roadmap unclear"]}'
    )
    paragraphs, cites, open_q = _parse_synthesis(raw, {1: "issue-1"})
    assert paragraphs == "Clean summary [1]."
    assert cites == {1: "issue-1"}
    assert open_q == ["Roadmap unclear"]
    assert "{" not in paragraphs


def test_planner_includes_pinned_issues(pytorch_profile):
    from casefile.models.assessment import AssessmentRequest

    request = AssessmentRequest(
        question="Is reviving torch.masked worth it?",
        repo="pytorch/pytorch",
        path="torch/masked",
    )
    plan = build_plan(request, pytorch_profile)
    issue_spec = next(s for s in plan.specs if s.retriever == "github_issues")
    joined = " ".join(issue_spec.queries)
    assert "is:issue 89734" in joined
    assert "label:" in joined


def test_validator_drops_off_topic_issues(pytorch_profile):
    from casefile.engine.validator import validate_evidence
    from casefile.models.assessment import AssessmentRequest

    request = AssessmentRequest(
        question="torch.masked MaskedTensor",
        repo="pytorch/pytorch",
        path="torch/masked",
    )
    items = [
        EvidenceItem(
            id="issue-1",
            kind=EvidenceKind.ISSUE,
            title="pytorch/ExampleRepo",
            url="https://github.com/pytorch/pytorch/issues/68063",
            snippet="unrelated template repo",
            source_retriever="test",
            relevance_score=0.5,
        ),
        EvidenceItem(
            id="issue-89734",
            kind=EvidenceKind.ISSUE,
            title="MaskedTensor docs tracking",
            url="https://github.com/pytorch/pytorch/issues/89734",
            snippet="torch.masked documentation",
            source_retriever="test",
            relevance_score=0.9,
            metadata={"pinned": True},
        ),
    ]
    kept, excluded = validate_evidence(items, pytorch_profile, request)
    assert len(kept) == 1
    assert kept[0].id == "issue-89734"
    assert len(excluded) == 1
    assert excluded[0].id == "issue-1"
    assert "off-topic" in excluded[0].metadata["exclusion_reason"]


def test_citation_checker_invalid():
    errors = check_citations(
        "Something [1].",
        {1: "missing-id"},
        {"issue-89734"},
    )
    assert errors


def test_citation_checker_rejects_unmapped_refs_in_text():
    """Bare [n] without a citations map entry is an error, not auto-filled."""
    errors = check_citations(
        "See [1] and [99].",
        {1: "issue-89734"},
        {"issue-89734"},
        id_by_num={1: "issue-89734"},
    )
    assert any("99" in e for e in errors)


def test_citation_checker_rejects_map_disagreeing_with_slot():
    """The Apertus failure: [1] in prose with citations[1] pointing at the wrong id."""
    errors = check_citations(
        'Ships "native tool-use" [1].',
        {1: "hf-disc-weights"},
        {"hf-disc-weights", "hf-disc-base-request"},
        id_by_num={1: "hf-disc-base-request", 17: "hf-disc-weights"},
        items_by_id={
            "hf-disc-weights": EvidenceItem(
                id="hf-disc-weights",
                kind=EvidenceKind.HF_DISCUSSION,
                title="Apertus-v1.5-8B",
                url="https://huggingface.co/x/discussions/17",
                snippet="ships native tool-use support",
                source_retriever="test",
            ),
            "hf-disc-base-request": EvidenceItem(
                id="hf-disc-base-request",
                kind=EvidenceKind.HF_DISCUSSION,
                title="Base model request",
                url="https://huggingface.co/x/discussions/1",
                snippet="please release base weights",
                source_retriever="test",
            ),
        },
    )
    assert any("disagrees" in e or "maps to" in e for e in errors)


def test_citation_checker_rejects_quote_not_in_cited_item():
    errors = check_citations(
        'Claim with "native tool use" [1].',
        {1: "issue-1"},
        {"issue-1"},
        id_by_num={1: "issue-1"},
        items_by_id={
            "issue-1": EvidenceItem(
                id="issue-1",
                kind=EvidenceKind.ISSUE,
                title="Unrelated",
                url="https://example.com/1",
                snippet="something else entirely",
                source_retriever="test",
            )
        },
    )
    assert any("not found" in e for e in errors)


def test_citation_checker_shares_quote_across_adjacent_markers():
    item_a = EvidenceItem(
        id="a",
        kind=EvidenceKind.ISSUE,
        title="One",
        url="https://example.com/1",
        snippet="the same span appears here",
        source_retriever="test",
    )
    item_b = EvidenceItem(
        id="b",
        kind=EvidenceKind.ISSUE,
        title="Two",
        url="https://example.com/2",
        snippet="the same span appears here too",
        source_retriever="test",
    )
    errors = check_citations(
        'Both pages state "the same span" [1][2].',
        {1: "a", 2: "b"},
        {"a", "b"},
        id_by_num={1: "a", 2: "b"},
        items_by_id={"a": item_a, "b": item_b},
    )
    assert errors == []


def test_citation_checker_ignores_curator_note_in_metadata():
    item = EvidenceItem(
        id="adj-1",
        kind=EvidenceKind.ADJACENT_PROJECT,
        title="Apertus-v1.5-8B",
        url="https://huggingface.co/swiss-ai/Apertus-v1.5-8B",
        snippet="Model card fetched from the Hub.",
        source_retriever="adjacent_projects",
        metadata={
            "curator_note": (
                "Primary 1.5 weights; documents native tool use and optional thinking mode."
            )
        },
    )
    errors = check_citations(
        'The release "documents native tool use and optional thinking mode." [1]',
        {1: "adj-1"},
        {"adj-1"},
        id_by_num={1: "adj-1"},
        items_by_id={"adj-1": item},
    )
    assert any("not found" in e for e in errors)


def test_render_markdown_minimal():
    request = AssessmentRequest(question="Test?", repo="o/r")
    item = EvidenceItem(
        id="issue-1",
        kind=EvidenceKind.ISSUE,
        title="Example",
        url="https://github.com/o/r/issues/1",
        snippet="snippet",
        source_retriever="test",
    )
    dropped = EvidenceItem(
        id="issue-2",
        kind=EvidenceKind.ISSUE,
        title="Off topic",
        url="https://github.com/o/r/issues/2",
        snippet="eos",
        source_retriever="test",
        metadata={"exclusion_reason": "off-topic: no token match"},
    )
    from casefile.models.assessment import AssessmentReport

    report = AssessmentReport(
        request=request,
        evidence=EvidenceBundle(items=[item], excluded=[dropped]),
        activity_forecast={
            "model_id": "vitals-logistic-v0",
            "probability": 0.649,
            "label": "likely_active",
        },
    )
    md = render_markdown(report)
    assert "# Assessment: Test?" in md
    assert "issue-1" in md
    assert "Retrieved but excluded" in md
    assert "Off topic" in md
    assert "## Activity forecast" not in md
    assert "vitals-logistic-v0" not in md


def test_excluded_adjacent_renders_with_reason(pytorch_profile):
    from casefile.engine.orchestrator import _heuristic_open_questions
    from casefile.models.assessment import AssessmentReport

    dropped = EvidenceItem(
        id="adjacent-nestedtensor",
        kind=EvidenceKind.ADJACENT_PROJECT,
        title="NestedTensor",
        url="https://pytorch.org/docs/stable/nested.html",
        snippet="Page could not be fetched.",
        source_retriever="adjacent_projects",
        metadata={"exclusion_reason": "adjacent page fetch failed"},
    )
    bundle = EvidenceBundle(items=[], excluded=[dropped])
    questions = _heuristic_open_questions(bundle, pytorch_profile)
    assert any("NestedTensor" in q and "excluded" in q for q in questions)
    md = render_markdown(
        AssessmentReport(
            request=AssessmentRequest(question="masked?", repo="pytorch/pytorch"),
            evidence=bundle,
        )
    )
    assert "Retrieved but excluded" in md
    assert "NestedTensor" in md
    assert "adjacent page fetch failed" in md


def test_validator_excludes_project_name_only_issues(pytorch_profile):
    """Apertus failure mode: 'EOS token' body mentions the project and truncated 'model'."""
    from casefile.engine.validator import validate_evidence
    from casefile.models.assessment import AssessmentRequest
    from casefile.profiles import load_profile
    from casefile.config import get_settings

    profile = load_profile(get_settings().resolved_profiles_dir(), "apertus", allow_stale=True)
    request = AssessmentRequest(
        question=(
            "Does swiss-ai/apertus-format still earn its place now that 1.5 ships "
            "native tool use and a thinking mode?"
        ),
        repo="swiss-ai/apertus-format",
        ecosystem="apertus",
    )
    items = [
        EvidenceItem(
            id="issue-1",
            kind=EvidenceKind.ISSUE,
            title="EOS token",
            url="https://github.com/swiss-ai/apertus-format/issues/1",
            snippet=(
                "fine-tuning Apertus end-of-sequence token. "
                "found a commit in the mode"  # truncated 'model'
            ),
            source_retriever="github_issues",
            relevance_score=0.25,
        ),
    ]
    kept, excluded = validate_evidence(items, profile, request)
    assert kept == []
    assert len(excluded) == 1
    assert "off-topic" in excluded[0].metadata["exclusion_reason"]


def test_profile_expanded_terms(pytorch_profile):
    terms = pytorch_profile.expanded_terms("work on masked tensor support")
    assert "MaskedTensor" in terms
