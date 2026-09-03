"""Web UI background job flow (mocked engine)."""

import time

import pytest

from casefile.models.assessment import AssessmentReport, AssessmentRequest
from casefile.models.evidence import EvidenceBundle, EvidenceItem, EvidenceKind
from casefile.web.app import create_app
from casefile.web.jobs import get_job


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_assess_redirects_to_status_and_completes(client, monkeypatch):
    async def fake_run(self, request, profile, clients=None):
        return AssessmentReport(
            request=request,
            evidence=EvidenceBundle(
                items=[
                    EvidenceItem(
                        id="issue-1",
                        kind=EvidenceKind.ISSUE,
                        title="Example",
                        url="https://github.com/o/r/issues/1",
                        snippet="body",
                        source_retriever="test",
                    )
                ]
            ),
            summary="Conditional summary [1].",
            citation_map={1: "issue-1"},
        )

    monkeypatch.setattr(
        "casefile.engine.orchestrator.AssessmentEngine.run",
        fake_run,
    )

    response = client.post(
        "/assess",
        data={
            "question": "Is it worth contributing?",
            "repo": "o/r",
            "ecosystem": "",
            "tier": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/assess/status/" in response.headers["Location"]

    job_id = response.headers["Location"].rsplit("/", 1)[-1]
    report_html = None
    for _ in range(50):
        record = get_job(job_id)
        assert record is not None
        if record.status == "done":
            report_html = client.get(f"/assess/status/{job_id}").data.decode()
            break
        if record.status == "error":
            pytest.fail(record.error)
        time.sleep(0.05)
    assert report_html is not None
    assert "Assessment report" in report_html
    assert "Conditional summary" in report_html
    assert "issue-1" in report_html or "/issues/1" in report_html
