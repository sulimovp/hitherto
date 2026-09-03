"""In-memory background jobs for long-running assessments (web UI)."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Callable

from casefile.models.assessment import AssessmentReport

JobRunner = Callable[[], AssessmentReport]


@dataclass
class JobRecord:
    status: str  # pending | done | error
    report: AssessmentReport | None = None
    error: str | None = None


_lock = threading.Lock()
_jobs: dict[str, JobRecord] = {}


def start_job(run_fn: JobRunner) -> str:
    job_id = uuid.uuid4().hex[:12]

    def runner() -> None:
        try:
            report = run_fn()
            with _lock:
                _jobs[job_id] = JobRecord(status="done", report=report)
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _jobs[job_id] = JobRecord(status="error", error=str(exc))

    with _lock:
        _jobs[job_id] = JobRecord(status="pending")
    threading.Thread(target=runner, daemon=True).start()
    return job_id


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _jobs.get(job_id)


def start_async_assessment(coro: Coroutine[Any, Any, AssessmentReport]) -> str:
    return start_job(lambda: asyncio.run(coro))
