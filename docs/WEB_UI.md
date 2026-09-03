# Web UI (Flask + Bootstrap)

Simple product shell around the same `AssessmentEngine` as the CLI.

## UX principles

1. **Evidence over verdict** — summary is labeled “verify citations”; open questions are prominent.
2. **One primary action** — “Run assessment” on a single form; sample cases reduce blank-page anxiety.
3. **Progressive disclosure** — tier / stale profile / synthesis are visible but not overwhelming; full markdown is collapsed.
4. **Trust signals** — footer shows GitHub/LLM config; `/health` for connectivity.
5. **Honest latency** — submit → progress page (3s auto-refresh) → report; copy says 30–90s.

## Sample cases (quick start)

Six curated scenarios live in [`eval/sample_cases.yaml`](../eval/sample_cases.yaml) and appear on the home page as cards. Use them when you are unsure what to try:

| Preset | Repo | Notes |
|--------|------|-------|
| PyTorch — torch.masked | `pytorch/pytorch` | Reference case; tier 2; synthesis on by default |
| PyTorch — torch.nested | `pytorch/pytorch` | Compare nested vs masked investment |
| NumPy — numpy.ma | `numpy/numpy` | `numpy/ma` path; synthesis off by default (retrieval-only) |
| NumPy — numpy.ma (strategy) | `numpy/numpy` | Tier 1 quick smoke |
| scikit-learn — metadata routing | `scikit-learn/scikit-learn` | SLEP006 |
| scikit-learn — generic feature | `scikit-learn/scikit-learn` | Template “feature X” question |

Selecting a card fills the form (question, repo, path, profile, tier, synthesis default). Edit before submitting if needed.

To add or change presets, edit the YAML only — `casefile.web.app` loads it at request time.

## Install and run

```bash
cd casefile
pip install -e ".[dev,web]"
# configure casefile/.env (CASEFILE_GITHUB_TOKEN, optional LLM keys)
casefile-web
# open http://127.0.0.1:5050
```

**Restart the server after code changes** — a long-running process will not pick up Python edits.

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CASEFILE_WEB_HOST` | `127.0.0.1` | Bind address |
| `CASEFILE_WEB_PORT` | `5050` | Port |
| `CASEFILE_FLASK_SECRET` | dev placeholder | Session/flash secret — **set in production** |
| `CASEFILE_WEB_DEBUG` | off | Flask debug (never on public internet) |

## Routes

| Path | Method | Purpose |
|------|--------|---------|
| `/` | GET | Assessment form + sample case cards (`?preset=<id>`) |
| `/assess` | POST | Start background job, redirect to status |
| `/assess/status/<id>` | GET | Poll until report ready (auto-refresh 3s) |
| `/health` | GET | GitHub + LLM ping status |

## Verification

Before claiming UI work done: [`VERIFICATION.md`](VERIFICATION.md) (`./scripts/verify.sh` + visual checklist).

## Future UI (backlog)

- Side-by-side compare two assessments
- Profile editor (YAML) with validation
- Export PDF / share link (read-only report id)
- Persistent job store for multi-worker deploys ([`DEPLOY.md`](DEPLOY.md))
