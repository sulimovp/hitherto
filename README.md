# Casefile

Evidence-first briefs for OSS contribution decisions. Given a question and a target repository, Casefile retrieves cited evidence (issues, commits, docs, adjacent projects) and optionally synthesizes a short summary. Humans keep the final call.

```bash
pip install hitherto
uvx hitherto assess -q "…" -r pytorch/pytorch -p torch/masked --no-synthesis
```

The PyPI name and CLI are `hitherto`. Python import stays `import casefile`.

**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

Product backlog and delivery dates live in the private Arraxis planning workspace, not in this tree.

## Install (dev)

```bash
pip install -e ".[dev]"      # CLI + tests
pip install -e ".[dev,web]"  # + Flask UI (hitherto-web)
```

## Configure

Copy the template and fill in secrets (never commit `.env`):

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `CASEFILE_GITHUB_TOKEN` | Yes (live runs) | Fine-grained PAT or classic `public_repo` |
| `CASEFILE_OPENAI_API_KEY` | Optional | Synthesis via OpenAI |
| `CASEFILE_ANTHROPIC_API_KEY` | Optional | Synthesis via Anthropic |
| `CASEFILE_LLM_PROVIDER` | Optional | `openai` or `anthropic` (default: `anthropic`) |

## Web UI

```bash
pip install -e ".[web]"
hitherto-web
# http://127.0.0.1:5050 â€” form, sample cases, cited report (Bootstrap)
```

**Not sure what to try?** Pick a card on the home page â€” six scenarios across `pytorch`, `numpy`, and `sklearn` ([`eval/sample_cases.yaml`](eval/sample_cases.yaml)).

See [docs/WEB_UI.md](docs/WEB_UI.md) and [docs/STATUS.md](docs/STATUS.md).

## Commands

```bash
hitherto ping
hitherto list-profiles

hitherto assess \
  --question "Is reviving torch.masked worth an upstream contribution?" \
  --repo pytorch/pytorch \
  --path torch/masked \
  --ecosystem pytorch \
  --tier 2 \
  --output report.md
```

Flags: `--no-synthesis`, `--json`, `--allow-stale-profile`.

## Test (mandatory before claiming done)

```bash
./scripts/verify.sh
```

See [docs/VERIFICATION.md](docs/VERIFICATION.md).

```bash
pytest -v
CASEFILE_RUN_LIVE=1 pytest tests/test_live.py -v
hitherto ping
```

## Ecosystem profiles

| Profile | Repo | Use case |
|---------|------|----------|
| `pytorch` | `pytorch/pytorch` | `torch.masked`, `torch.nested`, â€¦ |
| `numpy` | `numpy/numpy` | `numpy.ma`, NEPs, missing-data semantics |
| `sklearn` | `scikit-learn/scikit-learn` | SLEPs, metadata routing, estimator API |

```bash
hitherto list-profiles
./scripts/run_sample_assessments.sh   # needs .env
python scripts/validate_reports.py
```

## License

Apache-2.0. No CLA.

