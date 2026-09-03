# Casefile

Evidence-first briefs for OSS contribution decisions. Given a question and a target repository, Casefile retrieves cited evidence (issues, commits, docs, adjacent projects) and optionally synthesizes a short summary. Humans keep the final call.

```bash
# After PyPI publish (Track B W3):
uvx casefile assess -q "…" -r pytorch/pytorch -p torch/masked --no-synthesis
```

**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)  
**Public split steps:** [docs/SPLIT.md](docs/SPLIT.md)

Product backlog and delivery dates live in the private Arraxis planning workspace, not in this tree.

## Install (dev)

```bash
pip install -e ".[dev]"      # CLI + tests
pip install -e ".[dev,web]"  # + Flask UI (casefile-web)
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
casefile-web
# http://127.0.0.1:5050 — form, sample cases, cited report (Bootstrap)
```

**Not sure what to try?** Pick a card on the home page — six scenarios across `pytorch`, `numpy`, and `sklearn` ([`eval/sample_cases.yaml`](eval/sample_cases.yaml)).

See [docs/WEB_UI.md](docs/WEB_UI.md) and [docs/STATUS.md](docs/STATUS.md).

## Commands

```bash
casefile ping
casefile list-profiles

casefile assess \
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
casefile ping
```

## Ecosystem profiles

| Profile | Repo | Use case |
|---------|------|----------|
| `pytorch` | `pytorch/pytorch` | `torch.masked`, `torch.nested`, … |
| `numpy` | `numpy/numpy` | `numpy.ma`, NEPs, missing-data semantics |
| `sklearn` | `scikit-learn/scikit-learn` | SLEPs, metadata routing, estimator API |

```bash
casefile list-profiles
./scripts/run_sample_assessments.sh   # needs .env
python scripts/validate_reports.py
```

## License

Apache-2.0. No CLA.
