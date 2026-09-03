# Casefile verification (mandatory)

Every casefile change must pass **automated deep tests** and, when the UI or report layout changes, a **visual check** before you call the work done.

Agents and humans follow the same bar. CI enforces the automated part; the visual checklist is required for UI/report work even when CI is green.

## 1. Automated gate (required always)

From `casefile/`:

```bash
./scripts/verify.sh
```

This runs:

| Step | What it proves |
|------|----------------|
| `pytest -v` | Unit, integration, contract, web smoke, cache, job flow, sample-case YAML, held-out plan checks (`eval/sample_cases.yaml`) |
| `test_live.py` | Skipped unless `CASEFILE_RUN_LIVE=1` |
| Report contract | If `reports/*.md` exist, `validate_reports.py` checks format + golden needles |

Optional live smoke (needs `.env`):

```bash
CASEFILE_RUN_LIVE=1 pytest tests/test_live.py -v
./scripts/run_sample_assessments.sh
python scripts/validate_reports.py
```

## 2. Visual check (required for UI / report layout)

When you change `casefile/web/**`, `render/markdown.py`, or templates:

1. `pip install -e ".[web]"` then start the server (`casefile-web` or `python -m casefile.web.app`). **Restart** after code changes — a long-running process will not pick up edits.
2. Open http://127.0.0.1:5050
3. Confirm:
   - Home: presets, form, footer status
   - `/health`: GitHub + LLM status
   - Pick a sample case card (e.g. **NumPy — numpy.ma (strategy)**, tier 1, synthesis off) → progress page → report with evidence accordion
   - Download .md works
4. Note any UX bugs in the PR or session summary

Browser automation in CI is not required yet; **manual or agent browser review is mandatory** for UI changes.

## 3. Definition of done

- [ ] `./scripts/verify.sh` exit 0
- [ ] UI/report changes: visual checklist above completed
- [ ] Live-affecting changes: `CASEFILE_RUN_LIVE=1` smoke OR documented why skipped
- [ ] No secrets in git (`.env` stays local)

## 4. CI

GitHub Actions workflow `.github/workflows/assessor.yml` runs `./scripts/verify.sh` on pushes touching `casefile/`.
