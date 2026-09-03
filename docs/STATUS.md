# Casefile implementation status (August 2026)

Living snapshot of what is built vs what the Casefile product backlog still tracks as backlog. Checkboxes in `the product backlog` change only when a human confirms acceptance.

## Shipped

| Area | What works |
|------|------------|
| **CLI** | `casefile ping`, `list-profiles`, `assess` with `--tier`, `--ecosystem`, `--no-synthesis`, `--json` |
| **Profiles** | Four YAML files: `pytorch`, `numpy`, `sklearn`, `apertus`. Synonyms, pinned issues, path hints, adjacent projects, discourse / Hub. Re-verified 2026-08-30 after the 90-day stamp expired (day 92). Talk week still needs a second pass: 22 October is 53 days after this stamp. |
| **Retrieval** | GitHub issues/PRs/commits/files; adjacent URL validation plus fetched page text (curator `relevance` is labelled, not quoted; a failed fetch is excluded with a reason, not dropped); Hub discussions; vital signs; dev-discuss pinned + HTML search |
| **Forecast** | DemandÃ—supply topic rollup from realized outcomes (`## Topic trajectory` with provenance). Inflow and R1 samples are filtered with `automatic_assign` before counting. `vitals-logistic-v0` is not printed. Hazard score remains refused until a trained artifact ships. Assignment precision: PyTorch holdout `n=95`, precision 0.91 (recall omitted as unmeasured); Apertus `n=81`, precision 0.88 / recall 0.64 (recall below 0.80 refuses the rollup). Spec: [`docs/PREDICT.md`](PREDICT.md), wiring brief [`docs/PREDICT_QUADRANT.md`](PREDICT_QUADRANT.md) |
| **Extraction** | Block-6 frozen at v3/112 demand-side rows (`pilot_2026-09-02_â€¦`). Fill rates look high; distributions say ~three usable columns (`intent`, binarized `specificity`/`blocking_severity`). `affect` is a constant; `scope` unreliable (Îº 0.19). See [`eval/extraction/README.md`](../eval/extraction/README.md) . |
| **Engine** | Planner â†’ retrievers â†’ validator â†’ optional LLM synthesis â†’ citation checker â†’ markdown report |
| **Hardening** | GitHub search cache (1h TTL), rate-limit retries, citation id normalization, quote gate on fetched snippets |
| **Web UI** | Flask + Bootstrap on `:5050`; async jobs; 6 sample presets from [`eval/sample_cases.yaml`](../eval/sample_cases.yaml) |
| **Tests** | `./scripts/verify.sh` â€” pytest plus report contracts when `reports/*.md` exist |
| **CI** | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the same gate |
| **Verification** | [`VERIFICATION.md`](VERIFICATION.md) â€” mandatory deep tests + visual UI checklist |

## Sample cases (3 repos, 6 scenarios)

| Preset id | Repo | Purpose |
|-----------|------|---------|
| `pytorch-masked` | `pytorch/pytorch` | Reference dogfood; golden issues + dev-discuss |
| `pytorch-nested` | `pytorch/pytorch` | Compare nested/jagged vs masked path |
| `numpy-ma` | `numpy/numpy` | `numpy/ma` â€” `__array_function__` angle |
| `numpy-ma-strategy` | `numpy/numpy` | Shorter strategy wording; tier 1 quick smoke |
| `sklearn-routing` | `scikit-learn/scikit-learn` | SLEP006 / metadata routing |
| `sklearn-generic` | `scikit-learn/scikit-learn` | Template "feature X" question |

Live reports regenerated via [`scripts/run_sample_assessments.sh`](../scripts/run_sample_assessments.sh) â†’ `reports/*.md` (4 primary scenarios; subset of presets).

## Test coverage summary

| Layer | Runs when | Needs `.env` |
|-------|-----------|--------------|
| `./scripts/verify.sh` | Every PR / agent "done" | No |
| Mocked integration (`test_integration`, `test_ecosystems`) | `verify.sh` | No |
| Held-out plan checks (`test_held_out_eval`) | `verify.sh` | No |
| Live smoke (`test_live.py`) | `CASEFILE_RUN_LIVE=1` | Yes |
| Sample report regen | `run_sample_assessments.sh` | Yes |

## Phase mapping (the Casefile product backlog)

| Phase | Status |
|-------|--------|
| **0** â€” validation | **Not done** â€” maintainer interviews, proceed/kill ([`PHASE0_MAINTAINER.md`](PHASE0_MAINTAINER.md)) |
| **0.5** â€” dev setup | **Mostly done** â€” scaffold, ping, profiles; SQLite/LanceDB index deferred |
| **1** â€” CLI MVP | **In progress** â€” core retrieval + web UI built; human checkboxes open |
| **1.5** â€” Tier 2 | **Partial** â€” merged PR search, discourse search; issueâ€“PR linking thin |
| **2** â€” GitHub Action | **Cut from the talk path** (28 Aug, traded for `topic-hazard-v1`). Returns after 22 Oct. App remains Phase 2.5. |
| **3** â€” specialisation | **Partial** â€” four profiles; held-out scaffold (3 cases), not 10-question live eval |

## Docs index

| Doc | Audience |
|-----|----------|
| [`README.md`](../README.md) | Install, commands, profiles |
| [`WEB_UI.md`](WEB_UI.md) | UI routes, presets, UX |
| [`VERIFICATION.md`](VERIFICATION.md) | Mandatory test + visual gate |
| [`DEPLOY.md`](DEPLOY.md) | Gunicorn / hosting |
| [`PHASE0_MAINTAINER.md`](PHASE0_MAINTAINER.md) | Interview script |
| [`docs/samples/`](../docs/samples/) | Example decision write-ups |



