# Predict â€” topic trajectory from issue-level survival

Written 2026-08-24. Mechanism spec for topic-hazard prediction.

This exists because the current `vitals-logistic-v0` is not a model â€” it is nine hand-set constants over eight features, three of which are broken (project notes, 2026-08-23 Ã—3). Calling it ML on a stage that spends ten minutes attacking confident unsourced verdicts is the exact failure the talk is built to argue against.

---

## 1. What the Apertus run actually produced

Worth being precise, because it constrains everything below.

| Layer | Mechanism | Ran on Apertus? |
|---|---|---|
| Retrieval | Hub discussions API, adjacent projects, repo files | Yes |
| Ranking | `_relevance_score` â€” rule-based synonym + keyword + engagement | Yes |
| Synthesis | `gpt-oss-120b:fastest` via HF router | Yes |
| Activity forecast | Hand-set logistic on eight vital-signs features | **No** â€” `path_hints: {}`, no `--path`, vitals never fired |

So the Apertus result is retrieval + ranking + prose. No prediction happened, and the report is correct not to contain any.

Three properties of that evidence shape matter for the model design:

**It is tiny.** Seven threads on `Apertus-v1.5-8B`, thirty-three on the 1.0-era `Apertus-8B-Instruct-2509`. Any per-topic statistic over seven items has a standard error wider than the effect you are trying to measure.

**It is young.** The 1.5 weights are one month old. There is no history to reconstruct a point-in-time snapshot from, and no elapsed outcome window to label.

**The answer came from one artifact, not from an aggregate.** The redundancy question turned on draft PR [#5](https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5) and the `transformers` doc page still saying "Coming soon". No average over the thread set would have surfaced that; the fix logged in project notes was to *rank* better, not to *score* harder.

That last point is the strongest constraint on this whole design and it is stated up front so it is not lost: **for young or thin ecosystems, rank-and-surface beats score-and-aggregate.** The model below is for PyTorch-shaped corpora with years of resolved history. Apertus is where it refuses, and the refusal is a better slide than a number would be.

---

## 2. Verdict on the proposed design

The proposal has three parts. One is right, two need reformulating before they will hold.

### 2.1 LLM as feature extractor, small model on top â€” right, with three corrections

This is the correct architecture and it is worth saying why, because it is the part that survives contact with the talk's own thesis: the LLM converts unstructured text into a small number of schema-constrained fields; a tabular model that can be calibrated, backtested and traced does the prediction. The LLM never issues the verdict. That is the same separation the citation checker enforces on prose.

Three corrections:

**Tonality is the weakest field on the list.** Sentiment on issue text is confounded with issue *type* â€” bug reports read as negative, feature requests as positive â€” and it is what everybody reaches for first. The fields that will actually carry signal are speech-act and stance: what kind of request is this, does it contain a reproducer, does a maintainer reply and what does that reply commit to. `maintainer_stance âˆˆ {none, acknowledged, planned, deferred, declined, needs_info}` will almost certainly dominate every other extracted field. Keep one affect field, at ordinal 0â€“2, and expect it to rank low.

**A drifting extractor destroys an out-of-time backtest.** If the extraction model or prompt changes between building the training rows and building the test rows, the temporal split measures extractor drift and reports it as signal. The extractor version is a pinned string, stored as a column on every extracted row, and any change to model, prompt or schema forces full re-extraction. `gpt-oss-120b:fastest` behind an HF router alias is *not* a pin â€” the alias can move.

Pinned 2026-08-30 (last Pro day; Hub lookup in `docs/hf_snapshot/`):

- Hub revision of `openai/gpt-oss-120b`: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (`lastModified` 2025-08-26).
- Router id: `openai/gpt-oss-120b:groq` (groq was live on the catalog snapshot; `:fastest` is routing, not a model). The router does not take `@sha`.
- `extractor_version` model id: `openai/gpt-oss-120b:groq@b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (`predict/extractor.py` `PINNED_EXTRACTOR_MODEL_ID`).

`CASEFILE_LLM_MODEL` must be the router id. Any extraction row made against `:fastest` is scrap. The runner is `python -m casefile.predict.extract_run`.

**Reaction counts are salvageable, but only via the right endpoint.** `GET /repos/{o}/{r}/issues/{n}/reactions` returns one object per reaction carrying its own `created_at` â€” verified against the GitHub REST docs on 2026-08-24 â€” so reaction counts *are* reconstructable as of time *T* by filtering `reaction.created_at <= T`. The aggregate `reactions` block on the issue object is current-state and is a leak. Use the list endpoint, never the summary count. This is a real find: it removes the reason to drop the feature.

### 2.2 "Probability the issue is closed within X days" â€” reformulate the target

Closure is confounded in exactly the direction that breaks the product.

An issue closes for two very different reasons. It is *substantively resolved* â€” a merged PR touches the code, a maintainer answers the question, a design lands. Or it is *administratively closed* â€” stale bot, duplicate, wontfix, no-response, locked. These have opposite meanings for topic health, and dying modules generate the second kind at a high rate: backlogs get swept, stale bots are enabled precisely when nobody is triaging. A model trained on "closed" will learn that mass-closure predicts closure, score a dying module as healthy, and be most wrong exactly where the product's value is.

This is the same class of error as the mechanical-commit contamination the strategy note already flags on the positive class. Same fix, different table.

Replace the binary target with **competing risks**:

- **R1 â€” substantive resolution.** Closed with a linked merged PR or commit touching the topic's paths; or closed by a maintainer comment that the extractor labels as an answer; or, for Hub, a merged PR on the repo.
- **R2 â€” administrative closure.** Stale/duplicate/wontfix/invalid label at close, close by bot account, close with no linked code and no maintainer answer. Repos without `fixes #N` discipline may misclassify genuine fixes as R2 â€” treat that default as a **flag** (`unlabeled_closed_as_r2`, default on) and report false-R2 rate in the W3 hand-check.
- **R3 â€” censored.** Still open at the end of the observation window.

The quantity to predict is the hazard of R1. R2 is a competing event, not a positive, and not a row to delete â€” deleting it biases the R1 estimate upward on precisely the dying topics.

### 2.3 "Accumulate probabilities per topic" â€” reformulate the rollup

Two problems, one technical and one about what the product is for.

**Do not average predictions where you have outcomes.** For any issue whose window has elapsed you know what happened. Averaging model probabilities over those items produces a smoothed function of the features, not an estimate of topic health, and it inherits every model error without inheriting any of the data. The model earns its place only on the items that have *not* resolved yet â€” the censored ones. A survival estimator does this composition for you correctly: realized outcomes where they exist, modelled hazard where they do not.

**A scalar "perspective vs dying" score throws away the distinction Casefile exists to make.** The product question is *should this feature exist upstream*, and the answer is not "the topic is alive". Split the rollup on two axes:

- **Demand** â€” inflow of substantive items on the topic (support questions excluded via the extracted `intent` field), 6-month rate against the preceding 18 months (24-month lookback, non-overlapping). Compare Poisson rate intervals, not a bare ratio; refuse when a window is below a minimum count.
- **Supply** â€” hazard of R1 resolution, summarised as restricted mean time to substantive resolution at 180 days (RMST, which is defined under censoring), and its trend.

|  | Supply rising | Supply falling |
|---|---|---|
| **Demand rising** | Thriving â€” upstream is on it | **Gap â€” contribute here** |
| **Demand falling** | Maturing / solved | Dying |

The top-right cell is the entire product. `torch/masked` sits in it: demand persists, resolution capacity fell away after 2022, and NestedTensor absorbed the attention. A "dying topic" detector tells you to avoid that module. A gap detector tells you it is the best contribution target in the repo. Same data, inverted conclusion, and only the second one is worth putting on a slide.

So: the proposal's instinct is right, and stating it as a two-axis rollup rather than an accumulated probability is what makes it a product rather than a statistic.

---

## 3. Unit of prediction

Two units, chained.

**Item level** â€” a GitHub issue or Hub discussion, observed weekly from creation until R1, R2, or censoring, capped at 26 weeks. This is where the model lives.

**Topic level** â€” a `(repo, topic, T)` triple. Topics come from the profile: path prefixes for GitHub (`torch/masked`), synonym clusters for Hub (`apertus format`, `tool use`). This is where the report speaks.

Itemâ†’topic assignment is a source of error and must be measured, not assumed. Assign by path mentions in title and body, paths touched by linked PRs, and profile synonym hits. **Verify: hand-label 100 issues across three repos, report precision and recall of the assignment, and put both numbers in the report footer.** If assignment precision is below ~0.8 the topic rollup is measuring the wrong issues and nothing downstream is trustworthy.

---

## 4. Model

**Discrete-time competing-risks hazard, weekly periods.** Expand each item into one row per week alive. Each row carries `exposure_days` (â‰¤7; partial final weeks clip at `observation_end`). Train **two** hazard models on the same person-period table â€” `h1_j` for R1 in week j and `h2_j` for R2 in week j â€” or one multinomial head with three outcomes (R1 / R2 / still-at-risk). R2 is a competing event, not censoring; do not compose with `1 - Î (1 - h1_j)` alone, which treats administrative closure as if the issue could still resolve later.

The cumulative incidence of substantive resolution by week k is:

`CIFâ‚(k) = Î£_{jâ‰¤k} h1_j Â· Î _{i<j}(1 âˆ’ h1_i âˆ’ h2_i)`

That is the quantity to report as â€œP(resolved by week k)â€. RMST and trend summaries derive from this curve, not from a Kaplanâ€“Meier complement that ignores R2.

**Training contract.** Every person-period row carries `exposure_days` (â‰¤ 7; the final week of a censored item clips at `observation_end`). It must reach the model as a log-exposure offset â€” `log(exposure_days / 7)` via LightGBM `init_score` â€” or as a row weight of `exposure_days / 7`. A clipped week trained as a full week reintroduces the downward hazard bias that clipping removes, and the bias falls hardest on the most recent items, which are the ones a report is actually about. `to_training_row()` in `predict/person_period.py` is the canonical emitter; W3 uses it rather than reading the dataclass directly.

Why this shape:

- It answers "closed within X days" for *any* X from one model, which is what the original proposal wanted, without training a separate classifier per horizon.
- It handles censoring correctly, which matters because the freshest and most decision-relevant items are always the censored ones. the strategy note already reached this conclusion for the module-level framing; it applies unchanged here.
- It is a plain binary classifier on an expanded table, so LightGBM works out of the box, no `lifelines` dependency, and the tree dump â†’ pure-Python scorer export path stays intact.

Row count is manageable: ~40k issues Ã— ~8 weeks mean survival â‰ˆ 320k rows.

**Baselines, evaluated on identical folds** â€” without these "the model beat the baseline" is unfalsifiable:

1. Base rate per repo (the null).
2. The deterministic vital-signs decision rule, given an explicit threshold.
3. Discrete-time logistic on Blocks 1â€“4 only, i.e. no LLM features.

Baseline 3 is the one that matters. **The with/without-Block-6 ablation is the experiment that decides whether LLM feature extraction earns its cost**, and reporting a null result there is a good outcome, not a failure.

**Calibration is the metric.** Report time-dependent Brier score and integrated Brier, calibration slope at 30/90/180 days, and a per-fold reliability diagram. C-index may be reported but is rank-only and invariant to calibration; it does not decide anything. A number that goes next to a citation must be calibrated or it is a verdict with extra steps.

---

## 5. Feature blocks and leak audit

the strategy note requires one line per feature stating how it is reconstructed at *T* and which field could contaminate it. That table is the deliverable, not an appendix â€” it is what makes the backtest believable.

The scope cut that makes this affordable: **for item-level features, the GitHub REST API is already point-in-time.** `GET /repos/{o}/{r}/issues/{n}/timeline` returns every event with its own `created_at`, and the reactions list endpoint does the same (both verified 2026-08-24). Filtering both by `created_at <= T` reconstructs thread state at *T* without GH Archive. GH Archive is still needed for repo-level history and author priors â€” and is deferred out of v1 on that basis.

### Block 1 â€” item intrinsic (at creation)

| Feature | Reconstruction at *T* | Contamination risk |
|---|---|---|
| `age_days` | `T - created_at` | none |
| `body_len`, `title_len` | issue body at creation | edits are not versioned via API â€” accept, note |
| `has_code_block`, `has_traceback`, `has_version_info` | regex on body | same edit caveat |
| `is_pull_request` | issue object | none |
| `n_linked_refs` | timeline `cross-referenced` events â‰¤ *T* | **use timeline, not current refs** |

### Block 2 â€” author, point-in-time

| Feature | Reconstruction at *T* | Contamination risk |
|---|---|---|
| `author_prior_issues` | search `author:X created:<T` | rate-limit heavy; cache per author-quarter |
| `author_prior_merged_prs` | same, `is:pr is:merged` | as above |
| `author_is_maintainer_at_T` | CODEOWNERS blob at the commit that was HEAD at *T* | **reading CODEOWNERS from `main` today is a leak** |
| `author_is_bot` | login suffix + curated list | bot lists change; pin the list version |
| `author_account_age` | user `created_at` | none |

### Block 3 â€” engagement, strictly â‰¤ *T*

| Feature | Reconstruction at *T* | Contamination risk |
|---|---|---|
| `n_comments_le_T` | timeline `commented` events â‰¤ *T* | issue `comments` count is current-state â€” **never use it** |
| `n_participants_le_T` | distinct actors in timeline â‰¤ *T* | as above |
| `maintainer_replied_le_T` | intersect participants with `MaintainerSet(logins, as_of)` where `as_of <= T` | double leak if either side is read as-of-now; bare `set[str]` is not allowed in code |
| `hours_to_first_maintainer_reply` | first such event | undefined if none â€” encode as missing, not 0 |
| `reactions_{+1,heart,eyes,-1}_le_T` | reactions list endpoint, filter `created_at <= T` | the summary `reactions` block is current-state â€” **leak** |
| `label_set_at_T` | timeline `labeled`/`unlabeled` replay | current labels are the single most tempting leak in the dataset |

### Block 4 â€” topic context at *T*

| Feature | Reconstruction at *T* | Contamination risk |
|---|---|---|
| `commits_topic_3m/6m/12m` | `git log --before=T -- <paths>` on a local clone | **log1p, not raw** â€” raw counts are what saturate v0 to p=1.0 |
| `distinct_committers_12m` | same | none |
| `top_committer_active_6m` | same | none |
| `open_backlog_size_at_T` | replay open/close events | current open count is a leak |
| `median_backlog_age_at_T` | same | same |
| `inflow_3m / inflow_12m` | issue `created_at` histogram before *T* | none â€” this is the demand axis |
| `realized_R1_rate_prior_12m` | outcomes of items closed *before T* | safe by construction; do not let the window cross *T* |
| `has_codeowners_at_T` | blob at HEAD-as-of-*T* | as Block 2 |
| `topic_in_release_notes_6m` | releases with `published_at < T` | none |

`closure_rate` from the current implementation is **deleted**, not fixed. It is the ratio of two independently capped 30-item result sets and equals 0.5 on any repo with more than thirty of each (project notes, 2026-08-23). `realized_R1_rate_prior_12m` is its honest replacement.

### Block 5 â€” competitive displacement

The one feature the strategy note says is worth building carefully, and the only part of this design that is not in the repo-level literature.

| Feature | Reconstruction at *T* | Contamination risk |
|---|---|---|
| `adjacent_commit_trend` | same git log, adjacent paths from profile | profile `adjacent_projects` must be pinned per snapshot |
| `displacement_score` | adjacent rising Ã— self falling | none |
| `inflow_ratio_self_vs_adjacent` | issue histograms | none |
| `names_alternative_rate` | Block 6 field, aggregated over topic | extractor-dependent |

### Block 6 â€” LLM-extracted, from thread state â‰¤ *T*

Schema-constrained, temperature 0, one JSON object per item. Every field ordinal or categorical â€” no free text reaches the model.

| Field | Type | Note |
|---|---|---|
| `intent` | bug / feature_request / support / docs / design_proposal / integration_report / other | gates the demand axis â€” support noise is excluded from inflow |
| `specificity` | 0â€“3 | vague â†” reproducer with expected/actual |
| `proposed_solution_present` | bool | |
| `patch_offered` | bool | strong positive in most corpora |
| `blocking_severity` | 0â€“3 | curiosity â†” blocking production |
| `affect` | 0â€“2 | the "tonality" field; expect it to rank low |
| `maintainer_stance` | none / acknowledged / planned / deferred / declined / needs_info | from maintainer comments â‰¤ *T* only; likely the top extracted feature |
| `scope` | one_line_fix / contained / cross_cutting / requires_design | |
| `names_alternative` | bool + string | feeds Block 5 |
| `evidence_span` | verbatim â‰¤15-word quote | quotable fields only |
| `evidence_anchor` | closed rubric cell | judged fields: `specificity`, `blocking_severity`, `affect`, `scope` |

Rules that make this survive the backtest:

- **Input is reconstructed, not current.** Body plus comments with `created_at <= T`. Feeding today's thread is the most catastrophic leak available â€” a comment saying "fixed in #4471" predicts R1 perfectly and teaches the model nothing. The 2026-08-31 pilots did not do this: `extract_run` hashes title+body from the assignment gold YAML, which has no comment list, so `maintainer_stance` is structurally empty until the runner reconstructs comments.
- **`extractor_version` is a stored column.** Model id, prompt hash, schema version. Changing any forces re-extraction of the whole corpus. Current pin: Â§2.1 / `docs/hf_snapshot/extractor_pin.json`. Schema `block6-v3` (2026-08-31) â€” do not resume a v2 JSONL with the v3 runner and call it one study.
- **Demand-side fields are defined on issues and Hub questions, not pull requests.** Inflow and the person-period table are issues. A PR already carries a patch: `patch_offered` and `proposed_solution_present` are true by construction, and `blocking_severity` scores the defect being fixed (the `--limit 10` v3 smoke was ten PRs; #137890 is that confound). Assignment-gold PRs stay for the assigner eval; they do not enter the Block 6 corpus. `extract_run --kinds` defaults to `issue,hub`. `--stratify` draws across originÃ—kind so a smoke cannot repeat the prefix-slice mistake.
- **Extract once per (item, quarter), not per weekly row.** Thread content is near-static between comments; weekly re-extraction multiplies cost by eight for no signal.
- **Evidence is split.** Presence on quotable fields (`intent`, `proposed_solution_present=true`, `patch_offered=true`, a real `maintainer_stance`, `names_alternative=true`, `names_alternative_text`) still needs a verbatim â‰¤15-word quote. Absence (`false`, `maintainer_stance=none`) does not: there is nothing to quote. Judged fields (`specificity`, `blocking_severity`, `affect`, `scope`) need a rubric cell from the closed list in `predict/extractor.py`. v2 required a quote for every non-null field and emptied the ordinals (project notes 2026-08-31). v3 quote-on-false (fixed 2026-09-02) was the same mistake on booleans: Gemma filled `false` and the validator discarded 18/20 rows. The 112-row demand-side v3 pilot fills judged ordinals on ~94% of parsed rows; `maintainer_stance` stays ~7% because the gold YAML has no comments. #192516 is `title_only` / `nice_to_have` rather than a quoted title standing in for severity. #137890 (a test-only memory leak) scored `blocks_production`.
- **Measure the extractor's own reliability before trusting the features.** Re-run extraction on 200 items with a second model and report agreement (Krippendorff's Î± for ordinals, Cohen's Îº for categoricals). A field below Î± â‰ˆ 0.6 is noise wearing a schema and should be dropped. **Verify: agreement table published in the README alongside the calibration curve.**
  Not yet bought at n=200. The 2026-08-31 v2 smoke had n=0 on the ordinals. After the
  2026-09-02 quote-on-false fix, the v3 demand draw (`agreement_2026-09-01_gpt_oss_x_gemma_v3.json`)
  is overlap 20, valid 19: `intent` Îº 0.77 (n=17), `specificity` Î± 0.81 (n=19),
  `blocking_severity` Î± 0.95 (n=18), `scope` Îº 0.19 (n=17). `affect` is 0 on almost every
  row, so Î± is 0 despite 18/19 exact matches. `maintainer_stance` stays n=1 until comments
  exist. Gemma remains the second extractor. Do not spend 200 items until `scope` is kept
  or dropped on purpose, and not until comments â‰¤ T exist if stance stays in the schema.
  **Frozen 2026-09-03 at the 112-row demand pilot:** distributions show `affect` is a
  constant and the ordinals are near-binary (project notes). Buy neither Gemma-200 nor
  comment reconstruction before Track B (13 Sep) and Track A sends.

Cost sanity: ~40k items Ã— ~1.5k tokens in, ~300 out, extracted once. On a small hosted model that is tens of dollars, not thousands. It fits.

---

## 6. Evaluation protocol

Unchanged from the strategy note â€” it was right â€” with two additions for the survival framing.

- **Walk-forward, expanding window**, 4â€“6 folds. One split gives one number and no variance, and drift matters more than the pooled score.
- **Purge one full horizon.** Test snapshots begin at least 180 days after the last training snapshot, so training labels have fully resolved before test features are drawn.
- **Two headline numbers.** Temporal-only (re-scoring a module you have seen â€” the deployment case) and temporal + grouped-by-repo (a module never seen â€” the honest headline). Report both; they answer different questions.
- **New: competing-risk-aware metrics.** Brier and calibration computed against the cumulative incidence of R1, not against "closed". Scoring R1 predictions against a closed/not-closed ground truth silently rewards the confound this design exists to remove.
- **New: extractor-swap robustness.** Re-run the Block-6 ablation with the second extractor's features. If the gain from Block 6 only exists for one extractor, it is not a finding.

---

## 7. Refusal rules

Two gates. The rollup is descriptive (realized inflow Ã— realized R1 rates). The score is predictive (CIF over ~180 days) and inherits every rollup refusal.

### Rollup refusals (descriptive)

| Condition | Action |
|---|---|
| Any retrieval error in the feature set | refuse â€” absence of evidence and failed retrieval are different states (project notes, 2026-08-23) |
| No path-scoped topic (`--path` required) | refuse |
| Itemâ†’topic assignment precision unmeasured for this profile | refuse |
| `n_resolved_items` unknown after issue-search `total_count` | refuse |
| `n_resolved_items` on topic < 12 | refuse the topic rollup |
| Topic first appeared < 12 months before *T* | refuse â€” path history moves, and `torch/masked` did not exist before Nov 2022 |
| Hub items present and issue count is zero | refuse â€” Hub items are inference-only in v1 |
| Inflow rates (`inflow_3m` / `inflow_12m`) unknown | refuse â€” cannot place topic on the demand axis |
| Realized R1 rates (prior 12m / recent) unknown | refuse â€” cannot place topic on the supply axis |
| Demand and supply both flat | refuse â€” no trajectory to report (`Quadrant.UNKNOWN` must not render) |

### Score refusals (predictive, inherits all rollup refusals)

| Condition | Action |
|---|---|
| All rollup refusals above | refuse score as well |
| Trained topic-hazard artifact not shipped | refuse score (v0 product is the refuse path) |
| `extractor_version` differs from the trained artifact's | refuse Block 6 / score |
| Observation window < 2 Ã— horizon (360d for a 180d horizon), or unknown | refuse â€” implemented |

**Apertus hits shared refusals.** Seven threads on 1.5, no path, no elapsed window. It refuses rollup and score, on stage, with both reasons printed. the strategy note already argued that refusing to score Apertus is a better demo than scoring it, and this makes the refusal mechanical rather than a judgement call.

---

## 8. Packaging

No change to the plan in the strategy note: dump the trees to text, walk them in ~60 lines of pure Python, no `lightgbm`, `onnxruntime` or NumPy in the core install. Training lives in a separate repo that is never in the wheel. Model artifacts ship as Release assets, cached under `~/.cache/casefile/models/`, version pinned in the profile YAML so a stale model is as visible as a stale profile.

The LLM extractor is a *runtime* dependency for live scoring, which is new and needs a decision: either the extracted fields are computed at assess time (costs a call per item, needs a key â€” breaks the keyless demo path Track B W3 is protecting), or the Blocks 1â€“4 model is the shipped default and Block 6 is opt-in behind `--extract`. **Default to the second.** The keyless `uvx casefile assess` line is worth more to the talk than a few points of Brier.

---

## 9. What fits before 22 October, and what does not

Straight version first: **this reopens a decision that was closed.** the delivery roadmap Gate 0 accepted "vital signs before model" and put prediction after the talk. Eight weeks remain, Track B (PyPI) is the hard dependency and is not done, and Tracks D (slides, rehearsal) and A (maintainer conversations) are both calendar-bound. Four weeks of modelling has to come out of something. The only honest candidates are Track E (already optional, already deferred at W2) and the W6 GitHub Action. Cutting slides or rehearsal to build a model is how a talk fails.

If that trade is acceptable, this is the cut that is genuinely buildable and backtestable:

**In scope for v1 (`topic-hazard-v1`)**

- One corpus: 30â€“50 large Python repos with real module structure. Not "a few hundred".
- REST-only point-in-time reconstruction (timeline + reactions endpoints). **No GH Archive.** This is the single biggest saving and it is what makes the deadline plausible.
- Blocks 1â€“4 fully; Block 5 in reduced form (git-log trends only, no LLM-derived alternative mentions); Block 6 on a subsample, for the ablation.
- Discrete-time hazard, weekly, 26-week cap, R1 vs R2 vs censored.
- Walk-forward, 4 folds, purged. Both headline numbers.
- Refusal rules wired in, including the Apertus path.

**Out of scope, explicitly**

- GH Archive ingestion.
- Author priors at full fidelity (Block 2 becomes best-effort; rate limits, not correctness, are the reason â€” say so).
- Hub items in training. Hub discussions score at inference only, and the report must say the model was not trained on them.
- Anything at 70B+ or a neural sequence model.

**Week shape.** Do not use this table as the calendar. Modelling is 14 Sep â€“ 4 Oct with a 5â€“11 Oct buffer; Track B and model-independent slides finish 13 Sep. See the delivery roadmap Â§"Decision 2026-08-28".

| Window | Work | Verify |
|---|---|---|
| to 13 Sep | Track B + slides that do not need the model | `uvx casefile --help`; deck runs without a hazard number |
| 14 Sep â€“ 4 Oct | Corpus, labels, person-period, baselines, hazard, ablation | Kill criterion in writing Saturday 4 Oct |
| 5 â€“ 11 Oct | Buffer: rollup wiring / export, or empty | A slipping model stops here, not in rehearsal week |
| 12 â€“ 18 Oct | Live `torch/masked` quadrant, reports, rehearsal | Frozen Saturday 18 Oct |

**Kill criterion, evaluated Saturday 4 October in writing.** If the model does not beat the vital-signs baseline on Brier at 90 days on the grouped-by-repo folds, ship the deterministic scorecard and say on stage that the model did not earn its place. That is a *better* talk than a marginal model â€” it is the same argument the whole session makes, applied to your own work, and it costs nothing to say. Pre-committing to it now is what makes it sayable in October. Do not learn this after spending the slack and with slides unwritten.

**What the talk must not claim, in any version.**

- Not "we predict whether a feature will survive". It is a resolution-hazard forecast on a topic, over 180 days, with an interval.
- Not that the model is novel because prediction is novel. It is not â€” repo-level abandonment prediction is a published area with production tooling. The claim is *module and topic level*, which the literature explicitly does not cover, plus competing risks and the displacement feature.
- Not that it ran on Apertus. It refuses on Apertus, deliberately, and that is the slide.
- No number without its interval and its refusal rule visible next to it.

---

## 10. Open questions

- ~~Is the four-week trade against Track E and the W6 Action actually acceptable?~~ **Decided 2026-08-28: build the model, cut the W6 Action.** Schedule, tripwires and the pre-committed kill gate are in the delivery roadmap Â§"Decision 2026-08-28". The Gate checkbox there is ticked.
- Does itemâ†’topic assignment reach usable precision on `torch/masked`? If not, v1 is a per-issue tool and the topic rollup waits.
- Body edits are not retrievable per-version through the API. How much does that contaminate Block 1, and is it worth measuring on a sample?
- The affect/tonality field: keep it for the ablation even though it is expected to rank low, or drop it and save the schema complexity?
- Prior art on issue resolution-time prediction is well covered in the MSR literature and has **not** been surveyed for this file. Do that **before 14 Sep** â€” the strategy note Â§"Prior art" surveyed the repo-level abandonment work, not this. Checkbox in the delivery roadmap Decision 2026-08-28.


