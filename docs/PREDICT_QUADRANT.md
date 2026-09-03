# PREDICT â€” wiring the demandÃ—supply quadrant (implementation brief)

Written 2026-08-24. Follows the three fixes in the prior brief, which have landed. Companion to [`PREDICT.md`](PREDICT.md) Â§Â§2.3, 3, 7. This is a work order: delete it once the quadrant renders for `torch/masked`.

**What this delivers.** A real demandÃ—supply placement for a path-scoped topic, computed entirely from realized outcomes, printed next to a refusal for the hazard score. No training, no person-period table, no Blocks 1â€“6, no walk-forward harness. `score_refused` stays populated throughout â€” that is intended, and the report saying so is the point.

**What still blocks it today.** After the gate split, `assess_topic_hazard` refuses the rollup on five shared preconditions that nothing currently supplies: `assignment_precision_measured` is hardcoded `False` at `orchestrator.py:60`, `topic_first_seen` is never passed, and inflow and realized-R1 rates are never computed. This brief supplies all five.

---

## 1. Fix the inflow overlap before building on it

`demand_from_inflow(inflow_3m, inflow_12m)` divides by 3 and 12 and compares. The 12-month window *contains* the 3-month window, so the recent quarter is inside its own baseline and a genuine surge is damped toward 1.0. Mild, but the slide is going to be built on this ratio and it is cheaper to fix now.

Change `quadrant.py` to take rates directly, matching `supply_from_r1_rates` which already does:

```python
def demand_from_inflow(*, recent_per_month: float, baseline_per_month: float) -> DemandTrend
```

Keep the existing 1.25 / 0.8 thresholds for rate-only callers. Live rollup classifies from Poisson rate intervals (inflow counts) and Wilson intervals (R1 rates), with a minimum count per window. Update `build_topic_rollup` and `assess_topic_hazard` params to `demand_recent_per_month` / `demand_baseline_per_month`, and update `TopicRollup` fields to match. The rollup refusal reason R1 becomes `"inflow rates unknown â€” cannot place topic on the demand axis"` unchanged.

Windows, non-overlapping, defined once and reused by both axes:

| Window | Range | Months |
|---|---|---|
| recent | `[T âˆ’ 180d, T]` | 6 |
| baseline | `[T âˆ’ 730d, T âˆ’ 180d)` | 18 |

---

## 2. New module â€” `casefile/predict/topic_history.py`

One module computing every input the rollup needs, from the GitHub search and timeline APIs already wrapped in `clients/github.py`.

```python
@dataclass(frozen=True)
class TopicHistory:
    path: str
    demand_recent_per_month: float | None
    demand_baseline_per_month: float | None
    realized_r1_rate_recent: float | None
    realized_r1_rate_baseline: float | None
    n_resolved_items: int | None
    topic_first_seen: datetime | None
    # provenance â€” all of it goes in the report footer
    inflow_recent_total: int | None
    inflow_baseline_total: int | None
    r1_sampled_n: int
    r1_excluded_n: int          # labelled items dropped for missing PR paths
    search_truncated: bool
    fetch_errors: list[str]

async def compute_topic_history(
    clients, *, repo: str, path: str, topic_paths: tuple[str, ...],
    now: datetime, maintainers: MaintainerSet,
) -> TopicHistory: ...
```

### 2.1 Inflow

Two `search_issues_page` calls using `total_count`, never `len(items)`:

```
repo:{repo} is:issue "{path}" created:{recent_start}..{now}
repo:{repo} is:issue "{path}" created:{baseline_start}..{recent_start}
```

Keep the quoted-phrase path scoping from `vital_signs.py` â€” do not let `torch/masked` degrade into the free-text search `torch masked`.

**Saturation guard.** GitHub search `total_count` caps at 1000. If either window returns `>= 1000`, set `search_truncated=True` and leave both inflow fields `None`, so the rollup refuses. A rate computed from a capped count is the `closure_rate` mistake in a new window; do not compute it and do not report it.

### 2.2 Realized R1 rates

Per window, sample closed issues on the path and label them with the existing `label_issue_outcome`.

```
repo:{repo} is:issue is:closed "{path}" closed:{window_start}..{window_end}
```

Sampling must not be relevance-ranked â€” that is the same bias trap. Pass `sort=created&order=desc` and take up to `_R1_SAMPLE_CAP = 50` per window. Record `sampled_n` and the window's `total_count` separately; the rate is a sample estimate, the floor uses the total.

Per sampled issue:

1. `list_issue_timeline(...)` â†’ `labels_at_time(timeline, closed_at)` and `linked_merged_prs_le(timeline, closed_at, pr_paths=...)`.
2. `maintainer_answered` = any `commented` event by a login in `maintainers.for_snapshot(closed_at)`.
3. `label_issue_outcome(..., topic_paths=topic_paths, unlabeled_closed_as_r2=True)`.

Rate = `R1 / (R1 + R2)` over the labelled sample. R3 cannot occur here â€” the query is `is:closed` with a bounded `closed:` range.

**The `pr_paths` trap â€” read this before implementing.** R1 via linked PR requires `paths_touched âˆ© topic_paths` (fix 6). Populating `pr_paths` costs one `GET /pulls/{n}/files` per linked merged PR. When that call fails or is skipped, the PR arrives with empty paths, the intersection is empty, and the item silently becomes **R2**. That would read as collapsed maintainer capacity on exactly the topics the product cares about â€” the confound R1/R2 exists to remove, re-entering through a rate limit.

Required behaviour: if any linked merged PR on an item has unresolved paths, **exclude that item from both numerator and denominator** and increment `r1_excluded_n`. If `r1_excluded_n > 0.25 Ã— sampled_n` in either window, set both rate fields to `None` so the rollup refuses. Never let a fetch failure become an outcome.

Cap PR-file lookups at 3 per issue; an issue with more linked merged PRs than that is excluded rather than partially resolved.

### 2.3 `topic_first_seen`

First commit touching the path. Two calls, no full pagination:

1. `GET /repos/{o}/{r}/commits?path={path}&per_page=1` and read the `Link: rel="last"` header for the final page number.
2. Fetch that page; its last entry is the earliest commit. Use `commit.author.date`.

If the `Link` header is absent the history fits in one page and the last entry of that page is the answer. On any failure leave `None` â€” shared precondition S6 then refuses, which is correct.

`clients/github.py` currently discards response headers in `get_json`. Add a `get_json_with_headers` or have `list_commits` optionally return the link header; do not parse HTML or guess page counts.

### 2.4 Errors

Every failure appends to `fetch_errors` and leaves the corresponding field `None`. Nothing here may impute a zero. This is the 2026-08-23 project notes finding â€” retrieval failure and absence of evidence are different states â€” and it applies unchanged.

---

## 3. Assignment precision â€” the one piece of genuinely new work

`PREDICT.md` Â§3 requires itemâ†’topic assignment precision to be measured before any rollup renders. It is a hand-check, not code, and it is what actually gates the October slide.

### 3.1 Ground truth

`eval/topic_assignment/{profile}.yaml` â€” 100 issues across three repos, hand-labelled:

```yaml
profile: pytorch
labelled_at: 2026-09-??
labeller: <name>
items:
  - repo: pytorch/pytorch
    number: 12345
    topic: torch/masked        # or null when the issue is not about this topic
```

Sample the 100 as: 50 drawn from the automatic assignment's positives (measures precision) and 50 from path-adjacent issues it did *not* assign (measures recall). Record the draw method in the file â€” a convenience sample of positives only measures nothing about recall.

### 3.2 Scorer

`scripts/score_topic_assignment.py` â€” runs the automatic assignment over the labelled set, prints precision, recall, n, and a Wilson interval on precision. **Verify:** rerunnable offline from a fixture, and its output pastes into the profile block below.

### 3.3 Profile field

```yaml
assignment_precision:
  measured_at: 2026-09-??
  n: 100
  precision: 0.86
  recall: 0.71
```

Add to `models/profile.py`. `assignment_precision_measured` is then derived, never hardcoded:

```python
measured = (
    profile is not None
    and profile.assignment_precision is not None
    and profile.assignment_precision.precision >= _MIN_ASSIGNMENT_PRECISION  # 0.80
)
```

Below 0.80 the rollup refuses with the measured number in the reason string â€” `"itemâ†’topic assignment precision 0.62 < 0.80 (n=100, measured 2026-09-??)"`. A measured failure is a better report line than an unmeasured pass.

Stale check: reuse the existing `max_age_days` convention. If `measured_at` is older than the profile's `max_age_days`, treat it as unmeasured.

---

## 4. Orchestrator and renderer

`orchestrator.py:60` â€” delete the hardcoded `assignment_precision_measured=False` and pass the computed `TopicHistory` fields plus the derived precision flag. Call `compute_topic_history` only when `plan.resolved_path` is set; skip it entirely otherwise, since S2 refuses anyway and the calls are not free.

`observation_window_days` stays unset. C3 keeps the score refused, which is correct â€” there is no trained artifact either.

Renderer: under `## Topic trajectory`, after the quadrant line, add a provenance line so every number is traceable:

```
Demand **rising**, supply **falling** â†’ quadrant **gap**.

Demand rising, resolution capacity falling â€” contribution target.

Inflow 3.7/mo recent vs 1.9/mo baseline Â· R1 rate 0.31 (n=44, 6 excluded) vs 0.68 (n=50)
Â· assignment precision 0.86 (n=100, 2026-09-??)

_Hazard score refused:_ trained topic-hazard artifact not shipped (topic-hazard-v0-refuse)
```

That line is the difference between a chart and a citation. Do not ship the quadrant without it.

---

## 5. Tests

| Name | Asserts |
|---|---|
| `test_inflow_windows_do_not_overlap` | baseline query end == recent query start; 18-month and 6-month divisors |
| `test_inflow_saturation_refuses` | mocked `total_count=1000` â†’ both inflow fields `None`, `search_truncated` true, rollup refused |
| `test_r1_rate_excludes_unresolved_pr_paths` | one item with an unfetchable PR path â†’ `r1_excluded_n == 1`, item in neither numerator nor denominator |
| `test_r1_rate_refuses_when_exclusions_exceed_quarter` | 13 of 50 excluded â†’ rate `None` â†’ rollup refused |
| `test_r1_sample_is_created_sorted` | search called with `sort=created`, never relevance |
| `test_topic_first_seen_from_link_last_page` | mocked `Link: rel="last"` â†’ earliest commit date |
| `test_assignment_precision_below_floor_refuses` | precision 0.62 â†’ refused, reason carries the number and n |
| `test_assignment_precision_stale_refuses` | `measured_at` older than `max_age_days` â†’ treated as unmeasured |
| `test_fetch_error_never_imputes_zero` | search raising â†’ field `None` and a `fetch_errors` entry, not `0.0` |
| `test_quadrant_renders_with_provenance_line` | rendered report contains inflow, R1 n, exclusions, and precision |

**Verify end to end:** `casefile assess -q "â€¦" -r pytorch/pytorch -p torch/masked` prints a quadrant with the provenance line and a score refusal in the same section, and the Apertus run prints two refusals.

---

## 6. Cost and order

API calls per assessment, path-scoped: 2 inflow searches, 2 closed-issue searches, up to 100 timelines, up to 300 PR-file lookups, 2 commit calls. The PR-file lookups dominate. With `CASEFILE_GITHUB_TOKEN` at 5000/hr this is comfortable for one report and needs caching before the GitHub Action ships â€” the existing `cache/github_search.py` should cover the searches; PR files need a small addition.

Order: Â§1 (signature fix, isolated) â†’ Â§2 (module + tests against fixtures, no network) â†’ Â§3 (the hand-check, which is calendar work and should start in parallel because it is the only item that cannot be compressed) â†’ Â§4 (wiring).

Â§3 is the critical path. Start the labelling before the code is finished.

