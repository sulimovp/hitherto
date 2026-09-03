# Block-6 extraction pilots

Runs of `python -m casefile.predict.extract_run` over the existing assignment gold draws.
No new items were fetched for any of these; the corpus selection criterion is still an open
Corpus selection criteria are tracked outside this package.

The runner is `block6-v3` (quote for quotable fields, rubric cell for judged fields). The
files below are v1/v2 measurements. A v3 row will not match a v2 `extractor_version`, so
do not resume a v2 JSONL with the current runner and treat the mix as one study.

`smoke_block6_v3_gpt_oss.jsonl` is `--limit 10` over `pytorch_holdout.yaml`, and the first
45 entries there are pull requests, so the slice contains no issues and no Apertus items.
Read its fill rates as "the rubric fills on PRs", not as a discrimination check â€” `affect`
is 0 on all nine populated rows. See the dated notes in the extraction table captions.

`--stratify 20 --seed 0` with `--kinds issue,pr,hub` draws four items from each
originÃ—kind cell. The 31 Aug attempt wrote 20 rows of Hugging Face 402 (no inference
credit). Do not read that JSONL as a filled smoke. The runner stops a batch on HTTP
401/402 (`llm_http_status`, not a substring of the error body). Compaction keeps rows
already in the file that are not in the current draw.

`--kinds` defaults to `issue,hub`. Demand-side fields are not defined on pull requests.
The first spend after PAYG is funded should be a new file, not a resume of the 402 mix:

```bash
python -m casefile.predict.extract_run \
  -i eval/topic_assignment/pytorch_holdout.yaml eval/topic_assignment/apertus.yaml \
  -o eval/extraction/smoke_block6_v3_demand_gpt_oss.jsonl \
  --stratify 20 --seed 0 --kinds issue,hub
```

## Runs

| File | Schema | Model | Rows | Parse errors | Span failures |
|---|---|---|---|---|---|
| `pilot_2026-08-30.jsonl` | `block6-v1` | `openai/gpt-oss-120b:groq` | 176 | 0 | 90 (51%) |
| `pilot_2026-08-31_block6_v2_gpt_oss.jsonl` | `block6-v2` | `openai/gpt-oss-120b:groq` | 176 | 0 | 7 (4.0%) |
| `smoke_block6_v2_gpt_oss.jsonl` | `block6-v2` | `openai/gpt-oss-120b:groq` | 10 | 0 | 0 |
| `smoke_block6_v2_gemma.jsonl` | `block6-v2` | `google/gemma-4-31B-it:cerebras` | 20 | 0 | 2 |
| `smoke_block6_v2_qwen.jsonl` | `block6-v2` | `Qwen/Qwen3-14B:nscale` | 20 | 3 | 1 |
| `smoke_block6_v3_gpt_oss.jsonl` | `block6-v3` | `openai/gpt-oss-120b:groq` | 10 | 0 | 1 |
| `smoke_block6_v3_demand_gpt_oss.jsonl` | `block6-v3` | `openai/gpt-oss-120b:groq` | 20 | 0 | 1 |
| `secondary_block6_v3_gemma.jsonl` | `block6-v3` | `google/gemma-4-31B-it:cerebras` | 20 | 0 | 0 |
| `pilot_2026-09-02_block6_v3_demand_gpt_oss.jsonl` | `block6-v3` | `openai/gpt-oss-120b:groq` | 112 | 3 | 6 |

`smoke_block6_v3_demand_gpt_oss.jsonl` is `--stratify 20 --seed 0 --kinds issue,hub` (complete 1 Sep).
Strata: 6 `pytorch:issue`, 7 `other:issue`, 7 `other:hub` â€” no PRs. On all 20 rows, judged ordinals
fill (`specificity` 20/20, `blocking_severity` 19/20, `affect` 20/20, `scope` 18/20);
`maintainer_stance` 1/20 (gold YAML is title+body). After the 2 Sep revalidate (absence
needs no quote), one gpt-oss row still fails (`pytorch#39639`, intent quote not in the
thread). This is the first filled v3 smoke on demand-side items.

`secondary_block6_v3_gemma.jsonl` is the same draw on Gemma. Revalidate cleared 18
quote-on-false errors; 0 span failures remain. Agreement
(`agreement_2026-09-01_gpt_oss_x_gemma_v3.json`) is overlap 20, valid 19. Read `n` first:
`intent` Îº 0.77 (n=17), `specificity` Î± 0.81 (n=19), `blocking_severity` Î± 0.95 (n=18),
`scope` Îº 0.19 (n=17). `affect` matches on 18/19 rows but Î± is 0 because the value is
almost always 0. `maintainer_stance` n=1. This is a smoke, not the 200-item table.

`pilot_2026-09-02_block6_v3_demand_gpt_oss.jsonl` is every issue and Hub row in the assignment
gold (50 pytorch issues, 17 other issues, 45 Hub). No PRs. Seeded from the 20-row demand
smoke so those calls were not repeated. Of 112 rows, 109 have fields; 3 remain router
`json_validate_failed` 400s after one retry. On the 109 filled rows: `specificity` 96%,
`blocking_severity` 94%, `affect` 96%, `scope` 92%, `maintainer_stance` 7%. Span failures
6/112. v2 on the mixed 176-row file had `blocking_severity` at 6.2%.

**Frozen 2026-09-03.** Distributions, not fill rates, are the reading: `affect` is 102Ã—0 of 105
non-null; `specificity` and `blocking_severity` are near-binary; `scope` is 66% `one_line_fix`
with smoke Îº 0.19. Usable columns today: `intent` plus the two binarized ordinals. Do not
spend on Gemma-200 or comment reconstruction until the modelling window (14 Sep). Details in
the freeze note in this README.

Extractor pin for the gpt-oss runs:
`openai/gpt-oss-120b:groq@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
(`docs/hf_snapshot/extractor_pin.json`).

v2's 4.0% span-failure rate is not a quality win. On the repaired 176-row file
`blocking_severity` is populated on 6.2% of rows and `affect` on 3.4%. The v3 smoke
(n=10) fills those ordinals (`specificity` 10/10, `scope` 10/10, `blocking_severity`
9/10, `affect` 9/10) and still leaves `maintainer_stance` at 0/10. Read the
2026-08-31 notes before treating any of this as labels.

Resume treats rows without `repair_attempts` as incomplete, retries `parse_error` /
transport failures, and keeps span-invalid rows (they already spent the call). New rows
are appended, then the file is compacted once: current draw first, then any other rows
already present. `--revalidate` recomputes `span_errors` from stored fields (no LLM).

## Commands

v3 demand-side smoke (issues and Hub; this spends inference). `--limit` over the
holdout is all PRs for N â‰¤ 45. Default `--kinds` is `issue,hub`.

```bash
cd casefile && set -a; . ./.env; set +a
python -m casefile.predict.extract_run \
  -i eval/topic_assignment/pytorch_holdout.yaml eval/topic_assignment/apertus.yaml \
  -o eval/extraction/smoke_block6_v3_demand_gpt_oss.jsonl \
  --stratify 20 --seed 0 --kinds issue,hub
```

Demand-side v3 fill-rate (112 issue+Hub gold rows; seed from the 20-row smoke to skip those):

```bash
python -m casefile.predict.extract_run \
  -i eval/topic_assignment/pytorch_holdout.yaml eval/topic_assignment/apertus.yaml \
  -o eval/extraction/pilot_2026-09-02_block6_v3_demand_gpt_oss.jsonl \
  --kinds issue,hub
```

Second extractor, for the agreement study. Run it on v3 after a smoke shows the ordinals
fill, not against the v2 pilots:

```bash
python -m casefile.predict.extract_run \
  -i eval/topic_assignment/pytorch_holdout.yaml eval/topic_assignment/apertus.yaml \
  -o eval/extraction/secondary_block6_v3_gemma.jsonl \
  --router-model google/gemma-4-31B-it:cerebras \
  --extractor-model google/gemma-4-31B-it:cerebras@842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --stratify 20 --seed 0 --kinds issue,hub
```

Agreement (no network, no cost). v3 demand draw:

```bash
python -m casefile.predict.agreement \
  --primary eval/extraction/smoke_block6_v3_demand_gpt_oss.jsonl \
  --secondary eval/extraction/secondary_block6_v3_gemma.jsonl \
  --output eval/extraction/agreement_2026-09-01_gpt_oss_x_gemma_v3.json
```

v2 overlap (historical):

```bash
python -m casefile.predict.agreement \
  --primary eval/extraction/pilot_2026-08-31_block6_v2_gpt_oss.jsonl \
  --secondary eval/extraction/smoke_block6_v2_gemma.jsonl \
  --output eval/extraction/agreement_2026-08-31_gpt_oss_x_gemma.json
```

`agreement_2026-08-31_gpt_oss_x_gemma.json` and `..._x_qwen.json` are that command over the
20-item v2 smoke overlaps. `n` per field is the count of pairs where both runs are non-null,
and it is the number to read first.

