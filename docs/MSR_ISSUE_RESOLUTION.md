# MSR prior art — issue resolution-time prediction

Tripwire in `ROADMAP.md` Decision 2026-08-28 (before 14 Sep). `STRATEGY.md` already
surveyed **repo-level** abandonment; this note covers **issue-level** time-to-resolution.
Written 2026-09-03. Not a literature review — enough to keep the module/topic claim honest.

## What already exists

| Work | What it predicts | What it does not cover |
|------|------------------|------------------------|
| Giger et al., "Predicting the Fix Time of Bugs" (MSR / Empirical SE lineage; Eclipse, Mozilla, Gnome) | Fast vs slow bug fix from report attributes; decision trees | Module/topic path; competing risks (fix vs administrative close); adjacent-module displacement |
| Follow-on bug-fix-time literature (Eclipse/Mozilla/Jira) | Regression or classification of hours/days to FIXED | Same gaps; often mixes post-submission process features that leak if used at open |
| "Predicting Issue Resolution Time of OSS Using Multiple Features" (J. Softw. Evol. Process, DOI 10.1002/smr.2746) | Resolution time on GitHub issues with project/issue/developer features; static + dynamic | Topic hazard; R1 vs R2; path-scoped demand×supply |
| Process-mining of GitHub issue micro-processes (e.g. SMU / OSS process papers) | Patterns and duration of corrective maintenance | Module contribution decision; adjacent API |
| Leakage-aware IRT work (creation-time features only, temporal splits) | Early estimate without post-open contamination | Still tracker-native priority/assignee past — not ecosystem adjacent rise/fall |

Shared findings that we inherit, not reinvent:

1. Creation-time metadata predicts something; post-submission process features (comments, status thrash) improve accuracy and are the main leakage risk.
2. Tree ensembles are the usual baseline; random train/test overstates accuracy vs temporal splits.
3. Long-tail durations are normal; mean absolute error on hours is the common metric, not competing-risk CIF.

## What Casefile claims that this literature does not

- **Module / path level**, not issue-tracker row level alone — `torch/masked` vs the repo.
- **Competing risks** — R1 (linked merge touching the path) vs R2 (administrative close) vs censored, not "closed".
- **Displacement** — adjacent module rising while this one falls (NestedTensor pattern).
- **Refusal** when the window or assignment precision cannot support a score (Apertus path).

If a slide says "nobody predicts issue resolution," that slide is wrong. The honest line is:
issue-level time-to-fix is crowded; **topic-level competing-risk hazard with displacement
and explicit refusal** is the gap we are testing.

## Implication for 14 Sep – 4 Oct

Baselines must include a simple issue-level or person-period model without Block 6 / without
displacement, so a win is not just "we beat a mean." Kill criterion stays: beat vital-signs /
Blocks 1–4 on Brier at 90 days, grouped-by-repo.
