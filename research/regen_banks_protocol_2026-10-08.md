# Bank regeneration on MultiRoom and KeyCorridor

Status: prospective; generation local,
training submitted on Fir. Extends the DoorKey regeneration study
(`paper_optional_regeneration_2026-10-05.md`, O4) to the other two
MiniGrid source tasks.

## Question

Is the learning benefit of the selected MultiRoom and KeyCorridor banks a
property of the recipe, or of one lucky generation? Repeat each task's
frozen recipe five times and train students with every resulting bank.

## Generation (`scripts/regenerate_banks_20261008.py`)

Each repetition re-sends the SAME 36 consultation requests (copied from
the original run, checked against their pinned digest), then the
recipe's own self-check / blind-check export (pool sampling RNG 7) and
its own bank builder. GPT-5-mini, one attempt per request, no retries.
No repetition is selected, repaired or replaced; failed calls stay
missing, empty or duplicate banks are kept.

| Task | Recipe | Bank per student |
|---|---|---|
| MultiRoom-N6 | `conditional_rules_multiroom` (Sept 28) | PPO: `scoped` (all rules); Count-PPO: `blind_strict` |
| KeyCorridor-S3R3 | `conditional_rules_keycorridor_mem` (Sept 29) | both: `self_checked_pooled_valid` |

Reproducibility check before any paid call: feeding the ORIGINAL replies
through the same code reproduces the selected MultiRoom banks byte for
byte and the selected KeyCorridor bank's 28 rules and metadata exactly
(only local line endings differed), and every check request exactly.

Frozen banks: `research/rule_banks/regen_20261008/<task>/rep_<k>/`, LF
bytes, with `index.json` (rule counts, digests, recorded cost).

## Training (`scripts/run_regen_training_20261008.py`)

Cells copy the fresh cohort's (fix-wave addendum 15, replicates 80-89,
commit 875e5bb) completed selected-bank cells and change only
`rule_bank`, `rule_bank_sha256` and `experiment_id`: 5 banks x 10
replicates x 2 students = 100 cells per task, 200 in all, 5M transitions.
The batch code is `git archive 875e5bb`, required to equal the fresh
cohort's recorded source hashes file by file; cells are built by that
commit's own `cell()` and run by its own worker. The fresh cohort's
no-advice and selected-bank runs are reused as paired controls.

Cross-cluster caveat: controls ran on Vulcan, new cells on Fir. Runtime
identities are recorded for both; pairing is by training seed, and
initial-policy hashes are compared and reported.

## Analysis (fixed before outcomes)

Per task x student, as in O4: each bank's mean AUC and paired difference
to no advice; the average regenerated-minus-none difference with a crossed
percentile bootstrap (banks and seeds resampled separately, 10,000 draws,
RNG 20261008); the selected bank's paired difference reported separately.
Secondary: final greedy success, number of banks whose mean beats no
advice, rule counts and generation cost. No outcome threshold triggers
selecting or adding generations.
