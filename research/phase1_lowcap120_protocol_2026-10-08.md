# Phase 1 advisors at 120 GPT calls on three tasks

Status: prospective; preparation only.
Launcher: `scripts/run_phase1_lowcap120.py`. User wrapper:
`scripts/launch_phase1_lowcap120.sh`. Tests: `tests/test_phase1_lowcap120.py`.

## Question

The paper's motivating study compares online GPT-5-mini action advisors
(Phase 1: entropy importance, probability .20 mistake correction, uniform
random query times) on DoorKey-8x8 and MultiRoom-N6 at 480 consultations
per run. Does the same comparison hold on all three MiniGrid source tasks,
including KeyCorridor-S3R3, at a 120-consultation cap?

This is an exploratory, same-seed extension of completed cohorts. It is
not an independent confirmation and selects no method.

## What already exists (reused, not rerun)

All reused runs used the archived Phase 1 trainer `1ab5fc0d`, 9,999,360
transitions, local symbolic 7x7 view, CNN-GRU512, and the native advice
settings (labeled-mean imitation, coefficient 1.0 -> .01).

| Task | Seeds | No advice | random120 |
|---|---|---|---|
| DoorKey-8x8 | 7600000+100r | PPO, Count-PPO (`phase1_native_20260914_v1_r*`) | Count-PPO (`doorkey_gpt120_20260924_v1`) |
| MultiRoom-N6 | 8100000+100r | PPO, Count-PPO (`phase1_multiroom_native_20260916_v1`) | Count-PPO (`multiroom_gpt120_extension_20260923_v1`) |
| KeyCorridor-S3R3 | 11600000+100r | PPO, Count-PPO (`keycorridor_gpt_budget_20260924_v1`) | PPO, Count-PPO (same batch) |

## New cells (70 runs, 5 paired seeds each)

| Task | PPO | Count-PPO |
|---|---|---|
| DoorKey-8x8 | entropy120, probability120, random120 | entropy120, probability120 |
| MultiRoom-N6 | entropy120, probability120, random120 | entropy120, probability120 |
| KeyCorridor-S3R3 | entropy120, probability120 | entropy120, probability120 |

Each new cell copies a completed cell's archived arguments and changes only
the declared fields:

- DoorKey and MultiRoom start from the Phase 1 entropy480 cell of the same
  seed and student. KeyCorridor starts from the completed random120 cell of
  the same seed and student.
- All arms: `query_budget = advice_budget = 120`, a new `experiment_id`.
- entropy120: `advisor=importance`, `importance_source=entropy`,
  `uniform_queries=False`.
- probability120: as entropy120 plus `advisor=mistake`,
  `mistake_threshold=.2` (the Phase 1 probability20 recipe).
- random120: `advisor=unlimited`, `importance_source=none`,
  `uniform_queries=True` (the reviewed extension recipe).

KeyCorridor arms must equal the MultiRoom Phase 1 recipe for the same
student except task, seed, identity and cap. Tests enforce this.

Training source is built by the reviewed `run_budget_transfer` archive
builder and must be byte-identical to the `keycorridor_gpt_budget_20260924_v1`
snapshot for every file except that batch's own launcher/protocol records.

## Funding

Per guided run: 120 calls x $0.011096 worst-case call bound = $1.33152,
allocated $1.34. Batches: DoorKey $33.50, MultiRoom $33.50, KeyCorridor
$26.80; total hold $93.80 within the existing $453.13 ceiling (no ceiling
change). Expected standard-price spend from measured per-call costs
(DoorKey .0028, MultiRoom .0044, KeyCorridor .0050 USD) is about $34.
Holds stay open until completed usage is audited.

## Execution

One Slurm array per task; no array throttle. The launch wrapper delays the
MultiRoom and KeyCorridor arrays by 20 and 40 minutes (`--begin`) to spread
early entropy-paced requests. One attempt per consultation, zero SDK retries,
inherited 20% consultation-failure stop. Failed cells keep their artifacts;
at most one same-cell retry after a diagnosed infrastructure cause.

## Analysis (fixed before outcomes)

Primary: normalized teacher-free greedy-success AUC (trapezoid from 204,800
to 9,999,360 transitions). For each task x student, paired differences of
each advisor arm against no advice over the 5 seeds, 95% Student-t
intervals, Holm across the arms of that task x student family. Secondary:
final greedy success, sampled curves, AUC up to the 5M milestone, realized
consultations and delivered labels. Report 480-call Phase 1 arms beside
the 120-call arms where they exist; do not pool cohorts across tasks.
Plain-PPO cells with zero variance are reported descriptively.
