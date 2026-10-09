# Fresh complete Crafter progress factorial

Task W07.
Status: prospective implementation, no study runs or scientific results.

This is a complete bulk package, prepared without waiting for initial
outcomes. This protocol adds the missing count-student progress control
and independently repeats both students on ten new paired training seeds.
It does not change earlier experiments, inference families or outcomes.

Implementation: `scripts/paper_bulk_crafter_20261005.py`; focused tests:
`tests/test_paper_bulk_crafter.py`. The parent bulk runner owns frozen
source preparation, attempts, cluster scripts and user submission.
Standalone source admission is `python -m
scripts.paper_bulk_crafter_20261005 check`, expected: 60 cells, seeds
141 through 150, zero LLM calls and both bank hashes. Tests are
`python -m pytest tests/test_paper_bulk_crafter.py -q`; all must pass.
Working directory is the committed bulk integration checkout; environment
is the project Python environment with Crafter `1.8.3` or the observed
Vulcan build `1.8.3+computecanada` installed. The full version is frozen
at preparation and must match exactly in every worker. This runtime
amendment, 2026-10-05, fixes an admission failure before any batch
was prepared or submitted; it does not assert equivalence between builds.

## Frozen design and decision

This is prospective replication/extension using already selected banks and
coefficients, not a new hyperparameter search. All sixty cells run without
an outcome-dependent admission or stopping gate. Ten seeds 141--150 are
reserved across each of six combinations. Known historical Crafter
training cohorts use 1--40; no 141--150 assignment was found in the source
and protocol search. The launch manifest must preserve these identities.

| Student | Advice | Arm | Runs |
|---|---|---|---:|
| Plain PPO | None | `none` | 10 |
| Plain PPO | Full v3b bank | `full` | 10 |
| Plain PPO | Same bank without progress clauses | `no_progress` | 10 |
| Count PPO | None | `none` | 10 |
| Count PPO | Full v3b bank | `full` | 10 |
| Count PPO | Same bank without progress clauses | `no_progress` | 10 |

The cell identity includes student (`bonus=none/count`), arm and seed.
Trainer is `algos.ppo_crafter`; the symbolic observation is its existing
9 by 7 view and inventory vector. Count coefficient stays 0.01 and counts
the student's local view with the existing episodic reset. It is added to
the training reward with one value head, unlike MiniGrid's count setup.
The policy remains feedforward; progress counters are executor inputs.

Nominal training budget is 1,000,000 transitions, rounded by the original
16 by 128 rollout to 999,424 actual transitions. Each seed has 200 fixed
training worlds beginning at `40000000 + 1000 * seed`. Evaluation retains
the same ten previously used development/evaluation worlds beginning at
41000000; these are disjoint from the new training pools. Evaluation uses
sampled teacher-free actions, a 3,000-step cap, and the original 50,000-step
requested interval. The saved twenty panels start at 51,200 transitions
and end at 999,424. This trainer does not currently record step-zero or
dense early evaluations. No initial performance point is invented.

All original PPO settings are explicitly frozen in `frozen_args` rather
than inherited silently. Full and ablated advice use the selected
imitation coefficient 0.1 decaying to 0.001 by half the nominal horizon,
with advice off at 75 percent. The exact frozen banks are
`research/rule_banks/crafter_v3_20261001/v3b_preconditions.json` and
`research/rule_banks/progress_ablation_20261002/crafter_v3b_noprogress.json`.
Admission regenerates the expected ablation in memory and checks equality:
only progress conditions/exceptions are removed; actions, nonprogress
conditions, observer and rule count stay fixed. There are no paid calls.

The primary metric is the **arithmetic mean of twenty teacher-free mean
achievement panels**, retained as the historical `auc` compatibility key.
It is not a trapezoidal area and is not comparable in units to MiniGrid
success AUC. Per-training-seed differences receive paired 95 percent
t intervals. Four two-sided tests form one prespecified Holm family:

1. Plain full minus plain no-progress.
2. Count full minus count no-progress.
3. Plain full minus plain none.
4. Count full minus count none.

A positive estimate with Holm-adjusted p below .05 supports benefit in
that specified contrast. Otherwise the result remains negative or
inconclusive; no noninferiority or equivalence is inferred. Degenerate
zero-variance samples have undefined inferential p values, remain null in
the report, consume p=1 solely in family bookkeeping, and cannot establish
benefit. The historical arithmetic metric lies on a 1/200 lattice, which
is used to avoid floating-point differences inventing variance.

Secondary descriptive contrasts are no-progress minus none for each
student, count minus plain for unguided and full-bank arms, and the
student-by-progress interaction `(count full - count no-progress) -
(plain full - plain no-progress)`. Final achievements, final Crafter score,
label totals, count reward and wall time are retained as secondary
measurements. All sixty valid cells are required for statistical reporting;
failed, partial and unattempted runs remain separately accounted for by
the parent runner. There is no best-arm selection and no pooling with
earlier seeds or another bulk suite.

## Scope and failure handling

Prediction: both full-bank-minus-none means are positive; plain progress
benefit has prior evidence. Count progress direction and significance are
not predicted. A positive progress contrast supports the value of the
removed applicability clauses under this frozen recipe, not isolated
memory, perfect timing, matched exposure or a new algorithm. Conflicts,
exposure and student trajectories can change under clause removal. The
fixed bank and reused evaluation worlds limit generalization.

The entire finite grid is submitted without waiting for scientific pilot
results. A null or failed learning outcome is retained. No outcome-driven
seeds, weights or extensions are admitted. Parent bulk policy permits
only its finite, logged infrastructure retry mechanism; an incomplete run
does not authorize replacement of its seed. No cluster job or scientific
training is launched by an agent. Slurm time request remains the existing
Crafter value, 12:00:00 per run, one CPU and 8 GiB; elapsed runtime remains
unknown until actual receipts arrive. No array concurrency cap is added.

## Evidence and review state

| State | Evidence |
|---|---|
| Jobs completed | None at preparation. Job IDs unavailable. |
| Artifacts validated | Software fixtures only; no scientific artifacts yet. |
| Conclusions independently reviewed | Not applicable yet; no outcomes exist. |

The validator reuses the historical strict raw-panel checker in an
isolated module instance with only the frozen expected-argument contract
substituted. This avoids modifying the historical auditor's globals.
Additional checks require finite wall/metric/count values, exact count
coefficient and plausible totals, integer teacher counts, and every
checkpoint tensor's key, shape, dtype and finiteness. The parent runner
must also verify dispatch/exits and frozen-source provenance. Crafter's
unchanged trainer does not save an initial-policy hash, so initialization
pairing is by preserved implementation and identical training seed, not
an artifact-verified initial hash. No unavailable hash is fabricated.

Implementation reviewer and final committed revision: recorded by the
parent bulk integration receipt after independent review. This document
does not clear scientific results for the paper or project brief.

Implementation check, 2026-10-05: the focused suite passed all 17 cases in
8.60 seconds in the local project environment. It covers all six artifact
types, unchanged historical auditor globals, recipe/clock/metric/teacher/
count/checkpoint corruption, complete pairs, Holm bookkeeping and
degenerate differences. No scientific seed was trained; the checkpoint
fixture uses isolated engineering seed 8731991. The standalone admission
command also passed and reported the expected 60 cells and bank hashes.
Independent review remains the parent integration owner's next check.
