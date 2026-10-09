# Fresh teacher-view replication for the bulk paper batch

Status: prospective exploratory replication; implementation verification is
separate from scientific evidence. Work package W07.

## Decision and design

Question: with the same frozen rules, does reading their predicates from
the local student view or the full map change teacher-free learning? This
replicates the existing teacher-view intervention on fresh seeds. It is not
a new rule-writing method, an LLM-versus-other-teacher comparison, or a
causal isolation of the student's ability to imitate privileged targets.

Run all 120 cells without an outcome-dependent pilot: three source tasks
(DoorKey-8x8, MultiRoom-N6, KeyCorridor-S3R3), plain/count PPO, two execution
views, ten paired seeds. Both arms are advised; there is no no-advice arm
in this study, so this cohort alone cannot establish a net benefit over
unguided learning. Historical no-advice results remain separately labelled.

The saved October 4 strengthening plan reserved replicates 100--109 for
this comparison. No existing fix-wave suite or separate timing suite uses
these replicates. Training seeds are 14510000--14510900 for DoorKey,
14610000--14610900 for MultiRoom, and 14710000--14710900 for KeyCorridor,
each in increments of 100. Paired arms/students use the same seed within
each task. Teacher-off evaluation seeds retain the source trainer's
`training_seed + 50000` convention, with 50 episodes per panel.

The implementation `scripts/paper_bulk_view_20261005.py` supplies cells to
the frozen bulk archive/worker. The bulk worker's committed source and
manifest define the actual launch identity. One initial attempt is allowed
per cell, with no automatic retry. Infrastructure recovery needs a logged
amendment on the same seed; completed poor results are retained and are not
rerun. No paid API
calls are needed, and no array-concurrency cap is introduced.

## Treatment boundary

Use `fw.arm_args` for the current task/student-specific selected frozen
bank, 5M transitions, existing learning-rate schedule, imitation coefficient
0.1 to 0.001, existing decay/off schedule, labelled-mean normalization, and
student-selected actions. A full-map bank is the corresponding local bank
with identical conditions, exceptions, action labels, and executor mode;
its observer is replaced and source provenance appended. A preflight check
rejects a bank whose content differs from this transformation.

The student observation and architecture remain unchanged in every cell.
Full-map execution changes the object predicates available to the rule
matcher, including objects outside the window or behind walls. Immediate
front/carrying fields and the existing KeyCorridor progress source remain
unchanged. The KeyCorridor progress predicate reads episode grid state
(no locked doors remain); it is not an extra policy input and is not
replaced by a new history tracker in this study.

Enable the same read-only applicability/exposure diagnostics in both arms.
Extra teacher-off evaluations at frames 0, 10240, 25600, 51200, 76800,
102400, 128000, 153600, and 179200 include the initial policy. These panels
are isolated from training RNG/state and saved separately. The primary
regular evaluation grid and its AUC remain unchanged. Display this cohort's
complete curves, never splice early points into historical curves.

## Metrics, thresholds, and interpretation

Primary: six paired full-map minus local mean differences in regular
teacher-free greedy-success AUC, one per task/student background. Training
seed is the replication unit. Report all six estimates, paired 95% intervals,
and two-sided paired t tests with Holm correction across this
six-test family. A corrected p below .05 identifies a direction supported
within this family; otherwise report uncertainty, not equivalence. A
positive contrast favors full-map execution; a negative one favors local
execution. There is no directional nomination gate or data-driven expansion.
The bulk package is a broad exploratory analysis: correction within this
family does not correct selection across all study families.

Secondary descriptive measures: final greedy and sampled success, dense
early success curves including frame zero, accepted/retained label totals,
conflicts, abstentions, phase counts where available, and elapsed compute.
Keep failures, partial attempts, and unattempted cells separate. Do not
test a partial sample and then fill or select seeds according to outcomes.

Changing the teacher observer can change coverage, conflicts, chosen
targets, and later state visitation. This is the total effect of execution
view, not proof that information mismatch alone caused a difference.
Neither shared rules nor a shared coefficient guarantees matched exposure,
gradient, or quality. A full-map advantage is reported as such; it is not
grounds to hide a condition or revise the prospective analysis.

## Verification

Bounded checks: exact count/seed pairing, disjoint existing source and timing
cohorts, configuration deltas, all four selected bank pairs, and both
observers on short real-environment traces using unrelated seeds. Traces
check no mutation of the student image/environment, valid primitive labels,
and unchanged immediate/progress fields. MultiRoom uses oracle-controlled
diagnostic paths to encounter its sparse door-only checked rules; the other
tasks use random paths. These are engineering checks, not student behavior,
task-success experiments or study observations.

Local command from the repository root with the project Python environment:

```powershell
python -m pytest tests/test_paper_bulk_view.py -q -p no:cacheprovider
```

Expected: all eight test cases pass. A bank mismatch, missing real-env label,
student-observation mutation, or seed/configuration mismatch is a blocker.
Cluster commands and the source revision are those of the bulk batch
(research/paper_bulk_protocol_2026-10-05.md).

Author verification, 2026-10-05: final command passed all eight tests in
15.29 seconds using the shared project `.venv` Python. The initial run had
seven passes and one test-fixture failure: a short random MultiRoom walk
never encountered the checked bank's sparse door-only rules. Replacing that
engineering path with an explicitly oracle-controlled path exercised the
rules; no study treatment, seed, or success criterion changed. The seed
reservation check reads prior suite definitions without rebuilding all old
training commands. No scientific training ran during these checks.

Evidence states before launch: jobs completed = none; artifacts validated =
configuration/software only; conclusions independently reviewed = unavailable
because no scientific results exist. This document clears no efficacy or
novelty claim. Existing hypotheses and historical outcomes remain unchanged.
