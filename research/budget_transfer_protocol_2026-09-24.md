# Lower consultation budgets and KeyCorridor transfer, v1

Date: 2026-09-24.
Status: frozen exploratory design; user submits after implementation review.
No outcome from these new arms exists.

## Decision and scope

MultiRoom random120 passed its prospective development nomination rule with
75.1% fewer realized calls than random480, but the paired AUC interval does
not establish noninferiority. Now test30/60 to bracket a smaller usable dose;
do not infer that the5–16 calls preceding observed learning were sufficient.
Later advice may have sustained those policies. DoorKey120 tests whether the
reduction transfers to a prerequisite task. KeyCorridor tests the same direct
GPT recipe under hidden-key search and object pickup, on both PPO backgrounds.

These are new conditions, not replications of completed conditions. Reusing
development seeds permits pairing, not independent confirmation. We will not
select a winner by optional stopping or call the selected cap an optimum.
This is calibration for teaching/explanation research, not an explanation
benefit test or a novelty claim for finite advice budgets.

## Exact menu

| Batch | New conditions | Training seeds | New runs | Full hold |
|---|---|---|---:|---:|
| `multiroom_gpt30_60_20260924_v1` | Count-PPO, random30 and random60 | 8100000,8100100,8100200,8100300,8100400 | 10 | $5.05 |
| `doorkey_gpt120_20260924_v1` | Count-PPO, random120 | 7600000,7600100,7600200,7600300,7600400 | 5 | $6.70 |
| `keycorridor_gpt_budget_20260924_v1` | PPO and Count-PPO, each no advice/random120/random480 | 11600000,11600100,11600200,11600300,11600400 | 30 | $66.70 |

Total45 runs,7050 maximum GPT requests, $78.45 fully reserved. Five training
seeds per condition are a development screen, not a guaranteed precision.
No environment episodes are treated as independent training replications.
MultiRoom reuses15 completed cells (0/120/480); DoorKey reuses10 (0/480).
KeyCorridor's ten no-advice cells are new paired controls on fresh seeds;
another task's controls cannot substitute for them. Existing runs, failed
attempts, active lesson/progress studies and hypotheses remain unchanged.

## Teacher, advisor and learning channel

- Teacher: direct `llm_general` GPT-5-mini, full symbolic map/state and mission;
  original prompt/structured action schema. No BFS/subgoal planner selects
  the GPT action. The teacher's automatically returned rationale is logged
  but is not an explanation training target here.
- Advisor: inherited `UniformQueries`, sampling clock slots without replacement
  across active rollouts in the original first75% of training. Query budget
  and maximum delivered advice both equal the arm's cap. Reset-only slots
  are skipped without replacement; failures also consume opportunities.
  Different budgets sample independently from their own seeded schedule;
  their times are **not nested**. Thus these are complete budgeted-procedure
  comparisons, not isolated effects of deleting labels at identical states.
- Channel: the student executes its own actions. Add cross-entropy action
  distillation to PPO, averaged over labelled samples. Coefficient starts1,
  anneals to.01 by fraction.5, and is0 after fraction.75. No action override,
  label replay, auxiliary rationale loss or test-time teacher. No-advice
  controls have guidance off, zero budgets and uniform gate off.
- Keep original one attempt, zero SDK retries, input cap16384 and output3500,
  original timeout and consultation failure policy (>20% after40 samples).
  A30-cap arm cannot reach that40-sample threshold; its hard cap still bounds
  spend. Every failure is reported; an environment task failure and an API
  failure are separate. No automatic reruns or extra seeds for poor learning.

## Learner and environment settings

Use the original `1ab5fc0d9c9d22b7147d58ba92245bcc42c0b008` trainer plus the
previously audited opt-in uniform-query overlay. Source hashes are checked
against archived completed manifests. Keep that scientific source even if
the current trainer has changed. Only the new frozen worker's admission
ceiling is updated; historical original jobs/snapshots are not edited.

All runs: local egocentric7x7 symbolic image,512-unit recurrent GRU, no mission
encoder in this inherited policy, dual value heads;10,000,000 nominal and
9,999,360 actual transitions;8 environments ×128 steps per rollout;
4 PPO epochs ×4 minibatches; LR.00025 with linear annealing; gamma.999,
GAE.95; entropy.01; value coefficient.5; clip.2; max gradient norm.5;
normalized advantages and clipped value loss. Count-PPO uses policy-view
counts, intrinsic coefficient1, normalized intrinsic rewards and intrinsic
discount.99. PPO has bonus off. Call them PPO/Count-PPO, not universally weak
and strong: competence is an empirical property of this setting and horizon.

Tasks exactly `doorkey_8x8`, `multiroom_n6`, `keycorridor_s3r3` in the pinned
registry. The last maps to **BabyAI-KeyCorridorS3R3-v0** with constant mission
`pick up the ball`, not the similarly named MiniGrid registered variant.
It adds hidden-key exploration, remembering object locations and clearing
inventory before pickup. Do not call S3R3 simply a larger DoorKey or a test
of language comprehension. The matched local view avoids an accumulated
observation/inventory omission confound within these new comparisons.

## Metrics and frozen decisions

Primary: teacher-off greedy success AUC, trapezoidally integrated and divided
by support length, over the common evaluated steps204800..9999360. Evaluate
50 episodes every200 rollouts, plus inherited2M/5M milestones, evaluation
seed=trainseed+50000. Plot each seed, mean and95% pointwise interval. Do not
interpret pointwise bands as a simultaneous guarantee across the whole curve.
Pair training seeds for differences; report paired t95% intervals (df4) and
exact sign-flip sensitivity. With five pairs, two-sided sign-flip tests cannot
attain p<.05 even if all signs agree. No statistical superiority declaration
from a development point gate or selection across multiple doses.

Secondary: final success; first80% evaluation; second of two consecutive80%
evaluations (sustained80 confirmation); transitions, actual calls/delivered
labels and elapsed seconds to those milestones; failures, input/output tokens,
standard-rate cost with unknown-attempt bound, full runtime and hardware.
An unreached threshold is censored, not a made-up large finite step count.
Runtime across historical and new cohorts is descriptive because scheduling,
machines and provider latency differ. Distinguish sample efficiency from speed.

**MultiRoom nomination:** among30/60, select the smallest cap with at least4/5
final success>=.8, mean AUC>=historical random480 mean−.02, and mean AUC above
matched no advice. Compare each to120 secondarily. DoorKey120 uses the same
rule versus its own random480/no-advice controls. Show intervals for the
reference contrast against both0 and−.02; a mean satisfying the rule is not
evidence of noninferiority unless uncertainty supports that margin. If neither
new cap qualifies, retain120 as a MultiRoom candidate without expanding spend.
Missing/invalid pairs withhold nomination; all scientifically valid poor seeds
stay in the denominator. Completed results determine which *fresh-seed* study
to preregister next, not an automatic expansion or retrospective gate change.

**KeyCorridor:** primary contrast Count-PPO+480 minus Count-PPO alone; report
120 and both plain-PPO contrasts as declared secondary analyses. Nominate a
teaching configuration only if mean AUC improves by>=.03 with positive
differences on>=4/5 seeds and mean final success is no worse by>.02. If both
qualify within one background, select120. This is exploratory nomination,
not significance/confirmation. Floors/ceilings and inadequate coverage remain
possible; a null does not prove GPT teaching or explanations cannot work.
Report the two backgrounds separately and do not select a pooled winner.

## Funding, execution and completion

An additional 250 USD authorization adds to 203.13, giving an absolute
453.13 project ceiling. User-run launcher applies it **once idempotently**;
this is not an invoice or another250 on every execution. All old holds and
settled entries are preserved. Latest transferred ledger87 settled/111.89
reserved/4.24 free would become254.24 free, then175.79 after these holds;
live accounting, not this projection, controls admission. This projection
excludes separately reported 720-call local evidence screens (about
$0.74 known standard-rate usage, one reported failure; no ledger reservation).
That is unreconciled external activity, not a verified invoice or free credit;
retain it in the coordination note for the next accounting reconciliation.
It does not establish any new learning benefit. No reduction of
per-call worst-case bounds based on typical usage or sharing discounts.

Stage all scientific settings and validate original artifacts/source before
reserving. Submit45 cells without array concurrency limits; scheduler limits
still apply. Every new batch has separate funding and a stop marker; repeated
launcher use does not duplicate its reservation or job. Jobs finish their fixed
10M horizon regardless of promising intermediate evaluations. No test results
control consultation retirement. Reconcile only after terminal artifacts and
unknown costs are inspected. Agents make no live API calls or Slurm submissions.

## Three core tasks and language work in parallel

DoorKey (prerequisites), MultiRoom (navigation), and KeyCorridor (hidden-key
search/pickup) are the current three-task action-advising comparison. They are
complementary tasks within one gridworld family, not three unrelated domains.
There is no asserted AAMAS minimum environment count. GoToSeq remains a fourth
language/order candidate: preserve the already running25-cell native-progress
study and inspect its completion before a new paid language-learning grid.
Its sequence-only distribution and mission-conditioned learner differ from
this inherited trainer. The low/partial results do not justify hiding it or
calling the candidate a completed successful language extension.

Explanation development continues separately; these45 jobs test action-advice
budget/transfer, not a semantic explanation benefit. The current contrastive
reason-code target failed its promotion criterion; shuffled codes did as well
or better. An evidence-recoverability/novelty review should assess the next
information target before a new explanation grid is committed.

Primary environment references: [BabyAI KeyCorridor](https://minigrid.farama.org/environments/babyai/KeyCorridor/)
and [GoToSeq](https://minigrid.farama.org/environments/babyai/GoToSeq/).
The exact local registry and archived manifests remain the configuration
authority; prior comparison/novelty boundaries are in the project's literature
matrix and novelty ledger. No claim that these components are new is made.
