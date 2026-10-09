# MultiRoom matched advisor extension, version 2

Date: 2026-09-17.
Status: prospective exploratory extension; no new results.
Task P03.

## Decision and complete comparison

The completed `phase1_multiroom_native_20260916_v1` cohort already contains
five matched seeds, two student backgrounds, and both no-teacher and entropy
arms under the settings required here. Repeating those 20 runs would add an
independent replication, but it is unnecessary for completing the current
exploratory four-strategy comparison. Version 1 was therefore superseded
before submission.

The complete analysis will join:

| Source | Arms | Seeds | Runs |
|---|---|---:|---:|
| Completed array 959691 | no teacher, entropy | 8100000..8100400 | 20 |
| This extension | probability20, random | same five seeds | 20 |

Both sources cross PPO and Count-PPO. The joined matrix has 40 cells: five
paired seeds × two student backgrounds × four strategies. Existing outcomes
remain fixed; they are not rerun or selected by whether they looked favorable.
Exact-action correction is omitted from this extension, while its historical
results remain available as separate evidence.

## New arms

`probability20` uses the completed entropy arm's native uncertainty query
controller. After GPT answers, the label is delivered only if the student's
total probability on the teacher-endorsed action set is below .20. Because
GPT has no free action lookup, rejected labels still consume queries and API
cost. Report queries and delivered labels separately.

`random` selects 480 distinct environment-clock positions uniformly without
replacement from all rollouts where the imitation coefficient is positive.
It uses a private NumPy generator seeded by training seed + 1904117, so it
does not consume the learner's random stream. Reset placeholders are skipped
without replacement; actual calls can therefore be below 480. All proposed
and observed slots are saved. It makes no calls in the teacher-off tail.

Random timing is generally later and more dispersed than native entropy
timing. This compares complete strategies, including timing and label dose;
it does not isolate state selection alone. There are no fixed query windows.

## Exact matching contract

Each new cell is derived from the corresponding completed entropy cell for
the same seed and student background. Hold fixed:

- MultiRoom N6 and local symbolic 7x7 observations;
- CNN + GRU512, dual value heads, and the existing normalized count reward
  only in Count-PPO;
- 10M nominal / 9,999,360 actual transitions, eight environments, 128 steps,
  four minibatches and four PPO epochs;
- learning rate .00025 with annealing, gamma .999, GAE .95, PPO clip .2,
  entropy .01, value weight .5 and gradient norm .5;
- direct GPT-5-mini with full symbolic map and executed-action history;
- 480 maximum queries and 480 maximum delivered labels, one attempt, zero SDK
  retries, 16,384 input-token and 3,500 output-token ceilings;
- labeled-sample mean imitation cross-entropy, coefficient 1 decreasing to
  .01 by halfway and disabled after 75% of training;
- teacher-free evaluation: 50 greedy and 50 sampled episodes every 200
  rollouts, plus 2M, 5M and final milestones.

The archived base trainer is commit
`1ab5fc0d9c9d22b7147d58ba92245bcc42c0b008`. The snapshot stores both the
unmodified trainer and the explicit opt-in random gate. Probability20 leaves
that gate off and must reproduce the original trainer exactly under an
offline deterministic teacher test.

Before joining results, require matching task, seed, student background,
observation mode, optimizer/training settings, query/token limits, evaluation
schedule, initial-policy hash, and planned seed set. GPT replies may differ
between arms because calls are made at different states and times.

## Outcomes and inference

Primary metric: normalized trapezoidal area under teacher-free greedy success
from 204,800 through 9,999,360 transitions. Also report complete curves,
individual seeds, 2M/5M/final success, censored time-to-80%, calls, delivered
labels, failures, tokens, cost and wall time. Training seed is the replication
unit; evaluation episodes are not additional replications.

The completed entropy-versus-no-teacher result remains the original
exploratory contrast. New prespecified questions are:

1. Within Count-PPO, does probability20 improve on completed entropy?
2. Within Count-PPO, does entropy improve on random querying?

Report paired effects with ordinary 95% and Bonferroni 97.5% Student-t
intervals for these two questions, plus exact sign-flip sensitivity. PPO
background contrasts and random-versus-no-teacher are exploratory and use
clearly labeled unadjusted intervals. Five seeds do not guarantee precision.
Do not add seeds or select a different primary comparison after seeing these
outcomes. A joined exploratory result is not an independent replication of
the original entropy effect.

## Funding and launch

Each seed block adds four paid runs: probability20 and random for PPO and
Count-PPO. At $5.33 worst-case reservation per run, this is **$21.32 per
seed block, $106.60 for all five, and 9,600 maximum new calls**.

The initial $150 ceiling permitted two seed blocks after reconciliation.
**Dated amendment, 2026-09-17:** an explicit authorization set a **$203.13
total project ceiling** to admit all 20 missing runs together. Auditing
completed array 959691 retains known standard-rate usage plus the full bound
for unknown attempts: $21.25 rather than the old $53.30 hold. With the last
audited ledger, prior liabilities total $96.53, leaving exactly $106.60 for
the full extension. This is a reservation calculation, not an expected bill.

The launcher checks the live ledger and preserves unrelated holds. It raises
the allowance only to the authorized total, verifies the completed cohort,
and prepares all five blocks before submission. Arrays have no concurrency
throttle. The detached launch checkout shares the original results directory,
ledger and credential file while preserving the cluster branch/local edits.
No scientific settings, seed assignments or stopping rules change.

In addition to the opt-in random gate, the source archive records two
accounting repairs: the authorized ceiling in the worker admission check,
and decimal subtraction of stored dollar amounts. The latter prevents a
binary-float rounding error from rejecting the fifth $21.32 block when that
exact amount remains. It does not forgive overspending or release a hold.
