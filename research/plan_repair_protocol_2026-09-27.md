# Plan-target correction diagnostic, v1

Freeze date: 2026-09-27. Task: W05/P03.
Status: prospective development protocol; implementation and independent
launch review must pass before submission. No empirical outcome yet.

## Decision and motivation

The completed five-format study did not nominate a semantic mechanism.
Its full-state plan bank contains categorical errors (40/58 audit cases
exact), so that study did not test uniformly correct plan targets. This
diagnostic asks whether repairing those errors improves learning under the
**same input, cases, auxiliary head, scale and schedule**. It is not an
automatic expansion of the old study or a claim that errors caused its null.

We choose DoorKey8 Count-PPO, one background, to measure learning efficiency
above a capable baseline. Plain-PPO rescue and transfer to another task
remain separate decisions. No new teacher model, API call or embedding is
needed. Simulator-corrected targets are reference supervision, not improved
LLM output or a deployable language method.

## Four arms and fixed inputs

Twenty runs: five fresh paired training seeds **12700000, 12700100, 12700200,
12700300, 12700400**, each in these four arms:

| Arm | Auxiliary supervision | Action loss |
|---|---|---:|
| `ppo` | No applied auxiliary loss; matching replay schedule | 0 |
| `raw` | Original cached GPT-5-mini categorical plan labels | 0 |
| `corrected` | Native-state-corrected categorical plan labels | 0 |
| `permuted_corrected` | Corrected donor bundles from other episodes | 0 |

Reuse exactly the full-state bank's 182 training and 58 audit cases, their
order, images, endorsed/foil actions, source episode split, and donor mapping.
The original bank is byte-preserved. Correct only `plan`:
next-target object, egocentric direction and Manhattan-distance bucket.
These describe the fixed next-phase target under the existing convention,
not a guaranteed optimal plan or the consequence of following a new route.
Recompute phase from restored native state and cross-check stored geometry.
Rebuild permuted plan targets from corrected donors, preserving complete
bundle marginals. Report changed fractions and label distributions; do not
filter cases based on teacher errors, visibility, or audit outcomes.

The student acts using its local 7x7 symbolic observation and recurrent
policy. Auxiliary replay predicts from **current-image encoder features plus
the fixed action pair**, bypassing the GRU, exactly as before. No claim of
history-based reasoning or information unavailable in this input being
perfectly recoverable is made. Describe ambiguity; do not use observability
as an admission gate for privileged supervision.

Keep the previously frozen raw-plan loss scale **0.22896900710982449** for
every active arm. Do not recalibrate corrected labels or select a coefficient
from learning outcomes. The original calibration file and bank hashes are
sealed. Correction can change gradient magnitudes; record them rather than
normalizing away this part of the treatment. Head architecture and allocation
are checked with a detached-head engineering control, outside the 20 cells.

## Training and exposure

Retain the existing recipe: 10,000,000 nominal / **9,999,360 actual transitions**;
8 environments x 128 steps; 4 recurrent minibatches and 4 PPO epochs;
learning rate .00025 with linear annealing, gamma .999, GAE .95,
clip .2, entropy .01, value coefficient .5, gradient clipping .5,
normalized advantage and clipped value loss. Dual critics, observation-based
count bonus, intrinsic gamma .99/coefficient1/reward normalization unchanged.

One 32-case replay minibatch every four rollouts, for **1,830 opportunities**
at rollouts 4..7,320: **58,560 replay slots**, not new teacher calls.
The same private replay RNG draws the same case IDs in every arm of a seed.
PPO logs but does not apply replay supervision. No online guidance or reward
shaping. The final 2,503,680 transitions have no auxiliary replay.
This inherits the previous dose to isolate correction; no timing claim.

All new cells explicitly opt into isolated advisor RNG; online guidance is
off, so the RNG repair is not the target-correction treatment. Prior saved
studies and default legacy behavior remain unchanged. Separate real-trainer
tests require no-teacher and isolated zero-label sham scheduling to produce
identical policies, including the formerly problematic uniform query path.

## Metrics, data separation and rules

Primary: normalized trapezoidal greedy **teacher-off** success AUC on
[51,200, 9,999,360]; 50 evaluation episodes each 51,200 transitions plus
terminal, evaluation seed base training seed+50,000. Retain all failed-learning
seeds and the complete unsmoothed curves. Training seeds are replication units.
Report paired mean differences and descriptive 95% Student-t intervals, df4.
Do not use episode counts as independent n, change endpoints, or stop at a
favorable checkpoint. Report final success, wall time and gradient evidence.

Primary correction nomination: `corrected - raw` mean AUC >= .03 and
at least 4/5 positive pairs. A **useful semantic candidate** additionally
requires `corrected - ppo` >= .05 and `corrected - permuted_corrected` >= .03,
each with at least 4/5 positive pairs. These are development decision rules,
not significance tests, power guarantees or novelty clearance.

- All three rules pass: nominate this configuration for a new, separately
  frozen confirmation/transfer design; do not launch it automatically.
- Correction rule only: label repair may reduce harm; no useful explanation
  benefit claim or automatic expansion.
- Complete valid data but rules fail: stop expansion of this exact target,
  input and objective configuration. Do not generalize to all explanations.
- Missing/invalid artifacts: inconclusive; repair infrastructure evidence,
  not the observed outcome. No promotion from fewer than five complete pairs.

The old bank and its audit outcomes were already inspected during development.
Audit episodes remain excluded from updates/calibration and measure common
native-target accuracy for all arms; they are **not untouched confirmation
data**. Fresh training seeds do not turn a reused selected bank into an
independent test of general explanation quality. Head accuracy is diagnostic,
not a substitute for policy AUC or evidence of causal reasoning.

## Execution and stopping

One unthrottled array of 20 cells, CPU2/memory20G/four-day scheduler limit
per cell. Each cell executes once; a completed poor result is not retried.
Infrastructure failures are retained, reviewed and require an explicitly
versioned recovery decision; this launcher does not auto-retry or silently
resubmit. Repeated launch commands report an existing submission.
Archive committed source, immutable input copies, derived-bank report,
runtime package identity, cell dispatches, terminal checks and policy hashes.

API consultations/tokens/dollars: **0 / 0 / $0 new spend**. No reservation or
ledger access, no credentials loaded, no Slurm submissions by agents. User
runs the published launcher. No old control is rerun as an alleged new seed;
all four arms use the five new seeds for this newly frozen comparison.

Exact commands are recorded with the launch.
Novelty remains unestablished: LLM4Teach, explanation advising, auxiliary
prediction and belief-representation priors in `03_novelty_ledger.md` apply.
This experiment diagnoses a limitation; success alone is not an AAMAS
contribution or a new explanation algorithm.
