# DoorKey random-query extension

Date: 2026-09-18.
Status: prospective exploratory extension; no jobs submitted by agents.

## Decision and scope

Complete the missing random-query comparison on DoorKey without repeating the
40 completed native-advisor cells. Ask whether uniformly distributing the same
maximum 480 consultations changes learning relative to no advice, entropy
advice and probability-based correction. Do not replace the native study's
original primary contrast or the running MultiRoom confirmation (994232).

Add exactly ten cells: five existing paired training seeds
7600000/7600100/7600200/7600300/7600400, each with plain PPO and Count-PPO.
These are reused development seeds, not independent confirmation seeds.
Teacher: GPT-5-mini with the existing full symbolic teacher access. Advisor:
the existing UniformQueries sampler used for MultiRoom. Channel: action-target
cross entropy added to PPO; no explanation loss or executed-action override.
The sampler spreads consultation opportunities over the existing guidance
horizon; realized queries can be below 480. It is not mistake filtering.

## Matched configuration and provenance

Derive each new cell from its original native entropy cell. Preserve the
student's local symbolic view, network, optimizer, loss normalization,
coefficient schedule, teacher cutoff, evaluation settings and seed. Change
only run identity and advisor=unlimited, importance_source=none,
uniform_queries=true. Pin all five original manifests and the archived trainer
source; reject incompatible or missing originals. Reuse the existing reviewed
UniformQueries source patch. No arbitrary query windows or array throttle.

Random timing changes the states visited at consultation and the imitation
coefficient in effect then, because that coefficient decays. This compares
complete advising procedures, not state selection in isolation.

Each run retains 10,000,000 nominal / 9,999,360 actual transitions, 8 environments
and 128 steps per rollout. The consultation cap remains 480, independently of
labels delivered, cache hits and HTTP attempts. No early stopping on success.

## Outcomes and decisions fixed before the new runs

Primary descriptive learning measure: teacher-off greedy success AUC under the
same evaluation rule as the native comparison. Report all individual curves,
final success, paired effects and uncertainty. The newly specified random-minus-
no-advice comparisons for the two student backgrounds form an exploratory
two-contrast family: report 97.5% marginal intervals and paired differences.
Use paired Student-t intervals (df=4 with five complete pairs), alongside
individual differences and exact paired sign-flip sensitivity.
Random-versus-entropy/probability is secondary exploratory analysis. No claim of
an advisor winner without sufficient uncertainty evidence; five reused seeds
cannot confirm a newly selected hypothesis.

Retain the native >=80% success threshold at two consecutive evaluations.
Report first-high onset and the second, confirming checkpoint separately.
Count queries/labels/cost and elapsed time through the confirming checkpoint.
Include non-reaching runs as right-censored at the common horizon, not as zeros
or omitted observations. Threshold analyses newly added here are descriptive.
Do not infer that calls after reaching the threshold were unnecessary: stopping
consultations there has not been tested. Report standard-rate estimates apart
from provider-billed cost and retain uncertainty from timed-out requests.

If random advice improves AUC consistently but intervals remain broad, it
supports a fresh-seed confirmation proposal, not automatic expansion. A null or
negative result is retained. Stop after these ten cells. Infrastructure failure
does not permit replacement with favorable seeds; any retry needs its existing
attempt evidence and a separately bounded plan.

## Funding and launch

Whole-batch reservation: $53.30. Keep the authorized $203.13 total ceiling and
all active holds intact. Following the confirmed $87 reconciliation and the
running $106.60 confirmation plus $1.89 retained uncertainty, only $7.64 would
be free: preparation must refuse if the live ledger cannot cover $53.30.
Do not use account credit as permission to exceed this project ceiling.

The launcher previews without requests, verifies original manifests and checks
funding before creating paid work. Preparation creates an immutable snapshot
and user submission command; only the user runs sbatch. Existing output or
submission files must not be overwritten. Reconcile the active confirmation
only after completion and validated accounting; do not release it for this
extension. No paid rerun is required to compute the accompanying descriptive
efficiency report from already completed runs.
