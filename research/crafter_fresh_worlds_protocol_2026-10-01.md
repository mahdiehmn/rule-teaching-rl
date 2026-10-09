# Candidate Crafter evaluation on fresh worlds

Date: 2026-10-01, 11:58 UTC.
Existing task: A01. Status: prospective engineering candidate, NOT FROZEN
OR LAUNCHED. Independent implementation and complete-learning review are
required before freezing. If launched, this is the third and final new
scientific follow-up in the authorized ten-hour continuation.

## Decision and information state

The ongoing learning comparison uses ten shared evaluation worlds at
every checkpoint. Its fresh-training-seed confirmation also uses those
worlds. This candidate asks whether the final v2 policies exhibit a weak-
rule versus unguided performance difference on a new panel drawn from the
same environment generator. It is evaluation-world generalization within
the declared symbolic Crafter variant, not a new task family, new student
training, or independent confirmation across new training seeds.

At drafting, 20/30 exploratory runs are complete; no partial learning
contrasts have been inspected. Earlier interim notes exist and are
development context, not complete-cohort evidence. The candidate's question
arises from the known ten-world evaluation design. Selection to execute is
deferred until all30 learning artifacts and their primary inference have
independent clearance. Positive, null and negative primary learning results
all permit the same candidate if time admits; none permits dropping an arm,
changing the panel or replacing the original learning inference.

## Frozen choices for an eventual launch

- All30 final v2 models: seeds1--10, arms none/rules_weak/rules. No model
  selection by endpoint, AUC or checkpoint. Read-only SHA-256 identities.
- Fifty fresh shared world seeds44000000--44000049, disjoint from v1/v2
  construction30M blocks, training40M blocks, original evaluation41M, and
  v3 construction/check/dev43M blocks. This seed block is reserved for
  this study. Seeds are generated mechanically, never screened for difficulty.
- Same original c935b98 architecture and symbolic adapter materialized for
  the already accepted endpoint reproduction; all archived file hashes and
  the reproduction manifest/report identities remain pinned.
- Same sampled-policy action mode, local7x9 categorical grid and22-vector,
  17actions, rewards/dynamics/despawn fix, native death and3000evaluation
  cap. No teacher, advisor, optimizer, training or parameter update.
- One Torch generator per model, seeded by that model's training seed,
  carried across the50 worlds. No per-world reseeding or best-action mode.
  Models with the same seed therefore begin with the same sampling stream;
  different episode lengths can make their later stream positions differ.
- Run seeds in ascending order; arm order rotates across the three positions
  by(seed-1) modulo3. Exactly1500episodes, at most4.5Mtransitions. OneCPU
  worker and oneTorch thread; do not overlap the active DoorKey evaluator.

## Outcomes and interpretation

For each model, the endpoint metric is mean distinct achievements over the
50episodes. The primary contrast is rules_weak minus none, paired across
the ten training seeds. Compute differences from integer achievement totals
divided by50, a95% paired-t interval and a two-sided paired test. A positive
mean with p<.05 supports a difference on this fresh panel; otherwise report
the complete estimate as inconclusive or negative, as appropriate. This is
an exploratory endpoint test, not the original20-checkpoint learning AUC.
Do not pool it with learning AUC or turn it into a rescue of that hypothesis.

The stronger-rule arm remains in every table/figure. Its two additional
contrasts are secondary descriptive estimates, not additional routes to a
primary claim. Report every seed, each achievement's rate, geometric score,
episode lengths and deaths descriptively. Pointwise intervals condition on
this50-world panel and fixed bank; they do not estimate bank-generation
uncertainty. Zero empirical variance yields an undefined t-test, with no
superiority or equivalence inference. All failed and poor outcomes remain.

The original ten-world final means may be shown alongside the new-world
means with their distinct panel sizes and original sampling stream stated.
Do not perform a selected-world analysis, choose favorable achievement
subsets, or claim an isolated causal effect of changing world difficulty.

## Admission, time and failures

Before freeze, require a complete30-run audited learning report and a
separately authored acceptance receipt binding its SHA-256. The root
producer's self-audit alone is insufficient. Pin the report, all input
artifact hashes, runtime, code and this protocol in a new manifest.

Use a conservative scheduling forecast of60seconds for world generation
plus0.003seconds per expected transition, where expected transitions equal
50 times the sum of all30 original final mean episode lengths. This fixed
rate is an engineering allowance, not a measured throughput claim. Launch
only if the forecast is at most7200seconds and fits before16:30UTC. Repeat
the admission check at actual start. An optimistic forecast cannot extend
the hard limits.

Hard stop is the earlier of7200seconds after start or16:30UTC. Check it
before each world generation, each episode and at least every100steps.
Retain an incomplete cell and completed episode journals if time expires.
One attempt per cell; no automatic resumption or retries. A failed cell
ends the batch, with an explicit failure receipt. A complete poor result
is never retried. Report generation refuses incomplete or inconsistent
panels. No inference from a favorable completed subset.

## Evidence and review

Persist a manifest, start and final run receipts, per-cell start/end
receipts, one row per episode including world/seed/arm/action hash and
achievement set, and immutable cell summaries. Hash the policy before and
after; strictly load finite tensors. Independently aggregate the raw1500
episodes into seed means and all contrasts. A different reviewer checks
artifacts/inference before any paper, brief or abstract promotion. No new
LLM requests, spending or cluster jobs. v3 and all training/confirmation
cohorts are unchanged.

This candidate does not revise any prior study, no-harm rule, failure,
deadline or minimum-ten-hour commitment. The final launch decision and
actual completion status are recorded separately.
