# O2: fresh-world evaluation of all fifty historical Crafter policies

Task W07/A01.
Status: prospective implementation. No scientific evaluation is launched.

Code: `scripts/paper_optional_crafter_eval_20261005.py`. Tests:
`tests/test_paper_optional_crafter_eval.py`. The parent optional launcher
owns the committed source archive and user-submitted Slurm array. All
fifty input models must exist before admission; this study does not wait
for the newly submitted 640-run bulk experiment or train new students.

## Frozen scientific contract

This is an exploratory final-policy generalization study. Reuse exactly
the fifty final models in
`results/crafter_vulcan/crafter_vulcan_20261002`: seeds 31--40 in all five
arms `none`, `rules_v3b_weak`, `rules_noprog_weak`, `count_none` and
`count_rules_v3b`. There is no checkpoint or model selection by outcome.
Historical arguments, raw twenty-panel learning curves, terminal summaries,
finite model tensors, the complete manifest and all successful exit
receipts are required before freezing any evaluation input.

Every model receives the same fifty unscreened world seeds
45,500,000--45,500,049 in ascending order. These are disjoint from known
30M construction/check panels, 40M training pools (including new bulk
seeds 141--150), original 41M evaluation worlds, 43M revised-bank panels
and the previous 44M fresh-world evaluation. This is another panel from
the same symbolic Crafter generator, not a harder task distribution.
The panel is frozen before results; no worlds are dropped or replaced.

There are exactly 2,500 episodes: one array task per policy, fifty episodes
per task. Keep the original local 7 by 9 categorical grid and inventory/
facing/daylight/sleep vector, feedforward architecture, seventeen actions,
rewards, native termination and 3,000-step evaluation cap. Keep the
existing deterministic despawn fix. Sampling follows the original
`ppo_crafter.evaluate`: a dedicated Torch generator seeded once by the
model's training seed, carried across all fifty worlds. Paired policies
start from the same stream seed; different episode lengths can change
later stream positions. No teacher, advisor, imitation loss, optimizer,
policy update or paid API call is involved.

The endpoint is the mean number of distinct achievements per episode for
each policy, averaged over its fifty worlds. Training seed remains the
replication unit: ten paired model means per contrast, not 500 independent
episodes. Differences are computed from integer achievement totals divided
by fifty; report paired 95 percent t intervals and two-sided tests.
The three prespecified primary contrasts share a Holm correction:

1. Plain full bank minus plain unguided PPO.
2. Count full bank minus count unguided PPO.
3. Plain full bank minus plain no-progress bank.

Positive mean and adjusted p below .05 support the specified endpoint
contrast on this fresh panel. Other outcomes remain inconclusive or
negative; there is no equivalence/noninferiority claim. Undefined tests
from zero empirical variance remain null, consume p=1 only for family
bookkeeping and cannot establish superiority. No result changes the
historical learning-summary inference.

Secondary descriptive contrasts are plain no-progress minus plain none,
count none minus plain none, and count full minus plain full. Retain each
seed's outcome, individual achievement rates, Crafter geometric score,
episode lengths, deaths, actual transitions and action traces. All fifty
endpoints and all 2,500 raw episodes are required for inference. Intervals
condition on this fixed shared panel and existing banks; they do not
estimate uncertainty from bank generation or provide new training seeds.
No pooling with old learning curves, old fresh-world results or new bulk
training is allowed. Null outcomes complete the study without expansion.

## Input freezing and execution

`check(root=ROOT)` performs software, runtime and seed-panel admission
without historical model files. `preflight(source_batch, root=ROOT)`
inventories all required artifacts and validates all fifty historical runs
without creating an output directory or copying files. The parent launcher
runs it before creating its outer archive. Missing models in a metadata-only
sync therefore produce a clean admission error with original inputs intact.
`prepare(source_batch, out, root=ROOT)` reuses this same preflight, then
copies historical arguments, summaries,
evaluation journals and checkpoints into exclusive `out/inputs` folders.
Copies must match recorded SHA-256 values before an exclusive manifest is
written. A partially prepared output is preserved and requires inspection;
it is not overwritten. Historical output-path spellings are preserved as
provenance after copying; every scientific argument remains exact.

The manifest includes every original artifact hash and configuration,
source manifest, original successful exit hashes, canonical policy tensor
hashes, all relevant repository Python source hashes, this protocol, bank
hashes, the complete target panel and runtime. Accepted Crafter package
identifiers are exactly `1.8.3` and `1.8.3+computecanada`. Record the full
identifier rather than stripping the build suffix. Python, Torch, NumPy,
SciPy, opensimplex, Gymnasium, Pillow, Torch source version and CPU thread
count are also retained. Workers must equal the prepared runtime exactly
and import this module from the frozen archive.

`run_cell(index, out, root=ROOT)` executes one index 0--49. It verifies the
frozen input hashes, strictly loads finite tensors, hashes the frozen
policy, and takes exactly one attempt. An exclusive start receipt blocks
duplicate workers and silent resumption. Each episode is flushed to a
journal with model/manifest/world identity, achievements, termination,
reward, length and action-sequence hash. A failed or interrupted attempt
retains its partial evidence and consumes the one-attempt allowance;
there are no automatic retries or selected replacements. Parameter hashes
must match before and after evaluation. Final receipts record zero policy
updates, teacher use and API calls.

There is no inherited October 1 calendar deadline or expired replay gate.
Each task has a 7,000-second elapsed evaluation limit, checked before world
generation, before each episode and at least every hundred steps, inside
the two-hour Slurm allocation. At most 150,000 transitions are evaluated
per policy and 7,500,000 across the complete grid. Resource request is one
CPU, one Torch thread and 8 GiB per array task. No array concurrency cap
is introduced. User alone submits jobs. Runtime and completion are unknown
until actual receipts arrive; the bound is not a throughput forecast.

`validate_manifest(out, root=ROOT, worker=False)` checks frozen source,
configuration and records. Worker mode additionally requires the exact
prepared runtime. `report(out, root=ROOT)` can audit copied outputs under
another reporting runtime, but it requires matching source hashes and
checks each saved worker runtime against the manifest. It recomputes every
endpoint from raw episodes, confirms all receipts, preserved policy hashes,
time limits and frozen input hashes, then writes an exclusive report.
Repeated reporting is allowed only when the complete derived result is
identical. It never launches an episode or changes a historical result.

## Validation

Run from the repository root using the project environment:
`python -m pytest tests/test_paper_optional_crafter_eval.py -q`.
The software admission command is `python -m
scripts.paper_optional_crafter_eval_20261005 check`, expected: 50 cells and
2,500 episodes. The parent launch receipt records the committed revision,
archive identity and array command.

| State | Evidence |
|---|---|
| Jobs completed | None at preparation; job IDs unavailable. |
| Artifacts validated | Software fixtures only until source admission. |
| Conclusions independently reviewed | Pending; no new outcomes exist. |

The next scientific artifact check after collection is the complete
`report` command followed by independent review of its raw episodes and
inference. A different reviewer must clear conclusions before any paper
or project-brief promotion. Missing historical inputs block only this
optional source-dependent suite; they do not authorize new training.

Local implementation validation, 2026-10-05: all fourteen focused tests
passed in 16.37 seconds. Synthetic fixtures exercised all five original
recipes, complete fifty-model preparation, all fifty worker dispatches,
2,500 synthetic episode records, once-only dispatch, repeated identical
reporting, runtime/recipe/checkpoint/journal corruption and undefined-test
handling. A separate real four-step engineering-world rollout matched
the original sampled evaluator and preserved policy parameters. It used
world seed 87320019 and action seed 8731997, outside scientific panels;
no historical study model or new study world was evaluated. This is
software validation, not independent scientific clearance.

Preflight refinement, 2026-10-05: the complete focused suite now passes
fifteen tests in 21.38 seconds. The additional test removes one synthetic
checkpoint and confirms that both preflight and preparation reject it
without creating an evaluation output or changing any remaining input
file. Full fifty-model preflight also passes before the existing synthetic
worker/report test. This changes admission order only; source recipes,
policies, world panel and evaluation behavior remain unchanged.
