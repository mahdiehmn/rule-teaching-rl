# MultiRoom replication of native GPT advice

Date: 2026-09-16.
Status: array 959691_0..19 running as p1_multiroom on 2026-09-16.
The previous assertion that this was unsubmitted was incorrect. Its published
configuration uses direct GPT-5-mini llm_general, not the subgoal+BFS teacher.
Preserve the running snapshot; do not prepare duplicates. The proposed v2
replacement has been withdrawn. Actual consultation progress and completion
must be checked before deciding whether to cancel the existing v1 workers.

Progress update (worker journals, timestamps 2026-09-17 03:28-03:29 UTC;
not a completed-result audit): ten guided workers report 4,464/4,800
consultations, including 413 marked failed, leaving at most 336 consultation
slots. Guided summaries show 216,064-382,976 transitions. The previous checker
missed guided run names and incorrectly displayed zero calls. Recommendation:
finish the existing frozen batch to obtain matched learning curves; do not
increase its dose mid-run. Inspect failure metadata before attributing those
failures to model quality. Completed consultations are not equivalent to
delivered advice, and these progress counters establish no learning benefit.
Task: P03.

## Decision and limits

Test whether the native entropy-advice procedure selected from DoorKey
improves teacher-free learning on MultiRoom N6. This is transfer of a training
procedure, not transfer of trained weights. It is not a novelty claim or an
advisor-ranking study. Existing programmatic-teacher evidence motivates the
task; it does not establish GPT competence or a GPT learning benefit.

Run exactly 20 cells: five new paired seeds (8100000, 8100100, 8100200,
8100300, 8100400), two student backgrounds (none/count intrinsic reward),
and two arms (no teacher/entropy-guided GPT). No adaptive extension of seeds,
replacement of unsuccessful learning curves, or selection of a favorable
checkpoint. Keep inadequate-learning outcomes. Infrastructure failures are
reported separately and reviewed before any separately recorded retry.

The hypothesis is faster learning from GPT advice in each background.
Primary endpoint: normalized trapezoidal AUC of teacher-free greedy success
from the first common evaluation (204,800 transitions) to 9,999,360.
For each background report all five paired AUC differences, mean difference
and a two-sided Student-t 95% interval with four degrees of freedom.
A positive interval is preliminary support for that background; a negative
interval supports harm; an interval spanning zero leaves the effect uncertain.
Report both prespecified contrasts regardless of outcome; if claiming either
as a family-wise positive finding, also require a two-sided 97.5% interval
above zero (Bonferroni for two contrasts). This small exploratory selection
is not a confirmatory claim of cross-task generality.

Secondary: full curves, success at 2M/5M/final, descriptive first evaluation
reaching 80% (right-censor runs that never reach it), calls, delivered labels,
tokens, actual billing if available, wall-clock time, and query-position
histograms. Each seed is one replicate, not each evaluation episode. Plot
individual curves and pointwise seed-level intervals; do not describe these
as simultaneous confidence bands. Count-PPO's prior near-ceiling AUC leaves
little average improvement available; inspect learning time, not final alone.
An inconclusive outcome permits a proposed follow-up, not automatic spending.

## Exact intervention

Task `multiroom_n6` = `MiniGrid-MultiRoom-N6-v0`: six connected rooms.
Student local symbolic 7x7 observation resized to 56; existing CNN+GRU512
and two value heads in both backgrounds. Teacher GPT-5-mini sees the full
unmasked symbolic map and executed-action history. The student executes its
own sampled action throughout; no teacher override, explanation head or
replay. The teacher's returned action distribution enters the imitation
cross-entropy, averaged only over labeled minibatch samples.

| Setting | Value |
|---|---|
| Training | 10M nominal / 9,999,360 actual transitions; 9,765 rollouts |
| Rollout/update | 8 envs x 128 steps; 4 minibatches; 4 epochs |
| LR | .00025, linearly annealed, both backgrounds |
| Discount | extrinsic .999; intrinsic .99; GAE .95 |
| PPO | clip .2, entropy .01, value .5, max gradient norm .5 |
| Count | same existing observation-count reward; normalized; weight 1 |
| Imitation | coefficient 1 -> .01 over first half; .01 until 75%; then 0 |
| Evaluation | 50 greedy and 50 sampled episodes, teacher absent |
| Evaluation timing | every 200 rollouts; additional 2M/5M and final |
| Teacher budget | 480 consultations and at most 480 deliveries per guided run |
| Service | OpenAI default tier; one attempt; SDK retries zero |
| Token ceilings | input 16,384; output 3,500 per request |

No fixed query windows, query_interval=1. The existing entropy percentile
and delivery-pacing controller decides eligibility while guidance is active.
Its first 100 observations are calibration opportunities; its history window
is 2,000 observations and it updates the threshold every 25 opportunities.
All valid selected targets are delivered. Entropy here means student
uncertainty, not known teacher correctness. The 480 cap is retained for
comparability, not justified as sufficient or optimal. DoorKey exhausted it
early; record actual MultiRoom query times and do not promise spread timing.

## Source and cost

The JSON contains the four **deployed** original DoorKey argument templates,
not reconstructed defaults. Only task, training seed and experiment identity
change. Execute the byte-verified original trainer commit
`1ab5fc0d9c9d22b7147d58ba92245bcc42c0b008`; preparation archives it from Git.
New configuration/protocol files accompany the snapshot but no executable
training source is edited. The preparation commit is separately recorded.
Matching initialization is checked after runs; stochastic GPT supervision is
not guaranteed identical across trajectories. Environments differ, so neither
cross-environment observations nor outcomes are paired causal contrasts.

Ten controls make no API calls. Ten guided runs permit at most 4,800 calls;
each reserves $5.33, for **$53.30**. This is a standard-rate upper reservation,
not a predicted invoice or a claim that sharing discounts cover the batch.
The total project ceiling remains $150. Optional reconciliation checks all
40 completed DoorKey cells against scheduler state, pinned manifests, source,
dispatch/summary/journals and child ledgers. It retains recorded usage at
standard prices plus a full bound for each unknown-cost attempt, as OPEN
holds. It does not settle invoices, forgive failures, or touch Qwen jobs.

The existing worker stops subsequent starts after a failed cell. The trainer
stops if consultation failures exceed 20% after at least 40 samples. Do not
silently replace failed labels. No array concurrency cap is added.
