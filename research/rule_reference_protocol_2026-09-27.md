# Finite rule-teacher references, v1

Date: 2026-09-27.
Prospective protocol; no new learning results.

## Decision and scope

This batch adds reliable rule-based references alongside GPT advising and
the running explanation study (job 1201595). This batch measures
whether finite rule advice helps PPO and Count-PPO, and whether advisor timing
or withholding changes that effect. It is baseline evidence, not a novelty
claim, a replacement for explanation research, or proof of a perfect teacher.
Do not expand it automatically or substitute successful seeds for failures.

Twenty fresh training seeds per task are paired across conditions. This is a
fixed precision improvement over five, not a guaranteed power calculation or
a universal sufficient seed count. Evaluation episodes are not replications.

| Task | Rule teacher | Seed block (stride100,20seeds) | Student input |
|---|---|---|---|
| DoorKey8 | `oracle`, shortest-path rule planner |12800000..12801900|local symbolic7x7|
| MultiRoomN6 | `door_bfs`, stateless door/navigation planner |12900000..12901900|local symbolic7x7|
| KeyCorridorS3R3 | `bot`, native BabyAI planner |13000000..13001900|local symbolic7x7|

All students are recurrent, have dual value heads, and either no intrinsic
reward (`none`) or the existing count reward (`count`). These are operational
student backgrounds; do not promise that every no-count seed must fail.

## Frozen cells and advice semantics

Core: three tasks x two student backgrounds x four arms x twenty seeds =480.
Arms: no teacher; entropy importance480; probability mistake correction480;
uniform random480. All limits are selected consultations, not guaranteed labels.
No exact-action correction, early-only arm, dense advice arm, or second view.

Secondary: Count random120 on all three tasks (60cells); MultiRoom Count
random15/30/60 (60); KeyCorridor plain random120 (20). Total620 new cells.
These doses cover existing paid studies; there is no post-result search over
arbitrary budgets. Run order interleaves replicate/task/background/arm in the
manifest. Slurm may schedule any order, with no array concurrency throttle.

Entropy and probability use the original normalized-entropy importance signal,
native adaptive delivery pacing, starting threshold0 and advice_rate0.
Probability additionally withholds a returned selected action when the student
assigns it probability >=.20. The query has already been counted. The new
opt-in `advisor_no_teacher_peek=True` hides free `peek_action` and
`optimal_actions`; it does not let the rule arm get a free pre-query mistake
test that GPT lacks. The base probability test uses the selected hard action;
rule action targets and GPT probability distributions remain different teacher
outputs. No assertion of identical supervision is made.

Uniform random uses the existing seed-fixed clock-slot sampler throughout the
positive-coefficient interval. Terminal/reset slots are not replaced. Each
guided cell has equal query/advice caps, no query windows, query_interval1,
no teacher action override and no replay. The student executes its own action.
Only action cross-entropy enters learning; no explanation head is activated.

KeyCorridor's bot maintains state across executed actions. Its guided cells
therefore use `teacher_stream=True`: the planner processes every active
non-reset state, then the advisor selects at most its finite cap. Streaming
ends at the teacher cutoff, not when selected queries run out. Selected
consultations, delivered labels, actual reference calls, compute units and
reference wall time must be reported separately. It can make millions of
planner calls while delivering at most480 labels. This is not equivalent total
teacher-compute cost to480 API calls. DoorKey/MultiRoom query their stateless
planners directly. The bot also has visibility-history and privileged mission
internals; it is not an identical-information substitute for the full-map GPT.

## Learner and evaluation settings

- Native `algos.ppo_distill`: symbolic7x7 resized56, CNN+GRU, recurrent512,
  dual critics, CPU, eight environments,128steps,1024transitions/rollout.
- 10,000,000 nominal frames =>9,999,360 actual transitions,9765rollouts;
  four minibatches, four epochs, lr.00025 linearly annealed, gamma.999,
  intrinsic gamma.99, GAE.95, countcoef1 with reward normalization.
- PPO clip.2, entropycoef.01, valuecoef.5, gradient norm clipping.5,
  advantage normalization and clipped value loss.
- Imitation coefficient1 to.01 over first50%, .01 to75%, then zero;
  `distill_normalization=labeled` (mean over delivered labels). This protocol
  does not answer the separate loss-normalization question.
- Greedy and sampled teacher-off evaluation:50episodes each, every200rollouts,
  floor-to-rollout2M/5M milestones and final endpoint; reset seeds training+50000.
- Initial policy hashes, final finite checkpoint, complete native evaluation
  and consultation journals, source and runtime fingerprints are required.
- `advisor_rng_isolation=True` separates advisor ordering from PPO minibatch
  randomness. No sham scientific arm: actual no-label software controls already
  test this fix. Defaults of historical studies remain unchanged.

## Historical comparison and reuse

No completed exact run is relaunched. The historical180 free dense/exact cells
and20 MultiRoom dose cells have different seeds, budgets and/or advisor RNG
semantics. New seeds are needed for this prospective internally paired cohort.
Old GPT curves can provide context, but source/RNG, seed cohort, teacher output
and planner access differences prohibit presenting their subtraction from
these results as a matched causal rule-versus-LLM effect. A future direct
teacher comparison would need matched prospective GPT cells; not authorized
by this zero-API baseline batch. Do not hide a rule teacher that beats GPT.

## Analysis and decision rule

Primary outcome is normalized trapezoidal teacher-off greedy-success AUC from
the first evaluation (204800 transitions) through the common endpoint9999360.
There is no invented evaluation at step0. Reporter receives
only independently validated complete artifacts. Each seed supplies one AUC.
Final greedy success, sampled curves, actual label counts, planner computation
and wall time are secondary/descriptive. Plot separate task figures, plain and
count panels, unsmoothed mean curves, and explicitly pointwise95% t intervals.
Display clipping at[0,1] does not change raw interval statistics. These bands
are not simultaneous95% confidence over an entire learning curve.

Five predeclared paired AUC contrasts per task/background: entropy-none,
probability-none, random-none, probability-entropy, random-entropy.
Require all20 pairs in a stratum before its two-sided tests; report paired
95% t intervals, raw p, Holm over its five contrasts. For project-wide
selection, Holm over all30 core contrasts after all six strata finish.
Incomplete families withhold formal p-values. Raw95% intervals are not
family-adjusted. Across-task overall winners require the global correction;
an effect in one setting must not be generalized to all settings.

Evidence of benefit/harm: signed AUC effect, paired95% interval on that side
of zero, corresponding Holm-adjusted p<.05; global wording also needs global
Holm. Otherwise uncertain. No required positive result and no automatic paid
expansion. Low-dose versus random480 comparisons are exploratory intervals;
failure to detect a difference does not establish noninferiority. Precision
or a new hypothesis may motivate a separately frozen later cohort, not
outcome-dependent top-ups of this one.

One execution attempt per planned cell. Missing, failed, partial and invalid
artifacts stay in the inventory; zero task success is valid, not an execution
failure. No automatic requeue, no early stopping on success, no retry until
favorable. Infrastructure recovery, if needed, preserves the same seed and
earlier failed evidence under a dated amendment. Scheduler terminal receipts
are reconciled after transfer; software exit validation alone is not an
independent scientific conclusion.

API requests0, reservations0, new dollars0. Existing budget holds, running
explanation jobs, raw data and the manuscript are untouched. The launcher
archives committed code and refuses duplicate dispatch or ambiguous
submission retries.
