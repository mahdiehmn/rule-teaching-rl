# Progress-conditioned advice: label-exposure follow-up v1

Date: 2026-10-05. Existing task A01 (evidence); W07 (paper framing).
Status: prospective exploratory mechanism study; no scientific jobs run.

## Question and correction to the proposal

Does progress-conditioned advice retain a learning advantage over the
progress-ablated bank after reducing its label exposure? The frozen teacher
is the existing LLM-derived bank; advisor is unlimited; channel is masked
auxiliary action-label loss. Student chooses every environment action.

This supersedes the October 4 proposed *no-progress thinning direction*
(plan O3b). Twenty existing development runs show that the
full bank supplied MORE labels in aggregate: 14,344,645 versus 10,784,675.
Therefore thinning the no-progress bank cannot match the full-bank total.
No new efficacy evidence motivated this revision; the feasibility check
uses historical exposure counts. All original results remain unchanged.

Use the already implemented EvidenceGate with mode `none` and a fixed
keep fraction; this is not a new learned gate. It uniformly draws
round(fraction * emitted labels) without replacement from each rollout,
using its isolated NumPy generator seeded at training_seed + 7,654,321.
The chosen mask is stored once and reused by all PPO epochs. It reads no
progress, reward, oracle or eventual success. The rule executor itself
retains its existing conditions, progress clauses and conflict abstention.

## Frozen calibration and study cells

Calibration is `rule_timing_calibration_2026-10-05.json`: twenty plain-PPO
completed runs, replicates30--39 / seeds14703000--14703900 (stride100),
in `kc_mem_confirm` and `kc_prog`. The machine-readable receipt includes
source manifest, exit and summary hashes; final teacher_labels agree with
advisor-delivered counts. Development exposure is used only to choose one
pooled fraction, never a seed-specific fraction or a performance threshold.

Full-bank retention = 10,784,675 / 14,344,645 = 0.7518258555718876.
This aggregate calibration is feasible because it lies strictly in (0,1).
No per-interval historical exposure files were present in the inspected
sync; no temporal matching is claimed. Calibration cannot be overwritten
with different data by the script. A revision requires a new dated version.

Forty cells: KeyCorridorS3R3, plain recurrent PPO, four arms, ten paired
fresh replicates90--99 / seeds14709000--14709900 (stride100):

1. No advice.
2. Full frozen progress bank.
3. Same frozen bank with progress clauses removed (existing ablation).
4. Full frozen bank with random label thinning at the fraction above.

These seeds were reserved for timing in the 8ebaaf8 plan and do not
overlap the core140/optional80 or historical fix-wave suites. Local suite
audit is not a claim that remote jobs were inspected. Submission requires
checking no earlier attempt exists in the cluster workspace.
This is a standalone cohort; no reuse of core outcome-selected controls.

Engineering disclosure: an initial worker-contract test inadvertently used
seed14709000 for a2048-transition no-advice prefix; the three guided tests
aborted at their first consultations due to a logging bug. An independent
reviewer repeated this initial test before the correction. These are local
software artifacts, not study results; no learning outcome selected the
design, coefficient, retention fraction or seeds. Subsequent integration
tests use explicit nonstudy seeds9917900/9917970. Therefore14709000 is fresh
for the scientific full-horizon study, but not literally never initialized.

Cell settings inherit `kc_fresh`: 5M nominal frames (4,999,168 actual due
to rollout geometry), 8 environments x128 steps, original student symbolic
view, recurrent architecture, optimizer, evaluation worlds and schedules,
weak coefficient0.1 ->0.001 over first half, cutoff75%, `labeled` loss
normalization. No teacher action execution, replay, shaping or paid calls.
Evaluation worlds follow the existing seed+50,000 convention; this is
fresh training-seed evidence, not a new benchmark or held-out task claim.

Advice labels are averaged within labeled minibatches. Fewer labels do
NOT imply proportionally smaller expected auxiliary gradient strength.
Progress itself remains the historically used grid-derived unlock fact,
not an added student-policy input or a newly substituted history tracker.

## Metrics, diagnostics and inference

Primary contrast: full-thinned minus no-progress on teacher-off greedy
regular-grid learning AUC. Joint family of three two-sided paired tests:
that contrast, full minus none, and full minus no-progress; Holm correction
at0.05, mean differences and unadjusted95% paired-t intervals. Training
seeds are the units. No inference until all40 runs pass validation; no
favorable-seed replacement or early efficacy stopping.

The main mechanistic prediction is a positive primary effect. Promote only
if its Holm-adjusted p<0.05 and mean>0, with aggregate retained-label ratio
(full-thinned / no-progress) between0.90 and1.10. Outside that range, report
the performance contrast but reject an exposure-comparable interpretation.
This tolerance is an interpretation rule, not permission to recalibrate.
An imprecise/null contrast is inconclusive, not proof of equivalence.
Negative evidence narrows the claim; it does not trigger extra tuning.
Keep the overall full-minus-none result visible even if unfavorable.

On-policy visitation, conflict patterns and label composition may differ.
Even within the exposure tolerance, infer robustness to a generic reduction
in labels, not exact causal isolation of timing or matched gradients.

Read-only `rule_timing.jsonl` records per-rollout emitted/retained totals,
phase/action counts, no-rule/conflict abstentions and matching rule IDs.
`rule_timing_updates.jsonl` records occupied supervised minibatches and
mean auxiliary loss across executed PPO minibatches. Bounded examples
come from actual student trajectories; they are illustrative and cannot
estimate event prevalence. No oracle provides or selects training labels.
No diagnostic function changes targets or RNG state. Defaults are off.

All arms get dense early teacher-off evaluation at the same frames,
including0. Regular-grid AUC remains primary, as in the core package.
Additional early curves and sampled-policy metrics are secondary. Keep
the new complete cohort separate; never splice into historical curves.

## Execution, checks and stopping

Run order is seed-major, with all four arms adjacent in the manifest.
Exactly one attempt per planned cell; no automatic retry or extension.
Any infrastructure recovery needs a separate recorded amendment retaining
the failed attempt. Scientific budget200M nominal training transitions
plus evaluation; zero API calls. Slurm uses the existing CPU environment,
one unthrottled40-cell array, per-cell limit30hours. That is a cap, not a
runtime forecast. Agents only run bounded offline software tests locally.

Readiness: independent code/design review, real-trainer diagnostic-on/off
identity tests, each arm passing terminal validation, seed/config delta
checks, frozen-source archive verification and a committed source revision.
The separate runner does not alter the core suite definitions.

User commands, from the integrated committed source on the cluster:

```bash
python -m scripts.run_rule_timing_20261005 check
python -m scripts.run_rule_timing_20261005 prepare
python -m scripts.run_rule_timing_20261005 launch
```

`prepare` freezes committed source and refuses missing packet files or
tracked dirty changes. `launch` is run by hand only. No remote source
publication or cluster run is implied by this protocol. Report afterward:

```bash
python -m scripts.run_rule_timing_20261005 report \
  --batch results/rule_timing/rule_timing_20261005_v1 \
  --out results/rule_timing/rule_timing_20261005_v1/report.json
```
