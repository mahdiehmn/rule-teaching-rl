# Remaining useful paper studies, prospective version1

Existing task A01 / work package W07. This extends the 2026-10-04
strengthening plan (2026-10-05): implement the useful optional studies and
queue them together.
It does not change the submitted640-run bulk or its scientific recipes.
All additions are prospective exploratory robustness/comparison analyses.
No new novelty claim, hypothesis replacement, or result clearance is implied.

## Finite menu and decisions

| Plan | Group | Work | Decision informed |
|---|---|---|---|
| O2 | crafter_eval |50 historical final policies,50 new worlds each;2500 sampled teacher-off episodes|Whether historical endpoint differences persist across world seeds|
| O4 | regeneration |5 new unselected DoorKey banks; seven arms x10 seeds=70 new5M training runs|Whether the selected-bank evidence generalizes to repeated bank generation under one recipe|
| O6 | online |DoorKey/MultiRoom x plain/count x10 seeds=40 new5M online-advice runs|Whether the scoped online comparison persists at the main imitation weight|

The110 new training runs total550M nominal transitions; generation adds5
tasks and evaluation50 tasks. Four uncapped Slurm arrays:50 evaluation,
5 generation,70 regeneration training,40 online training. The70 training
jobs have an `afterok` dependency on all5 generation tasks, a technical
input dependency only. No scientific-result gate chooses whether to run
an arm, coefficient or seed. All failed, empty-bank, duplicate-bank,
negative and null results remain in the record; one attempt per cell and
one request attempt per planned paid call. No implicit retries or requeues.

O1 is covered prospectively by the already submitted60-cell fresh Crafter
factorial, including count/no-progress; do not duplicate10 historical-seed
follow-ups. O3/O3b are submitted view120/timing40. O5 has the submitted200
finite coefficient/horizon sensitivity cells; it is not the plan's exact
screen-then-select-confirmation design. A later selected confirmation needs
an explicit selection rule and new protocol, not automatic favorable retries.
Thus coverage is purposeful; it is not a claim to have executed every literal
historical suggestion. Submission itself is not completion.

## O2 and O4 detailed contracts

See `paper_optional_crafter_eval_2026-10-05.md` and
`paper_optional_regeneration_2026-10-05.md`. O2 reuses original training
seeds31--40 and all five historical arms, evaluates worlds45500000--45500049,
and tests three declared paired contrasts with Holm correction after all50
policies validate. World episodes are not new independent training seeds.
The historical run/checkpoint artifacts must exist before preparation;
preparation copies/hashes their inputs without altering originals.

O4 uses replicates130--139, ten fresh shared seeds per arm. Five generation
repetitions are the bank sample, not fifty independent banks. Crossed
resampling accounts for bank and training-seed variation. The historical
selected bank is reported separately from the five-bank population summary.
Positive interval bounds support a narrowly conditional robustness statement;
intervals including zero are inconclusive; negative effects or wide bank
variation constrain the selected-bank claim. Neither outcome triggers another
generation, seed replacement, bank selection or claim of general model robustness.

## O6 exact comparison and limitations

Freeze the already submitted `dk_fresh` and `mr_fresh` manifests before
any new model call. Reuse their ten paired replicates80--89, both exploration
settings, no-advice and selected-bank controls. These are same-seed follow-up
comparisons, not a new independent confirmation of earlier selected recipes.
No learning outcomes from these controls are used to choose this menu.

Change the advising channel to `llm_scoped` with uniform query slots and
dated model `gpt-5-mini-2025-08-07`. Keep5M nominal frames,8 environments,
128 rollout steps, local student observation, isolated advisor RNG,
student-sampled actions, labelled-mean imitation loss, main weight0.1 to
0.001, decay/off fractions0.5/0.75, optimizer and evaluation settings from
the corresponding selected-bank arm. Measure teacher-off greedy AUC over
the complete regular evaluation curve as primary; sampled curves, final
success, frame-zero and dense early evaluations are secondary diagnostics.
Pairing also requires identical initial policy hashes; missing/mismatched
controls withhold inference. Dense evaluations include the untrained policy.

Online query/advice caps per run equal the selected bank's recorded generation
and checking calls: DoorKey/plain62, DoorKey/count62, MultiRoom/plain36,
MultiRoom/count67. Total2270 maximum online request attempts. The online
teacher uses the original full-map consultation prompt and supplies action_now;
its generated rule is discarded. It has at most10000 input tokens and2048
output tokens, service tier default, one attempt, SDK retries zero. This is
call-cap matching; dollars, tokens, labels, queried states and teacher/student
information are not equalized. Record them separately. KeyCorridor is omitted
because its main bank includes progress memory absent from this online prompt;
these40 runs cannot establish a six-cell equal-information comparison.

The four primary contrasts are selected-bank minus online AUC, separately
for task x exploration, paired on ten training seeds; use Holm4. Four online
minus no-advice contrasts form a separate secondary Holm4 family. Report
effects and95% intervals. Constant paired differences have undefined
variance-based p-values, not p=0. Withhold a family until all required cells
and paired controls validate. A positive adjusted contrast supports the
specified scoped advantage; intervals spanning zero are inconclusive; a
negative contrast constrains the paper's comparison. All outcomes stay;
none authorize a new coefficient, prompt or seed. Section4.6's final wording
depends on these results and independent scientific review.

Uniform clock slots falling on episode reset are skipped without replacement.
Validate persisted planned-slot hash and observed journal coordinates. Count
planned slots, observed consultations, skipped slots, request failures,
failed responses and labels separately. The existing technical stop is more
than30% failed responses after10 consultations. For inferential admission,
transport request failures must not exceed5% of the planned call cap and at
least one consultation must have occurred. This criterion does not imply95%
label delivery. Preserve/describe incomplete or inadmissible runs; do not
silently exclude individual seeds from an otherwise reported family.

## Money, attempts and source

Frozen prices: standard GPT-5-mini $0.25/M input and $2/M output, verified
2026-10-05 against the official
[pricing](https://developers.openai.com/api/docs/pricing) and
[model](https://developers.openai.com/api/docs/models/gpt-5-mini) pages.
Cached-input discounts are not assumed. The committed price table records
this source and date. Generation: at most360 requests, each16384 input /
8192 output tokens, five rounded child allocations $1.48 = $7.40. Online:
twenty62-call children at$0.41, ten36-call at$0.24, ten67-call at$0.45 =
$15.10. Combined reservation $22.50; at most2630 requests. This is a
conservative token liability bound, not predicted spend or a measured cost.

Before any submission, prepare all selected groups and reserve each entire
paid group from the existing project ledger. Both whole-group holds total
$22.50. Child ledgers are subdivisions of those holds, not new project
allowances. Admission fails without enough unreserved balance; it never
changes the authorized project ceiling ($453.13), releases old holds, or
resets a ledger. Workers check intact parent holds and pristine children.
Paid training retains its child hold for reconciliation; successful generation
books known cost or a conservative full bound for unknown usage. Parent
holds remain until a separate cost audit reconciles all attempts, including
partial/unknown provider charges. Do not rerun paid attempts after an
ambiguous response, missing receipt, interruption or successful poor result.

2026-10-05 accounting recovery, after user admission failure: a separately
reviewed helper can reduce the completed historical DoorKey random-extension
OPEN hold from53.30 to14.23 after verifying its exact original artifacts and
live scheduler completion. It retains full bounds for every non-usage-proven
slot and preserves all other entries and the453.13 ceiling. This is a bounded
audit of an old liability, not a new allowance or provider invoice settlement.
It is opt-in through `VLM_RECONCILE_DK_HOLD=1`; the ordinary launch never
releases prior reservations automatically.

Each group archives committed source and hashes all source files, protocols
and frozen input manifests. Workers run from the archive with exact runtime
equality. Archive/input/dispatch checks and raw-artifact validation precede
inference. Wall-clock caps: O2 two hours/cell, generation two hours/bank,
O4 training30hours, online36hours; these are administrative limits, not
runtime predictions. Slurm requests are uncapped arrays and no requeue.

## Execution and evidence

Use `scripts/launch_paper_optionals.sh` from a fetched/pinned commit in the
existing Vulcan repository. Default groups are all three; individual group
names allow the free evaluator or paid studies to be queued separately.
Paid groups require the existing shared ledger and an OpenAI credential in
the existing repository `.env` or environment; dotenv is read as data and
credentials are never copied into the source archive. O2 requires no key.

`python -m scripts.run_paper_optionals_20261005 report` writes a new dated
report under `results/paper_optionals/paper_optionals_20261005_v1/reports`.
It separates unattempted, dispatched/no-exit, failed, worker-completed pending
artifact check, and validated states. Repeated reports preserve past snapshots.
No result enters the paper until an independent reviewer
checks the actual persisted artifacts and inference. Software tests and
source review clear implementation only; they are not scientific evidence.
