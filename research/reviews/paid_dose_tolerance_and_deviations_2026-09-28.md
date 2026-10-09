# Paid-run admission by dose fidelity, and deviations from the frozen plan

Status: decision
note. **Written before any AUC is recomputed under the rule it declares.**
It changes an admission rule and records three deviations; it reports no
new outcome and promotes nothing.

## 1. The defect

Two parts of the pipeline disagree about what a broken paid run is.

| Component | Rule | Source |
|---|---|---|
| Launcher | a run stops above **30%** failed consultations, after 10 | `FAIL_FAST` in `run_rule_bank_study_20260928.py` |
| Report | **any single** failed call discards the run, never pairs it | `load()` in `report_rule_bank_paper_20260928.py` |
| Rerun batch | same zero tolerance selects the cells to recollect | `rerun_cells()` |

The zero-tolerance rule was written on 2026-09-28 for the credit outage,
where an account with no balance fails every remaining call and the run
genuinely trains on a different dose. It is the wrong rule for transient
rate limiting, where a run loses about 1% of its calls and trains on
essentially its promised dose.

Observed on the cluster the same day: the `budget_online_*_b` cells now
running carry 2-5 failed calls against a 480-call budget, across all three
tasks and both students. Under the launcher's contract they are healthy and
will run to completion. Under the report's contract every one of them is
discarded. The same mismatch built the 145-cell rerun batch, so an unknown
share of those cells were never broken.

A count of 2-5 in 480 also rules out credit exhaustion as the cause: an
exhausted balance fails every subsequent call, which would show as hundreds
of failures per run, not four.

## 2. The rule, declared now

A paid run is **admitted** when it delivered at least `1 - t` of its arm's
promised consultations, where the promised count is the arm's own budget
(62 / 67 / 61 for `llm_action_online_eq` on DoorKey / MultiRoom /
KeyCorridor; 480 for the `_480` arms). Lost calls are those
`request_failures()` counts: `metadata.failed` and
`metadata.outcome == 'request_failure'`. Truncated or malformed replies stay
in, unchanged: they are the teacher's own abstentions.

`t` is **not a free parameter chosen from outcomes.** Every result computed
under this rule is reported at all three levels below, and the primary
number is the middle one.

| Level | `t` | Why this number and not another |
|---|---:|---|
| Strict | 0.00 | The existing rule. Kept so nothing is hidden. |
| **Primary** | **0.05** | Dose fidelity: a run within 5% of its promised calls trained on its arm's dose. Chosen as the smallest round bound that covers observed transient loss (~1%) while excluding any run that lost a material share. |
| Permissive | 0.30 | The launcher's own pre-existing frozen tolerance, adopted rather than invented. |

**Pre-commitment.** A conclusion is reported as supported only if it holds
at all three levels. Where the three disagree, the disagreement is the
result and is printed, not resolved by preferring a level. Runs excluded at
a given level are listed with their lost-call counts. No level is chosen
per task, per student, per arm or per contrast.

**What this does not do.** It does not recover a run that lost a large share
of its calls; those cells still need recollection. It does not change any
free arm, since only paid arms can lose calls. It does not alter a single
frozen bank, manifest, spec or seed, and it does not re-open the
pre-registered contrast set.

## 3. Deviations from the frozen confirmation plan

[The confirmation plan](confirmation_plan_2026-09-28.md) is the frozen
reference. Three things have since changed. All three are recorded here
**before** the confirmation numbers are final, and the plan's own
pre-registered analysis is reported alongside the new primary in every case.

### D1. The primary rule set is selected on development replicates

The plan pre-registers **P1** as `llm_rules_blind_strict - none`, with Holm
over P1-P6 within each environment x student. The paper's goal is
learning gains rather than rule precision, and the
unanimous blind check is treated as a tunable setting rather than an
objective: it keeps 7 of 32 door-only rules on MultiRoom plain PPO and
delivers 23k labels where the raw bank delivers 3.1M.

**What is now primary.** For each (task, student) the check strictness
(checked or unchecked, or rules off when neither beats `none`) is chosen on
**replicates 0-4** and scored on **replicates 5-19**, with Holm over the
resulting 18 contrasts. This is `report_rule_bank_paper_20260928.py
--selected`.

**Why it is admissible.** Selection uses development replicates only and
scoring uses fifteen disjoint replicates, so the reported interval is not
the one the selection optimised. The selection rule is a function of
development outcomes alone and is stated before the confirmation replicates
are read.

**What is reported anyway.** The pre-registered P1-P6 on
`llm_rules_blind_strict`, unselected, with its own Holm family. If the two
disagree the paper says so. The selected view is not described as a
confirmation of a pre-registered hypothesis; it is a development-selected
setting evaluated on held-out replicates, and it is labelled that way.

### D2. `online_rules` was launched although the plan excluded it

The plan lists online rule accumulation under "not in this plan", as "too
risky to build and validate before the deadline". The third wave launched it
anyway (`online_rules_{task}`, replicates 0-4). It is therefore an
**exploratory** batch: five replicates, no pre-registered contrast, and it
cannot enter the primary family. Reported as an extension with its status
stated.

### D3. `schedule_online` was cancelled

The plan's entropy-importance arm for MultiRoom Count-PPO was cancelled
on cost grounds, with the in-house schedule studies standing in.
The advising-schedule comparison is therefore absent for the LLM teacher,
and the paper does not claim a best schedule for it. The partial
`schedule_online` cells that did run are not reported as an arm.

## 4. What is still needed, and what it costs

At the measured rate (`EXPECTED_USD_PER_CALL = .0025`; measured
$0.0018-0.0022):

| Need | Cells | Calls | Bound | Status |
|---|---:|---:|---:|---|
| P3 `llm_action_online_eq`, replicates 5-19, 3 tasks x 2 students | 90 | 5,700 | **$14.25** | collected; admission decides how many pair |
| `_480` arm, replicates 5-9 (`budget_online_*_b`) | 30 | 14,400 | **$36.00** | running; secondary |
| Rerun of genuinely broken cells | <= 145 | <= 16,000 | **<= $40** | running; most may be unnecessary under the rule above |

P3 is the core claim and is the cheapest item on the list. The `_480` arm is
a secondary headline and can ship at replicates 0-4. The exact residual is
produced by `scripts/paid_cell_accounting_20260928.py`, which is read-only
and makes no calls.

## 5. Order of operations

1. Run the accounting on the cluster. It reports the lost-call distribution
   and the cells admitted at `t = 0, .05, .30`.
2. Cancel the rerun cells the rule admits at `t = .05`, if any, and keep the
   rest.
3. Recompute the report at all three tolerances.
4. Only then read the confirmation numbers.

Steps 1-3 do not depend on any AUC, which is why this note is written first.
