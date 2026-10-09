# O4 bounded recovery amendment

This amendment follows the failed original generation task 1303027_2 from
source 2c419fa1edbdff1365ec20309c5e9b4c556dbca0. It preserves the original
single-attempt protocol and its observed completion of four of five banks.
It does not retrospectively label that original protocol successful.
The amended experiment is exploratory, with no student outcomes inspected
to choose this recovery. Existing tasks A01/W07 remain the task records.

## Observed failure and decision

The receipts hold 36 START and 36 END records for bank_2 consultations,
with exactly one failed request: consult_018, ValueError, served model
gpt-5-mini-2025-08-07, 1493 input tokens and 8192 output tokens. No blind
requests were sent for that bank and no success receipt was created. The
generation failure receipt retains its original reservation. This is
consistent with an incomplete response at the output cap; the collector
overwrote the provider status, so the precise provider reason is unavailable.
It is not established to be a transient transport failure or a bad rule.

Frozen recovery decision: permit exactly one additional attempt for that
same consultation, with byte-identical request data, model, low reasoning
effort, default service tier, 8192 output cap, 300-second timeout, and no SDK
retry. Keep the other 35 successful consultation responses. Keep all four
completed banks verbatim. After a technically complete retry, execute only
bank_2's previously unattempted blind checks, at most 36, once each under
the original prompt, corpus, RNG7 matching and strict filter.

Any failed retry or blind request stops the recovery and leaves the amended
cohort incomplete. There is no additional retry, higher token cap, replacement
bank, synthetic abstention, semantic repair or winner selection in this
amendment. Valid model abstentions, duplicate banks, empty banks and weak
rules remain valid outcomes. This changes the permitted attempt count for
one observed technical failure and must be disclosed when reporting it.

## Source, evidence and execution

Use a separate recovery batch and a new committed source archive. Original
source archives, input manifests, successful banks, raw failed receipts,
original job IDs and reservations remain intact. Verify the pinned original
manifest, source hashes, runtime, original worker receipts and each request's
START/END identity. Admission requires the exact specified failure and all
35 other consultation replies technically complete. No original bank-training
task may have started. Preserve any ambiguity and stop admission.

All original attempts are retained with immutable hashes. An effective
generation view may select the 35 original valid replies and the single
valid retry for downstream use, but must have an explicit lineage receipt
mapping each selected reply to its original attempt. It is derived data,
not the original single-attempt journal. Preserve the failed attempt's cost
and tokens separately; never subtract them from total generation cost.
Every training worker and report rechecks this lineage and its source hashes.

The original pending training array 1303032 cannot satisfy its failed
afterok dependency. It is retained as an unstarted original submission;
any cancellation must first verify that no original training attempt has
started. Recovery uses a new generation job and a new dependent 70-cell
training array. Repeated launches must not duplicate submissions or paid
calls. Preserve failed, ambiguous and interrupted recovery attempts.

## Budget

Reserve a fresh $0.76 parent allocation from the existing shared project
ledger before recovery dispatch. The bound is 37 requests times $0.02048
= $0.75776, rounded upward, using the existing frozen input/output caps
and price table. The one repeated consultation is included in that count.
This additional allocation is conservative because the original $7.40
group reservation, including bank_2's $1.48 child hold, remains intact.
Do not reset ledgers, increase the existing $453.13 project ceiling, release
old holds or infer new allowance from the provider balance.

The recovery child starts with $0.76 and reserves the entire allowance
before calls. Record known usage and full bounds for unknown charges.
Failures retain their liability; later settlement must account for every
attempt. Report original and additional generation costs separately and
their sum, distinguishing usage-proven cost, unknown bounds and reservations.
No paid calls or scheduler submissions are made by implementing agents.

## Learning design and reporting

Retain the exact 70 original slots: no-advice, historical selected bank,
and five generated banks, each on the same ten fresh training replicates
130--139 (14513000--14513900 in increments of 100). Retain original 5M
transitions, local observations, main imitation coefficient/schedule,
labelled-mean normalization, student-selected actions, greedy teacher-off
AUC and frame-zero/dense diagnostics. The four original banks and all
successful consultation outputs are reused, never regenerated. The
historical selected-bank arm remains a separate comparator.

The primary descriptive estimate and crossed bank-by-seed bootstrap from
the original O4 protocol remain unchanged. Only a complete, validated
70-cell amended cohort permits its five-bank summary; report every bank
and seed, including null or negative outcomes. This is evidence under a
recipe with one documented recovery, not five successful single-attempt
generations. Original single-attempt completion remains four of five.

Scheduler completion, artifact validation and independent scientific review
are separate. Offline implementation tests and independent software review
do not establish any learning result. If recovery fails, preserve four
complete original banks and the failed fifth as incomplete-study evidence;
no four-bank replacement primary inference or new generation is authorized
by this amendment.

## Verification

Before publication, run offline tests on disposable fixtures and ledgers:
successful reuse/retry, failed retry and blind stage, exact paid-call count,
unchanged old artifacts/holds, input/lineage tampering, original-training
activity rejection, source/runtime binding and idempotent submission.
