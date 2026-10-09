# Five unselected DoorKey bank repetitions

Status: prospective exploratory implementation for the remaining optional
paper studies. Existing work package W07; the execution record holds the
source revision and cluster commands.

## Question and fixed menu

Measure how much learning varies when the complete frozen DoorKey bank
writing/checking recipe is repeated five times. Generate all five banks
without selecting, ranking, repairing, or replacing any bank according to
offline precision or student success. An empty but technically valid bank
and duplicate banks are retained. This probes variability conditional on
one corpus, model snapshot, prompt recipe, and environment; it does not
establish robustness over other models or consultation corpora.

Seventy plain-PPO DoorKey-8x8 training cells: no advice, the historical
selected blind-strict bank, and five newly generated blind-strict banks,
each on ten paired fresh training seeds. Replicates 130--139 correspond to
14513000--14513900 in increments of100. All arms keep the source5M horizon,
learning-rate schedule, imitation weight0.1 to0.001 and its existing
decay/off schedule, labelled-mean loss normalization, student-selected
actions, local policy observation, regular50-episode teacher-off evaluation,
and dense early evaluations including frame zero. Fresh banks differ only
in frozen bank file/hash and experiment identity from the historical-bank
arm. Historical-bank selection is disclosed and it is not counted among
the five unselected generation repetitions.

## Generation recipe and provenance

The committed fixture contains only the original36 consultation states and
343 experience-pool states. Original full panels were inspected at
`vlm-rl-bench-langpilot-20260927/results/conditional_rules_v3_20260928/panels.json`,
with raw SHA256
`312dcd8f080543ddaaf12daab44a0984252d108440443d94f169d00c3ff76aa4`.
Canonical JSON SHA256 of the exact consultation/pool subset is
`d1105d126a8137cbc491a923102a4982323c9070165772d3299473f801de5a70`.
Author/date metadata wrap this subset in the committed fixture without
altering its contents. No new development or confirmation worlds are
generated or scored to select banks.

Each repetition submits the same36 original `scoped_rule_v3` requests to
`gpt-5-mini-2025-08-07`, low effort, default service tier,8192 output tokens.
No variant hint is added to prompts. After their replies, reuse the original
`export_refine` experience matching and RNG7 choice of up to five situations
per rule, then submit only its corresponding blind-check requests. No
refinement API requests are sent. Apply the original
`blind_filtered(..., strict=True)` rule filter. This repeats writing plus
checking, not just writing noise. No model-side seed is supplied, and
separate API repetitions are not guaranteed to produce different outputs.

Only technically complete collection permits freezing: all36 consultation
attempts and every requested blind check must return the requested model,
completed status, valid JSON/schema, and durable matching receipts. Model
abstentions and checker rejection of rules are valid outcomes. A failed
generation is retained, never automatically retried or replaced; the
prequeued training stage remains blocked by technical dependency. A later
infrastructure recovery requires a dated amendment and preserves attempts.

## Paid contract and execution boundary

At most72 requests per bank,360 across five banks; exactly one attempt per
case, zero SDK retries, six request threads per bank,300-second timeout.
Wire guard accepts only the fixed OpenAI Responses endpoint and refuses
serialized bodies above15360 bytes, leaving1024 framing allowance under
the16384-token conservative input bound. Largest original consultation is
6362 serialized bytes; a conservative longest-five-state blind prompt is
3069 bytes. Output cap is8192. Price rates are the separately verified
optional-package rates: $0.25/M input and $2/M output.

Reserve$1.48 per bank, total$7.40, before collection. The per-call bound is
$0.02048;72 calls cost at most$1.47456 under this token/price contract. The
root orchestrator reserves the aggregate amount from the existing shared
project pool and creates five separately capped child ledgers before
submitting the five-task generation array. The worker verifies its child
allowance/hold; root verifies the parent reservation before dispatch.
Known usage plus full bounds for unknown-cost completed attempts are
settled on success. Failures before settlement retain the full child hold
for separate receipt reconciliation; a later receipt-write failure records
that settlement already occurred. Other project holds and budget ceilings are not
released or increased by this module.

Durable START and END records bind every request to its case, condition,
body digest, served model, status, raw output, and usage where available.
All inputs, bank files, and receipts are frozen and checked. A training
plan fixes all70 index/arm/seed slots before generation. The root launcher
can queue the70 training tasks behind the generation array using `afterok`;
each training worker revalidates all five completed generations, resolves
their actual hashes, and records them with its own dispatch. This is a
technical dependency, with no interim learning-result gate.

## Reporting and interpretation

Primary descriptive quantity: the average paired regular greedy-success
AUC improvement over no advice across the five generated banks and ten
training seeds. Report every bank's mean AUC and paired mean difference,
and an exploratory crossed percentile bootstrap interval, independently
resampling the bank axis and the paired seed axis10000 times with fixed
reporting RNG20261005. This preserves shared control observations across
banks; it does not treat50 trained bank/seed combinations as50 independent
banks. Only five generation repetitions are available, so the interval's
coverage is uncertain and no population-wide robustness claim follows.

Report the historical selected-bank minus no-advice paired estimate with
its usual seed-based95% interval separately. It is a reference comparator,
not an unselected generation repetition. Report final success, early curves,
label exposure, bank sizes, costs, duplicate banks, and all failures
descriptively. Incomplete cohorts are provisional and receive no complete
five-bank inference. Keep all valid weak or negative results; there is no
outcome threshold that triggers selecting a winner or adding generations.
If all observed historical-bank paired differences are identical, its
paired-t test is undefined and the report gives no p-value.

Evidence before launch: jobs completed = none; artifacts validated = local
software/fixture checks only; scientific conclusions independently reviewed
= unavailable because no study results exist.

## Local verification

Run from the repository root using the project environment:

```powershell
python -m pytest tests/test_paper_optional_regeneration.py -q -p no:cacheprovider
```

Tests use only mocked API responses and temporary ledgers. They cover all
five complete receipt paths, strict bank reconstruction, preserved empty
and duplicate banks, fixed70-cell pairing, unchanged treatment settings,
failed-attempt retention/no retry, input tampering, actual wire guards,
receipt completeness, and the bank-versus-seed reporting distinction.

Author validation, 2026-10-05: final fourteen O4 cases passed as part of a
joint28-case O4/O2 run in32.05 seconds, using the project `.venv` and an
explicit workspace `--basetemp`. The first invocation's default Windows
pytest temporary directory was inaccessible (five tests passed, six setup
errors); switching only the temporary directory resolved that environment
issue. All paid responses and ledgers were synthetic and temporary.

Independent-review corrections implemented before the final run: a failed
receipt write now reports whether the child hold was actually retained
after settlement; constant historical-bank paired differences no longer
inherit the legacy reporter's artificial p=0; all70 AUC values, including
the historical comparator, must be finite and within[0,1]. Frozen blind
request plans are reconstructed from the exact corpus, replies, and RNG7
selection before training admission. Additional tests check missing usage
is settled at full request bounds and the actual SDK constructor disables
retries and installs the outgoing wire guard.
