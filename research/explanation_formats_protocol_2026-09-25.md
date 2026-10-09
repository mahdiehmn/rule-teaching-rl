# Explanation formats, writer access and replay timing: frozen protocol

Version 3, **frozen before any collection,
calibration or learning outcome for these studies**. It supersedes the v2
draft (`explanation_formats_protocol_draft_2026-09-25.md`, kept as history)
and resolves that draft's open decisions (2026-09-25): the privileged
teacher is required, and all corrected versions are run. Launcher:
`scripts/run_explanation_formats_20260925.py` (worker
`scripts/submit_explanation_formats.sh`, user entry point
`scripts/launch_explanation_formats.sh`). The frozen explanation-isolation
study, the cached categorical reasons and every earlier study are
unchanged.

## Questions

1. **Formats (primary).** When a privileged, full-state teacher explains a
   fixed endorsed/foil action pair, does training the student's encoder to
   predict the explanation (no explicit action loss) speed teacher-off
   learning, and does that depend on the explanation's format?
2. **Writer access (secondary).** Does the same training with explanations
   written by a teacher that sees only the student's 7x7 view do as well?
3. **Timing (secondary, free).** Is the same amount of action-only replay
   worth more early or late in training?

## Fixed design

- Task and student: DoorKey-8x8, local 7x7 symbolic observation, CNN-GRU,
  dual critics, PPO exactly as in the isolation study (10M nominal
  transitions, 9,765 rollouts of 1,024), no-bonus and count-bonus
  backgrounds. Seeds 12,300,000 + 100 r, r = 0..4 (fresh; the isolation
  study used 12,200,000-based seeds, the memory studies 12,400,000+).
- Cases: the 256 cases of `contrastive_lessons_20260924_v1/panel.json`
  (SHA-256 `71825bc9...a9225a`), their 192/64 episode-disjoint split and
  their fixed action pairs from the full-state shortest-path planner.
- Replay: one minibatch of 32 cases every 4 rollouts from rollout 4 to
  7,320 (1,830 updates), encoder-only, identical sampled case stream for
  every arm of a seed; nothing after 75% of training.
- Targets are predicted from the current encoded image plus the two action
  one-hots. A privileged target (for example the direction of an
  out-of-view goal) is only partly predictable from that image by design.

## Writers and requests

One `gpt-5-mini` request per case and writer (reasoning effort low, 4,096
output tokens, strict JSON schema, every categorical field with an
`unknown` option, one attempt, zero SDK retries). Request version
`explanation_formats_request_v2`:

| Writer | Sees | Request SHA-256 | Worst-case bound |
|---|---|---|---|
| full_state (primary) | true full map, pose, inventory, objects, plus the student-visibility mask; may explain beyond it | `caa16581...` | $2.4073 |
| local_only (comparison) | only the student's 7x7 cells | `5fa5f7b9...` | $2.4794 |

Total worst case **$4.8868** including embeddings (`text-embedding-3-small`).
The launcher checks the live ledger read-only at preparation; each writer
job reserves its full chat bound before its first request and its
embedding bound before embedding, and settles known costs plus the full
bound of any unknown-cost attempt. The two writer jobs run one after the
other so ledger admissions never coincide.

## Formats and targets

| Format | Trained target | Loss | Check |
|---|---|---|---|
| plain | embedding of why the endorsed action helps | 1 - cosine | independent audit |
| contrastive | embeddings of why endorsed beats foil / why foil is worse | 1 - cosine (mean of two) | independent audit |
| subgoal | current target, relation, completion predicate | CE (3 heads) | phase |
| consequence | one-step movement/inventory/door change under each action | CE (6 heads) | native restoration |
| plan | next subgoal's object, egocentric direction, distance bucket | CE (3 heads) | full-state positions |

## Decisions resolving the v2 draft

1. **Primary family and thresholds: the isolation study's**, per format and
   within each background. A format is a development candidate if mean
   teacher-off AUC(aligned) - AUC(PPO) >= 0.05 with >= 4/5 positive seed
   pairs **and** AUC(aligned) - AUC(permuted) >= 0.03 with >= 4/5 positive
   pairs. Nomination thresholds, not significance tests; all arms and
   contrasts are shown with paired 95% t intervals over the five seeds.
2. **Local-only comparison collected now** (user: do it all).
3. **Panel kept.** The September 24 panel admitted only cases whose current
   target is visible; 115/256 next targets are hidden, so the plan format
   still carries privileged content. A panel admitting hidden current
   targets is deferred and stated as a limitation.
4. **Consequence format kept**, reported as the low-information format
   (movement dominates: inventory unchanged 233/256, door 251/256).
5. **Second teacher deferred.**
6. **Sensitivity scale not run.** Exclusions are only the frozen rules below.
7. **No pairing across writers (changed from the v2 draft).** Each writer's
   bank is frozen on its own intersection of cases known in all five
   formats. Restricting the full-state bank to cases the local writer also
   knows would drop mostly the cases whose next target is hidden, which is
   exactly the privileged content under study. The access comparison is
   therefore writer against writer: explanation content and coverage both
   differ, and both are reported (cases per bank, overlap, hidden fraction).

## Gates, frozen before collection

A writer's bank is admitted to learning only if: train cases >= 120
(full_state) or >= 48 (local_only); audit cases >= 40 or >= 16; embedding
dimension 1,536; unpaired; access label matches; and, for the primary, the
permuted control changes every format's targets on some case. The primary
floor also guarantees the 64 calibration cases. A failed gate writes
`GATE_FAILED.json` with the full record, cancels only that writer's
dependents, and is never retried automatically.

## Calibration

Once, on the admitted full-state bank, before any learning: at each of the
five training seeds' actual initial policy, the shared-encoder gradient
norm of each unscaled format loss on 64 hash-selected training cases is
compared with the isolation study's aligned code loss (0.1 x positive CE +
foil CE) on the same cases; scale = median over seeds of the ratio. The
same scales serve the access comparison. Never revised after learning
starts.

## Menus (fixed; no seed-count option)

- **Formats, 120 cells** (full-state bank): per seed and background PPO,
  action-only (coefficient 1), and each format x {aligned, bundle-permuted}
  as explanation-only at its calibrated scale.
- **Access, 50 cells** (local-only bank): each format, aligned, same seeds,
  backgrounds, scales, schedule and heads as the primary aligned cells.
- **Timing, 30 cells** (cached categorical lessons, SHA-256 `594983dc...`,
  images and action pairs only; no API): PPO, action-only early (rollouts
  4..4,880) and late (2,445..7,321), 1,220 updates each at stride 4, equal
  length, both ending before the teacher-free final quarter.

## Outcomes and analysis

Primary: teacher-off greedy success AUC, 50 episodes every 50 rollouts and
at the final transition, normalized over the observed support 51,200 to
9,999,360 transitions. Secondary: final success, first 80%/95% crossings
(censored if unreached), wall time, audit-head metrics. Every cell's
curve, dose, schedule, replay stream and checkpoint are validated at exit.

Secondary contrasts, same nomination style (>= 0.03 with >= 4/5 pairs):
full-state aligned minus local-only aligned per format (access);
action-only early minus late (timing); each timing arm minus PPO;
aligned minus action-only per format (explanation versus imitation).

If permuted matches aligned, no claim that explanation meaning caused a
gain. If every arm is at floor, report the floor; it is not proof that
explanations fail. Candidates go to a fresh-seed confirmation on this task
and replication on a second task before any paper claim. Prose quality is
not machine-certified: the independent audit of `audit_sheet.csv` is
required before any claim about explanation content. No seed replacement,
retuning or silent extension; failed cells keep their artifacts and get
an explicit recovery record.

## Execution (user-owned)

```bash
repo=/project/<allocation>/<user>/vlm-rl-bench
git -C "$repo" fetch origin master && \
  git -C "$repo" show FETCH_HEAD:scripts/launch_explanation_formats.sh \
  | bash -s -- "$repo"
```

This prepares both batches in `results/explanation_lessons/` and submits
the timing array (30 cells) and the formats chain: the full-state writer
job, then the 120-cell array and the local-only writer job, then the
50-cell array. `--suites timing` or `--suites formats` limits it; a rerun
reports what was already submitted.
