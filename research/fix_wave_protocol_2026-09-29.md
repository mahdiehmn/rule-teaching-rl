# Main rule-teaching training study, `fix_wave_20260929_v1` (frozen before launch)

Runner: `scripts/run_fix_wave_20260929.py`. Launcher:
`scripts/launch_fix_wave.sh`. **Written before any run of this wave.**
Development evidence, not confirmation; no novelty claim.

## Why

Under LLM rule imitation, KeyCorridor-S3R3 fails in two separate ways
(rule-bank study, replicates 0-9 unless stated):

1. **Harm to a competent student.** The count-bonus student solves the task
   alone (.952). Dense rule labels take it to .529. The LLM's own actions
   replayed over the same predicates also harm it (.484), while the same
   rules capped at 480 labels do not (.962). The harm follows dense,
   predicate-matched labels, not who wrote them.
2. **Nothing teaches plain PPO.** Every teaching arm is .000 at 5M: rules,
   LLM action advice at 15/30/61/120 calls, and replay (at most .048).
   The one exception is the valid-action rules, at .118 on replicates 20-29.

A fixed imitation dose cannot serve both students: MultiRoom plain PPO
needs the dense dose (.867 dense vs .000 capped). ADVISOR did not rescue
S3R3 with dense planner advice (evidence pilot 09-25: .019 vs .858).

**Selection disclosure.** The 09-27 potential pilot was read on 09-29,
**before** this protocol was written, and it motivates the shaping arms.
On KeyCorridor with the count student (2M, 13.3M seed series) subgoal
shaping gave .954 vs .768 for no teacher, positive on 6/6 seeds, where
action advice at the same budget gave .672. This wave is therefore a
fresh-seed replication at the paper's 5M horizon plus a generality test,
not an independent discovery.

## Design

Every cell is built from the rule-bank study's own cells for the same
task, student and seed; each arm may differ from its reference only in
its own teaching channel (enforced when building, tested). Rule banks are
byte-identical to the paper's (committed LF hashes equal the frozen
manifests'). No API calls; 255 free runs in four independent arrays.

| Suite | Task | Students | Replicates | Horizon | Arms |
|---|---|---|---|---|---|
| `kc` | KeyCorridor-S3R3 | plain, count | 5-14 | 5M | none, shaping_dense, shaping_dense_wrong, shaping_480, rules, rules_plus_shaping, rules_weak |
| `mr` | MultiRoom-N6 | plain, count | 5-9 | 5M | none, shaping_dense, shaping_dense_wrong, rules_plus_shaping, rules_weak |
| `dk` | DoorKey-8x8 | plain, count | 5-9 | 5M | same as `mr` |
| `kc_long` | KeyCorridor-S3R3 | plain | 5-9 | 15M | none, shaping_dense, rules |

- `shaping_*`: subgoal potential `F = gamma*Phi(s',m') - Phi(s,m)` on the
  extrinsic reward only (`algos/ppo_potential.py`,
  `teachers/subgoal_potential.py`), no imitation, teacher-off unshaped
  evaluation. Dense = a message at every episode start and stage change;
  `_480` = 480 messages over training; `_wrong` = every message names a
  wrong stage (content control).
- Rule arms imitate the development-selected rule set of
  `report_rule_bank_paper_20260928 --selected`: DoorKey checked (both
  students), MultiRoom unchecked (plain) and checked (count). KeyCorridor's
  selection turned rules off, so its fixes start from the valid-action
  rules. `rules_weak` imitates the same rules at a tenth of the
  paper's weight: .1 -> .001 instead of 1 -> .01, the rule-bank study's
  actual schedule (the trainer default of 10 -> .1 is not used there). `rules_plus_shaping` imitates them AND receives
  dense shaping (`potential_with_guidance`, default off).
- The re-run `none` must reproduce the rule-bank study's `none` exactly on
  the same seeds. **If it does not, every comparison with the rule-bank
  study is void** and only within-wave contrasts are reported.

## Questions and decision rules

Paired by seed, teacher-off AUC over the observed support, two-sided 95%
paired t intervals, n=10 (`kc`) or 5 (`mr`, `dk`, `kc_long`). These are
development rules, not significance tests.

| # | Question | Contrast | Counts as a fix when |
|---|---|---|---|
| Q1 | Does shaping avoid imitation's harm on the competent student? | KC/count: `shaping_dense - none`, and `shaping_dense - rules` | first mean >= -.02, AND second > 0 with >= 8/10 positive |
| Q2 | Does shaping teach plain PPO KeyCorridor? | KC/plain: `shaping_dense - none` (5M; 15M reported separately) | mean > 0 and the interval excludes 0 |
| Q3 | Is it the content? | `shaping_dense - shaping_dense_wrong` wherever Q1, Q2 or Q4 passes | > 0 with >= 8/10 (KC) or >= 4/5 positive |
| Q4 | Does shaping generalize? | `shaping_dense - none`, all six cells; vs the paper's rules on the same seeds | reported descriptively; no rule |
| Q5 | Can the rules channel be repaired? | `rules_weak - rules` and `rules_plus_shaping - rules` on KC/count; `rules_weak - none` on DK and MR plain | KC/count harm reduced (> 0, >= 8/10) while DK and MR plain keep >= .5 of their gain |

Every arm and contrast is reported, including failures. A cell where both
arms score 0 on every seed is uninformative, not a result.

## What this wave cannot establish

- **The shaping teacher is a scripted rule with privileged state** (true
  stage timing and full-map BFS distances), not an LLM. A result here
  supports the *channel*. It is not yet an LLM-teacher result; that needs
  an LLM-specified stage plan feeding the same potential.
- Policy invariance of potential-based shaping is an asymptotic property.
  It does not predict finite-time AUC or rule out slower learning
  (Hounwanou et al. 2026, section 6.1). That is what the wrong-message
  control and the paired seeds are for.
- Replicates 5-14 overlap the rule study's confirmation cohort, by design
  (pairing). This wave does not re-confirm that study's claims.

## Operations

`bash scripts/launch_fix_wave.sh <repo>` from the cluster checkout
at FETCH_HEAD. Each array runs once, unthrottled, with no requeue; a
completed poor result is never retried. Report:
`python -m scripts.run_fix_wave_20260929 report --rule-bank results/efficiency`.

## Addendum 1 (2026-09-29): shaping along the LLM's own plans

Written before any run of the suites below and before any result of the
original four suites existed.

**Why.** The shaping arms above follow a hand-coded subgoal plan, so they
test the channel, not the LLM teacher. GPT-5-mini (the dated snapshot that
wrote the rule banks, low effort, strict JSON schema) was asked for each
task's plan from the mission text and mechanics only, in the vocabulary the
potential can measure, with distractor options
(`scripts/llm_stage_plan_20260929.py`; 5 samples per task, 15 calls,
$0.013; record `research/stage_plans/stage_plans_20260929.json`).

**Finding: the LLM's plans differ from the hand-coded one.**

| Task | Modal plan (count) | Hand-coded |
|---|---|---|
| DoorKey | key -> unlock -> **enter the next room** -> goal (5/5) | key -> unlock -> goal |
| KeyCorridor | key, **searched room by room** -> unlock -> **enter the ball's room** -> ball (4/5) | key -> unlock -> ball |
| MultiRoom | per room: **open the door**, enter -> goal (3/5) | per room: enter -> goal |

The modal plan per task (most frequent; a tie goes to the first sampled,
declared before freezing) is frozen with a content digest in
`research/stage_plans/llm_plans_20260929.json`.

**Implementation.** `teachers.subgoal_potential.PlanStages` drives the
unchanged potential from a written plan. Given the hand-coded plan it
reproduces `TaskStages` at every stage of all three tasks, and the shaping
stream is bit-identical at wrong-message rates 0, .5 and 1
(`tests/test_llm_stage_plan.py`); with the LLM's plans each episode's
discounted shaping still telescopes to -Phi(s0) in the real trainer.
Interpretations, fixed in the code and the frozen file:

- `each_room_before_the_goal_room` expands per room in MultiRoom only; on
  KeyCorridor the LLM's "search room by room" marker on its key step means
  once, and the potential targets the key's true cell, which is privileged
  information the plan did not assume.
- `door_unlocked` means unlocked for a door locked at episode start, and
  open for a door never locked (MultiRoom; the LLM's rationale: "open and
  pass through each closed door").
- KeyCorridor-S3R3 rooms are 3x3 with walls: the ball fills the ball room's
  only interior cell, so standing in its opened doorway counts as having
  entered.

**Suites (40 free runs).** `kc_llm` (KeyCorridor, both students,
replicates 5-14), `mr_llm` and `dk_llm` (both students, replicates 5-9),
arm `shaping_llm_plan`: dense correct shaping along the LLM's plan. Each
cell equals the matching `shaping_dense` cell except the plan; reports merge
each addendum with its suite, so the arm pairs with that suite's `none`
and `shaping_dense` by seed.

**Q6. Does the LLM-written plan fix what the hand-coded plan fixes?** The
arm counts as the fix under exactly the rules of Q1 (KeyCorridor count
student) and Q2 (KeyCorridor plain PPO), substituting `shaping_llm_plan`
for `shaping_dense`. `shaping_llm_plan - shaping_dense` is reported
descriptively on every cell, with no rule: the plans differ, and either
may shape better.

**What it cannot establish.** The LLM contributes the subgoal sequence;
the environment measures progress through it with privileged distances.
That is a thinner LLM contribution than the rule banks, and the paper says
so.

## Addendum 2 (2026-09-29): fresh-seed confirmation of weaker imitation

Written after reading `results/reports/20260929_0641/fix_wave.txt` and
before any run of the suites below. **Selection disclosure:** the arm
confirmed here was chosen because it passed Q5 on development seeds.

**What the development seeds showed.** The same LLM rules imitated at a
tenth of the paper's weight (`rules_weak`, .1 -> .001 instead of 1 -> .01)
were the only arm to work on both KeyCorridor students. Count student:
.908 vs .965 alone (-.056 [-.130, +.019], 4/9), against .468 at the paper
weight (-.496, 0/9). Plain PPO: .386 vs .000 (+.367 [+.180, +.554], 7/9;
final success .80). Across all six cells it had an interval above zero in
four (DoorKey both students, MultiRoom plain, KeyCorridor plain) and no
significant harm in the two cells whose students already reach .94-.97
alone. Replicates there were 5-14 (KeyCorridor) and 5-9 (DoorKey,
MultiRoom), overlapping the rule study's confirmation cohort, with n=5 on
DoorKey and MultiRoom. Shaping, by contrast, failed Q2 (KeyCorridor plain,
interval includes 0) and hurt MultiRoom's count student in finite time.

**Design.** Suites `kc_confirm`, `mr_confirm`, `dk_confirm`: both
students, **fresh replicates 30-39** (unused by any study: rule-bank 0-19,
valid-action KeyCorridor 20-29), arms `none`, `rules` (paper weight) and
`rules_weak`, the same rule banks as the fix wave (DoorKey checked,
MultiRoom unchecked for plain and checked for count, KeyCorridor
valid-action). 180 free runs. The weight was fixed by the original
protocol before any result and is not tuned here.

**Primary family (six cells, Holm).** `rules_weak - none` per cell, paired
by seed, two-sided paired t. A cell **HELPS** when the mean is positive and
p_holm < .05; it shows **NO HARM** when the 95% interval's lower bound is
above -.05; otherwise the effect is not shown. Frozen predictions: HELPS
in DoorKey plain, DoorKey count, MultiRoom plain and KeyCorridor plain;
NO HARM in MultiRoom count and KeyCorridor count.

**Decision.** If every prediction holds, the paper recommends the tenth
weight as its imitation setting and reports this cohort as its
confirmation. Otherwise it reports weaker imitation as a development
finding, with this cohort's numbers as they are. Secondary, descriptive:
`rules_weak - rules` per cell, and `rules - none` (the paper weight
re-measured on fresh seeds).

**Limits.** One alternative weight, not a sweep. The equal-call LLM advice
comparison (P3) was run at the paper weight and is not re-run at a tenth
(paid); a claim that rules beat advice therefore stays at the shared
paper weight.

## Addendum 3 (2026-09-29): KeyCorridor bank with the `door_unlocked` progress fact

Written before any run of the suites below.

**Bank** (scripts/conditional_rules_keycorridor_mem.py; the base KeyCorridor
pipeline's states, task text, model, settings, rule form and checks
otherwise unchanged):
1. One predicate, `door_unlocked`: yes once the locked door has been
   unlocked in this episode. Only the agent can unlock it, so the value is
   determined by the agent's own history; the executor reads it from the
   episode (teachers/minigrid/rule_bank.py, observer `keycorridor_mem_v1`).
2. The action list states MiniGrid's exact preconditions (pickup does
   nothing while carrying; drop needs an EMPTY floor cell, not a doorway).
3. Self-check and blind check (5 pool situations per rule), with identical
   rules' checks pooled: a rule stands unchanged only if the check of every
   copy left it unchanged.
GPT-5-mini, 96 calls, $0.2253 (consult $0.0776, self-check $0.0831,
blind check $0.0646), from the analysis machine, outside the ledger.

**Offline results** (results/conditional_rules_keycorridor_mem_20260929/
evaluation.json). The selected bank, pooled and valid-restricted
(`self_checked_pooled_valid`, 28 rules), solves 55/200 maps as a policy and
48/200 on a second, held-out map set (random useful actions alone: 2/200);
it finishes 55 of the 57 maps where it unlocks the door; the failures are
the key search (72 maps never find the key; 71 find it but do not reach the
locked door). Confirmation panel: 587 correct labels of 762 (.770).

**Selection.** Declared before any check reply was read: train the bank
variant that solves the most of 200 fresh maps (seeds 31,000,000+), gate
50/200. Pooling was introduced during this development comparison, after
the unpooled variants had been scored (best unpooled: 37/200), so the
selected pooled bank is also reported on the held-out map set (48/200, just
below the gate). No training seed was used in any of this.

**Design.** Suites `kc_mem` (replicates 5-14, the kc suite's seeds) and
`kc_mem_confirm` (fresh replicates 30-39, kc_confirm's seeds), both
KeyCorridor students, arms `rules_mem` (paper weight, 1 -> .01) and
`rules_mem_weak` (a tenth, .1 -> .001). Each cell equals the same seed's
`rules` / `rules_weak` cell except the bank. 80 free runs. They pair
by seed with none / rules / rules_weak already run on those seeds.

**Family (fresh seeds 30-39, Holm over four).** `rules_mem - none` and
`rules_mem_weak - none` in KeyCorridor plain and count; verdicts as in
addendum 2 (HELPS: mean > 0 and p_holm < .05; NO HARM: interval above
-.05). Frozen predictions: HELPS for both arms on plain PPO; NO HARM for
`rules_mem_weak` on the count student; no prediction for `rules_mem` on
the count student (the base bank at that weight: -.496). A speed-up of the
count student is not predicted; it is reported descriptively (success
over the first 1M and 2M steps, steps to 80%), as for every cell.

**Secondary, descriptive.** The same contrasts on replicates 5-14; the
bank with against without the progress fact at equal weight
(`rules_mem - rules`, `rules_mem_weak - rules_weak`), on both seed sets.

**Decision.** If the predictions hold, the paper's KeyCorridor result is
this bank. The base-bank KeyCorridor results stay reported. Otherwise the
cohort is reported as it is.

## Addendum 4 (2026-09-30): the DoorKey rules, unchanged, on DoorKey-16x16

Written before any run of the suite below. It uses ten paired seeds in one
cohort.

**Why.** The rules read only the student's own 7x7 view, so nothing in them
depends on the map's size. Run alone as a policy on fresh maps, the checked
DoorKey-8x8 bank solves .34 of DoorKey-16x16 episodes against .03 for random
useful actions (docs/assets/rule_speed_2026-09-30/rules_alone_vs_student.json,
`larger_maps_zero_shot`). DoorKey-16x16 has four times the area and a
2560-step limit.

**Design.** Suite `dk16`: DoorKey-16x16, both students, fresh replicates
40-49 (no suite has used them on any DoorKey map), arms `none` and
`rules_weak` (imitation .1 -> .001, the setting confirmed on DoorKey-8x8 in
addendum 2), 5M steps, 40 free runs. Every cell equals the same student's
DoorKey-8x8 confirmation cell except the map, the seed and the run name
(tests/test_fix_wave.py checks this). The bank is the checked DoorKey bank,
unchanged: research/rule_banks/v3_20260928/blind_strict.json, sha256 of the
committed file 1357545cf30c020f05ff0342504281d561ea7fb9f0b42feee33deb2d2e1e4c2e.
No LLM call, no new rule, no weight or bank choice from this cohort.

**Family (Holm over two).** `rules_weak - none` per student, paired by seed,
two-sided paired t; verdicts as in addendum 2 (HELPS: mean > 0 and
p_holm < .05; NO HARM: interval above -.05). Frozen predictions: HELPS for
plain PPO and HELPS for the count-bonus student.

**Secondary, descriptive.** Median steps to 80% and 95% teacher-off success,
mean success over the first 1M and 2M steps, final success, and the
taught students' final success beside the bank's .34 as a policy.

**Decision.** Every run is reported, unsolved ones included. Whatever the
verdicts, this is the paper's transfer result: a bank written for 8x8, used
unchanged on 16x16. A second cohort would need its own frozen addendum.

**Limits.** One larger map of one task; the larger KeyCorridor maps are not
tested in training (the memory bank as a policy solves .03 / .01 / .00 of
S4R3 / S5R3 / S6R3 episodes).

## Addendum 5 (2026-10-01): the KeyCorridor memory bank, unchanged, on S4R3

Written before any run of the suite below. Same kind of test as addendum 4:
the teaching knowledge (the frozen bank) transfers to a harder map with a
fresh student; frozen students (no training) are tested on larger maps
separately.

**Why.** On KeyCorridor-S4R3 (4x4 rooms, 480-step limit) the S3R3 memory
bank, unchanged, labels .51 of states sampled along oracle routes with
detours, .690 of them optimal (S3R3: .57 and .761;
docs/assets/rule_speed_2026-09-30/larger_map_labels.json). As a policy it
solves only .03 of S4R3 episodes, but MultiRoom showed that a bank that
finishes no episode alone can still speed learning.

**Design.** Suite `kc_s4`: both students, replicates 40-49, arms `none` and
`rules_mem_weak` (.1 -> .001), 5M steps, 40 free runs. Each cell equals the
same student's S3R3 confirmation cell (kc_confirm / kc_mem_confirm) except
the map, the seed and the run name (tested). Bank
research/rule_banks/keycorridor_mem_20260929/self_checked_pooled_valid.json,
unchanged; the unlocking memory is read from the S4R3 episode the same way.

**Family (Holm over two).** `rules_mem_weak - none` per student, verdicts
as in addendum 2. Frozen predictions: HELPS for plain PPO, NO HARM for the
count-bonus student.

**Decision.** Every run is reported; this is a transfer result whatever the
verdicts. Secondary: speed (steps to 80% and 95%, first 1M and 2M steps).

## Addendum 6 (2026-10-01): teacher-view ablation, the same rules checked on the full map

Written before any run of the suites below, after a novelty search found
the
paper's distinct claim to be the restriction itself: a privileged teacher's
rules are checked on what the student observes, and an unseen object is
unknown, so the rule abstains. No result so far isolates that restriction.
Rules beat the same LLM calls spent on per-state advice, but that advice
also covers far fewer states. This ablation holds the rules fixed and
changes only where their conditions are read.

**Design.** Suites `dk_tv`, `mr_tv`, `kc_tv`: both students, replicates
30-39 (the confirmation seeds), one arm each, 5M steps, 60 free runs, no
API call. `rules_weak_tv` (DoorKey, MultiRoom) and `rules_mem_weak_tv`
(KeyCorridor) equal, setting for setting, the completed confirmation cells
`rules_weak` (dk_confirm, mr_confirm) and `rules_mem_weak` (kc_mem_confirm)
of the same student and seed, except the bank file and the run name
(tested). Each bank in research/rule_banks/teacher_view_20261001/ has its
source's rules and mode byte for byte (tested); only its `observer`
differs (teachers/minigrid/teacher_view.py). That observer computes the
same predicate names from the full map in the agent's frame: every object
on the grid counts as visible, at its true offset, through walls and beyond
the 7x7 window; `front`, `carrying` and the unlocking memory are the
student's own. An object behind the agent gets ahead='behind' and a
negative fwd, values no rule tests, so only side-only rules fire on it.
Geometry check: wherever the student sees an object, the two observers
agree (DoorKey 4,534 and KeyCorridor 2,607 comparisons, MultiRoom 3,881
seen-object offsets, no disagreement; also a unit test).

**Offline, before this addendum (descriptive, selects nothing).** Same
banks on states sampled along oracle routes with short random detours
(layout seeds 38,000,000+, about 480 states per task;
scripts/teacher_view_20261001.py labels;
docs/assets/teacher_view_2026-10-01/labels.json):

| Bank | Coverage SV / TV | Precision SV / TV | Full-map-only labels (precision) | Student-view-only |
|---|---|---|---|---|
| DoorKey blind_strict | .486 / .659 | .965 / .843 | 120 (.658) | 38 |
| MultiRoom scoped (plain) | .806 / .854 | .757 / .734 | 35 (.514) | 12 |
| MultiRoom blind_strict (count) | .200 / .198 | .969 / .937 | 5 (.400) | 6 |
| KeyCorridor memory | .561 / .525 | .764 / .744 | 34 (.735) | 51 |

No state got a different action from the two views; they differ in which
states are labelled. Labels that only the full map gives are less often
optimal than the student-view labels in every bank. In KeyCorridor the
full map also removes 51 labels: more rules fire at once and conflict.

**Family (Holm over six).** `student view - teacher view` per task and
student: `rules_weak - rules_weak_tv` (DoorKey, MultiRoom) and
`rules_mem_weak - rules_mem_weak_tv` (KeyCorridor), paired by seed, AUC of
teacher-off success. Verdicts as in addendum 2, read for this contrast:
HELPS = checking on the student's view beats checking on the full map
(mean > 0, p_holm < .05); NO HARM = the restriction costs at most .05 AUC
(lower 95% bound > -.05); otherwise NOT SHOWN.

**Frozen predictions.** Positive mean in every cell. HELPS in DoorKey for
plain PPO, where the full map adds the most labels and the largest
precision drop; NO HARM in the other five. MultiRoom with the count-bonus
student (blind_strict bank) is expected near zero: its two views label
almost the same states.

**Decision.** Every run is reported. One or more HELPS: the paper may say
that checking rules on the student's own view improves learning in those
cells, named. No HELPS: the restriction stays a stated design choice,
supported by the offline precision above, and the paper says the learning
effect was not shown. Secondary, descriptive: `teacher view - none` per
cell (do full-map rules still help?), steps to 80% and 95%, final success.

**Limits.** The full-map arm reuses rules written for the student's view;
it tests where rules are checked, not a bank written for the full map,
which would need new consultations in a new vocabulary.

**Operations.** Launched on Vulcan after `git fetch origin
language-pilot-20260927 && git merge --ff-only FETCH_HEAD`:
`bash scripts/launch_fix_wave.sh <repo> dk_tv mr_tv kc_tv`. Report:
`python -m scripts.run_fix_wave_20260929 report` (the confirm group gains
the addendum 6 family).

## Results of addenda 4 and 5 (2026-10-01 ~18:00 MDT, Vulcan sync 23:49 UTC)

Report: `run_fix_wave_20260929 report` on the synced batch
(docs/assets/transfer_2026-10-01/fix_wave_report_20261001.txt). Archived
files the result sync does not pull (non-artifact extensions, `.env.example`)
were restored from git at each batch's own commit, each written only when
its hash equalled the manifest's; none differed.

**Addendum 4, DoorKey-16x16, the 8x8 bank unchanged.** Count-bonus
student: **HELPS**, +0.984 [+0.978, +0.990], 10/10, p_holm 9.6e-20; the
taught students reach final success 1.000 where the untaught reach 0.000.
Plain PPO: NO HARM so far, +0.030 [-0.036, +0.096], 5/9, p_holm .33
(INTERIM: replicate 40's pair, cells 0-1, still running at 2.5M steps;
final when synced). Frozen prediction HELPS for plain PPO: not met.

**Addendum 5, KeyCorridor-S4R3, the S3R3 memory bank unchanged (final,
40/40).** Count-bonus student: **HELPS**, +0.849 [+0.744, +0.953], 10/10,
p_holm 3.8e-8; final success 0.998 vs 0.194. Plain PPO: +0.000, both arms
0 in every run (NO HARM by the margin, no learning in either arm). Frozen
predictions (HELPS plain, NO HARM count): plain not met; count exceeded.

Reading: on the larger maps the roles reverse. The count-bonus student,
which solves the source maps alone, fails or nearly fails alone on the
larger ones, and the unchanged rules let it solve them; plain PPO learns
neither larger map with or without rules within 5M steps.

## Addendum 7 (2026-10-01): the weak student on the larger maps, stronger imitation

Written before any run of the suites below, after the addendum 4/5 results.
On DoorKey-16x16 plain PPO with the tenth weight received 3-7x fewer labels
than on 8x8 (0.24-0.56M vs ~1.8M per run): rules speak only when the key or
door is in view, and a student without an exploration bonus rarely gets
there. With the weight decaying from .1 over the first 2.5M steps, the few
labels arrive when imitation is almost off; 1 of 9 runs still learned the
task and 4 more reached first successes without consolidating. The
question: does the paper's imitation weight (1 -> .01, ten times
stronger), with the same banks unchanged, let the weak student learn the
larger maps?

**Design.** Suites `dk16_w` (DoorKey-16x16, arm `rules`, the 8x8 bank) and
`kc_s4_w` (KeyCorridor-S4R3, arm `rules_mem`, the S3R3 memory bank): plain
PPO only, replicates 40-49, 5M steps, 20 free runs. Each cell equals the
same seed's `rules_weak` / `rules_mem_weak` cell of dk16 / kc_s4 except the
imitation weight and the run name (tested); it pairs by seed with those
suites' `none` cells (report groups dk16 and kc_s4).

**Test.** Per map, `rules - none` (DoorKey-16x16) and `rules_mem - none`
(KeyCorridor-S4R3), plain PPO, paired AUC of teacher-off success, verdicts
as in addendum 2, each its own family of one. Frozen predictions: HELPS on
DoorKey-16x16; on KeyCorridor-S4R3 no prediction (both plain arms were 0 in
every run). Secondary: final success, steps to first success, labels
delivered, and the stronger-weight arm against the tenth-weight arm by seed.

**Decision.** Every run is reported. A HELPS lets the paper say the
unchanged rules also teach the weak student on that larger map, at the
paper's imitation weight; otherwise the weak student's larger-map result
stays as reported in addenda 4 and 5.

## Addendum 8 (2026-10-01): RLingua-style teacher execution against label-only teaching

Written before any run of the suites below, after reading RLingua (Chen et al., RA-L 2024), the closest prior work. RLingua's
LLM-written controller sees the same full state as the learner and, during
training, its action is executed instead of the learner's "with a
probability of p_LLM", annealed exponentially, while an imitation loss pulls
the actor toward the controller. Our teacher never acts: the student
executes every action and rules only label states. This addendum isolates
that one design choice under partial observability.

**Design.** Suites `dk_act`, `mr_act`, `kc_act`: both students, replicates
30-39 (the confirmation seeds), 5M steps, 60 free runs, no API call. Arms
`rules_weak_act` (DoorKey, MultiRoom) and `rules_mem_weak_act`
(KeyCorridor) equal, setting for setting, the completed confirmation cells
`rules_weak` / `rules_mem_weak` of the same student and seed, except
`execute_teacher_start` = .75 and the run name (tested): same bank, same
imitation loss and weight, same seeds. Where a rule labels a state, its
action is executed instead of the student's with probability
p = .75 * 0.01^(progress / .5): exponential annealing to 1% of the start at
mid-training, and 0 once distillation stops (75%). The executed action is
stored with the student's own log-probability of it (the update's
log-ratio clamp bounds the importance ratio), as an override PPO must. The
start value .75 is our earlier override default; RLingua's initial p_LLM
and annealing rate are not given in its main text. Execution draws use their
own random stream. algos/ppo_distill.py `execute_teacher_*`; default 0
leaves every other arm unchanged (tested end to end).

**Family (Holm over six).** `labels only - teacher acts` per task and
student: `rules_weak - rules_weak_act` (DoorKey, MultiRoom) and
`rules_mem_weak - rules_mem_weak_act` (KeyCorridor), paired by seed, AUC of
teacher-off success. Verdicts as in addendum 2, read for this contrast:
HELPS = label-only teaching beats letting the teacher act (mean > 0,
p_holm < .05); NO HARM = label-only is not worse by more than .05;
otherwise NOT SHOWN.

**Frozen predictions.** Positive mean in every cell (label-only at least as
good), on the grounds of our June pilot (an oracle teacher executing actions
with annealed probability left DoorKey students at 0% once it stopped) and
of DAgger's analysis of learning from teacher-driven trajectories. No
per-cell significance is predicted: here the teacher acts only where a rule
fires, and the imitation loss continues after execution fades, either of
which may prevent a collapse.

**Secondary, descriptive.** `teacher acts - none` per cell (does execution
still help?); the gap between training-time episode success (teacher
acting) and teacher-off evaluation success over the first half of training
(the masking signature); teacher actions executed per run; steps to 80% and
95% teacher-off success.

**Decision.** Every run is reported. One or more HELPS: the paper may say
that, under partial observability, teaching through labels outperforms
letting the teacher act as in RLingua-style integration, in those cells,
named, and only as "RLingua-style execution" (their full system, robots and
TD3 are not reproduced). No HELPS: the paper reports the comparison as is
and does not claim an advantage of label-only teaching.

**Operations.** Launched on Vulcan after `git fetch origin
language-pilot-20260927 && git merge --ff-only FETCH_HEAD`:
`bash scripts/launch_fix_wave.sh <repo> dk_act mr_act kc_act`.

## Addendum 9 (2026-10-01): the larger environments of all three families, 15M steps

Written before any run of the suites below, to show that the method extends to larger environments the way it was confirmed on
the source tasks, with both students, all three families and three times
the training. Addenda 4 and 5 ran two larger maps at 5M steps; there the
count-bonus student was taught to solve both, and plain PPO learned neither
(its labels were 3-7x scarcer and arrived as the imitation weight decayed).
With the schedule expressed in fractions of training, 15M steps gives three
times the steps at each imitation weight.

**Design.** Suites `dk16_long` (DoorKey-16x16, the 8x8 bank), `mr10_long`
(MultiRoom with 10 rooms, the N6 banks: scoped for plain PPO, blind-checked
for the count student, as on N6) and `kc_s4_long` (KeyCorridor-S4R3, the
S3R3 memory bank). Both students, arms `none` and the unchanged bank at the
tenth weight (`rules_weak`; `rules_mem_weak` in KeyCorridor), replicates
50-59 (no suite has used them), 15M steps, 120 free runs, no API call. Each
cell equals the same student's source-task confirmation cell except the
map, the seed, the horizon and the run name (tested). MultiRoom-N10 is
MiniGrid's MultiRoomEnv with 10 rooms (step limit 200, same 25x25 grid and
7x7 view), registered in envs/registry.py.

**Offline, before this addendum (descriptive).** The N6 MultiRoom banks,
unchanged, on states sampled along oracle routes with short detours
(about 320 states each): scoped coverage .84 / .81 / .80 and precision
.789 / .815 / .773 on N6 / N8 / N10; blind-checked coverage .19 / .18 / .19,
precision .934 / .966 / .935. The DoorKey and KeyCorridor banks on their
larger maps: precision .993 and .690 (docs/assets/rule_speed_2026-09-30/
larger_map_labels.json).

**Family (one per environment, Holm over two students each).**
`rules - none` per student, paired by seed, AUC of teacher-off success,
verdicts as in addendum 2. Frozen predictions: HELPS for the count-bonus
student in all three environments; for plain PPO, HELPS in MultiRoom-N10
and no prediction on DoorKey-16x16 and KeyCorridor-S4R3 (no plain arm
learned either at 5M).

**Decision.** Every run is reported. The paper may say the unchanged
explanations extend to larger environments for each student and
environment with HELPS, named. Secondary: final success, steps to 80% and
95%, labels delivered, and the 5M results of addenda 4 and 5 beside these.

**Operations.** Launched on Vulcan after `git fetch origin
language-pilot-20260927 && git merge --ff-only FETCH_HEAD`:
`bash scripts/launch_fix_wave.sh <repo> dk16_long mr10_long kc_s4_long`.

### Addendum 8, qualification (2026-10-02, before any addendum-8 result was synced or inspected)

An independent review confirmed a formulation issue in the execution arm. On a labelled step the
behavior distribution is q = (1 - p) pi_old + p 1[a = y], but the stored
log-probability and the PPO ratio use pi_old, and GAE uses the mixed
trajectory without correction; this holds on every labelled step, including
those where the coin kept the student's action. The arm is therefore an
execution-augmented PPO heuristic (the same pattern as the earlier override
trainer, algos/ppo_teacher.py), not RLingua's off-policy learner and not an
isolated test of student control. Withdrawn from the text above: that the
addendum "isolates that one design choice", and that student log-probabilities
are what "an override PPO must" store. Unchanged: the frozen arm, seeds,
family, predictions and verdict rule. Any result is reported as label-only
teaching against this specified heuristic, with the mismatch disclosed, and
supports no claim of superiority over RLingua or over teacher execution in
general. A principled variant needs its own addendum.

## Addendum 10 (2026-10-02): progress-only training ablation in KeyCorridor

Written before any run of the suite below. A reviewed retrospective
analysis (scripts/analyze_executable_teaching_20261002.py) showed at the
level of advice that the memory bank's progress clauses suppress repeated
key-pickup labels after unlocking: 55 of 323 saved post-unlock states with
the base bank, 0 with the memory bank, 55 again with the memory bank minus
its progress clauses. Addendum 3's bank differs from the base bank in
several ways (the progress predicate, the action-validity restriction,
pooled self-checks); this isolates the progress clauses' effect on
learning.

**Design.** Suite `kc_prog`: KeyCorridor-S3R3, both students, replicates
30-39 (the confirmation seeds), 5M steps, 20 free runs, no API call. Arm
`rules_mem_weak_noprog` equals, setting for setting, the completed
`kc_mem_confirm` cell `rules_mem_weak` of the same student and seed, except
the bank file and the run name (tested). Its bank
(research/rule_banks/progress_ablation_20261002/keycorridor_mem_noprogress.json,
scripts/progress_ablation_20261002.py) is the memory bank with every
condition and exception on `door_unlocked` removed (15 conditions, no
exceptions) and nothing else changed: same rules otherwise, mode, observer
and imitation weight; its rules equal the offline counterfactual
`parsed_rules(strip_progress=True)` exactly (tested).

**Family (Holm over two).** `rules_mem_weak - rules_mem_weak_noprog` per
student, paired by seed, AUC of teacher-off success; verdicts as in addendum
2, read as: HELPS = the progress clauses improve learning (mean > 0,
p_holm < .05). Frozen predictions: positive mean for both students, HELPS in
at least one. Secondary: `rules_mem_weak_noprog - none` per student; labels
delivered; final success.

**Decision.** Every run is reported. HELPS in a cell lets the paper say the
progress clauses, isolated from the bank's other differences, improve
learning for that student; otherwise the learning gain stays attributed to
the bank as a whole, with the advice-level counterfactual reported as such.

## Addendum 11 (2026-10-02): the KeyCorridor memory bank on S5R3 and S6R3, 15M steps

Written before any run of the suites below: the larger-environment test of addendum 9, extended to two more KeyCorridor
sizes. Offline (docs/assets/rule_speed_2026-09-30/larger_map_labels.json) the
unchanged memory bank labels S5R3 / S6R3 states with precision .727 / .653.
Addendum 9's 15M runs took 12-17 h on Vulcan.

**Design.** Suites `kc_s5_long` (BabyAI-KeyCorridorS5R3, replicates 60-69)
and `kc_s6_long` (BabyAI-KeyCorridorS6R3, replicates 70-79): both students,
arms `none` and `rules_mem_weak` (the S3R3 memory bank unchanged), 15M steps,
80 free runs. Each cell equals the same student's S3R3 confirmation cell
except the map, the seed, the horizon and the run name (tested).

**Family (one per map, Holm over two students).** `rules_mem_weak - none`,
verdicts as in addendum 2. Frozen predictions: HELPS for the count-bonus
student on S5R3; no prediction for S6R3 or for plain PPO.

**Operations for addenda 10 and 11.** Launched on Vulcan after
`git fetch origin language-pilot-20260927 && git merge --ff-only FETCH_HEAD`:
`bash scripts/launch_fix_wave.sh <repo> kc_prog kc_s5_long kc_s6_long`.

## Results of addenda 6, 7, 8 and 9 (2026-10-02, Vulcan sync 21:15 UTC)

Report: `run_fix_wave_20260929 report` on the synced batch
(docs/assets/results_2026-10-02/fix_wave_report_20261002.txt); every cell of
these suites validated (dk_tv/mr_tv/kc_tv 60/60, dk16_w/kc_s4_w 20/20,
dk_act/mr_act/kc_act 60/60, dk16_long/mr10_long/kc_s4_long 120/120; dk16 now
40/40). Archived files the result sync does not pull were restored from git at
each batch's commit, each only when its hash equalled the manifest's; none
differed.

**Addendum 9, larger environments at 15M (Holm over two per map).**
DoorKey-16x16: count **HELPS** +0.901 [+0.774, +1.028] 10/10 (final success
.988 vs .112); plain PPO **HELPS** +0.463 [+0.337, +0.590] 10/10 (final .878
vs .000). MultiRoom-N10: plain **HELPS** +0.534 [+0.439, +0.629] 10/10 (final
.644 vs .000); count NO HARM +0.023 [-0.011, +0.057] 8/10 (it solves N10 alone,
AUC .970). KeyCorridor-S4R3: count **HELPS** +0.614 [+0.335, +0.893] 10/10
(final .996 vs .502); plain 0 vs 0 in every run. Frozen predictions: count
HELPS in all three (met for DoorKey and KeyCorridor; MultiRoom NO HARM because
the count student solves it alone); plain HELPS in MultiRoom-N10 (met); no
prediction for plain DoorKey-16x16 (HELPS) or KeyCorridor-S4R3 (no learning).

**Addendum 7, plain PPO at the paper's weight on the larger maps (5M).**
DoorKey-16x16 `rules - none` **HELPS** +0.146 [+0.005, +0.286] 5/10 (final
.394); the direct weight contrast `rules_weak - rules` -0.119 [-0.284, +0.047]
3/10 (secondary, n.s.). KeyCorridor-S4R3: 0 vs 0. Prediction (HELPS on
DoorKey-16x16) met.

**Addendum 6, student view minus teacher view, same rules (Holm over six).**
DoorKey plain **HELPS** +0.038 [+0.017, +0.059] 10/10, p_holm .014; DoorKey
count +0.347 [+0.052, +0.642] 7/10 (NO HARM, p_holm .10); KeyCorridor count
+0.001 (NO HARM); KeyCorridor plain +0.244 [-0.091, +0.579] 6/10 (NOT SHOWN);
MultiRoom plain +0.069 [-0.017, +0.155] 7/10 (NO HARM); MultiRoom count
-0.007 [-0.010, -0.003] 0/10 (NO HARM by the margin; teacher view slightly
ahead). Frozen predictions: positive mean in every cell (5 of 6; MultiRoom
count slightly negative), HELPS for DoorKey plain (met), NO HARM elsewhere
(met in four of five; KeyCorridor plain not shown). By the decision rule the
paper may say checking rules on the student's view improves learning for plain
PPO on DoorKey; elsewhere no learning advantage is demonstrated.

**Addendum 8, label-only minus the execution heuristic (Holm over six; see
the qualification above).** No cell differs: DoorKey count -0.004, plain
-0.002; KeyCorridor count +0.003, plain -0.162 [-0.294, -0.031] 2/10 (NOT
SHOWN; execution ahead, n.s. after Holm); MultiRoom count -0.001, plain +0.092
(n.s.). The frozen prediction (positive mean everywhere) failed in four cells.
Reported as: no evidence that label-only teaching outperforms this
execution-augmented PPO heuristic.

## Results of addenda 10 and 11 (2026-10-03, Vulcan sync 20:53 UTC)

Report: docs/assets/results_2026-10-03/fix_wave_report_20261003.txt; every
cell validated (kc_prog 20/20, kc_s5_long 40/40, kc_s6_long 40/40). Archived
files the result sync does not pull were restored from git at each batch's
commit, each only when its hash equalled the manifest's; none differed.

**Addendum 10, progress-only training ablation (Holm over two).** With minus
without the progress clauses, all else identical: plain PPO **HELPS** +0.702
[+0.580, +0.825] 10/10, p_holm 8.0e-7 (AUC .733 vs .031; final success .994
vs .086); count **HELPS** +0.009 [+0.004, +0.015] 10/10, p_holm .0047. Without
its progress clauses the bank barely teaches plain PPO (`noprog - none`
+0.031 [-0.039, +0.101]); for the count student it still helps a little
(+0.030 [+0.015, +0.044] 10/10). Frozen predictions (positive mean for both,
HELPS in at least one) met, both HELPS. By the decision rule the paper may say
the progress clauses, isolated from the bank's other differences, improve
learning for both students.

**Addendum 11, the S3R3 memory bank on larger KeyCorridor maps at 15M.**
S5R3: count **HELPS** +0.744 [+0.700, +0.788] 10/10, p_holm 5.5e-11 (final
.984 vs .000: the count student does not solve S5R3 alone); plain 0 vs 0.
S6R3: count **HELPS** +0.178 [+0.056, +0.299] 10/10, p_holm .018 (final .490
vs .000); plain 0 vs 0. Frozen prediction (count HELPS on S5R3) met; S6R3 had
no prediction.

## Addendum 12 (2026-10-04): content control, the confirmed banks with deranged targets

Written before any run of the suites below. A read-only review and a
simulated-reviewer report (2026-10-04) both name a
matched content control as the most important missing experiment: nothing in
the confirmed cohort separates the value of WHAT the rules recommend from the
value of receiving an auxiliary cross-entropy target on the same states. The
existing shuffled-rule control (rule-bank study, replicates 0-19) is not a
matched control for the confirmed method: it used the earlier raw banks, the
paper weight (1 -> .01), and it permuted actions between rules, which changes
which states conflict and abstain.

**Design.** Suites `dk_cc`, `mr_cc`, `kc_cc`: replicates 30-39 (the
confirmation seeds), both students, 5M steps, 60 free runs, no API call. Arm
`rules_weak_cc` (DoorKey, MultiRoom) / `rules_mem_weak_cc` (KeyCorridor)
equals, setting for setting, the completed confirmed cell of the same student
and seed (`dk_confirm` / `mr_confirm` `rules_weak`; `kc_mem_confirm`
`rules_mem_weak`) except the bank file, the run name and addendum 13's extra
evaluations (tested). Its bank (research/rule_banks/content_control_20261004/,
scripts/content_control_banks_20261004.py) is the confirmed bank with every
rule's action replaced by the fixed derangement a -> (a + 3) mod 6 (left ->
pickup, right -> drop, forward -> toggle, and back; no action maps to itself);
conditions, exceptions, mode and observer are unchanged. Because the map is a
bijection, the executor labels and abstains on exactly the same states as the
confirmed bank and its target is always the deranged original (tested on
random-walk states of all three tasks). In MultiRoom each student receives
the derangement of its own selected bank.

**Family (Holm over six).** Confirmed minus deranged, per cell, paired by
seed, AUC of teacher-off greedy success; verdicts as in addendum 2, read as:
HELPS = the content of the targets matters. Frozen predictions: positive mean
in all six cells; HELPS in at least the five cells where the confirmed arm
helped against no teacher (DoorKey both students, MultiRoom plain PPO,
KeyCorridor both students). Secondary family (Holm over six): deranged minus
none; prediction: HELPS in no cell (wrong targets do not help), with harm
possible for Count-PPO. Also reported: labels delivered per run.

**Decision.** Every run is reported. If the primary predictions hold, the
paper may say that the learning gains depend on the actions the rules
recommend, not merely on auxiliary supervision at the same states (scope:
these banks, weight and budget). If deranged minus none HELPS in a cell, the
paper reports that auxiliary targets alone help there and restricts the
content claim to cells where confirmed minus deranged HELPS. Training
trajectories differ between arms, so label exposure is reported, not held
fixed. The test concerns action-target content, not natural-language semantics.

## Addendum 13 (2026-10-04): the confirmed cells re-run with dense early evaluation

Written before any run of the suites below. The first regular evaluation is at 204,800 transitions (200 rollouts), when taught
DoorKey students are already near 1.0, so the published curves cannot show
how learning begins. No initial score was ever saved, and none is imputed.

**Design.** Suites `dk_dense`, `mr_dense`, `kc_dense`: replicates 30-39, both
students, arms `none` and the confirmed rule arm (`rules_weak`;
`rules_mem_weak` in KeyCorridor), 5M steps with the unchanged horizon and
advice schedule, 120 free runs. Each cell equals the confirmed cell of the
same arm, student and seed except the run name and
`diagnostic_eval_frames = 0, 10240, 25600, 51200, 76800, 102400, 128000,
153600, 179200` (tested). The extra evaluations (the 2026-10-01
instrumentation) use the regular evaluator (greedy and sampled, 50 episodes
each, the same evaluation seeds), isolate every random-number generator,
check that the student's parameters are unchanged, and are saved apart in
diagnostic_evaluations.jsonl; the regular evaluations and AUC inputs keep
their original definition. Addenda 12 and 14 carry the same extra frames.

**Reproduction rule (revised 2026-10-05, before any run, after review).** The first version allowed splicing the early points onto the
original curves when every re-run AUC equalled the original's. That
inference is invalid: different learning curves can share an area, and even
identical evaluation curves do not certify identical training. Therefore the
re-runs are always reported as their own **replay cohort** (group `dense`, its
own family and its own complete curves from step 0), never spliced into or
averaged with the original runs, never chosen over them by outcome; the
primary table keeps the original cohort. The report's reproduction line is
informational only: per re-run, whether the initial-policy hash, the full
greedy and sampled regular-evaluation curves, the labels delivered and the
AUC equal the original's. Any figure that shows replay curves says so.

**Family (Holm over six).** Rule arm minus none in group `dense`; verdicts as
in addendum 2. Frozen prediction: the original cohort's six verdicts.

## Addendum 14 (2026-10-04): MultiRoom bank crossover

Written before any run of the suite below. The two MultiRoom students imitate different banks, each selected for its student on
development seeds (plain PPO: the 32-rule unchecked `scoped` bank; Count-PPO:
the 7-rule `blind_strict` bank). Reviewers will ask whether one bank could
serve both students.

**Design.** Suite `mr_cross`: replicates 30-39, both students, 5M steps, 20
free runs. Arm `rules_weak_xbank` equals the confirmed `mr_confirm`
`rules_weak` cell of the same student and seed except that it imitates the
OTHER student's selected bank (and the run name and addendum 13's frames;
tested).

**Family (Holm over two).** `rules_weak_xbank - none` per student.
Secondary: `rules_weak - rules_weak_xbank` per student (own selected bank
minus the other). Frozen predictions: Count-PPO with the 32-rule bank:
positive mean against none; plain PPO with the 7-rule bank: no HELPS against
none (in the rule-bank study the strict bank delivered few labels to plain
PPO and its AUC stayed at 0), so the crossover is expected to show a
coverage-precision trade-off between the two banks.

**Decision.** If the 32-rule bank HELPS Count-PPO, the paper may report one
MultiRoom bank teaching both students and keep the per-student selection as
a secondary result; otherwise the per-student selection stays, disclosed
with these crossover results.

## Addendum 15 (2026-10-05): a fresh-seed source cohort with measured early learning

Written before any run of the suites below, after a review of addenda
12-14 that recommended one new batch on seeds no
study has used, rather than more arms on the reused confirmation seeds: the
progress-clause effect (addendum 10) is a follow-up on reused seeds, and the
dense measurements of addendum 13 are a replay, not new evidence. This
addendum combines the 140-run package (no teacher, confirmed arm and
KeyCorridor without progress clauses) with the content control (addendum 12)
and MultiRoom's crossover (addendum 14) in the same fresh cohort, so every
source comparison pairs within one cohort. It is recommended **instead of**
the historical-seed suites of addenda 12-14, not in addition to them.

**Design.** (The grouping into suites in this paragraph is superseded by
the packaging note at the end of this addendum: core `dk_fresh` 40,
`mr_fresh` 40, `kc_fresh` 60, and optional suites of 20 each; arms, seeds
and settings are unchanged.) Suites `dk_fresh` (60 runs), `mr_fresh` (80) and `kc_fresh` (80):
replicates 80-89 (unused by every fix-wave suite and by the rule-bank study,
tested), both students, 5M steps with the unchanged horizon, optimizer and
advice schedule, 220 free runs, no API call. Arms per cell: `none`; the
confirmed arm (`rules_weak`; `rules_mem_weak` in KeyCorridor); its deranged
twin (addendum 12); in KeyCorridor `rules_mem_weak_noprog` (addendum 10); in
MultiRoom `rules_weak_xbank` (addendum 14). Every cell equals its
historical twin of replicate 30 except the seed, the run name and the dense
evaluation frames of addendum 13 (tested). Every arm carries the dense
frames (0 to 179,200 transitions, RNG-isolated, saved apart).

**Families (Holm within each; verdicts as in addendum 2).**
1. Replication (six cells): confirmed arm minus none. Frozen predictions:
   HELPS in the five cells where the original cohort found HELPS (DoorKey
   both students, MultiRoom plain PPO, KeyCorridor both students); positive
   mean for MultiRoom Count-PPO.
2. Progress (two cells): `rules_mem_weak - rules_mem_weak_noprog`.
   Prediction: positive mean for both students; HELPS for plain PPO.
3. Content (six cells): confirmed arm minus its deranged twin. Prediction:
   positive mean in all six; HELPS in the five cells of family 1.
   Secondary (six cells): deranged twin minus none; prediction: HELPS in no
   cell.
4. Crossover (two cells): `rules_weak_xbank - none`; predictions as in
   addendum 14.

**Primary metric** as in every earlier addendum: greedy teacher-off success
AUC over the regular evaluations (204,800 to 5M). The dense frames give
secondary, descriptive summaries only (greedy and sampled success at each
early frame and the area over 0-204,800 including the measured step 0); no
new primary metric is introduced.

**Reporting.** This cohort is reported as its own cohort, beside (never
spliced into) the original confirmation cohort. A pooled 20-pair analysis of
the original and fresh cohorts may be shown as a secondary, labelled
analysis; it is not a test of any frozen prediction. Every run is reported,
including failures; infrastructure failures follow the dated-amendment
recovery rule used for the Crafter confirmations.

**What the outcomes permit.** Family 1 replicating lets the paper state the
source results on two independent cohorts. Family 2 HELPS gives a fresh
confirmation of the progress-clause effect, still subject to addendum 10's
caveat that removing clauses also changes coverage and conflicts. Family 3
HELPS shows the gains are sensitive to the content of the targets (not an
exact exposure-matched or LLM-specific claim; a remapped target is
different, not always wrong). Family 4 tells whether one MultiRoom bank can
serve both students.

**Packaging (2026-10-05, before any run).** The cohort is
split into separately launchable suites so a launch command states its real
size. CORE, 140 runs: `dk_fresh` and `mr_fresh` (none, confirmed arm; 40
each) and `kc_fresh` (none, memory arm, memory arm without progress clauses;
60): families 1 and 2. OPTIONAL, 80 runs on the same seeds: `dk_fresh_cc`,
`mr_fresh_cc`, `kc_fresh_cc` (deranged twins; 20 each) and `mr_fresh_cross`
(20): families 3 and 4. The arms, seeds and settings above are unchanged;
only their grouping into suites is new. Which package runs is chosen before
launch, not after seeing outcomes; a family whose suites are not launched is
reported as not run. If the optional suites are launched later, they are
reported as a same-seed follow-up to the core, not as fresh confirmation.

## Addendum 16 (2026-10-05): the RLingua baseline under partial observability

Written before any run of the suites below, and revised the same evening
after a read-only review of commit 7c89c19
(VERDICT NOT READY, three blocking issues; responses below), still before
any suite run. RLingua (Chen et al., RA-L 2024) is the closest published
method: an LLM writes a complete rule-based controller once, the controller
takes some of the actions during RL, and the student clones them. Addendum
8 was not RLingua (it executed our partial rules where they fired, kept our
labels on the student's states, used p = .75, and mixed teacher steps into
PPO's update). This addendum ports RLingua itself, as specified in its
Algorithm A-1 and Table A-II, to our tasks and students.

**The question, in plain terms.** In our setting the teacher knows more
than the student: the student sees a 7x7 window, while the teacher can
know the whole map. RLingua passes that knowledge on by ACTING: its
controller reads the full simulator state and chooses some of the actions,
which the student imitates. We pass it on by ADVISING in the student's own
terms: rules over what the student sees (and one progress fact), applied
only where they hold, with the student choosing every action. Two
comparisons:

- A, a full-state RLingua port (the setting RLingua was published in):
  its controller reads the full simulator state at every step it acts
  (`rlingua_full`, suites `*_rlfull`). This needs privileged state at
  training time, every step, which our rules never read.
- B, RLingua at matched runtime information: its controller reads only
  what our rules read, the student's view (plus the action actually
  executed, and a memory of its own) (`rlingua_view`, suites `*_rl` and
  `*_rl_long`).

**Controllers.** Written by GPT-5-mini, the model our banks were written
with, given the same task knowledge, by RLingua's recipe (its Appendix B.A:
a phase decomposition from a fill-in template, then the controller code
given the exact input variables, then feedback;
scripts/rlingua_controllers_20261005.py). RLingua's non-expert human
feedback is replaced by automatic feedback from 20 test episodes (seeds
36,000,000+: success count, raised errors, where the agent got stuck), up to
three rounds, keeping the best version; no researcher edits the code. A
view controller's prompt says that another policy acts at many steps and
that `obs['last_action']` reports what was executed; its feedback also
reports 20 episodes in which a random action replaces its own at 25% of the
steps (an interruption stress test, not the training mixture, in which the
controller acts at no more than 25% of the steps), and it is selected on
those episodes first, then alone. The first
view controllers (not told the executed action, selected alone) are
superseded (review blocking issue 1): their memory advanced on actions that
never ran, e.g. MultiRoom's stepped through its phases at an unchanged
doorway. Each env runs its own controller instance, recreated at every
episode start: the MultiRoom full-state controller keeps a plan and a
visited set on its function object, which were shared across envs and
episodes before this revision. Costs: $0.27 and $0.14 (receipts and
superseded files in research/rlingua_controllers_20261005/).

Success of each frozen controller by itself on 100 fresh layouts (seeds
37,000,000+), alone and, for view controllers, with a random action at 25%
of the steps; the same controller serves the family's larger map, as our
unchanged bank does:

| controller (by itself) | DoorKey-8x8 | MultiRoom-N6 | KeyCorridor-S3R3 | DoorKey-16x16 | MultiRoom-N10 | KeyCorridor-S4R3 |
|---|---|---|---|---|---|---|
| A: full state | 1.00 | 1.00 | 0.87 | 1.00 | 1.00 | not run (0.38 s per call) |
| B: student view, alone | 0.37 | 0.00 | 0.00 | 0.86 | 0.00 | 0.00 |
| B: student view, another policy at 25% of steps | 0.94 | 0.00 | 0.08 | 0.97 | 0.00 | 0.00 |
| superseded view controller, alone / mixed | 0.67 / 0.90 | 0.00 / 0.00 | 0.11 / 0.20 | | | |

The MultiRoom view controller succeeded in 0 of 20 feedback episodes in all
eleven rounds across both generations; the KeyCorridor one in at most 4 of
20 with another policy acting. The superseded KeyCorridor controller did
better with a random policy acting (0.20 against 0.08) although its memory
assumed its own actions ran; both are reported, and the selection followed
the rule above. The second review found a defect in the selected
KeyCorridor view controller (it compares a stored tuple with an image list,
so it treats every successful forward move as failed). Two feedback rounds
reporting that defect produced versions that succeeded in 0 of 20 feedback
episodes, alone and with another policy acting, so the selection rule kept
the original (4 of 20 with another policy acting); the defect remains and
is reported.

With the full map GPT-5-mini wrote search-based planners that solve the
tasks by themselves; the complete controllers it wrote from the student's
view did not, except in DoorKey.

**Speed-only feedback (KeyCorridor, full state).** The selected KeyCorridor
full-state controller (20/20 feedback episodes) searched by copying the
whole grid at every node: about 0.17-0.27 s per call, which at about 250,000
calls per run adds 12-18 hours per run. It received speed-only feedback (its
time per call; its decisions were not criticised): a first single attempt
was rejected (6/20, 21.5 ms; its reply text was not stored, only these
statistics), then up to three rounds gave 0/20 (0.09 ms), 0/20 (0.05 ms) and
19/20 (57 ms). The acceptance limit, first 10 ms, was raised to 60 ms to
accept the 19/20 version, re-checked before installation (19/20, 43 ms per
call). On KeyCorridor-S4R3 it still takes about 0.38 s per call (about 26
extra hours per 15M run), so `rlingua_full` does not run on the larger maps.

**Mechanism (algos/ppo_distill.py `rlingua_*`, default off).** At every real
transition the controller acts instead of the student with probability
p = 0.25 * 0.999999^n, n the number of real transitions so far (Table
A-II, RLBench row; RLBench's multi-million-step horizon matches ours;
Gymnasium's autoreset ticks, whose action the env discards, are neither
counted nor drawn). Its transitions are stored in a persistent buffer
(RLingua's R_LLM, 1M entries). Each minibatch update adds behavior cloning,
cross-entropy on a same-sized batch drawn from that buffer at constant
weight 1.0 (lambda_IM), replayed from the stored recurrent state (no
burn-in; cross-entropy replaces TD3's squared error, so the numeric weight
does not establish equal strength). The PPO surrogate, value,
intrinsic-value and entropy terms, the advantage normalization and the KL,
clip-fraction and explained-variance diagnostics use only the student's
own real transitions (RLingua trains its actor's RL term and its critic on
R_RL only); autoreset ticks are excluded too (review blocking issue 2; the
other arms keep the trainer's original convention, which includes them;
the effect of this difference was not measured). The student's advantages
and value targets stop before a step
where the controller acted and bootstrap from the value of that state (a
tree-backup cut), so neither the critic nor the policy-gradient term
learns from controller transitions; the continuing intrinsic stream is cut
at controller steps only. A view controller is called at every step (it
sees every frame and is told the executed action) and acts only when
drawn; a full-state controller is called only when drawn. No rule bank, no
execution heuristic, no labels from our method.

**Design.** Suites `dk_rlfull`, `mr_rlfull`, `kc_rlfull` (A; replicates
30-39, both students, 5M steps, 60 runs) and `dk_rl`, `mr_rl`, `kc_rl` (B;
the same cells, 60 runs) pair by seed with the confirmation cohort's
`none`, confirmed arm and execution heuristic. Suites `dk16_rl_long`,
`mr10_rl_long`, `kc_s4_rl_long` (B; replicates 50-59, both students, 15M
steps, 60 runs) pair with addendum 9's `none` and unchanged bank. Every
RLingua cell equals the no-teacher cell of its seed except the RLingua
settings (tested). Each group of three suites is separately launchable.

**Pre-run information (not confirmation data;
docs/assets/rlingua_pilots_2026-10-05/pilot_curves.json).** (a) A 41k-step
timing run of `rlingua_full` (DoorKey, plain PPO, replicate 30) reached
0.94 teacher-free success; its curve and checkpoint hash are archived, and
the `dk_rlfull` cell of replicate 30, plain PPO, is development-exposed
(reported, flagged). (b) Pilots on replicates 200-202 (outside every
protocol block), plain PPO, with the first trainer version (no trace cut,
autoreset ticks included) and the superseded view controllers:
`rlingua_full` reached 0.82 in KeyCorridor at 51,200 steps; `rlingua_view`
reached 0.78, 0.98 and 0.84 in DoorKey at 51,200 steps, 0.00 in MultiRoom
up to 204,800 steps, and at most 0.06 in KeyCorridor up to 153,600 steps;
with the trace cut, DoorKey reached 0.64 at 51,200 and 102,400 steps
(replicate 200). For comparison, the confirmation cohort's confirmed arm
(10 seeds, plain PPO) is at 0.80, 0.60 and 0.00 at 204,800 steps (DoorKey,
MultiRoom, KeyCorridor), reaches 0.96-1.00 at 5M, and its AUCs are 0.939,
0.767 and 0.733; with the count bonus it is at 0.94-0.96 at 204,800 steps
in every task. (c) An offline imitability analysis
(scripts/rlingua_imitability_20261005.py,
docs/assets/rlingua_imitability_2026-10-05/imitability.json), on states of
100 fresh layouts per task visited by a half-random behaviour:

our bank labels 45%, 82% and 46% of DoorKey, MultiRoom and KeyCorridor
states, 0.98, 0.84 and 0.72 of them in the shortest-path optimal set (in
KeyCorridor that set uses the hidden key location, which no view-based
teacher knows); the view controllers label every state, 0.83, 0.74 and
0.67 of them optimal; the full-state controllers 0.96, 0.97 and 0.85. How
well each teacher's targets can be predicted from the student's current
view alone, by a lookup fitted on half of the episodes and scored on the
other half's states whose view it has seen (62%, 45% and 19% of them):
ours 1.00 in all three tasks (the KeyCorridor bank also reads its progress
fact); the full-state controller 0.95, 0.84 and 0.92; the view controller
0.98, 1.00 and 0.88. In-sample agreement, an optimistic bound, is in the
JSON.

**Families (Holm within each; verdicts as in addendum 2, read as HELPS =
ours better).** A: our confirmed arm minus `rlingua_full` over the six
confirmation cells. B: our confirmed arm minus `rlingua_view` over the six
confirmation cells; on each larger map, the same contrast over both
students (two cells each). Secondary: RLingua minus none; our execution
heuristic minus `rlingua_full`; controller-executed steps, buffer size,
cloning loss and controller errors per run. A family's verdicts are issued
only when all of its declared contrasts have all ten seed pairs; until
then the report prints PROVISIONAL (review blocking issue 3).

**Frozen predictions.** (A) Its controller solves the tasks by itself, so
its student imitates a near-oracle: RLingua better (negative means) for
plain PPO in all three tasks; |mean| < 0.05 with the count bonus, where
both students are near ceiling (a prediction about the point estimate, not
an equivalence test). (B) Ours HELPS in MultiRoom for both
students and in KeyCorridor with the count bonus (by themselves the view
controllers succeed 0.00 and 0.00 alone, 0.00 and 0.08 with another
policy acting, and cloning them at constant weight 1.0 pulls the student
towards them, while our rules abstain where the view does not decide the
action). No direction predicted in DoorKey (its view controller succeeds
0.37 alone and 0.94 with another policy acting) or in KeyCorridor with
plain PPO (the pilot student cloning the view controller succeeded earlier
than ours does).
(B, larger maps) positive means in MultiRoom-N10 and KeyCorridor-S4R3 with
the count bonus; no direction elsewhere (plain PPO on KeyCorridor-S4R3
stays at 0 under our bank).

**Decision.** Every run is reported. HELPS in B lets the paper say our
teaching outperforms this RLingua port when both teachers read only what
the student sees, in the named cells. A is reported as a full-state
RLingua port: what a controller that reads the full simulator state and
acts achieves, against teaching that needs neither runtime privileged state
nor control; where RLingua wins, the paper says so. Claims stay within the
stated adaptation (PPO, discrete actions, automatic feedback); no claim
about RLingua's robotics results or its TD3 implementation.

**Review, 2026-10-05 (read-only, commit 7c89c19), and responses.**
Blocking: (1) view-controller memory advanced on unexecuted actions:
controllers regenerated, told the executed action, selected under
mixing; per-env, per-episode instances. (2) autoreset ticks received
student losses and counted in the decay: excluded from RLingua's student
terms and from the decay count. (3) incomplete cohorts could receive
verdicts: verdicts withheld until every declared contrast has all ten
pairs. Non-blocking, adopted: the trace cut judged defensible (kept);
KL and clip-fraction diagnostics masked; the imitability wording limited
to measured agreement, with a held-out estimate added; the receipt note
about the missing first speed reply; replicate 30's exposure recorded.
Noted as limitations: the cross-entropy weight's strength relative to
TD3's squared error, and stale stored-state recurrent replay.

**Second review (read-only, commit efd6d91), and responses.**
Blocking: families were selected by arm names, so KeyCorridor's
confirmation cohort contributed both `rules_weak` and `rules_mem_weak`
contrasts and a family could issue verdicts with DoorKey absent; addendum
16 families now count only each task's confirmed arm (`family_members`,
`CONFIRMED_ARM`), and a family with a missing, extra or repeated
task-student cell is PROVISIONAL (regression tests). Older families keep
counting every named contrast. Non-blocking, adopted: explained variance
masked; the mixed feedback described as an interruption stress test; A
called a full-state RLingua port; the controller-writing claim limited to
these generated candidates; the effect of excluding autoreset ticks left
untested rather than asserted; the count-bonus prediction marked as not
an equivalence test; the KeyCorridor view controller's defect reported to
GPT-5-mini in two feedback rounds (no improvement, original kept).


## Addendum 17 (2026-10-06): RLingua B with matched design-time information

Written before any run of the suites below, after noting that B matched
our rules' information during training
but not while the teacher was written: our rule writer saw 36 full-map
example situations ("You see the full map; the student sees only its 7x7
view"), while RLingua's recipe shows its LLM only the task description and
test-episode feedback. Check B (research/rlingua_controllers_20261005/
checks_20261005/, 2026-10-05, a development check) removed that difference
at the controller level: the same RLingua recipe plus one message with 36
example situations, each with the full map and the agent's own 7x7 view,
from fresh layouts (seeds 40,000,000+). Those controllers, by themselves on
the 100 standard layouts (seeds 37,000,000+), alone and with a random action
at 25% of the steps: DoorKey 1.00 (1.00), MultiRoom 0.00 (0.00),
KeyCorridor 0.00 (0.02). This addendum trains with them.

**Design.** Suites `dk_rlex`, `mr_rlex`, `kc_rlex`: replicates 30-39, both
students, 5M steps, 60 free runs, no API call. Arm `rlingua_view_ex` equals,
setting for setting, the completed addendum-16 B cell (`rlingua_view`, suites
`*_rl`) of the same student and seed except the controller file (the Check B
controller, hashes in its receipt and in the archived sources) and the run
name (tested). Everything else is addendum 16's RLingua port unchanged.

**Families (Holm over six each; verdicts as in addendum 2).**
1. Primary: our confirmed arm (`rules_weak`; `rules_mem_weak` in
   KeyCorridor) minus `rlingua_view_ex`, paired by seed, teacher-free AUC.
2. Secondary: `rlingua_view_ex - rlingua_view` (did the full-map examples
   change what RLingua's student learned?).
Also reported, descriptive: `rlingua_view_ex - none`.

**Frozen predictions.** Primary: HELPS in MultiRoom for both students and in
KeyCorridor with the count bonus (as in B; these controllers still solve
0.00 alone); no direction for KeyCorridor plain PPO; negative means in
DoorKey for both students (the example-informed DoorKey controller solves
1.00 alone, as the full-state one does, and A's DoorKey means were -0.020
and -0.007); no significance predicted in DoorKey. Secondary: positive
means in DoorKey for both students (0.37 -> 1.00 alone); no direction in
MultiRoom and KeyCorridor.

**Decision.** Every run is reported. If the MultiRoom and KeyCorridor
predictions hold, the paper may say that with the same design-time
information as our rule writer and the same runtime information, RLingua's
complete controller still taught less than our rules in the named cells,
and that in DoorKey, where the example-informed view controller solves the
task, RLingua matched or exceeded our rules (as observed). Otherwise the
results are reported as they are.

**Review.** No independent review. Tests:
tests/test_fix_wave.py (cells equal the B cells except the controller;
controller hashes match their receipts and are archived; family wiring;
an end-to-end training run of `kc_rlex`).

**Operations.** Launched on Vulcan from the commit that adds this
addendum: `bash scripts/launch_fix_wave.sh <repo> dk_rlex mr_rlex kc_rlex`
(via `git show FETCH_HEAD:scripts/launch_fix_wave.sh | bash -s -- ...`).
