# Crafter wave on Vulcan (frozen before any run)

Date: 2026-10-02. Code:
scripts/run_crafter_vulcan_20261002.py (cells, commands, report),
scripts/launch_crafter_vulcan.sh and scripts/submit_crafter_vulcan.sh (launched
on Vulcan), algos/ppo_crafter.py (`count_coef`), scripts/
progress_ablation_20261002.py (the ablated bank). Tests:
tests/test_crafter_vulcan.py.

**Why.** Crafter has one confirmed result: the v3b bank (v2's rules re-checked
with the agent's progress facts) improves plain PPO on fresh seeds 21-30
(+0.446 [+0.152, +0.741], 8/10, p .0076; an independent audit
reproduces it). The MiniGrid study has two students and a progress-only
ablation; Crafter had neither. This wave adds both, plus a replication.

**Design.** Fresh seeds 31-40 (exploratory 1-10, v2 confirmation 11-20 and v3b
confirmation 21-30 used the others). Every setting is the v3b confirmation's
(symbolic Crafter, PPO, 1M steps, 200 seeded training worlds per seed, the
same 10 held-out evaluation worlds, evaluation every 50k steps, imitation
.1 -> .001); 50 runs, no LLM call. Arms:

| Arm | Student | Bank |
|---|---|---|
| `none` | plain | none |
| `rules_v3b_weak` | plain | v3b (unchanged) |
| `rules_noprog_weak` | plain | v3b minus its 3 `done_*` conditions, nothing else changed |
| `count_none` | count | none |
| `count_rules_v3b` | count | v3b |

**The count student.** An episodic count bonus `count_coef / sqrt(n)` over the
student's own 9x7 view (n its visits to that exact view this episode), as
MiniGrid's count student counts its own view. `count_coef` = .01, not tuned:
chosen so a fully novel episode's bonus (one per step) is of the order of an
episode's achievement reward. Crafter's trainer has one value head, so the
bonus is added to the training reward; MiniGrid's count student uses separate
intrinsic and extrinsic value heads with a normalized bonus. Evaluation and all
reported achievements use the game's own reward.

**Tests.** Metric: paired AUC of teacher-free mean achievements per episode.
Primary family (Holm over two): (1) `rules_v3b_weak - rules_noprog_weak`, the
progress clauses for plain PPO; (2) `count_rules_v3b - count_none`, the rules
for the count student. Verdicts as in the fix-wave protocol (HELPS: mean > 0,
p_holm < .05; NO HARM: lower 95% bound > -.05). Replication, its own test:
`rules_v3b_weak - none`, REPLICATED if mean > 0 and two-sided p < .05.
Secondary, descriptive: `count_none - none` (is the count student stronger?),
`rules_noprog_weak - none`, `count_rules_v3b - rules_v3b_weak`; final
achievements and Crafter score at 1M.

**Frozen predictions.** (1) HELPS. (2) positive mean; no significance
predicted. Replication: REPLICATED. No prediction for `count_none - none`.

**Decision.** Every run is reported; none is added, replaced or rerun except
for a dated infrastructure amendment written before any rerun, as for the
earlier Crafter cohorts. (1) HELPS lets the paper say the progress clauses
themselves improve Crafter learning; (2) HELPS lets it name the count student
in Crafter; a REPLICATED replication is reported beside the v3b confirmation,
not pooled with it.

**Operations.** On Vulcan, once: `pip install crafter==1.8.3` in the project
environment. Then `git fetch origin language-pilot-20260927 && git merge
--ff-only FETCH_HEAD` and `bash scripts/launch_crafter_vulcan.sh <repo>`.
Results land in `results/crafter_vulcan/crafter_vulcan_20261002/` and are
synced with `--batch crafter_vulcan/crafter_vulcan_20261002`.

## Result (2026-10-03, Vulcan job 1283831, all 50 runs complete, every exit code 0)

Report: `run_crafter_vulcan_20261002 report` (docs/assets/results_2026-10-03/
crafter_vulcan_report.json). Mean AUC / final achievements / Crafter score at
1M: none 5.540 / 6.48 / 5.52; rules_v3b_weak 6.157 / 7.05 / 7.91;
rules_noprog_weak 5.818 / 7.11 / 7.73; count_none 5.763 / 6.68 / 6.21;
count_rules_v3b 6.271 / 7.17 / 7.97.

**Primary family (Holm over two).** (1) `rules_v3b_weak - rules_noprog_weak`
**HELPS** +0.339 [+0.186, +0.491] 10/10, p_holm .0007: the progress clauses
improve Crafter learning. (2) `count_rules_v3b - count_none` **HELPS** +0.507
[+0.336, +0.678] 10/10, p_holm .0002: the rules help the count student.

**Replication.** `rules_v3b_weak - none` +0.617 [+0.327, +0.907] 9/10,
p .00096: **REPLICATED** (reported beside the v3b confirmation on seeds 21-30,
+0.446, not pooled).

**Secondary.** `count_none - none` +0.224 [-0.016, +0.464] 6/10, p .064 (the
count student is not significantly stronger than plain PPO in Crafter);
`rules_noprog_weak - none` +0.279 [-0.009, +0.566] 7/10, p .056;
`count_rules_v3b - rules_v3b_weak` +0.114 [-0.124, +0.352] 6/10, p .31.

**Frozen predictions.** (1) HELPS: met. (2) positive mean: met (and HELPS).
Replication: met.
