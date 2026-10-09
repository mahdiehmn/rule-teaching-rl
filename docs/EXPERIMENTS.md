# Experiments and their code

Each appendix section (`paper/appendix/sections/`) with the code and suites
that produce it. Suites of the main training study are defined in
`scripts/run_fix_wave_20260929.py` (`SUITES`) and documented, addendum by
addendum, in `research/fix_wave_protocol_2026-09-29.md`.

## Bank construction (Appendix: Construction and Implementation)

| Task | Generation and checks | Bank |
|---|---|---|
| DoorKey | `scripts/conditional_rules_v3.py` (blind check) | `research/rule_banks/v3_20260928/` |
| MultiRoom | `scripts/conditional_rules_multiroom.py` | `research/rule_banks/multiroom_20260928/` |
| KeyCorridor | `scripts/conditional_rules_keycorridor_mem.py` (pooled self-check, action preconditions) | `research/rule_banks/keycorridor_mem_20260929/` |
| Crafter | `scripts/crafter_rules_v3_20261001.py` (self-check; runtime preconditions) | `research/rule_banks/crafter_v3_20261001/` |

Each pipeline has the stages `build` (requests, no calls), `collect` (paid
GPT-5-mini requests, each sent once), the check exports, and `banks` (frozen
bank files, no calls). The executor is `teachers/minigrid/rule_bank.py`;
predicates are computed from the student's observation, plus the progress
facts `door_unlocked` (KeyCorridor) and `done_<achievement>` (Crafter).
The advice loss and schedule are in `algos/ppo_distill.py` (MiniGrid) and
`algos/ppo_crafter.py` (Crafter).

## Cohorts and evaluation; original confirmation and larger tasks

- Source confirmation (replicates 30-39): suites `dk_confirm`, `mr_confirm`,
  `kc_mem_confirm`; dense early evaluation: `dk_dense`, `mr_dense`, `kc_dense`.
- Larger maps with the source bank unchanged: `dk16`, `kc_s4` (5M);
  `dk16_long`, `mr10_long`, `kc_s4_long`, `kc_s5_long`, `kc_s6_long` (15M);
  plain PPO at the paper weight: `dk16_w`, `kc_s4_w`.
- Offline label checks on larger maps: `scripts/larger_map_labels_20261001.py`.

## Fresh replication and action content

- Fresh cohort (replicates 80-89): `dk_fresh`, `mr_fresh`, `kc_fresh`
  (core) and `dk_fresh_cc`, `mr_fresh_cc`, `kc_fresh_cc`, `mr_fresh_cross`
  (optional). Batch runner: `scripts/run_paper_bulk_20261005.py`.
- Content control (deranged action targets): `scripts/content_control_banks_20261004.py`;
  suites `dk_cc`, `mr_cc`, `kc_cc`.
- MultiRoom bank crossover: `mr_cross`.
- Early-learning curves: `scripts/plot_fresh_early_curves_20261005.py`.

## Generated-controller comparison (RLingua)

- Controller generation: `scripts/rlingua_controllers_20261005.py`; the
  controllers and receipts are in `research/rlingua_controllers_20261005/`.
- A, full simulator state: `dk_rlfull`, `mr_rlfull`, `kc_rlfull`.
  B, the student's view: `dk_rl`, `mr_rl`, `kc_rl` and the 15M larger-map
  suites `dk16_rl_long`, `mr10_rl_long`, `kc_s4_rl_long`.
  B with example situations: `dk_rlex`, `mr_rlex`, `kc_rlex`.
- Offline imitability: `scripts/rlingua_imitability_20261005.py`.

## Progress conditions and advice quantity

- Progress-only ablation bank: `scripts/progress_ablation_20261002.py`;
  suite `kc_prog` (and `kc_fresh` on the fresh cohort).
- Label-exposure follow-up: `scripts/run_rule_timing_20261005.py`.
- Advice-level counterfactual: `scripts/analyze_executable_teaching_20261002.py`.

## Online advice at a fixed request budget

- Request-matched online advice: `scripts/paper_optional_online_20261005.py`
  (run through `scripts/run_paper_optionals_20261005.py`).

## Additional Crafter results

- Confirmation: `scripts/run_crafter_v3b_confirm_20261001.py`.
- Progress ablation, Count-PPO student and replication:
  `scripts/run_crafter_vulcan_20261002.py`, `scripts/paper_bulk_crafter_20261005.py`.
- Fresh-world evaluation of trained policies: `scripts/paper_optional_crafter_eval_20261005.py`.

## Bank regeneration

- DoorKey, five repetitions: `scripts/paper_optional_regeneration_20261005.py`
  (`scripts/recover_paper_regeneration_20261005.py` for the recovered repetition).
- MultiRoom and KeyCorridor, five repetitions each:
  `scripts/regenerate_banks_20261008.py`, `scripts/run_regen_training_20261008.py`,
  `scripts/analyze_regen_banks_20261008.py`; banks in `research/rule_banks/regen_20261008/`.

## Earlier action-advising and execution comparisons

- Online GPT advisors at 120 requests: `scripts/run_phase1_lowcap120.py`,
  `scripts/analyze_phase1_lowcap120.py`; strategies in `advising/`.
- Teacher execution (RLingua-style mixing of our rules): `dk_act`, `mr_act`, `kc_act`.
- Teacher-view ablation (rules checked on the full map):
  `teachers/minigrid/teacher_view.py`, `scripts/teacher_view_20261001.py`;
  suites `dk_tv`, `mr_tv`, `kc_tv`.

## Fixed-policy baselines and recorded timing

- Banks and random actions as policies: `scripts/rules_alone_vs_student_20260930.py`,
  `scripts/bank_alone_mr10_20261004.py`.
- Cost, time and learning speed: `scripts/paper_cost_and_speed_20260930.py`.
