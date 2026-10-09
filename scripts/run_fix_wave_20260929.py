"""Main rule-teaching training study (study id fix_wave_20260929_v1).

Protocol:
research/fix_wave_protocol_2026-09-29.md (frozen before any run).
Structure follows scripts/run_potential_channel_20260927.py: the
committed source is archived, every cell is resolved and round-tripped
before submission, each Slurm array task runs its cell once, and reports
use only runs that pass the rule-bank study's strict terminal checker.
No API calls anywhere; every teacher here is a rule or a rule bank.

Every cell is built from the rule-bank study's own cells for the same
task, student and seed (scripts/run_rule_bank_study_20260928.py), so
each run pairs with that study's rule arms by seed, and the report
checks the initial policies agree before pairing anything.

Suites (independent arrays):
  kc       KeyCorridor-S3R3, plain + count, 5M, replicates 5-14 (+ rules)
  mr       MultiRoom-N6,     plain + count, 5M, replicates 5-9
  dk       DoorKey-8x8,      plain + count, 5M, replicates 5-9
  kc_long  KeyCorridor-S3R3, plain only,   15M, replicates 5-9

Arms (see ARMS_OF):
  none                 the rule-bank no-teacher cell, re-run for pairing
  shaping_dense        subgoal potential, a message at every episode start
                       and stage change (algos.ppo_potential)
  shaping_dense_wrong  every message names a wrong stage: content control
  shaping_480          480 messages over training (KeyCorridor only)
  rules_plus_shaping   the cell's rule bank imitated AND dense shaping
  rules_weak           the same rules at a tenth of the paper's imitation
                       weight (.1 -> .001, not 1 -> .01): is the harm its
                       strength?
  rules                (kc_long only) the rules at the long horizon

Addendum 3 (KeyCorridor bank with the door_unlocked progress fact; suites
kc_mem on the kc seeds and kc_mem_confirm on the fresh confirmation seeds):
  rules_mem            the KeyCorridor rule bank generated with the
                       door_unlocked progress fact and pooled self-checks
                       (scripts/conditional_rules_keycorridor_mem.py), at
                       the paper's imitation weight
  rules_mem_weak       the same bank at a tenth of that weight

Addendum 4 (suite dk16): the DoorKey-8x8 checked bank, unchanged, on
DoorKey-16x16 (arms none and rules_weak, replicates 40-49). Transfer tasks
(TRANSFER) take every setting from their source task's cells; only the map
changes. Addendum 5 (suite kc_s4): the KeyCorridor-S3R3 memory bank,
unchanged, on KeyCorridor-S4R3 (arms none and rules_mem_weak).

Addendum 6 (suites dk_tv, mr_tv, kc_tv; the confirmation seeds 30-39):
the teacher-view ablation. The confirmed arm's bank, the same rules byte
for byte, with conditions checked on the full map instead of the
student's view (teachers/minigrid/teacher_view.py,
scripts/teacher_view_20261001.py):
  rules_weak_tv        DoorKey and MultiRoom, against rules_weak
  rules_mem_weak_tv    KeyCorridor, against rules_mem_weak
Each pairs by seed with the completed confirmation cells of dk_confirm,
mr_confirm and kc_mem_confirm (report group 'confirm').

Addendum 7 (suites dk16_w, kc_s4_w; plain PPO only, replicates 40-49): the
weak student on the larger maps with the same unchanged banks at the
paper's imitation weight (1 -> .01): arms rules (DoorKey-16x16) and
rules_mem (KeyCorridor-S4R3), paired by seed with dk16 / kc_s4.

Addendum 8 (suites dk_act, mr_act, kc_act; the confirmation seeds 30-39):
RLingua-style teacher execution (Chen et al., RA-L 2024). The confirmed
arm with one change: where a rule labels a state, its action is EXECUTED
instead of the student's with probability .75, annealed exponentially to
1% of that by mid-training (algos/ppo_distill.py execute_teacher_*):
  rules_weak_act       DoorKey and MultiRoom, against rules_weak
  rules_mem_weak_act   KeyCorridor, against rules_mem_weak
Same rules, imitation loss, seeds and students: the teacher acting versus
the teacher only labelling. Report group 'confirm'.

Addendum 9 (suites dk16_long, mr10_long, kc_s4_long; replicates 50-59,
15M steps): the larger environments of all three families with three times
the training, both students, arms none and the unchanged source bank
(rules_weak; rules_mem_weak in KeyCorridor): DoorKey-16x16 (8x8 bank),
MultiRoom-N10 (N6 banks), KeyCorridor-S4R3 (S3R3 memory bank).

Addendum 10 (suite kc_prog; the confirmation seeds 30-39): the progress-only
training ablation. The KeyCorridor memory bank with only its
progress clauses removed (scripts/progress_ablation_20261002.py), arm
rules_mem_weak_noprog against kc_mem_confirm's rules_mem_weak by seed.

Addendum 11 (suites kc_s5_long, kc_s6_long; replicates 60-69 / 70-79, 15M
steps): the S3R3 memory bank, unchanged, on KeyCorridor-S5R3 and S6R3, both
students, arms none and rules_mem_weak.

Addendum 12 (suites dk_cc, mr_cc, kc_cc; the confirmation seeds 30-39):
the content control. The confirmed arm with one change: its bank's action
targets mapped by a fixed derangement (scripts/content_control_banks_
20261004.py), so the same states are labelled and abstained on, with the
wrong target:
  rules_weak_cc        DoorKey and MultiRoom, against rules_weak
  rules_mem_weak_cc    KeyCorridor, against rules_mem_weak
Report group 'confirm'.

Addendum 13 (suites dk_dense, mr_dense, kc_dense; the confirmation seeds
30-39): the six confirmed cells (none and the confirmed rule arm) re-run
unchanged except for extra student-only evaluations at DENSE_FRAMES,
including step 0. They are RNG-isolated and saved apart
(diagnostic_evaluations.jsonl); the regular evaluations and AUC inputs stay
as they were. Report group 'dense', plus a reproduction check against
'confirm' (same seeds). Addendum 12 and 14 cells carry the same extra
evaluations.

Addendum 14 (suite mr_cross; the confirmation seeds 30-39): MultiRoom's
bank crossover. Each student imitates the bank selected for the OTHER
student (plain PPO the 7-rule blind-strict bank, Count-PPO the 32-rule
scoped bank), everything else as its confirmed arm:
  rules_weak_xbank     against rules_weak and none. Report group 'confirm'.

Addendum 15 (FRESH replicates 80-89): one source cohort on seeds no study
has used, every arm with addendum 13's dense evaluations, report group
'fresh'. CORE, 140 runs (FRESH_CORE): dk_fresh and mr_fresh (none and the
confirmed arm) and kc_fresh (none, the memory arm and the memory arm
without progress clauses): fresh replication with measured early learning,
and a fresh confirmation of the progress effect. OPTIONAL, 80 runs
(FRESH_OPTIONAL), launchable separately on the same seeds: dk_fresh_cc,
mr_fresh_cc, kc_fresh_cc (deranged twins, addendum 12) and mr_fresh_cross
(MultiRoom crossover, addendum 14). Recommended instead of, not in addition
to, the historical-seed suites of addenda 12-14.

Addendum 16 (suites dk_rl, mr_rl, kc_rl and dk_rlfull, mr_rlfull, kc_rlfull
on replicates 30-39; dk16_rl_long, mr10_rl_long, kc_s4_rl_long on replicates
50-59 at 15M): the RLingua baseline (Chen et al., RA-L 2024, Algorithm A-1,
Table A-II), compared two ways. A, as published (suites *_rlfull,
confirmation seeds): a complete GPT-5-mini controller reading the full
simulator state at every step it acts. B, at matched runtime information
(suites *_rl and *_rl_long): a complete controller reading only what the
student sees (its view and the action actually executed, with a memory of
its own), as our rules do. Each RLingua cell is the no-teacher cell plus
only the RLingua settings. Report groups 'confirm' and the larger maps'
addendum-9 groups.

Addendum 17 (suites dk_rlex, mr_rlex, kc_rlex on replicates 30-39): B with
the student-view controllers GPT-5-mini wrote after seeing 36 full-map
example situations (research/rlingua_controllers_20261005/checks_20261005/
examples/), the kind of information our rule writer saw: matched design-
time and runtime information. Each cell is the matching *_rl cell with
only the controller file changed. Report group 'confirm'.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
from scipy.stats import t
import tyro

from algos import ppo_distill as ppo
from algos import ppo_potential as pot
from scripts.run_explanation_formats_20260925 import runtime_identity
from scripts.run_explanation_grid import trainer_argv
from scripts.run_plan_repair_20260927 import digest, inside, read, write

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'fix_wave_20260929_v1'
PROTOCOL = 'research/fix_wave_protocol_2026-09-29.md'
# Batches prepared before addendum 1 archived only these; each batch is
# checked against the list it was prepared with (manifest 'required').
REQUIRED_V1 = (
    'scripts/run_fix_wave_20260929.py', 'scripts/launch_fix_wave.sh',
    'scripts/submit_fix_wave.sh', 'tests/test_fix_wave.py', PROTOCOL,
    'algos/ppo_distill.py', 'algos/ppo_potential.py',
    'teachers/subgoal_potential.py', 'teachers/minigrid/rule_bank.py',
    'scripts/run_rule_bank_study_20260928.py',
)
REQUIRED = REQUIRED_V1 + (
    'scripts/llm_stage_plan_20260929.py',
    'research/stage_plans/llm_plans_20260929.json',
    'research/stage_plans/stage_plans_20260929.json',
    # addendum 3
    'scripts/conditional_rules_keycorridor_mem.py',
    'research/rule_banks/keycorridor_mem_20260929/'
    'self_checked_pooled_valid.json',
    # addendum 6
    'teachers/minigrid/teacher_view.py', 'scripts/teacher_view_20261001.py',
    'research/rule_banks/teacher_view_20261001/doorkey_blind_strict.json',
    'research/rule_banks/teacher_view_20261001/multiroom_scoped.json',
    'research/rule_banks/teacher_view_20261001/multiroom_blind_strict.json',
    'research/rule_banks/teacher_view_20261001/'
    'keycorridor_mem_pooled_valid.json',
    # addendum 10
    'scripts/progress_ablation_20261002.py',
    'research/rule_banks/progress_ablation_20261002/'
    'keycorridor_mem_noprogress.json',
    # addenda 12-14
    'scripts/content_control_banks_20261004.py',
    'research/rule_banks/content_control_20261004/'
    'doorkey_blind_strict_deranged.json',
    'research/rule_banks/content_control_20261004/'
    'multiroom_scoped_deranged.json',
    'research/rule_banks/content_control_20261004/'
    'multiroom_blind_strict_deranged.json',
    'research/rule_banks/content_control_20261004/'
    'keycorridor_mem_pooled_valid_deranged.json',
    'monitoring/evaluation_diagnostics.py',
    # addendum 16
    'teachers/minigrid/rlingua_controller.py',
    'scripts/rlingua_controllers_20261005.py',
    'research/rlingua_controllers_20261005/dk_full.py',
    'research/rlingua_controllers_20261005/dk_view.py',
    'research/rlingua_controllers_20261005/mr_full.py',
    'research/rlingua_controllers_20261005/mr_view.py',
    'research/rlingua_controllers_20261005/kc_full.py',
    'research/rlingua_controllers_20261005/kc_view.py',
    # addendum 17
    'research/rlingua_controllers_20261005/checks_20261005/examples/'
    'dk_view.py',
    'research/rlingua_controllers_20261005/checks_20261005/examples/'
    'mr_view.py',
    'research/rlingua_controllers_20261005/checks_20261005/examples/'
    'kc_view.py',
)
# Addendum 3: the KeyCorridor bank written with the unlocking memory, after
# v1's self-check with identical rules' checks pooled, under the same
# valid-action restriction as the KeyCorridor rules arm (chosen offline:
# the most maps solved as a policy; evaluation.json of the memory study).
MEM_BANK = ('research/rule_banks/keycorridor_mem_20260929/'
            'self_checked_pooled_valid.json')
# GPT-5-mini's subgoal plans (scripts/llm_stage_plan_20260929.py): the
# modal plan of 5 samples per task, frozen with a content digest.
PLAN_FILE = 'research/stage_plans/llm_plans_20260929.json'
HORIZON = 5_000_000
LONG = 15_000_000
BUDGET = 480
# (task, students, replicates, horizon) per suite
SUITES = {
    'kc': ('keycorridor_s3r3', ('none', 'count'), range(5, 15), HORIZON),
    'mr': ('multiroom_n6', ('none', 'count'), range(5, 10), HORIZON),
    'dk': ('doorkey_8x8', ('none', 'count'), range(5, 10), HORIZON),
    'kc_long': ('keycorridor_s3r3', ('none',), range(5, 10), LONG),
    # Addendum 1: the same cells, shaped along the LLM-written plans.
    'kc_llm': ('keycorridor_s3r3', ('none', 'count'), range(5, 15), HORIZON),
    'mr_llm': ('multiroom_n6', ('none', 'count'), range(5, 10), HORIZON),
    'dk_llm': ('doorkey_8x8', ('none', 'count'), range(5, 10), HORIZON),
    # Addendum 2: weaker imitation on FRESH replicates 30-39, which no
    # study has used (rule-bank 0-19, valid-action KeyCorridor 20-29).
    'kc_confirm': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                   HORIZON),
    'mr_confirm': ('multiroom_n6', ('none', 'count'), range(30, 40),
                   HORIZON),
    'dk_confirm': ('doorkey_8x8', ('none', 'count'), range(30, 40),
                   HORIZON),
    # Addendum 3: the memory rules on the kc seeds and on kc_confirm's.
    'kc_mem': ('keycorridor_s3r3', ('none', 'count'), range(5, 15), HORIZON),
    'kc_mem_confirm': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                       HORIZON),
    # Addendum 4: the DoorKey-8x8 bank, unchanged, on DoorKey-16x16;
    # replicates 40-49, which no suite has used.
    'dk16': ('doorkey_16x16', ('none', 'count'), range(40, 50), HORIZON),
    # Addendum 5: the KeyCorridor-S3R3 memory bank, unchanged, on S4R3.
    'kc_s4': ('keycorridor_s4r3', ('none', 'count'), range(40, 50), HORIZON),
    # Addendum 6: teacher-view ablation on the confirmation seeds, paired
    # with the completed student-view cells there.
    'dk_tv': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_tv': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    'kc_tv': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
              HORIZON),
    # Addendum 7: the weak student on the larger maps at the paper's
    # imitation weight, paired with dk16 / kc_s4 by seed.
    'dk16_w': ('doorkey_16x16', ('none',), range(40, 50), HORIZON),
    'kc_s4_w': ('keycorridor_s4r3', ('none',), range(40, 50), HORIZON),
    # Addendum 8: RLingua-style execution on the confirmation seeds.
    'dk_act': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_act': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    'kc_act': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
               HORIZON),
    # Addendum 9: the larger environments, unchanged banks, 15M steps,
    # fresh replicates 50-59.
    'dk16_long': ('doorkey_16x16', ('none', 'count'), range(50, 60), LONG),
    'mr10_long': ('multiroom_n10', ('none', 'count'), range(50, 60), LONG),
    'kc_s4_long': ('keycorridor_s4r3', ('none', 'count'), range(50, 60),
                   LONG),
    # Addendum 10: progress-only ablation on the confirmation seeds.
    'kc_prog': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                HORIZON),
    # Addendum 11: two more KeyCorridor sizes, unchanged memory bank.
    'kc_s5_long': ('keycorridor_s5r3', ('none', 'count'), range(60, 70),
                   LONG),
    'kc_s6_long': ('keycorridor_s6r3_babyai', ('none', 'count'),
                   range(70, 80), LONG),
    # Addendum 12: content control on the confirmation seeds.
    'dk_cc': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_cc': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    'kc_cc': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
              HORIZON),
    # Addendum 13: the confirmed cells re-run with dense early evaluation.
    'dk_dense': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_dense': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    'kc_dense': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                 HORIZON),
    # Addendum 14: MultiRoom bank crossover on the confirmation seeds.
    'mr_cross': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    # Addendum 15: a fresh-seed source cohort (replicates 80-89, unused by
    # every fix-wave suite and the rule-bank study), every arm with the
    # dense early evaluations. CORE (140 runs): dk_fresh, mr_fresh,
    # kc_fresh. OPTIONAL, separately launchable on the same seeds (80
    # runs): the content controls *_fresh_cc and mr_fresh_cross.
    'dk_fresh': ('doorkey_8x8', ('none', 'count'), range(80, 90), HORIZON),
    'mr_fresh': ('multiroom_n6', ('none', 'count'), range(80, 90), HORIZON),
    'kc_fresh': ('keycorridor_s3r3', ('none', 'count'), range(80, 90),
                 HORIZON),
    'dk_fresh_cc': ('doorkey_8x8', ('none', 'count'), range(80, 90),
                    HORIZON),
    'mr_fresh_cc': ('multiroom_n6', ('none', 'count'), range(80, 90),
                    HORIZON),
    'kc_fresh_cc': ('keycorridor_s3r3', ('none', 'count'), range(80, 90),
                    HORIZON),
    'mr_fresh_cross': ('multiroom_n6', ('none', 'count'), range(80, 90),
                       HORIZON),
    # Addendum 16: the RLingua baseline on the confirmation seeds (pairs
    # with none, the confirmed arm and the execution heuristic there) and,
    # student-view controller only, on addendum 9's larger maps (pairs with
    # none and the unchanged bank; the full-state KeyCorridor controller
    # takes ~0.4 s per call on S4R3, about 26 extra hours per run).
    'dk_rl': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_rl': ('multiroom_n6', ('none', 'count'), range(30, 40), HORIZON),
    'kc_rl': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
              HORIZON),
    'dk_rlfull': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_rlfull': ('multiroom_n6', ('none', 'count'), range(30, 40),
                  HORIZON),
    'kc_rlfull': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                  HORIZON),
    'dk16_rl_long': ('doorkey_16x16', ('none', 'count'), range(50, 60),
                     LONG),
    'mr10_rl_long': ('multiroom_n10', ('none', 'count'), range(50, 60),
                     LONG),
    'kc_s4_rl_long': ('keycorridor_s4r3', ('none', 'count'), range(50, 60),
                      LONG),
    # addendum 17: B with the example-informed student-view controllers
    'dk_rlex': ('doorkey_8x8', ('none', 'count'), range(30, 40), HORIZON),
    'mr_rlex': ('multiroom_n6', ('none', 'count'), range(30, 40),
                HORIZON),
    'kc_rlex': ('keycorridor_s3r3', ('none', 'count'), range(30, 40),
                HORIZON),
}
# Addendum 15's packages (the launch commands name these explicitly).
FRESH_CORE = ('dk_fresh', 'mr_fresh', 'kc_fresh')
FRESH_OPTIONAL = ('dk_fresh_cc', 'mr_fresh_cc', 'kc_fresh_cc',
                  'mr_fresh_cross')
TIME = {'kc': '1-00:00:00', 'mr': '1-00:00:00', 'dk': '1-00:00:00',
        'kc_long': '3-00:00:00', 'kc_llm': '1-00:00:00',
        'mr_llm': '1-00:00:00', 'dk_llm': '1-00:00:00',
        'kc_confirm': '1-00:00:00', 'mr_confirm': '1-00:00:00',
        'dk_confirm': '1-00:00:00', 'kc_mem': '1-00:00:00',
        'kc_mem_confirm': '1-00:00:00',
        # longer episodes (2560-step limit) make evaluation slower
        'dk16': '1-12:00:00', 'kc_s4': '1-12:00:00',
        'dk_tv': '1-00:00:00', 'mr_tv': '1-00:00:00', 'kc_tv': '1-00:00:00',
        'dk16_w': '1-12:00:00', 'kc_s4_w': '1-12:00:00',
        'dk_act': '1-00:00:00', 'mr_act': '1-00:00:00',
        'kc_act': '1-00:00:00',
        # 15M steps; the 16x16 / S4R3 runs took ~13.5 h per 5M
        'dk16_long': '3-00:00:00', 'mr10_long': '3-00:00:00',
        'kc_s4_long': '3-00:00:00',
        'kc_prog': '1-00:00:00',
        'kc_s5_long': '3-00:00:00', 'kc_s6_long': '3-00:00:00',
        # nine extra evaluations each: allow for the slower start
        'dk_cc': '1-06:00:00', 'mr_cc': '1-06:00:00', 'kc_cc': '1-06:00:00',
        'dk_dense': '1-06:00:00', 'mr_dense': '1-06:00:00',
        'kc_dense': '1-06:00:00', 'mr_cross': '1-06:00:00',
        'dk_fresh': '1-06:00:00', 'mr_fresh': '1-06:00:00',
        'kc_fresh': '1-06:00:00', 'dk_fresh_cc': '1-06:00:00',
        'mr_fresh_cc': '1-06:00:00', 'kc_fresh_cc': '1-06:00:00',
        'mr_fresh_cross': '1-06:00:00',
        # a view controller runs every step: allow extra time
        'dk_rl': '1-12:00:00', 'mr_rl': '1-12:00:00', 'kc_rl': '1-12:00:00',
        # the full-state KeyCorridor controller takes ~45 ms per call
        'dk_rlfull': '1-12:00:00', 'mr_rlfull': '1-12:00:00',
        'kc_rlfull': '1-18:00:00',
        'dk16_rl_long': '3-00:00:00', 'mr10_rl_long': '3-00:00:00',
        'kc_s4_rl_long': '3-00:00:00',
        'dk_rlex': '1-12:00:00', 'mr_rlex': '1-12:00:00',
        'kc_rlex': '1-12:00:00'}
# Suites whose cells share task, horizon and seeds are reported together,
# so an addendum arm pairs with the original suite's none by seed.
GROUP = {'kc': 'kc', 'kc_llm': 'kc', 'mr': 'mr', 'mr_llm': 'mr',
         'dk': 'dk', 'dk_llm': 'dk', 'kc_long': 'kc_long',
         # fresh seeds: reported apart, one Holm family over six cells
         'kc_confirm': 'confirm', 'mr_confirm': 'confirm',
         'dk_confirm': 'confirm',
         'kc_mem': 'kc', 'kc_mem_confirm': 'confirm',
         'dk16': 'dk16', 'kc_s4': 'kc_s4',
         'dk_tv': 'confirm', 'mr_tv': 'confirm', 'kc_tv': 'confirm',
         'dk16_w': 'dk16', 'kc_s4_w': 'kc_s4',
         'dk_act': 'confirm', 'mr_act': 'confirm', 'kc_act': 'confirm',
         'dk16_long': 'dk16_long', 'mr10_long': 'mr10_long',
         'kc_s4_long': 'kc_s4_long',
         'kc_prog': 'confirm',
         'kc_s5_long': 'kc_s5_long', 'kc_s6_long': 'kc_s6_long',
         'dk_cc': 'confirm', 'mr_cc': 'confirm', 'kc_cc': 'confirm',
         'mr_cross': 'confirm',
         # a re-run of the confirmation cells: its own group, so its pairs
         # never mix with the original runs of the same seeds
         'dk_dense': 'dense', 'mr_dense': 'dense', 'kc_dense': 'dense',
         # addendum 15: fresh seeds, their own group
         'dk_fresh': 'fresh', 'mr_fresh': 'fresh', 'kc_fresh': 'fresh',
         'dk_fresh_cc': 'fresh', 'mr_fresh_cc': 'fresh',
         'kc_fresh_cc': 'fresh', 'mr_fresh_cross': 'fresh',
         'dk_rl': 'confirm', 'mr_rl': 'confirm', 'kc_rl': 'confirm',
         'dk_rlfull': 'confirm', 'mr_rlfull': 'confirm',
         'kc_rlfull': 'confirm',
         'dk16_rl_long': 'dk16_long', 'mr10_rl_long': 'mr10_long',
         'kc_s4_rl_long': 'kc_s4_long',
         'dk_rlex': 'confirm', 'mr_rlex': 'confirm', 'kc_rlex': 'confirm'}
ARMS_OF = {
    # KeyCorridor adds its own rules arm: the valid-action rules ran only
    # on replicates 20-29, so the rule fixes need a same-seed reference.
    'kc': ('none', 'shaping_dense', 'shaping_dense_wrong', 'shaping_480',
           'rules', 'rules_plus_shaping', 'rules_weak'),
    'mr': ('none', 'shaping_dense', 'shaping_dense_wrong',
           'rules_plus_shaping', 'rules_weak'),
    'dk': ('none', 'shaping_dense', 'shaping_dense_wrong',
           'rules_plus_shaping', 'rules_weak'),
    'kc_long': ('none', 'shaping_dense', 'rules'),
    'kc_llm': ('shaping_llm_plan',),
    'mr_llm': ('shaping_llm_plan',),
    'dk_llm': ('shaping_llm_plan',),
    'kc_confirm': ('none', 'rules', 'rules_weak'),
    'mr_confirm': ('none', 'rules', 'rules_weak'),
    'dk_confirm': ('none', 'rules', 'rules_weak'),
    'kc_mem': ('rules_mem', 'rules_mem_weak'),
    'kc_mem_confirm': ('rules_mem', 'rules_mem_weak'),
    'dk16': ('none', 'rules_weak'),
    'kc_s4': ('none', 'rules_mem_weak'),
    'dk_tv': ('rules_weak_tv',),
    'mr_tv': ('rules_weak_tv',),
    'kc_tv': ('rules_mem_weak_tv',),
    'dk16_w': ('rules',),
    'kc_s4_w': ('rules_mem',),
    'dk_act': ('rules_weak_act',),
    'mr_act': ('rules_weak_act',),
    'kc_act': ('rules_mem_weak_act',),
    'dk16_long': ('none', 'rules_weak'),
    'mr10_long': ('none', 'rules_weak'),
    'kc_s4_long': ('none', 'rules_mem_weak'),
    'kc_prog': ('rules_mem_weak_noprog',),
    'kc_s5_long': ('none', 'rules_mem_weak'),
    'kc_s6_long': ('none', 'rules_mem_weak'),
    'dk_cc': ('rules_weak_cc',),
    'mr_cc': ('rules_weak_cc',),
    'kc_cc': ('rules_mem_weak_cc',),
    'dk_dense': ('none', 'rules_weak'),
    'mr_dense': ('none', 'rules_weak'),
    'kc_dense': ('none', 'rules_mem_weak'),
    'mr_cross': ('rules_weak_xbank',),
    'dk_fresh': ('none', 'rules_weak'),
    'mr_fresh': ('none', 'rules_weak'),
    'kc_fresh': ('none', 'rules_mem_weak', 'rules_mem_weak_noprog'),
    'dk_fresh_cc': ('rules_weak_cc',),
    'mr_fresh_cc': ('rules_weak_cc',),
    'kc_fresh_cc': ('rules_mem_weak_cc',),
    'mr_fresh_cross': ('rules_weak_xbank',),
    'dk_rl': ('rlingua_view',),
    'mr_rl': ('rlingua_view',),
    'kc_rl': ('rlingua_view',),
    'dk_rlfull': ('rlingua_full',),
    'mr_rlfull': ('rlingua_full',),
    'kc_rlfull': ('rlingua_full',),
    'dk16_rl_long': ('rlingua_view',),
    'mr10_rl_long': ('rlingua_view',),
    'kc_s4_rl_long': ('rlingua_view',),
    'dk_rlex': ('rlingua_view_ex',),
    'mr_rlex': ('rlingua_view_ex',),
    'kc_rlex': ('rlingua_view_ex',),
}
# The rule-bank study spec whose seeds and bank each task pairs with.
BASE_SPEC = {'doorkey_8x8': 'confirm_doorkey',
             'multiroom_n6': 'confirm_multiroom',
             'keycorridor_s3r3': 'confirm_keycorridor',
             'doorkey_16x16': 'confirm_doorkey',
             'keycorridor_s4r3': 'confirm_keycorridor',
             'multiroom_n10': 'confirm_multiroom',
             'keycorridor_s5r3': 'confirm_keycorridor',
             'keycorridor_s6r3_babyai': 'confirm_keycorridor'}
# Transfer tasks: every setting, the bank included, is the source task's;
# only the task itself (the map) changes.
TRANSFER = {'doorkey_16x16': 'doorkey_8x8',
            'keycorridor_s4r3': 'keycorridor_s3r3',
            'multiroom_n10': 'multiroom_n6',
            'keycorridor_s5r3': 'keycorridor_s3r3',
            'keycorridor_s6r3_babyai': 'keycorridor_s3r3'}
# The rule set each fix arm imitates: the development-selected set of
# report_rule_bank_paper_20260928 --selected, except KeyCorridor, where
# that selection turned rules off; there the fixes start from the best
# KeyCorridor rules, the valid-action restriction.
RULE_ARM = {('doorkey_8x8', 'none'): 'llm_rules_blind_strict',
            ('doorkey_8x8', 'count'): 'llm_rules_blind_strict',
            ('multiroom_n6', 'none'): 'llm_rules_scoped',
            ('multiroom_n6', 'count'): 'llm_rules_blind_strict',
            ('keycorridor_s3r3', 'none'): 'llm_rules_scoped_valid',
            ('keycorridor_s3r3', 'count'): 'llm_rules_scoped_valid',
            ('doorkey_16x16', 'none'): 'llm_rules_blind_strict',
            ('doorkey_16x16', 'count'): 'llm_rules_blind_strict',
            ('keycorridor_s4r3', 'none'): 'llm_rules_scoped_valid',
            ('keycorridor_s4r3', 'count'): 'llm_rules_scoped_valid',
            ('multiroom_n10', 'none'): 'llm_rules_scoped',
            ('multiroom_n10', 'count'): 'llm_rules_blind_strict',
            ('keycorridor_s5r3', 'none'): 'llm_rules_scoped_valid',
            ('keycorridor_s5r3', 'count'): 'llm_rules_scoped_valid',
            ('keycorridor_s6r3_babyai', 'none'): 'llm_rules_scoped_valid',
            ('keycorridor_s6r3_babyai', 'count'): 'llm_rules_scoped_valid'}
POTENTIAL_KEYS = {'potential_scale', 'potential_queries',
                  'potential_wrong_rate', 'potential_truncation',
                  'potential_trace_episodes', 'potential_with_guidance',
                  'potential_plan', 'potential_plan_sha256'}
WEIGHT_KEYS = {'distill_coef_start', 'distill_coef_min'}
BANK_KEYS = {'rule_bank', 'rule_bank_sha256'}
MEM_ARMS = ('rules_mem', 'rules_mem_weak')
# Addendum 6: each teacher-view arm and the student-view arm it copies.
TV_ARMS = {'rules_weak_tv': 'rules_weak',
           'rules_mem_weak_tv': 'rules_mem_weak'}
# Addendum 8: each RLingua-style arm and the label-only arm it copies.
ACT_ARMS = {'rules_weak_act': 'rules_weak',
            'rules_mem_weak_act': 'rules_mem_weak'}
EXEC_KEYS = {'execute_teacher_start', 'execute_teacher_fraction'}
# Addendum 10: the memory bank with only its progress clauses removed.
NOPROG_BANK = ('research/rule_banks/progress_ablation_20261002/'
               'keycorridor_mem_noprogress.json')
EXECUTE_START = .75
# Addendum 12: each confirmed bank and its deranged twin (same conditions,
# every action target mapped by scripts/content_control_banks_20261004.py).
_CC = 'research/rule_banks/content_control_20261004/'
CC_BANKS = {
    'research/rule_banks/v3_20260928/blind_strict.json':
        _CC + 'doorkey_blind_strict_deranged.json',
    'research/rule_banks/multiroom_20260928/scoped.json':
        _CC + 'multiroom_scoped_deranged.json',
    'research/rule_banks/multiroom_20260928/blind_strict.json':
        _CC + 'multiroom_blind_strict_deranged.json',
    MEM_BANK: _CC + 'keycorridor_mem_pooled_valid_deranged.json',
}
CC_ARMS = {'rules_weak_cc': 'rules_weak',
           'rules_mem_weak_cc': 'rules_mem_weak'}
# Addendum 14: MultiRoom's bank crossover.
XBANK_ARM = 'rules_weak_xbank'
# Addendum 13: extra student-only evaluations (transitions; rounded up to
# whole 1,024-transition rollouts by the trainer), step 0 included. They
# fill the first regular evaluation's gap (204,800) and are saved apart
# from the regular evaluations, which, with every AUC, stay unchanged.
DENSE_FRAMES = '0,10240,25600,51200,76800,102400,128000,153600,179200'
# Addendum 16: RLingua baseline (Chen et al., RA-L 2024). Each arm is the
# no-teacher cell plus only the RLingua settings: a complete controller
# written by GPT-5-mini with RLingua's prompts (scripts/rlingua_
# controllers_20261005.py), executed with p = .25 * .999999 ** transitions,
# behavior cloning on its persistent buffer at constant weight 1.0, and RL
# terms on the student's own steps (Table A-II, RLBench row).
RLINGUA_DIR = 'research/rlingua_controllers_20261005/'
RLINGUA_ARMS = {'rlingua_full': 'full', 'rlingua_view': 'view',
                'rlingua_view_ex': 'view'}
# Addendum 17: the view controllers written after 36 full-map examples.
RLINGUA_ARM_DIR = {
    'rlingua_view_ex': RLINGUA_DIR + 'checks_20261005/examples/'}
RLINGUA_KEYS = {'rlingua_controller', 'rlingua_controller_sha256',
                'rlingua_variant', 'rlingua_p0', 'rlingua_decay',
                'rlingua_bc_coef', 'rlingua_buffer'}
RLINGUA_FAMILY_TASK = {'doorkey_8x8': 'dk', 'multiroom_n6': 'mr',
                       'keycorridor_s3r3': 'kc'}
DENSE_SUITES = ('dk_dense', 'mr_dense', 'kc_dense', 'dk_cc', 'mr_cc',
                'kc_cc', 'mr_cross', 'dk_fresh', 'mr_fresh', 'kc_fresh',
                'dk_fresh_cc', 'mr_fresh_cc', 'kc_fresh_cc',
                'mr_fresh_cross')


def rule_bank_spec(task, suite):
    from scripts.run_rule_bank_study_20260928 import SPECS
    base = SPECS[BASE_SPEC[task]]
    return dict(study=f'{STUDY}_{suite}', task=task, seed0=base['seed0'],
                banks=base['banks'], bonuses=SUITES[suite][1],
                summary_only=True)


def shaping(args, experiment_id, queries=-1, wrong=0.0, guided=False):
    values = asdict(args)
    values.update(experiment_id=experiment_id, potential_scale=1.0,
                  potential_queries=queries, potential_wrong_rate=wrong,
                  potential_truncation='zero', potential_trace_episodes=0,
                  potential_with_guidance=guided)
    return pot.Args(**values)


def arm_args(suite, arm, r, bonus, root=ROOT):
    """One cell's settings; addendum 13's suites add dense evaluations."""
    args = _arm_args(suite, arm, r, bonus, root)
    if suite in DENSE_SUITES:
        args = replace(args, diagnostic_eval_frames=DENSE_FRAMES)
    return args


def _arm_args(suite, arm, r, bonus, root=ROOT):
    """One cell's settings, built from the rule-bank study's own cells."""
    from scripts import run_rule_bank_study_20260928 as rb
    task, _students, _reps, horizon = SUITES[suite]
    spec = rule_bank_spec(task, suite)
    none, guided = rb.base_args(TRANSFER.get(task, task), bonus)
    if task in TRANSFER:
        none, guided = replace(none, task=task), replace(guided, task=task)
    name = f'{STUDY}_{suite}_{bonus}_{arm}'
    base = replace(rb.arm_args(spec, 'none', r, bonus, none, guided, root,
                               ''), total_timesteps=horizon,
                   experiment_id=name)
    rule = replace(rb.arm_args(spec, RULE_ARM[(task, bonus)], r, bonus,
                               none, guided, root, ''),
                   total_timesteps=horizon, experiment_id=name)
    if arm == 'rules_mem_weak_noprog':
        # The memory arm's exact settings; only the bank file
        # differs: the same rules without their progress clauses.
        from teachers.minigrid.rule_bank import file_sha256
        return replace(arm_args(suite, 'rules_mem_weak', r, bonus, root),
                       experiment_id=name, rule_bank=NOPROG_BANK,
                       rule_bank_sha256=file_sha256(Path(root) / NOPROG_BANK))
    if arm in ACT_ARMS:
        # The label-only arm's exact settings; the teacher's labelled
        # action is now executed with an annealed probability.
        return replace(arm_args(suite, ACT_ARMS[arm], r, bonus, root),
                       experiment_id=name,
                       execute_teacher_start=EXECUTE_START)
    if arm in RLINGUA_ARMS:
        # The no-teacher cell plus only the RLingua settings; the same
        # controller serves a family's larger maps (as the bank does).
        family = RLINGUA_FAMILY_TASK[TRANSFER.get(task, task)]
        variant = RLINGUA_ARMS[arm]
        directory = RLINGUA_ARM_DIR.get(arm, RLINGUA_DIR)
        path = f'{directory}{family}_{variant}.py'
        data = (Path(root) / path).read_bytes()
        return replace(
            base, rlingua_controller=path,
            rlingua_controller_sha256=hashlib.sha256(
                data.replace(b'\r\n', b'\n')).hexdigest(),
            rlingua_variant=variant, rlingua_p0=0.25, rlingua_decay=0.999999,
            rlingua_bc_coef=1.0, rlingua_buffer=1_000_000)
    if arm in CC_ARMS:
        # The confirmed arm's exact settings; only the bank file differs:
        # the same rules and scopes, every action target deranged.
        from teachers.minigrid.rule_bank import file_sha256
        twin = arm_args(suite, CC_ARMS[arm], r, bonus, root)
        bank = CC_BANKS[twin.rule_bank]
        return replace(twin, experiment_id=name, rule_bank=bank,
                       rule_bank_sha256=file_sha256(Path(root) / bank))
    if arm == XBANK_ARM:
        # The confirmed arm's exact settings with the OTHER student's
        # selected MultiRoom bank.
        if task != 'multiroom_n6':
            raise ValueError('The bank crossover is a MultiRoom arm')
        twin = arm_args(suite, 'rules_weak', r, bonus, root)
        other = arm_args(suite, 'rules_weak', r,
                         'count' if bonus == 'none' else 'none', root)
        return replace(twin, experiment_id=name, rule_bank=other.rule_bank,
                       rule_bank_sha256=other.rule_bank_sha256)
    if arm in TV_ARMS:
        # The student-view arm's exact settings; only the bank file
        # differs: the same rules, their conditions read from the full map.
        from scripts.teacher_view_20261001 import BANKS as TV_BANKS
        from teachers.minigrid.rule_bank import file_sha256
        student_view = arm_args(suite, TV_ARMS[arm], r, bonus, root)
        bank = TV_BANKS[student_view.rule_bank]
        return replace(student_view, experiment_id=name, rule_bank=bank,
                       rule_bank_sha256=file_sha256(Path(root) / bank))
    if arm == 'none':
        return base
    if arm == 'shaping_dense':
        return shaping(base, name)
    if arm == 'shaping_dense_wrong':
        return shaping(base, name, wrong=1.0)
    if arm == 'shaping_480':
        return shaping(base, name, queries=BUDGET)
    if arm == 'shaping_llm_plan':
        from teachers.subgoal_potential import plan_sha256
        entry = json.loads((Path(root) / PLAN_FILE).read_text(
            encoding='utf-8'))['tasks'][task]
        if plan_sha256(entry['plan']) != entry['sha256']:
            raise ValueError('Frozen LLM plan differs from its digest')
        values = asdict(shaping(base, name))
        values.update(potential_plan=PLAN_FILE,
                      potential_plan_sha256=entry['sha256'])
        return pot.Args(**values)
    if arm == 'rules_plus_shaping':
        return shaping(rule, name, guided=True)
    if arm in MEM_ARMS:
        if TRANSFER.get(task, task) != 'keycorridor_s3r3':
            raise ValueError('The memory bank is a KeyCorridor bank')
        from teachers.minigrid.rule_bank import file_sha256
        rule = replace(rule, rule_bank=MEM_BANK, rule_bank_sha256=file_sha256(
            Path(root) / MEM_BANK))
    if arm in ('rules_weak', 'rules_mem_weak'):
        # The rule-bank study imitates at 1 -> .01 (not the trainer's
        # 10 -> .1 default); a tenth of that isolates imitation strength.
        return replace(rule,
                       distill_coef_start=rule.distill_coef_start / 10,
                       distill_coef_min=rule.distill_coef_min / 10)
    if arm in ('rules', 'rules_mem'):
        return rule
    raise ValueError(f'Unknown arm {arm}')


def reference(suite, arm, r, bonus, root=ROOT):
    """The cell each arm must equal outside its own teaching channel."""
    return arm_args(suite, 'rules' if arm in (
        'rules_plus_shaping', 'rules_weak', 'rules', *MEM_ARMS,
        *TV_ARMS, *ACT_ARMS, *CC_ARMS, XBANK_ARM,
        'rules_mem_weak_noprog') else 'none', r, bonus, root)


def allowed(arm):
    if arm in ('none', 'rules'):
        return {'experiment_id'}
    if arm in RLINGUA_ARMS:
        return {'experiment_id'} | RLINGUA_KEYS
    if arm.startswith('shaping'):
        return {'experiment_id'} | POTENTIAL_KEYS
    if arm == 'rules_plus_shaping':
        return {'experiment_id'} | POTENTIAL_KEYS
    if arm == 'rules_mem':
        return {'experiment_id'} | BANK_KEYS
    if arm in ('rules_mem_weak', 'rules_mem_weak_noprog', XBANK_ARM) or \
            arm in TV_ARMS or arm in CC_ARMS:
        return {'experiment_id'} | BANK_KEYS | WEIGHT_KEYS
    if arm in ACT_ARMS:
        return {'experiment_id'} | BANK_KEYS | WEIGHT_KEYS | EXEC_KEYS
    return {'experiment_id'} | WEIGHT_KEYS           # rules_weak


def cell(cells, suite, arm, args, r):
    trainer = ('algos.ppo_potential' if isinstance(args, pot.Args)
               else 'algos.ppo_distill')
    cls = pot.Args if isinstance(args, pot.Args) else ppo.Args
    if asdict(tyro.cli(cls, args=trainer_argv(asdict(args)),
                       console_outputs=False)) != asdict(args):
        raise ValueError('Trainer command changed resolved settings')
    if trainer == 'algos.ppo_potential':
        pot.validate_args(args)
    cells.append(dict(index=len(cells), suite=suite, arm=arm,
                      replicate=r, seed=args.seed, task=args.task,
                      bonus=args.bonus, trainer=trainer, args=asdict(args)))


def suite_cells(suite, root=ROOT):
    """Every arm per student and replicate, seed-major for early pairs."""
    _task, students, reps, _horizon = SUITES[suite]
    cells = []
    for r in reps:
        for bonus in students:
            for arm in ARMS_OF[suite]:
                args = arm_args(suite, arm, r, bonus, root)
                ref = asdict(reference(suite, arm, r, bonus, root))
                mine = asdict(args)
                extra = set(mine) - set(ref)
                changed = {k for k in ref if mine[k] != ref[k]}
                if (extra | changed) - allowed(arm):
                    raise ValueError(
                        f'{suite}/{bonus}/{arm}/r{r} differs from its '
                        f'reference beyond its channel: '
                        f'{sorted((extra | changed) - allowed(arm))}')
                cell(cells, suite, arm, args, r)
    return cells


def batch_dir(root, suite):
    return Path(root) / 'results/fix_wave' / STUDY / suite


def prepare(root, suite):
    root = Path(root).resolve()
    batch = batch_dir(root, suite)
    if batch.exists():
        # Checked, not rebuilt: a later commit may add settings, and the
        # batch runs its own archived source anyway.
        verify(batch, rebuild=False)
        return batch
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete fix-wave packet first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    manifest = dict(
        study=STUDY, suite=suite, commit=commit, runtime=runtime_identity(),
        cells=suite_cells(suite, batch / 'code'), protocol=PROTOCOL,
        api_calls=0, required=list(REQUIRED),
        source_hashes={p.relative_to(batch / 'code').as_posix(): digest(p)
                       for p in (batch / 'code').rglob('*') if p.is_file()})
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    return batch


def verify(batch, rebuild=True):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    if (digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or manifest['study'] != STUDY
            or (batch / 'READY').read_text().strip() != manifest['commit']
            or not set(manifest.get('required', REQUIRED_V1)) <= set(
                manifest['source_hashes'])):
        raise ValueError('Frozen preparation identity differs')
    if rebuild and manifest['cells'] != suite_cells(manifest['suite'],
                                                    batch / 'code'):
        raise ValueError('Frozen cells differ from the archived builder')
    for name, expected in manifest['source_hashes'].items():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    return manifest


def args_of(record):
    cls = pot.Args if record['trainer'] == 'algos.ppo_potential' \
        else ppo.Args
    return cls(**record['args'])


def validate(run, record):
    """The rule-bank study's strict checker, plus the shaping evidence."""
    from scripts.run_rule_bank_pilot_20260928 import read_jsonl, validate_run
    from scripts.run_explanation_formats_20260925 import derived_of
    metrics = validate_run(run, record['args'])
    # The whole regular-evaluation curve (addendum 13's reproduction check
    # compares curves, not just their area; validate_run has already
    # checked the steps and the teacher-off contract of these rows).
    metrics['curve'] = [
        [r['global_step'], r['success_rate'], r.get('sampled_success_rate')]
        for r in read_jsonl(Path(run) / 'evaluations.jsonl')]
    if record['trainer'] == 'algos.ppo_potential':
        args = args_of(record)
        derived = derived_of(args)
        endpoint = derived['num_iterations'] * derived['batch_size']
        potential = read(Path(run) / 'potential_summary.json')
        if (potential['global_step'] != endpoint
                or potential['wrong_rate'] != args.potential_wrong_rate
                or potential['budget'] != args.potential_queries
                or (args.potential_queries >= 0 and
                    potential['queries_used'] > args.potential_queries)
                or potential['queries_used'] <= 0):
            raise ValueError('Potential teacher evidence differs')
        metrics['potential'] = {k: potential[k] for k in (
            'queries_used', 'wrong_messages', 'shaping_abs_sum')}
    return metrics


def run_cell(batch, index):
    batch = Path(batch).resolve()
    manifest = verify(batch, rebuild=False)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute the archived source')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker environment differs from preparation')
    record = manifest['cells'][index]
    args = args_of(record)
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    write(directory / 'dispatch.json', dict(
        cell=record, commit=manifest['commit'],
        manifest_sha256=digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    pattern = f'*{args.experiment_id}__{args.seed}__*/run_summary.json'
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        if list((ROOT / 'results/runs').glob(pattern)):
            raise FileExistsError('An earlier attempt exists for this cell')
        process = subprocess.run([sys.executable, '-u', '-m',
                                  record['trainer'],
                                  *trainer_argv(asdict(args))], cwd=ROOT)
        outcome['returncode'] = process.returncode
        runs = list((ROOT / 'results/runs').glob(pattern))
        outcome['runs'] = [p.parent.relative_to(batch).as_posix()
                           for p in runs]
        if process.returncode == 0 and len(runs) == 1:
            outcome['metrics'] = validate(runs[0].parent, record)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit(batch):
    batch = Path(batch).resolve()
    manifest = verify(batch, rebuild=False)
    submitted = batch / 'SUBMITTED_JOB'
    if submitted.exists():
        print(f"{manifest['suite']} already submitted: "
              f'{submitted.read_text().strip()}')
        return
    verify(batch)                   # a fresh batch: rebuild and compare
    n = len(manifest['cells'])
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write(f'One unthrottled array, {n} free cells.\n')
    command = [
        'sbatch', '--parsable', f"--job-name=fx_{manifest['suite']}",
        f'--array=0-{n - 1}', f"--time={TIME[manifest['suite']]}",
        f'--output={batch}/slurm/%x_%A_%a.out',
        str(batch / 'code/scripts/submit_fix_wave.sh'), str(batch)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f"{manifest['suite']}: {job} ({n} cells)")


# ------------------------------------------------------------------ report

def validated(batch):
    """Cells whose worker validated them, re-checked here from disk."""
    manifest = verify(batch, rebuild=False)
    rows = {}
    for record in manifest['cells']:
        exit_path = Path(batch) / 'cells' / str(record['index']) / 'exit.json'
        if not exit_path.exists():
            continue
        saved = read(exit_path)
        if saved['artifact_status'] != 'terminal_contract_validated':
            continue
        metrics = validate(inside(batch, saved['runs'][0]), record)
        if (not np.isclose(metrics['auc'], saved['metrics']['auc'])
                or metrics['initial_sha256'] !=
                saved['metrics']['initial_sha256']):
            raise ValueError('Recomputed metrics differ from the worker')
        key = (record['task'], record['bonus'], record['arm'])
        rows.setdefault(key, {})[record['seed']] = metrics
    return manifest, rows


def paired(a, b, label):
    """Paired by seed; refuses pairs whose initial policies differ."""
    keys = sorted(set(a) & set(b))
    bad = [k for k in keys
           if a[k]['initial_sha256'] != b[k]['initial_sha256']]
    if bad:
        return dict(contrast=label, n=0, error=f'{len(bad)} seeds have '
                    'different initial policies; not pairable')
    delta = np.array([a[k]['auc'] - b[k]['auc'] for k in keys])
    n = len(delta)
    mean = float(delta.mean()) if n else None
    sd = float(delta.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    if n > 1 and sd > 0:
        p = float(2 * t.sf(abs(mean) / (sd / np.sqrt(n)), n - 1))
    elif n > 1:
        p = 1.0 if mean == 0 else 0.0     # every pair identical
    else:
        p = None
    return dict(contrast=label, n=n, mean=mean,
                ci95=None if half is None else [mean - half, mean + half],
                positive=int((delta > 0).sum()), p=p)


CONTRASTS = (
    ('shaping_dense', 'none'),
    ('shaping_dense', 'shaping_dense_wrong'),
    ('shaping_dense_wrong', 'none'),
    ('shaping_480', 'none'),
    ('rules_plus_shaping', 'none'),
    ('rules_weak', 'none'),
    ('rules', 'none'),
    ('rules_plus_shaping', 'rules'),
    ('rules_weak', 'rules'),
    ('shaping_dense', 'rules'),
    # addendum 1: the LLM-written plan against no teacher, the hand-coded
    # plan and the rules, on the same seeds
    ('shaping_llm_plan', 'none'),
    ('shaping_llm_plan', 'shaping_dense'),
    ('shaping_llm_plan', 'rules'),
    # addendum 3: the memory rules against no teacher, and against the
    # rules without memory at the same imitation weight
    ('rules_mem', 'none'),
    ('rules_mem_weak', 'none'),
    ('rules_mem', 'rules'),
    ('rules_mem_weak', 'rules_weak'),
    ('rules_mem_weak', 'rules_mem'),
    # addendum 6: student view against teacher view (the same rules), and
    # the teacher view against no teacher
    ('rules_weak', 'rules_weak_tv'),
    ('rules_mem_weak', 'rules_mem_weak_tv'),
    ('rules_weak_tv', 'none'),
    ('rules_mem_weak_tv', 'none'),
    # addendum 8: label-only against RLingua-style execution (same
    # rules), and execution against no teacher
    ('rules_weak', 'rules_weak_act'),
    ('rules_mem_weak', 'rules_mem_weak_act'),
    ('rules_weak_act', 'none'),
    ('rules_mem_weak_act', 'none'),
    # addendum 10: with progress clauses minus without them (same rules
    # otherwise), and the ablated bank against no teacher
    ('rules_mem_weak', 'rules_mem_weak_noprog'),
    ('rules_mem_weak_noprog', 'none'),
    # addendum 12: the confirmed bank minus its deranged twin (the same
    # states labelled, the wrong targets), and the twin against no teacher
    ('rules_weak', 'rules_weak_cc'),
    ('rules_mem_weak', 'rules_mem_weak_cc'),
    ('rules_weak_cc', 'none'),
    ('rules_mem_weak_cc', 'none'),
    # addendum 14: each student's selected MultiRoom bank minus the other
    # student's, and the other student's bank against no teacher
    ('rules_weak', 'rules_weak_xbank'),
    ('rules_weak_xbank', 'none'),
    # addendum 16: our confirmed arm minus the RLingua baseline (A, as
    # published: full-state controller; B, matched runtime information:
    # student-view controller), RLingua against no teacher, and our
    # execution heuristic against RLingua
    ('rules_weak', 'rlingua_full'),
    ('rules_mem_weak', 'rlingua_full'),
    ('rules_weak', 'rlingua_view'),
    ('rules_mem_weak', 'rlingua_view'),
    ('rlingua_full', 'none'),
    ('rlingua_view', 'none'),
    ('rules_weak_act', 'rlingua_full'),
    ('rules_mem_weak_act', 'rlingua_full'),
    # addendum 17: our confirmed arm minus RLingua with the example-informed
    # view controller; that controller against the original and no teacher
    ('rules_weak', 'rlingua_view_ex'),
    ('rules_mem_weak', 'rlingua_view_ex'),
    ('rlingua_view_ex', 'rlingua_view'),
    ('rlingua_view_ex', 'none'),
)
# Against the rule-bank study's arms on the same seeds (--rule-bank):
# every rule arm that study ran for the cell, plus a reproduction check
# (the re-run none must equal the study's none: expect +0.000).
CROSS_RULES = ('llm_rules_blind_strict', 'llm_rules_scoped')
CROSS = ('shaping_dense', 'shaping_llm_plan', 'rules_plus_shaping',
         'rules_weak')


def holm(ps):
    """Holm step-down adjustment; None stays None."""
    order = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    out, running = [None] * len(ps), 0.0
    for rank, (p, i) in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p))
        out[i] = running
    return out


# Addendum 2's primary family, one contrast per cell (six cells).
PRIMARY = ('rules_weak', 'none')
NO_HARM_MARGIN = -.05
# Addendum 3's family on the fresh seeds: both memory arms against no
# teacher, in both KeyCorridor cells (four contrasts).
MEM_FAMILY = (('rules_mem', 'none'), ('rules_mem_weak', 'none'))
# Addendum 6's family: student view minus teacher view, the same rules, in
# all six confirmation cells (DoorKey, MultiRoom, KeyCorridor x student).
TV_FAMILY = (('rules_weak', 'rules_weak_tv'),
             ('rules_mem_weak', 'rules_mem_weak_tv'))
# Addendum 8's family: label-only minus RLingua-style execution, six cells.
ACT_FAMILY = (('rules_weak', 'rules_weak_act'),
              ('rules_mem_weak', 'rules_mem_weak_act'))
# Addendum 12's family: the confirmed bank minus its deranged twin, six cells;
# the secondary family: the deranged twin against no teacher, six cells.
CC_FAMILY = (('rules_weak', 'rules_weak_cc'),
             ('rules_mem_weak', 'rules_mem_weak_cc'))
CC_NONE_FAMILY = (('rules_weak_cc', 'none'), ('rules_mem_weak_cc', 'none'))
# Addendum 13's family: the re-run's confirmed arm minus no teacher, six cells.
DENSE_FAMILY = (('rules_weak', 'none'), ('rules_mem_weak', 'none'))
# Addendum 16's families: our confirmed arm minus RLingua. A: RLingua as
# published, its controller reading the full simulator state whenever it
# acts. B: RLingua at the runtime information our rules use (the student's
# view), its controller reading only that.
RL_FULL_FAMILY = (('rules_weak', 'rlingua_full'),
                  ('rules_mem_weak', 'rlingua_full'))
RL_VIEW_FAMILY = (('rules_weak', 'rlingua_view'),
                  ('rules_mem_weak', 'rlingua_view'))
# Addendum 17: our confirmed arm minus RLingua whose student-view controller
# was written after 36 full-map examples (matched design-time and runtime
# information); secondary: that controller minus the original view one.
RL_VIEW_EX_FAMILY = (('rules_weak', 'rlingua_view_ex'),
                     ('rules_mem_weak', 'rlingua_view_ex'))
RL_EX_VS_VIEW_FAMILY = (('rlingua_view_ex', 'rlingua_view'),)
# Declared sizes of the addenda 12-15 families. With fewer complete
# contrasts (cells still running, or suites not launched) the report says
# PROVISIONAL, because Holm then adjusts over a smaller family.
DECLARED_FAMILY_SIZE = {
    'addendum 12 family, confirmed bank - deranged twin': 6,
    'addendum 12 secondary, deranged twin - none': 6,
    'addendum 13 family, re-run with dense evaluation, rules - none': 6,
    'addendum 14 family, MultiRoom other-student bank - none': 2,
    'addendum 15 primary, fresh seeds, rules - none': 6,
    'addendum 15 progress, with - without progress clauses': 2,
    'addendum 15 content, confirmed bank - deranged twin': 6,
    'addendum 15 content secondary, deranged twin - none': 6,
    'addendum 15 crossover, other-student bank - none': 2,
    'addendum 16 A, ours - RLingua as published (full-state controller)': 6,
    'addendum 16 B, ours - RLingua at matched runtime information '
    '(student-view controller)': 6,
    'addendum 16, DoorKey-16x16 at 15M, ours - RLingua student-view': 2,
    'addendum 16, MultiRoom-N10 at 15M, ours - RLingua student-view': 2,
    'addendum 16, KeyCorridor-S4R3 at 15M, ours - RLingua student-view': 2,
    'addendum 17, ours - RLingua at matched design-time and runtime '
    'information (example-informed student-view controller)': 6,
    'addendum 17 secondary, example-informed - original student-view '
    'controller': 6,
}


# Seed pairs each declared contrast is planned to have (ten replicates).
PLANNED_PAIRS = 10


def provisional(title, family):
    """A warning line when a declared family is incomplete, else None.

    `family` is the list of present contrasts (or just their number). A
    family is incomplete when contrasts are missing or any present one has
    fewer than PLANNED_PAIRS seed pairs; its verdicts are then withheld."""
    declared = DECLARED_FAMILY_SIZE.get(title)
    if not declared:
        return None
    complete = family if isinstance(family, int) else len(family)
    short = [] if isinstance(family, int) else [
        c for c in family if c.get('n', 0) < PLANNED_PAIRS]
    cells = [] if isinstance(family, int) else [
        (c['task'], c['bonus']) for c in family]
    repeated = len(cells) - len(set(cells))
    if complete != declared or short or repeated:
        return (f'   PROVISIONAL: {complete} of {declared} declared '
                f'contrasts present, {len(short)} with fewer than '
                f'{PLANNED_PAIRS} seed pairs, {repeated} repeated '
                'task-student cells; Holm adjusts over the present ones only '
                'and no verdict is final')
    return None


# Addendum 16 compares RLingua with OUR CONFIRMED arm of each task. The
# KeyCorridor confirmation cohort also ran rules_weak, which must not stand
# in for (or be counted beside) its confirmed memory bank there.
OURS = ('rules_weak', 'rules_mem_weak')
CONFIRMED_ARM = {'doorkey_8x8': 'rules_weak', 'multiroom_n6': 'rules_weak',
                 'keycorridor_s3r3': 'rules_mem_weak',
                 'doorkey_16x16': 'rules_weak', 'multiroom_n10': 'rules_weak',
                 'keycorridor_s4r3': 'rules_mem_weak'}


def family_members(rows, contrasts, confirmed_only=False):
    """The rows of one family: its named contrasts with at least two seed
    pairs; with `confirmed_only`, a contrast whose first arm is one of our
    banks counts only for the task whose confirmed arm it is."""
    names = {' - '.join(c) for c in contrasts}
    keep = []
    for c in rows:
        if c['contrast'] not in names or c.get('n', 0) <= 1:
            continue
        first = c['contrast'].split(' - ')[0]
        if (confirmed_only and first in OURS
                and CONFIRMED_ARM.get(c['task']) != first):
            continue
        keep.append(c)
    return keep


def reproduction(groups, a='dense', b='confirm', tol=1e-9):
    """Re-run cells against the original runs of the same task, student,
    arm and seed. Informational only (addendum 13, revised): equal AUCs,
    or even equal curves, do not certify identical training, so re-runs
    are always reported as their own cohort and never spliced."""
    if a not in groups or b not in groups:
        return None
    out = dict(total=0, same_auc=0, same_initial=0, same_curve=0,
               same_labels=0, max_abs_auc_diff=0.0)
    for key, seeds in groups[a].items():
        for seed, m in seeds.items():
            other = groups[b].get(key, {}).get(seed)
            if other is None:
                continue
            out['total'] += 1
            gap = abs(m['auc'] - other['auc'])
            out['same_auc'] += gap < tol
            out['max_abs_auc_diff'] = max(out['max_abs_auc_diff'], gap)
            out['same_initial'] += (m['initial_sha256'] ==
                                    other['initial_sha256'])
            out['same_curve'] += (m.get('curve') is not None and
                                  m.get('curve') == other.get('curve'))
            out['same_labels'] += (m.get('labels_delivered') is not None and
                                   m.get('labels_delivered') ==
                                   other.get('labels_delivered'))
    print(f"#### {a} vs {b} reproduction (informational): of "
          f"{out['total']} re-runs, {out['same_initial']} same initial "
          f"policy, {out['same_curve']} identical greedy+sampled curves, "
          f"{out['same_labels']} same labels delivered, {out['same_auc']} "
          f"same AUC; largest |AUC diff| {out['max_abs_auc_diff']:.3g}")
    return out


def verdict(c, q):
    return ('HELPS' if c['mean'] > 0 and q < .05 else
            'NO HARM' if c['ci95'][0] > NO_HARM_MARGIN else 'NOT SHOWN')


def fmt(c):
    if c.get('error'):
        return f"{c['contrast']:44s} {c['error']}"
    if not c['n']:
        return f"{c['contrast']:44s} no complete pairs yet"
    ci = ('' if c['ci95'] is None else
          f" [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    return (f"{c['contrast']:44s} {c['mean']:+.3f}{ci} "
            f"{c['positive']}/{c['n']} positive")


def report(root, suites, rule_bank=None, out=None):
    """Validated runs only; missing runs are reported, never zeros.

    Suites of one GROUP (a suite and its addendum) are merged, so an
    addendum arm pairs with the original suite's cells by seed; paired()
    still refuses any pair whose initial policies differ.
    """
    result, bank_rows, groups = {}, {}, {}
    if rule_bank:
        from scripts.report_rule_bank_paper_20260928 import load
        bank_rows, _problems = load(rule_bank, replicates=(5, 15),
                                    tolerance=0.05)
    for suite in suites:
        batch = batch_dir(root, suite)
        if not (batch / 'manifest.json').exists():
            print(f'== {suite}: not prepared')
            continue
        manifest, rows = validated(batch)
        done = sum(len(v) for v in rows.values())
        print(f"== {suite}: {done}/{len(manifest['cells'])} cells validated")
        merged = groups.setdefault(GROUP[suite], {})
        for key, seeds in rows.items():
            merged.setdefault(key, {}).update(seeds)
    for group, rows in groups.items():
        print(f'#### {group}')
        for task, bonus in sorted({(k[0], k[1]) for k in rows}):
            student = 'count bonus' if bonus == 'count' else 'plain PPO'
            print(f'   -- {task} / {student}')
            arms = {arm: v for (tk, bn, arm), v in rows.items()
                    if (tk, bn) == (task, bonus)}
            for arm, seeds in sorted(arms.items(),
                                     key=lambda kv: -np.mean(
                                         [m['auc'] for m in kv[1].values()])):
                aucs = [m['auc'] for m in seeds.values()]
                finals = [m['final'] for m in seeds.values()]
                print(f'      {arm:22s} n={len(aucs):2d} AUC '
                      f'{np.mean(aucs):.3f}  final {np.mean(finals):.3f}')
            for a, b in CONTRASTS:
                if a in arms and b in arms:
                    c = paired(arms[a], arms[b], f'{a} - {b}')
                    result.setdefault(group, []).append(
                        dict(c, task=task, bonus=bonus))
                    print('      ' + fmt(c))
            if bank_rows:
                pairs = [('none', 'none')] + [
                    (a, b) for a in CROSS for b in dict.fromkeys(
                        (RULE_ARM[(task, bonus)], *CROSS_RULES))]
                for a, b in pairs:
                    other = bank_rows.get((task, bonus, b), {})
                    if a in arms and other:
                        c = paired(arms[a], other, f'{a} - rule-bank {b}')
                        result.setdefault(group, []).append(
                            dict(c, task=task, bonus=bonus))
                        print('      ' + fmt(c))
    reproduced = reproduction(groups)
    if reproduced:
        result['reproduction_dense_vs_confirm'] = reproduced
    for group, title, contrasts in (
            ('confirm', 'primary family, rules_weak - none', (PRIMARY,)),
            ('confirm', 'addendum 3 family, memory rules - none', MEM_FAMILY),
            ('dk16', 'addendum 4 family, transferred rules - none',
             (PRIMARY,)),
            ('kc_s4', 'addendum 5 family, transferred memory rules - none',
             (('rules_mem_weak', 'none'),)),
            ('confirm', 'addendum 6 family, student view - teacher view',
             TV_FAMILY),
            ('confirm', 'addendum 8 family, labels only - teacher acts',
             ACT_FAMILY),
            ('dk16_long', 'addendum 9, DoorKey-16x16 at 15M, rules - none',
             (PRIMARY,)),
            ('mr10_long', 'addendum 9, MultiRoom-N10 at 15M, rules - none',
             (PRIMARY,)),
            ('kc_s4_long', 'addendum 9, KeyCorridor-S4R3 at 15M, memory '
             'rules - none', (('rules_mem_weak', 'none'),)),
            ('confirm', 'addendum 10 family, with - without progress clauses',
             (('rules_mem_weak', 'rules_mem_weak_noprog'),)),
            ('kc_s5_long', 'addendum 11, KeyCorridor-S5R3 at 15M, memory '
             'rules - none', (('rules_mem_weak', 'none'),)),
            ('kc_s6_long', 'addendum 11, KeyCorridor-S6R3 at 15M, memory '
             'rules - none', (('rules_mem_weak', 'none'),)),
            ('dk16', 'addendum 7, paper-weight rules - none (plain PPO)',
             (('rules', 'none'),)),
            ('kc_s4', 'addendum 7, paper-weight memory rules - none '
             '(plain PPO)', (('rules_mem', 'none'),)),
            ('confirm', 'addendum 12 family, confirmed bank - deranged '
             'twin', CC_FAMILY),
            ('confirm', 'addendum 12 secondary, deranged twin - none',
             CC_NONE_FAMILY),
            ('dense', 'addendum 13 family, re-run with dense evaluation, '
             'rules - none', DENSE_FAMILY),
            ('confirm', 'addendum 14 family, MultiRoom other-student bank '
             '- none', ((XBANK_ARM, 'none'),)),
            ('fresh', 'addendum 15 primary, fresh seeds, rules - none',
             DENSE_FAMILY),
            ('fresh', 'addendum 15 progress, with - without progress '
             'clauses', (('rules_mem_weak', 'rules_mem_weak_noprog'),)),
            ('fresh', 'addendum 15 content, confirmed bank - deranged twin',
             CC_FAMILY),
            ('fresh', 'addendum 15 content secondary, deranged twin - none',
             CC_NONE_FAMILY),
            ('fresh', 'addendum 15 crossover, other-student bank - none',
             ((XBANK_ARM, 'none'),)),
            ('confirm', 'addendum 16 A, ours - RLingua as published '
             '(full-state controller)', RL_FULL_FAMILY),
            ('confirm', 'addendum 16 B, ours - RLingua at matched runtime '
             'information (student-view controller)', RL_VIEW_FAMILY),
            ('dk16_long', 'addendum 16, DoorKey-16x16 at 15M, ours - '
             'RLingua student-view', RL_VIEW_FAMILY),
            ('mr10_long', 'addendum 16, MultiRoom-N10 at 15M, ours - '
             'RLingua student-view', RL_VIEW_FAMILY),
            ('kc_s4_long', 'addendum 16, KeyCorridor-S4R3 at 15M, ours - '
             'RLingua student-view', RL_VIEW_FAMILY),
            ('confirm', 'addendum 17, ours - RLingua at matched design-time '
             'and runtime information (example-informed student-view '
             'controller)', RL_VIEW_EX_FAMILY),
            ('confirm', 'addendum 17 secondary, example-informed - original '
             'student-view controller', RL_EX_VS_VIEW_FAMILY)):
        family = family_members(result.get(group, []), contrasts,
                                confirmed_only=title.startswith(
                                    ('addendum 16', 'addendum 17,')))
        if not family:
            continue
        adjusted = holm([c['p'] for c in family])
        print(f'#### {group}: {title}, Holm over {len(family)} contrasts '
              f'(no-harm margin {NO_HARM_MARGIN})')
        warning = provisional(title, family)
        if warning:
            print(warning)
        for c, q in zip(family, adjusted):
            c['p_holm'] = q
            label = 'PROVISIONAL' if warning else verdict(c, q)
            print(f"   {c['task']:18s} {c['bonus']:5s} {c['contrast']:26s}"
                  f" {c['mean']:+.3f} [{c['ci95'][0]:+.3f}, "
                  f"{c['ci95'][1]:+.3f}] {c['positive']}/{c['n']}  "
                  f"p_holm {q:.3g}  {label}")
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('check', 'prepare', 'launch',
                                        'report', 'run-cell'))
    cli.add_argument('--suite', choices=tuple(SUITES))
    cli.add_argument('--batch', type=Path)
    cli.add_argument('--index', type=int)
    cli.add_argument('--root', type=Path, default=ROOT)
    cli.add_argument('--rule-bank', type=Path,
                     help='results/efficiency: also pair with the rule-bank '
                          'study arms on the same seeds')
    cli.add_argument('--out', type=Path)
    args = cli.parse_args()
    if args.action == 'run-cell':
        return run_cell(args.batch, args.index)
    if args.action == 'report':
        report(args.root, [args.suite] if args.suite else list(SUITES),
               args.rule_bank, args.out)
        return 0
    if not args.suite:
        cli.error('--suite is required')
    if args.action == 'check':
        cells = suite_cells(args.suite, args.root)
        arms = sorted({c['arm'] for c in cells})
        print(f"PASS {STUDY}/{args.suite}: {len(cells)} cells; "
              f"arms {arms}; students {sorted({c['bonus'] for c in cells})}")
        return 0
    batch = prepare(args.root, args.suite)
    if args.action == 'launch':
        submit(batch)
    else:
        print('Prepared', batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
