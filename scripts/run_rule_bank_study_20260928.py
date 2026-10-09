"""Learning studies with frozen LLM rule banks: KeyCorridor, DoorKey Count.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? This runner holds the studies that complete
the paper's environment-by-student grid, with the same design as the
DoorKey plain-PPO pilot (scripts/run_rule_bank_pilot_20260928.py) and the
MultiRoom study (scripts/run_rule_bank_multiroom_20260928.py):

  keycorridor    KeyCorridor-S3R3, plain PPO and Count-PPO. GPT-5-mini
                 action advice collapsed a working Count-PPO student
                 here; banks from scripts/conditional_rules_keycorridor.py.
  doorkey_count  DoorKey-8x8 Count-PPO with the frozen DoorKey banks
                 (research/rule_banks/v3_20260928), so DoorKey has both
                 students.

Every guided arm reads a bank written before training from the same 36
GPT-5-mini consultations (the blind-strict bank also used the blind
checks); training makes no API calls. Five fresh paired seeds, a 5M
horizon, the reviewed rule-reference settings for the task and student,
and the pilot's distillation settings. Arms, per student:

  none                    no teacher
  llm_action_replay       the LLM's action_now on identical views only
  llm_rules_scoped        all of the LLM's rules, v3 scope semantics
  llm_rules_shuffled      the same scopes with actions permuted
  llm_rules_blind_strict  the rules the blind strict check kept
  llm_rules_random_subset as many raw rules, chosen at random (a
                          different frozen subset per replicate)

KeyCorridor also runs `llm_action_predicate_replay`: the LLM's action
labels reused wherever the student's predicates equal a consultation
state's (scripts/build_predicate_replay_banks_20260928.py), the strongest
action-only reuse over the rules' own features. The `*_controls` studies
add that arm alone for DoorKey (plain and Count-PPO) and MultiRoom, on the
seeds and settings of the running studies; preparation refuses unless
each cell equals that seed's `llm_rules_scoped` cell apart from the bank.

The paid `*_online` studies run `llm_action_online_eq`, the equal-budget
action-only control. GPT-5-mini is consulted DURING training at uniformly
scheduled visited states (the reviewed random-timing schedule), through
the offline consultation prompt (teachers/minigrid/llm_scoped.py). It
gets as many calls as the blind-strict bank cost (DoorKey 62, MultiRoom
67, KeyCorridor 61), and each call labels only that state. Preparation
refuses unless the shared budget ledger covers every run's worst case
and the key file exists. Cells differ from the paired scoped cells only
in the teaching channel and its budget fields.

Frozen rule, per student (as MultiRoom): the verified explanation is
useful beyond replay if `llm_rules_blind_strict` beats BOTH
`llm_action_replay` and `llm_rules_shuffled` with a positive mean
teacher-off AUC and at least 4/5 positive pairs; verification, not mere
pruning, matters if it also beats `llm_rules_random_subset` by the same
bar. KeyCorridor adds a harm check: action advice harmed Count-PPO here,
so `llm_rules_blind_strict - none` is reported with its interval. A
contrast in which every run scores zero is uninformative.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import functools
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

from scripts.run_explanation_formats_20260925 import runtime_identity
from scripts.run_explanation_grid import trainer_argv
from scripts.run_plan_repair_20260927 import digest, inside, read, write
from scripts.run_rule_bank_pilot_20260928 import (find_run, paired,
                                                  validate_run)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('none', 'llm_action_replay', 'llm_rules_scoped',
        'llm_rules_shuffled', 'llm_rules_blind_strict',
        'llm_rules_random_subset')
BANK_OF = {'llm_action_replay': 'replay', 'llm_rules_scoped': 'scoped',
           'llm_rules_shuffled': 'shuffled',
           'llm_rules_blind_strict': 'blind_strict',
           'llm_rules_random_subset': 'random_subset_r{r}',
           'llm_action_predicate_replay': 'predicate_replay'}
N_SEEDS = 5
HORIZON = 5_000_000
PREDICATE_REPLAY = ('llm_action_predicate_replay',)
SPECS = {
    'keycorridor': dict(
        study='rule_bank_keycorridor_20260928_v1', task='keycorridor_s3r3',
        bonuses=('none', 'count'), seed0=14_700_000,
        banks='research/rule_banks/keycorridor_20260928',
        observer_source='scripts/conditional_rules_keycorridor.py',
        arms=ARMS + PREDICATE_REPLAY),
    # v1 was prepared by a runner whose identity check compared the
    # JSON manifest (lists) with this tuple-valued spec, so it refused
    # itself and was never submitted; v2 is the same study.
    'doorkey_count': dict(
        study='rule_bank_doorkey_count_20260928_v2', task='doorkey_8x8',
        bonuses=('count',), seed0=14_500_000,
        banks='research/rule_banks/v3_20260928',
        observer_source='scripts/conditional_rules_v3.py'),
    'doorkey_controls': dict(
        study='rule_bank_doorkey_controls_20260928_v1', task='doorkey_8x8',
        bonuses=('none', 'count'), seed0=14_500_000,
        banks='research/rule_banks/v3_20260928',
        observer_source='scripts/conditional_rules_v3.py',
        arms=PREDICATE_REPLAY),
    'multiroom_controls': dict(
        study='rule_bank_multiroom_controls_20260928_v1',
        task='multiroom_n6', bonuses=('none', 'count'), seed0=14_600_000,
        banks='research/rule_banks/multiroom_20260928',
        observer_source='scripts/conditional_rules_multiroom.py',
        arms=PREDICATE_REPLAY),
}
BANK_KEYS = {'experiment_id', 'rule_bank', 'rule_bank_sha256'}

# Paid equal-budget control: GPT-5-mini consulted DURING training at
# uniformly scheduled visited states, through the offline consultation
# prompt (teachers/minigrid/llm_scoped.py), as many calls as the
# blind-strict bank cost (its consultations plus blind checks); only the
# action label is used. Seeds and settings pair with the rule-bank
# studies; only the teaching channel and its budget fields differ.
ONLINE = ('llm_action_online_eq',)
ONLINE_MODEL = 'gpt-5-mini-2025-08-07'
ONLINE_PRICES = 'configs/prices_llm_scoped_2026-09-28.json'
ONLINE_LIMITS = dict(max_input_tokens=10_000, max_output_tokens=8192,
                     max_attempts=1)
PAID_KEYS = BANK_KEYS | {
    'teacher', 'teacher_model', 'uniform_queries', 'query_budget',
    'advice_budget', 'budget_ledger', 'price_table', *ONLINE_LIMITS,
    'consultation_failure_limit', 'consultation_failure_min_samples'}
# Batches prepared after the 2026-09-28 credit outage stop a paid run whose
# calls fail (the trainer's failure limit) instead of training on without
# labels; online rules stop after three failed calls in a row.
FAIL_FAST = dict(consultation_failure_limit=.3,
                 consultation_failure_min_samples=10)
for _name, _base in (('doorkey', 'doorkey_count'),
                     ('multiroom', 'multiroom_controls'),
                     ('keycorridor', 'keycorridor')):
    SPECS[f'{_name}_online'] = dict(
        {k: SPECS[_base][k] for k in ('task', 'seed0', 'banks',
                                      'observer_source')},
        study=f'rule_bank_{_name}_online_20260928_v1',
        bonuses=('none', 'count'), arms=ONLINE, paid=True)
# Second wave (confirmation plan, 2026-09-28): fresh
# confirmation replicates 5..19 of the core arms, and the budget and dose
# arms on replicates 0..9. Cells are ordered seed by seed, so complete
# paired seeds finish first. Paid arrays are throttled so the ledger only
# ever holds the worst case of the runs actually in flight.
#   online arm: (LLM calls or 'bank', output-token ceiling, schedule)
ONLINE_ARMS = {
    'llm_action_online_eq': ('bank', 8192, 'uniform'),
    'llm_action_online_480': (480, 4096, 'uniform'),
    'llm_action_online_480_entropy': (480, 4096, 'entropy'),
    # third wave: the remaining budgets and advising schedules
    'llm_action_online_15': (15, 4096, 'uniform'),
    'llm_action_online_30': (30, 4096, 'uniform'),
    'llm_action_online_120': (120, 4096, 'uniform'),
    'llm_action_online_480_mistake': (480, 4096, 'mistake'),
    'llm_action_online_480_regular': (480, 4096, 'regular'),
}
# How each schedule departs from the uniform one (the in-house entropy480,
# probability480 and regular-spacing clock arms, with the LLM teacher).
SCHEDULES = {
    'uniform': {},
    'entropy': dict(advisor='importance', importance_source='entropy',
                    uniform_queries=False),
    'mistake': dict(advisor='mistake', importance_source='entropy',
                    mistake_threshold=.2, uniform_queries=False),
    'regular': dict(uniform_queries=False, query_clock='random',
                    query_clock_spacing='regular'),
}
CAPPED = {'llm_rules_scoped_cap480': 480}     # rule labels, uniform slots
# Rules written during training (teachers/minigrid/llm_rules_online.py), at
# the verified bank's call budget: arm -> blind-check the rules?
ONLINE_RULE_ARMS = {'llm_rules_online_blind': True,
                    'llm_rules_online_raw': False}
ONLINE_RULE_KEYS = {'online_rule_calls', 'online_rule_blind',
                    'journal_paid_only'}
BANK_OF.update({'llm_rules_scoped_cap480': 'scoped',
                'llm_rules_scoped_12': 'scoped_12',
                'llm_rules_blind_strict_12': 'blind_strict_12',
                'llm_rules_direct': 'direct',
                'llm_rules_scope_checked': 'scope_checked'})
CONFIRM_ARMS = ('none', 'llm_rules_blind_strict', 'llm_rules_scoped',
                'llm_action_replay', 'llm_action_predicate_replay',
                'llm_rules_random_subset')
BUDGET_ARMS = ('llm_rules_scoped_cap480', 'llm_rules_scoped_12',
               'llm_rules_blind_strict_12')
for _name, _task, _seed0, _banks, _source in (
        ('doorkey', 'doorkey_8x8', 14_500_000,
         'research/rule_banks/v3_20260928', 'scripts/conditional_rules_v3.py'),
        ('multiroom', 'multiroom_n6', 14_600_000,
         'research/rule_banks/multiroom_20260928',
         'scripts/conditional_rules_multiroom.py'),
        ('keycorridor', 'keycorridor_s3r3', 14_700_000,
         'research/rule_banks/keycorridor_20260928',
         'scripts/conditional_rules_keycorridor.py')):
    _base = dict(task=_task, seed0=_seed0, banks=_banks,
                 observer_source=_source, bonuses=('none', 'count'),
                 seed_major=True)
    SPECS[f'confirm_{_name}'] = dict(
        _base, study=f'rule_bank_confirm_{_name}_20260928_v1',
        arms=CONFIRM_ARMS, replicates=(5, 20))
    # Pool revision before any paid launch (about $100 available): the
    # paid wave-2 arms cap output at 2048 tokens, as consultations used
    # at most 1171, which cuts each run's worst-case hold about 3-4x; the
    # 480-call arm runs replicates 0-4 with a throttle of 5.
    SPECS[f'confirm_online_{_name}'] = dict(
        _base, study=f'rule_bank_confirm_online_{_name}_20260928_v1',
        arms=('llm_action_online_eq',), replicates=(5, 20), paid=True,
        throttle=15, max_output_tokens=2048)
    SPECS[f'budget_{_name}'] = dict(
        _base, study=f'rule_bank_budget_{_name}_20260928_v1',
        arms=BUDGET_ARMS, replicates=(0, 10))
    SPECS[f'budget_online_{_name}'] = dict(
        _base, study=f'rule_bank_budget_online_{_name}_20260928_v1',
        arms=('llm_action_online_480',), replicates=(0, 5), paid=True,
        throttle=5, max_output_tokens=2048)
    # Deferred for the pool (launch when money allows): replicates 5-9 of
    # the 480-call arm, and replicates 5-19 of the shuffled-content control
    # (free; deferred for cluster time).
    SPECS[f'budget_online_{_name}_b'] = dict(
        SPECS[f'budget_online_{_name}'], replicates=(5, 10), fail_fast=True,
        study=f'rule_bank_budget_online_{_name}_b_20260928_v1')
    SPECS[f'confirm_shuffled_{_name}'] = dict(
        _base, study=f'rule_bank_confirm_shuffled_{_name}_20260928_v1',
        arms=('llm_rules_shuffled',), replicates=(5, 20), summary_only=True)
    # Third wave (2026-09-28): the
    # remaining online budgets and advising schedules for EVERY task and
    # student, and the weaker rule variants beyond DoorKey plain PPO.
    SPECS[f'sweep_online_{_name}'] = dict(
        _base, study=f'rule_bank_sweep_online_{_name}_20260928_v1',
        arms=('llm_action_online_15', 'llm_action_online_30',
              'llm_action_online_120'), replicates=(0, 5), paid=True,
        throttle=15, max_output_tokens=2048, fail_fast=True)
    SPECS[f'schedule_online_{_name}'] = dict(
        _base, study=f'rule_bank_schedule_online_{_name}_20260928_v1',
        arms=('llm_action_online_480_entropy',
              'llm_action_online_480_mistake',
              'llm_action_online_480_regular'), replicates=(0, 5),
        paid=True, throttle=5, max_output_tokens=2048, fail_fast=True)
    SPECS[f'variants_{_name}'] = dict(
        _base, study=f'rule_bank_variants_{_name}_20260928_v1',
        arms=('llm_rules_direct', 'llm_rules_scope_checked'),
        replicates=(0, 5), summary_only=True,
        # DoorKey plain PPO already ran both variants in the pilot
        bonuses=('count',) if _name == 'doorkey' else ('none', 'count'))
    SPECS[f'online_rules_{_name}'] = dict(
        _base, study=f'rule_bank_online_rules_{_name}_20260928_v1',
        arms=tuple(ONLINE_RULE_ARMS), replicates=(0, 5), paid=True,
        throttle=10, max_output_tokens=2048)
# Fourth wave (2026-09-28; the goal is faster learning, not rule
# precision). KeyCorridor's frozen raw rules restricted to the front cells
# where their action has an effect (scripts/valid_action_rules_20260928.py;
# no LLM calls), on fresh replicates 20-29 with their own no-teacher and
# unrestricted-rule cells, so every contrast is paired on new seeds.
BANK_OF['llm_rules_scoped_valid'] = 'scoped_valid'
SPECS['valid_keycorridor'] = dict(
    task='keycorridor_s3r3', seed0=14_700_000, bonuses=('none', 'count'),
    banks='research/rule_banks/keycorridor_20260928',
    observer_source='scripts/conditional_rules_keycorridor.py',
    seed_major=True, summary_only=True,
    study='rule_bank_valid_keycorridor_20260928_v1',
    arms=('none', 'llm_rules_scoped', 'llm_rules_scoped_valid'),
    replicates=(20, 30))
# Reruns after the 2026-09-28 credit outage (07:43 UTC until credit was
# added): every cell of a launched paid batch whose run had a paid call
# that never reached the model. Found when the batch is prepared; each is
# rerun with its original settings, so it pairs with everything else. The
# report never pairs the original.
RERUN_FROM = tuple(name for name, spec in SPECS.items() if spec.get('paid'))
# v1 was prepared empty (the run list was rebuilt from the frozen code copy,
# which has no results) and never submitted; v2 is the same batch.
SPECS['rerun_paid'] = dict(study='rule_bank_rerun_paid_20260928_v2',
                           task='mixed', rerun=True, paid=True, throttle=15)
# Completion (2026-09-29). The v2 rerun refused cells 73-144 at start with
# BudgetError: $320 of the $453 allowance was held by ENDED runs whose holds
# were never settled ($133 spent). Two outage cells were also cancelled
# mid-run without a lost call, so no rerun covered them. These batches rerun,
# verbatim, every paid advice cell of the given arms that has no usable run
# in ANY batch (the paper report's admission at 5%), except the schedule
# arms, cancelled on purpose. Settle the holds first
# (scripts/settle_paid_runs_20260928.py). Defined after RERUN_FROM, so they
# are never themselves a source of reruns.
SPECS['complete_paid_eq'] = dict(
    study='rule_bank_complete_paid_eq_20260929_v1', task='mixed',
    rerun='complete', arms=('llm_action_online_eq',), paid=True,
    throttle=15)
SPECS['complete_paid_480'] = dict(
    study='rule_bank_complete_paid_480_20260929_v1', task='mixed',
    rerun='complete', arms=('llm_action_online_480',), paid=True,
    throttle=5)
CANCELLED_SPECS = tuple(f'schedule_online_{t}' for t in
                        ('doorkey', 'multiroom', 'keycorridor'))
RUNNER = ('scripts/run_rule_bank_study_20260928.py',
          'scripts/submit_rule_bank_study.sh',
          'scripts/launch_rule_bank_study.sh',
          'teachers/minigrid/rule_bank.py',
          'tests/test_rule_bank_study.py')


def replicates(spec):
    return range(*spec.get('replicates', (0, N_SEEDS)))


def seeds(spec):
    return tuple(spec['seed0'] + 100 * r for r in replicates(spec))


def arms_of(spec):
    return spec.get('arms', ARMS)


def bank_names(spec):
    names = set()
    for arm in arms_of(spec):
        if arm in ONLINE_ARMS or arm in ONLINE_RULE_ARMS:
            names.add('blind_strict')       # the equal budget comes from here
        elif arm != 'none':
            names |= {BANK_OF[arm].format(r=r) for r in replicates(spec)}
    return sorted(names)


def request_failures(run):
    """Paid calls that never reached the model (no credit, network)."""
    path = Path(run) / 'consultations.jsonl'
    count = 0
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if '"request_failure"' in line:
                    meta = json.loads(line).get('metadata', {})
                    count += bool(meta.get('failed') and
                                  meta.get('outcome') == 'request_failure')
    return count


def completion_cells(root=ROOT, arms=('llm_action_online_eq',)):
    """Paid cells of `arms` with no usable run in any batch, verbatim.

    Coverage is the study inventory's classification, the paper report's
    own checks with paid runs admitted at the declared 5% dose tolerance.
    """
    from scripts.study_inventory_20260929 import PRIMARY, inventory
    _batches, entries = inventory(Path(root) / 'results/efficiency',
                                  PRIMARY, jobs=None)
    covered = {key for key, found in entries.items()
               if any(e['state'] == 'valid' for e in found)}
    out, seen = [], set()
    for name in RERUN_FROM:
        if name in CANCELLED_SPECS:
            continue
        batch = batch_dir(root, SPECS[name])
        if not (batch / 'manifest.json').exists():
            continue
        manifest = read(batch / 'manifest.json')
        for cell in manifest['cells']:
            args = cell['args']
            key = (args['task'], cell.get('bonus', args.get('bonus')),
                   cell['arm'], cell['seed'])
            if cell['arm'] not in arms or key in covered or key in seen:
                continue
            seen.add(key)
            out.append(dict(cell, index=len(out),
                            origin=dict(study=manifest['study'],
                                        index=cell['index'])))
    return out


def rerun_cells(root=ROOT):
    """Cells of launched paid batches whose runs lost calls to the API."""
    out = []
    for name in RERUN_FROM:
        batch = batch_dir(root, SPECS[name])
        if not (batch / 'manifest.json').exists():
            continue
        manifest = read(batch / 'manifest.json')
        for cell in manifest['cells']:
            try:
                run = find_run(batch / 'code/results/runs', cell['args'])
            except FileNotFoundError:
                continue                     # not started: will be clean
            lost = request_failures(run)
            if lost:
                out.append(dict(cell, index=len(out), lost_calls=lost,
                                origin=dict(study=manifest['study'],
                                            index=cell['index'])))
    return out


def required(spec):
    if spec.get('rerun'):
        return (*RUNNER, 'teachers/minigrid/llm_scoped.py',
                'teachers/minigrid/llm_rules_online.py', ONLINE_PRICES,
                'tests/test_llm_scoped.py', 'scripts/conditional_rules_v3.py',
                'scripts/conditional_rules_multiroom.py',
                'scripts/conditional_rules_keycorridor.py')
    paid = [a for a in arms_of(spec)
            if a in ONLINE_ARMS or a in ONLINE_RULE_ARMS]
    extra = ('teachers/minigrid/llm_scoped.py', ONLINE_PRICES,
             'tests/test_llm_scoped.py') if paid else ()
    if any(a in ONLINE_RULE_ARMS for a in paid):
        extra += ('teachers/minigrid/llm_rules_online.py',)
    return (*RUNNER, spec['observer_source'], *extra,
            *(f"{spec['banks']}/{name}.json" for name in bank_names(spec)))


def paid_calls(args):
    """LLM calls a run may make (0 for a free teacher).

    Cells frozen by older runners lack settings added later, such as
    online_rule_calls, so a missing setting means the trainer default.
    """
    if args['teacher'] == 'llm_scoped':
        return args['query_budget']
    if args['teacher'] == 'llm_rules_online':
        return args.get('online_rule_calls', 0)
    return 0


def job_active(job_id):
    """Is this Slurm job still queued or running? (True if unknown.)"""
    try:
        out = subprocess.run(['squeue', '-h', '-j', str(job_id)],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(out.stdout.strip())


def allowed_keys(arm):
    """Settings an arm may change relative to its seed's scoped cell."""
    if arm in ONLINE_RULE_ARMS:
        return PAID_KEYS | ONLINE_RULE_KEYS
    if arm in ONLINE_ARMS:
        return PAID_KEYS | set(SCHEDULES[ONLINE_ARMS[arm][2]])
    if arm in CAPPED:
        return BANK_KEYS | {'uniform_queries', 'query_budget',
                            'advice_budget'}
    return BANK_KEYS


def online_budget(spec, root=ROOT):
    """LLM calls behind the blind-strict bank: consultations + checks."""
    cost = read(Path(root) / spec['banks'] / 'blind_strict.json')['cost']
    return sum(stage['calls'] for stage in cost.values())


def reference_cells(spec):
    """The running studies' cells that a control study pairs with."""
    if spec.get('rerun'):
        return []                    # reruns copy their original settings
    task_study = {'doorkey_8x8': 'doorkey', 'multiroom_n6': 'multiroom',
                  'keycorridor_s3r3': 'keycorridor'}[spec['task']]
    if spec['study'] in (SPECS['doorkey_controls']['study'],
                         SPECS['doorkey_online']['study']):
        from scripts import run_rule_bank_pilot_20260928 as pilot
        return ([dict(c, bonus='none') for c in pilot.cells()]
                + cells(SPECS['doorkey_count']))
    if spec['study'] in (SPECS['multiroom_controls']['study'],
                         SPECS['multiroom_online']['study']):
        from scripts import run_rule_bank_multiroom_20260928 as multiroom
        return multiroom.cells()
    if spec['study'] == SPECS[f'{task_study}_online']['study']:
        return cells(SPECS[task_study])
    return []


def paired_cells(reference, built, allowed=BANK_KEYS):
    """Each control cell must equal its seed's scoped cell but `allowed`."""
    scoped = {(c['bonus'], c['seed']): c['args'] for c in reference
              if c['arm'] == 'llm_rules_scoped'}
    for c in built:
        other = scoped[(c['bonus'], c['seed'])]
        if set(c['args']) != set(other) or {
                k for k in other if c['args'][k] != other[k]} - allowed:
            raise ValueError(f"Control cell {c['index']} differs from the "
                             'paired scoped cell beyond its teaching channel')


def spec_of(study):
    hits = [name for name, s in SPECS.items() if s['study'] == study]
    if len(hits) != 1:
        raise ValueError(f'Unknown study {study}')
    return SPECS[hits[0]]


# ------------------------------------------------------------------ cells

@functools.lru_cache(maxsize=None)
def reference_menu():
    """The reviewed rule-reference settings, built once per process."""
    from scripts.run_rule_reference_20260927 import menu
    return tuple(c for c in menu() if c['replicate'] == 0)


def base_args(task, bonus):
    from algos.ppo_distill import Args
    cells = {c['arm']: c['args'] for c in reference_menu()
             if c['task'] == task and c['bonus'] == bonus}
    return Args(**cells['none']), Args(**cells['random480'])


def arm_args(spec, arm, r, bonus, none, guided, root, ledger):
    from teachers.minigrid.rule_bank import file_sha256
    seed = spec['seed0'] + 100 * r
    name = f"{spec['study']}_{bonus}_{arm}"
    if arm == 'none':
        return replace(none, seed=seed, total_timesteps=HORIZON,
                       experiment_id=name)
    if arm in ONLINE_ARMS:
        calls, ceiling, schedule = ONLINE_ARMS[arm]
        ceiling = spec.get('max_output_tokens', ceiling)
        budget = online_budget(spec, root) if calls == 'bank' else calls
        args = replace(
            guided, seed=seed, total_timesteps=HORIZON,
            experiment_id=name, teacher='llm_scoped',
            teacher_model=ONLINE_MODEL, teacher_stream=False,
            advisor='unlimited', uniform_queries=True,
            query_budget=budget, advice_budget=budget,
            advisor_no_teacher_peek=False,
            importance_source='none', audit_explanations=False,
            budget_ledger=str(ledger), price_table=ONLINE_PRICES,
            **dict(ONLINE_LIMITS, max_output_tokens=ceiling))
        if spec.get('fail_fast'):
            args = replace(args, **FAIL_FAST)
        return replace(args, **SCHEDULES[schedule])
    if arm in ONLINE_RULE_ARMS:
        return replace(
            guided, seed=seed, total_timesteps=HORIZON,
            experiment_id=name, teacher='llm_rules_online',
            teacher_model=ONLINE_MODEL, teacher_stream=False,
            advisor='unlimited', uniform_queries=False, query_budget=0,
            advice_budget=0, advisor_no_teacher_peek=False,
            importance_source='none', audit_explanations=False,
            budget_ledger=str(ledger), price_table=ONLINE_PRICES,
            online_rule_calls=online_budget(spec, root),
            online_rule_blind=ONLINE_RULE_ARMS[arm], journal_paid_only=True,
            # blind prompts carry up to five full-state situations
            max_input_tokens=20_000, max_attempts=1,
            max_output_tokens=spec.get('max_output_tokens', 2048))
    bank = f"{spec['banks']}/{BANK_OF[arm].format(r=r)}.json"
    args = replace(
        guided, seed=seed, total_timesteps=HORIZON,
        experiment_id=name, teacher='rule_bank',
        teacher_stream=False, rule_bank=bank,
        rule_bank_sha256=file_sha256(Path(root) / bank),
        advisor='unlimited', query_budget=0, advice_budget=0,
        uniform_queries=False, advisor_no_teacher_peek=False,
        importance_source='none', audit_explanations=False)
    if arm in CAPPED:
        args = replace(args, uniform_queries=True, query_budget=CAPPED[arm],
                       advice_budget=CAPPED[arm])
    if spec.get('summary_only'):       # no per-step journal rows
        args = replace(args, offline_summary_only=True)
    return args


def order(spec):
    reps, bonuses, arms = replicates(spec), spec['bonuses'], arms_of(spec)
    if spec.get('seed_major'):
        return [(r, b, a) for r in reps for b in bonuses for a in arms]
    return [(r, b, a) for b in bonuses for r in reps for a in arms]


def cells(spec, root=ROOT, ledger=''):
    from algos.ppo_distill import Args
    import tyro
    if spec.get('rerun') == 'complete':
        return completion_cells(root, spec['arms'])
    if spec.get('rerun'):
        return rerun_cells(root)
    base = {b: base_args(spec['task'], b) for b in spec['bonuses']}
    out, parsed = [], set()
    for r, bonus, arm in order(spec):
        args = arm_args(spec, arm, r, bonus, *base[bonus], root, ledger)
        # Cells of one arm and student differ only in seed and bank file,
        # so one trainer-parser round trip per arm and student suffices.
        if (bonus, arm) not in parsed:
            if asdict(tyro.cli(Args, args=trainer_argv(asdict(args)),
                               console_outputs=False)) != asdict(args):
                raise ValueError('Trainer command changed settings')
            parsed.add((bonus, arm))
        out.append(dict(index=len(out), bonus=bonus, arm=arm,
                        seed=args.seed, trainer='algos.ppo_distill',
                        args=asdict(args)))
    return out


def fresh_pairs(spec, built, root=ROOT):
    """Every guided cell equals its seed's scoped cell but its channel."""
    base = {b: base_args(spec['task'], b) for b in spec['bonuses']}
    for c in built:
        if c['arm'] == 'none':
            continue
        r = (c['seed'] - spec['seed0']) // 100
        scoped = asdict(arm_args(spec, 'llm_rules_scoped', r, c['bonus'],
                                 *base[c['bonus']], root, ''))
        differ = {k for k in scoped if c['args'][k] != scoped[k]}
        if set(c['args']) != set(scoped) or differ - allowed_keys(c['arm']):
            raise ValueError(f"Cell {c['index']} ({c['arm']}) differs from "
                             f'its scoped cell in {sorted(differ)}')


# ------------------------------------------------------------ preparation

def batch_dir(root, spec):
    return Path(root) / 'results/efficiency' / spec['study']


EXPECTED_USD_PER_CALL = .0025   # measured: $0.0018-0.0022 per consultation


def paid_admission(spec, root, ledger, credential, built=None):
    """Refuse a paid study the pool or the credential cannot cover.

    Each run reserves its own worst case when it starts and settles when
    it ends, so a throttled array only ever holds `throttle` reservations
    at once. The pool must cover that peak (all runs when unthrottled).
    """
    from scripts.run_advising_strength_grid import (backend_environment,
                                                    check_pool)
    from teachers.budget import PriceTable
    if not (ledger and credential):
        raise ValueError('A paid study needs --ledger and --credential-file')
    backend_environment('openai', credential)     # presence only
    prices = PriceTable.load(str(Path(root) / ONLINE_PRICES))
    built = built if built is not None else cells(spec, root, str(ledger))
    runs = sorted((prices.call_bound(ONLINE_MODEL, a['max_input_tokens'],
                                     a['max_output_tokens'])
                   * a['max_attempts'] * paid_calls(a))
                  for a in (c['args'] for c in built)
                  if paid_calls(a))[::-1]
    peak = sum(runs[:spec.get('throttle') or len(runs)])
    check_pool(ledger, peak)
    calls = sum(paid_calls(c['args']) for c in built)
    return dict(ledger=str(Path(ledger).resolve()),
                credential_file=str(Path(credential).resolve()),
                worst_case_usd=sum(runs), peak_reserved_usd=peak,
                throttle=spec.get('throttle'), calls=calls,
                expected_usd=calls * EXPECTED_USD_PER_CALL)


def prepare(root, spec, ledger=None, credential=None):
    root = Path(root).resolve()
    batch = batch_dir(root, spec)
    if batch.exists():
        verify(batch)
        return batch
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                   text=True).strip()
    data = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         head], cwd=root)
    # A rerun batch copies cells found in the LAUNCHED batches, so it scans
    # the real results tree, once; the frozen code archive has no results.
    reruns = cells(spec, root) if spec.get('rerun') else None
    if reruns is not None and not reruns:
        raise ValueError('No paid cell lost calls; nothing to rerun')
    paid = paid_admission(spec, root, ledger, credential, reruns) \
        if spec.get('paid') else None
    ledger = paid['ledger'] if paid else ''
    allowed = PAID_KEYS if paid else BANK_KEYS
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        if not set(required(spec)) <= set(zipped.namelist()):
            raise ValueError('Commit the complete packet (with banks) first')
        # Fingerprint the archive in memory: the extracted files are the
        # same bytes, and re-reading them is slow on a shared filesystem.
        hashes = {n: hashlib.sha256(zipped.read(n)).hexdigest()
                  for n in zipped.namelist() if not n.endswith('/')}
        reference = reference_cells(spec)
        paired_cells(reference, cells(spec, root, ledger) if reference
                     else [], allowed)
        if spec.get('seed_major'):
            fresh_pairs(spec, cells(spec, root, ledger), root)
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    built = reruns if reruns is not None else cells(spec, batch / 'code',
                                                    ledger)
    if not built:
        raise ValueError('A batch needs at least one cell')
    paired_cells(reference, built if reference else [], allowed)
    if spec.get('seed_major'):
        fresh_pairs(spec, built, batch / 'code')
    manifest = dict(study=spec['study'], spec=spec, head=head,
                    runtime=runtime_identity(), cells=built, paid=paid,
                    api_calls_in_training=paid['calls'] if paid else 0,
                    source_hashes=hashes)
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(head + '\n')
    verify(batch, sources=False)       # every worker re-checks all sources
    return batch


def verify(batch, sources=True):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    if (digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or manifest['spec'] != json.loads(json.dumps(
                spec_of(manifest['study'])))
            or manifest['study'] != batch.name
            or (batch / 'READY').read_text().strip() != manifest['head']):
        raise ValueError('Frozen preparation identity differs')
    for name, expected in manifest['source_hashes'].items() if sources \
            else ():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    return manifest


def run_cell(batch, index):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute the archived source')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker environment differs from preparation')
    record = manifest['cells'][index]
    args = record['args']
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    write(directory / 'dispatch.json', dict(
        cell=record, manifest_sha256=digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        runs_root = ROOT / 'results/runs'
        if runs_root.exists() and list(runs_root.glob(
                f"*{args['experiment_id']}*__{args['seed']}__*")):
            raise FileExistsError('An earlier attempt exists for this cell')
        env = None
        if manifest.get('paid'):
            from scripts.run_advising_strength_grid import \
                backend_environment
            env = backend_environment(
                'openai', manifest['paid']['credential_file'])
        process = subprocess.run([sys.executable, '-u', '-m',
                                  record['trainer'], *trainer_argv(args)],
                                 cwd=ROOT, env=env)
        outcome['returncode'] = process.returncode
        run = find_run(runs_root, args)
        outcome['runs'] = [run.relative_to(batch).as_posix()]
        if process.returncode == 0:
            outcome['metrics'] = validate_run(run, args)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit(batch):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    submitted = batch / 'SUBMITTED_JOB'
    if submitted.exists():
        print(f'Already submitted: {submitted.read_text().strip()}')
        return
    n = len(manifest['cells'])
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write(f'One array, {n} cells, throttle '
                     f"{manifest['spec'].get('throttle')}.\n")
    throttle = manifest['spec'].get('throttle')
    command = ['sbatch', '--parsable',
               f"--job-name=rb_{manifest['spec']['task'][:10]}",
               f'--array=0-{n - 1}' + (f'%{throttle}' if throttle else ''),
               '--time=3-00:00:00',
               f'--output={batch}/slurm/%x_%A_%a.out',
               str(batch / 'code/scripts/submit_rule_bank_study.sh'),
               str(batch)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f"{manifest['study']}: {job} ({n} cells)")


# ----------------------------------------------------------------- report

def report(root, spec, batch=None):
    batch = Path(batch or batch_dir(root, spec))
    manifest = verify(batch)
    by = {}
    for record in manifest['cells']:
        try:
            run = find_run(batch / 'code/results/runs', record['args'])
            by.setdefault((record['bonus'], record['arm']), {})[
                record['seed']] = validate_run(run, record['args'])
        except (FileNotFoundError, ValueError, KeyError):
            continue
    pairs = [('llm_rules_blind_strict', 'llm_action_replay'),
             ('llm_rules_blind_strict', 'llm_rules_shuffled'),
             ('llm_rules_blind_strict', 'llm_rules_random_subset'),
             ('llm_rules_blind_strict', 'llm_rules_scoped'),
             ('llm_rules_blind_strict', 'none'),
             ('llm_rules_scoped', 'llm_action_replay'),
             ('llm_rules_scoped', 'none'),
             ('llm_action_replay', 'none'),
             ('llm_rules_blind_strict', 'llm_action_predicate_replay'),
             ('llm_rules_scoped', 'llm_action_predicate_replay'),
             ('llm_action_predicate_replay', 'none')]
    result = dict(study=spec['study'], students={})
    for bonus in spec['bonuses']:
        arms = {arm: dict(n=len(rows), mean_auc=float(np.mean(
            [m['auc'] for m in rows.values()])), mean_labels=float(np.mean(
                [m['labels_delivered'] for m in rows.values()])))
            for (b, arm), rows in by.items() if b == bonus}
        contrasts = {f'{a} - {c}': paired(by.get((bonus, a), {}),
                                          by.get((bonus, c), {}),
                                          f'{a} - {c}') for a, c in pairs}

        def passes(*labels):
            return all(contrasts[k]['n'] == N_SEEDS
                       and contrasts[k]['mean'] > 0
                       and contrasts[k]['positive'] >= 4 for k in labels)
        result['students'][bonus] = dict(
            arms=arms, contrasts=list(contrasts.values()),
            verified_explanation_beyond_replay=passes(
                'llm_rules_blind_strict - llm_action_replay',
                'llm_rules_blind_strict - llm_rules_shuffled'),
            verification_beyond_pruning=passes(
                'llm_rules_blind_strict - llm_rules_random_subset'))
    result['inference'] = 'development study; five seeds per student'
    print(json.dumps(result, indent=2))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('check', 'prepare', 'launch',
                                        'report', 'run-cell'))
    cli.add_argument('--spec', choices=sorted(SPECS))
    cli.add_argument('--batch', type=Path)
    cli.add_argument('--index', type=int)
    cli.add_argument('--ledger', type=Path,
                     help='paid studies: the shared budget ledger')
    cli.add_argument('--credential-file', type=Path,
                     help='paid studies: dotenv file with OPENAI_API_KEY')
    args = cli.parse_args()
    if args.action == 'run-cell':
        return run_cell(args.batch, args.index)
    spec = SPECS[args.spec]
    if args.action == 'check':
        built = cells(spec)
        if spec.get('seed_major'):
            fresh_pairs(spec, built)
        if spec.get('rerun'):
            # Originals still running spend on runs the report never uses.
            for c in built:
                cell = (batch_dir(ROOT, spec_of(c['origin']['study'])) /
                        'cells' / str(c['origin']['index']))
                if not (cell / 'exit.json').exists() and \
                        (cell / 'dispatch.json').exists() and job_active(
                            read(cell / 'dispatch.json')['slurm_job_id']):
                    job = read(cell / 'dispatch.json')['slurm_job_id']
                    print(f"scancel {job}   # {c['origin']['study'][10:]} "
                          f"cell {c['origin']['index']}, "
                          f"{c.get('lost_calls', '?')} lost calls, "
                          'still running')
        paid = sorted({paid_calls(c['args']) for c in built} - {0})
        calls = f'; {paid} LLM calls per run' if paid else ''
        print(f"PASS {spec['study']}: {len(built)} resolved commands; "
              f"students {sorted({c['bonus'] for c in built})}; arms "
              f"{sorted({c['arm'] for c in built})}{calls}")
        return 0
    if args.action == 'report':
        report(ROOT, spec, args.batch)
        return 0
    batch = prepare(ROOT, spec, args.ledger, args.credential_file)
    print(f'Prepared {batch}')
    if args.action == 'launch':
        submit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
