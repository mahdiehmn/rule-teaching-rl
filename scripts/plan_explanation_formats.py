"""Resolve the proposed explanation-format, access and timing menus.

Menus frozen by
research/explanation_formats_protocol_2026-09-25.md; the launcher
scripts/run_explanation_formats_20260925.py prepares and runs them.
`--check` resolves every cell and confirms it survives the trainer
command line.

Primary format menu (120 cells), full-state (privileged) explanation
writer: DoorKey-8x8, local symbolic GRU, no-bonus and count-bonus PPO,
five development seeds; per seed and background: PPO, action-only, and
each of the five formats (plain, contrastive, subgoal, consequence, plan)
as explanation-only with aligned and bundle-permuted targets at the
frozen calibrated scale. Every shared setting repeats the
explanation-isolation study.

Access comparison (50 cells, opt-in): the same five formats, aligned,
with the local-only writer's bank, on the same seeds, backgrounds, heads,
scales and schedule as the primary aligned cells; only the bank and its
access label differ.

Timing menu (30 cells, cached lessons, no new collection): PPO,
action-only early (iterations 4..4,880) and late (2,445..7,321): 1,220
updates each at stride 4, equally long, both before the teacher-free
final quarter.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import tyro

from algos.ppo_lesson_formats import Args, lesson_schedule
from scripts.explanation_formats import FORMATS
from scripts.run_explanation_grid import trainer_argv

SEEDS = [12_300_000 + 100 * i for i in range(5)]
BACKGROUNDS = ('none', 'count')
STUDY = 'explanation_formats_20260925'
TIMING_STUDY = 'explanation_timing_20260925'
ISOLATION_SHARED = dict(
    task='doorkey_8x8', obs_mode='symbolic', agent_view_size=7,
    recurrent=True, dual_value=True, guidance=False, teacher='oracle',
    teacher_model='', cuda=False, torch_deterministic=True,
    total_timesteps=10_000_000, learning_rate=0.00025, num_envs=8,
    num_steps=128, num_minibatches=4, update_epochs=4, anneal_lr=True,
    gamma=0.999, gae_lambda=0.95, norm_adv=True, clip_coef=0.2,
    clip_vloss=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    int_gamma=0.99, int_coef=1.0, norm_int_reward=True,
    count_observation='policy', eval_interval=50, eval_episodes=50,
    eval_sampled=False, record_initial_policy=True, query_budget=0,
    budget_ledger='', explanation='none', consequence='none',
    advice_replay=False, lesson_batch=32,
)
ITERATIONS = 10_000_000 // (8 * 128)          # 9,765
LATE_OFFSET = ITERATIONS // 4                  # 2,441: starts after 25%
TIMING_EXPOSURES = 1_220


def cell(cells, seed, bonus, condition, study, **values):
    args = Args(**ISOLATION_SHARED, seed=seed, bonus=bonus,
                lesson_every=4,
                experiment_id=f'{study}_{bonus}_{condition}', **values)
    cells.append(dict(index=len(cells), seed=seed, bonus=bonus,
                      condition=condition,
                      trainer='algos.ppo_lesson_formats', args=asdict(args)))


def format_menu(bank, bank_sha256, scales):
    """Primary: full-state (privileged) explanation writer."""
    cells = []
    for seed in SEEDS:
        for bonus in BACKGROUNDS:
            common = dict(format_bank=bank, format_bank_sha256=bank_sha256,
                          format_access='full_state', lesson_offset=0,
                          lesson_exposures=1830)
            cell(cells, seed, bonus, 'ppo', STUDY, lesson_mode='ppo',
                 **common)
            cell(cells, seed, bonus, 'actions', STUDY,
                 lesson_mode='actions', lesson_action_coef=1.0, **common)
            for fmt in FORMATS:
                for targets in ('aligned', 'permuted'):
                    cell(cells, seed, bonus, f'{fmt}_{targets}', STUDY,
                         lesson_mode='explanation', lesson_format=fmt,
                         lesson_targets=targets, lesson_scale=scales[fmt],
                         **common)
    return cells


def access_menu(local_bank, local_sha256, scales):
    """Opt-in comparison: local-only writer, same everything else."""
    cells = []
    for seed in SEEDS:
        for bonus in BACKGROUNDS:
            for fmt in FORMATS:
                cell(cells, seed, bonus, f'{fmt}_aligned_local', STUDY,
                     format_bank=local_bank,
                     format_bank_sha256=local_sha256,
                     format_access='local_only', lesson_offset=0,
                     lesson_exposures=1830, lesson_mode='explanation',
                     lesson_format=fmt, lesson_targets='aligned',
                     lesson_scale=scales[fmt])
    return cells


def timing_menu(lessons, lessons_sha256):
    """Cached categorical lessons serve as the bank for ppo/actions."""
    arms = (('ppo', 'ppo', 0), ('actions_early', 'actions', 0),
            ('actions_late', 'actions', LATE_OFFSET))
    cells = []
    for seed in SEEDS:
        for bonus in BACKGROUNDS:
            for condition, mode, offset in arms:
                cell(cells, seed, bonus, condition, TIMING_STUDY,
                     format_bank=lessons, format_bank_sha256=lessons_sha256,
                     lesson_mode=mode,
                     lesson_action_coef=1.0 if mode == 'actions' else 0.0,
                     lesson_offset=offset,
                     lesson_exposures=TIMING_EXPOSURES)
    return cells


def schedule_summary():
    early = lesson_schedule(ITERATIONS, 0, 4, TIMING_EXPOSURES)
    late = lesson_schedule(ITERATIONS, LATE_OFFSET, 4, TIMING_EXPOSURES)
    return dict(early=[early[0], early[-1]], late=[late[0], late[-1]],
                teacher_free_from=int(ITERATIONS * 0.75),
                equal_length=early[-1] - early[0] == late[-1] - late[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', default='corpora/format_bank_full.json')
    parser.add_argument('--bank-sha256', default='0' * 64)
    parser.add_argument('--bank-local',
                        default='corpora/format_bank_local.json')
    parser.add_argument('--bank-local-sha256', default='1' * 64)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--lessons', default='corpora/original.json')
    parser.add_argument('--lessons-sha256', default='0' * 64)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    scales = ({f: 1.0 for f in FORMATS} if args.calibration is None else
              json.loads(args.calibration.read_text())['primary_scale'])
    primary = format_menu(args.bank, args.bank_sha256, scales)
    access = access_menu(args.bank_local, args.bank_local_sha256, scales)
    timing = timing_menu(args.lessons, args.lessons_sha256)
    print(f'primary (full-state) menu: {len(primary)} cells; access '
          f'comparison (opt-in): {len(access)} cells; timing: '
          f'{len(timing)} cells; schedules {schedule_summary()}')
    if args.calibration is None:
        print('NOTE: scales are placeholders (1.0) until a frozen '
              'calibration file is supplied.')
    if args.check:
        for c in primary + access + timing:
            parsed = tyro.cli(Args, args=trainer_argv(c['args']))
            if asdict(parsed) != c['args']:
                raise ValueError('CLI changed a resolved setting')
        print('All cells survive the trainer CLI. Nothing prepared or '
              'submitted; use scripts/run_explanation_formats_20260925.py.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
