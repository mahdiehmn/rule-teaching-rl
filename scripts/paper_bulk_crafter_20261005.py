"""Fresh complete symbolic-Crafter progress factorial for the bulk batch.

The parent bulk runner owns preparation,
source archives, attempts, subprocess execution and Slurm submission.
This module freezes cells and validates every artifact before reporting.
"""

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from scripts import audit_crafter_learning_20261001 as audit


ROOT = Path(__file__).resolve().parents[1]
SUITE = 'crafter_fresh'
STUDY = 'paper_bulk_crafter_20261005_v1'
TIME = '12:00:00'
PROTOCOL = 'research/paper_bulk_crafter_2026-10-05.md'
SEEDS = tuple(range(141, 151))
BANK = 'research/rule_banks/crafter_v3_20261001/v3b_preconditions.json'
NOPROG = ('research/rule_banks/progress_ablation_20261002/'
          'crafter_v3b_noprogress.json')
ARMS = {
    'none': '',
    'full': BANK,
    'no_progress': NOPROG,
}
STUDENTS = ('none', 'count')
PRIMARY = (
    ('plain_full', 'plain_noprog'),
    ('count_full', 'count_noprog'),
    ('plain_full', 'plain_none'),
    ('count_full', 'count_none'),
)
SECONDARY = (
    ('plain_noprog', 'plain_none'),
    ('count_noprog', 'count_none'),
    ('count_none', 'plain_none'),
    ('count_full', 'plain_full'),
)


def frozen_args(seed, bonus, bank):
    """
    Spell out the original recipe independently of mutable defaults.
    """

    return dict(
        seed=seed, total_steps=1_000_000, num_envs=16, num_steps=128,
        learning_rate=.0003, gamma=.99, gae_lambda=.95, update_epochs=4,
        num_minibatches=4, clip_coef=.2, ent_coef=.01, vf_coef=.5,
        max_grad_norm=.5, rule_bank=bank, distill_start=.1,
        distill_min=.001, distill_decay_end=.5, distill_off=.75,
        pool_size=200, world_seed0=40_000_000, eval_every=50_000,
        eval_episodes=10, eval_cap=3000, eval_seed0=41_000_000,
        count_coef=.01 if bonus == 'count' else 0.)


def cells(root=ROOT):
    """
    Produce all sixty cells with ten paired seeds and explicit arguments.
    """

    root = Path(root)
    for bank in (BANK, NOPROG):
        if not (root / bank).is_file():
            raise FileNotFoundError(root / bank)
    result = []
    for seed in SEEDS:
        for bonus in STUDENTS:
            for arm, bank in ARMS.items():
                result.append(dict(
                    index=len(result), suite=SUITE, arm=arm,
                    replicate=seed, seed=seed, task='Crafter-Symbolic-v1',
                    bonus=bonus, trainer='algos.ppo_crafter',
                    experiment_id=f'{STUDY}_{bonus}_{arm}',
                    args=frozen_args(seed, bonus, bank)))
    return result


def _checker(expected):
    """
    Reuse strict historical metric checks without shared-global changes.
    """

    # Only the declared argument contract changes. A private module
    # instance preserves concurrent validators and historical audits.
    spec = importlib.util.spec_from_file_location(
        '_bulk_crafter_metric_audit', audit.__file__)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.expected_args = lambda arm, seed, folder: dict(expected)
    return checker


@lru_cache(maxsize=1)
def _checkpoint_schema():
    """
    Derive expected tensor shapes without changing the caller's RNG.
    """

    import torch
    from algos.ppo_crafter import Net

    with torch.random.fork_rng(devices=[]):
        state = Net().state_dict()
    return {name: (tuple(value.shape), value.dtype)
            for name, value in state.items()}


def validate(run, record):
    """
    Reject changed recipes, invalid metrics and damaged checkpoints.
    """

    run = Path(run).resolve()
    arm, seed = record['arm'], record['seed']
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unplanned Crafter cell')
    bonus, bank = record['bonus'], ARMS[arm]
    if bonus not in STUDENTS:
        raise ValueError('Unplanned Crafter student')
    expected = frozen_args(seed, bonus, bank)
    if (record['suite'] != SUITE or record['bonus'] != bonus
            or record['replicate'] != seed
            or record['task'] != 'Crafter-Symbolic-v1'
            or record['trainer'] != 'algos.ppo_crafter'
            or record['args'] != expected):
        raise ValueError('Crafter cell differs from the frozen recipe')
    expected['out'] = str(run)
    checker_arm = 'rules_weak' if bank else 'none'
    row = _checker(expected).validate_run(run, checker_arm, seed)
    summary = audit.read(run / 'run_summary.json')
    for name in ('wall_time_sec', 'sps', 'auc'):
        if not math.isfinite(summary[name]):
            raise ValueError(f'Nonfinite Crafter summary: {name}')
    count = summary['count_bonus']
    if (count['coef'] != expected['count_coef']
            or not math.isfinite(count['total'])
            or count['total'] < 0
            or (count['total'] > 0) != (bonus == 'count')
            or count['total'] > expected['count_coef'] * 999424):
        raise ValueError('Invalid count-bonus accounting')
    teacher = summary['teacher']
    if any(type(teacher[name]) is not int
           for name in ('asked', 'labelled', 'llm_calls')):
        raise ValueError('Teacher counts must be integers')
    import torch

    tensors = torch.load(run / 'final_model.pt', map_location='cpu',
                         weights_only=True)
    schema = _checkpoint_schema()
    if not isinstance(tensors, dict) or set(tensors) != set(schema):
        raise ValueError('Wrong Crafter checkpoint keys')
    for name, value in tensors.items():
        if (not isinstance(value, torch.Tensor)
                or (tuple(value.shape), value.dtype) != schema[name]
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid Crafter checkpoint tensor: {name}')
    row.update(arm=arm, seed=seed, bonus=bonus, suite=SUITE,
               count_bonus=count, api_dollars=0.,
               metric_name='mean_achievements',
               finite_checkpoint_schema_checked=True)
    return row


def _paired(values, label, role):
    """
    Summarize paired training seeds without inventing degenerate p values.
    """

    values = np.asarray(values, dtype=float)
    estimate = audit.interval(values)
    pvalue = (None if estimate['zero_empirical_variance'] else
              float(stats.ttest_1samp(values, 0).pvalue))
    return dict(contrast=label, role=role, n=len(SEEDS),
                seeds=list(SEEDS), differences=values.tolist(),
                positive=int((values > 0).sum()), p_two_sided=pvalue,
                **estimate)


def report(rows):
    """
    Withhold inference until every planned artifact has been validated.
    """

    expected = {(bonus, arm, seed) for bonus in STUDENTS
                for arm in ARMS for seed in SEEDS}
    keys = [(row['bonus'], row['arm'], row['seed']) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Require all sixty unique Crafter cells')
    # Each mean averages 20 panels of 10 episodes. Integer achievement
    # units avoid inventing variance for equal differences.
    grid = {}
    for row in rows:
        value = row['auc'] * 200
        if not math.isfinite(value) or not np.isclose(
                value, round(value), rtol=0, atol=1e-9):
            raise ValueError('Crafter metric is off its 1/200 lattice')
        student = 'plain' if row['bonus'] == 'none' else 'count'
        arm = 'noprog' if row['arm'] == 'no_progress' else row['arm']
        grid[f'{student}_{arm}', row['seed']] = int(round(value))

    def contrast(pair, role):
        a, b = pair
        differences = [(grid[a, seed] - grid[b, seed]) / 200
                       for seed in SEEDS]
        return _paired(differences, f'{a} - {b}', role)

    primary = [contrast(pair, 'primary_Holm4') for pair in PRIMARY]
    # Undefined tests consume a family position as p=1 for bookkeeping;
    # their reported p remains null and they cannot establish benefit.
    order = sorted(range(4), key=lambda i: (
        primary[i]['p_two_sided'] if primary[i]['p_two_sided'] is not None
        else 1.))
    running = 0.
    for rank, i in enumerate(order):
        raw = primary[i]['p_two_sided']
        running = max(running, min(1., (4 - rank) * (
            1. if raw is None else raw)))
        primary[i]['p_holm4'] = running if raw is not None else None
        primary[i]['positive_effect_supported'] = bool(
            raw is not None and primary[i]['mean'] > 0 and running < .05)
    secondary = [contrast(pair, 'secondary_descriptive')
                 for pair in SECONDARY]
    interaction = [(grid['count_full', seed]
                    - grid['count_noprog', seed]
                    - grid['plain_full', seed]
                    + grid['plain_noprog', seed]) / 200 for seed in SEEDS]
    return dict(
        suite=SUITE, complete=len(rows), primary=primary,
        secondary=secondary,
        interaction=_paired(interaction, 'count progress - plain progress',
                            'secondary_descriptive'),
        metric='Mean of 20 teacher-free achievement panels',
        evaluation_steps=audit.expected_steps(),
        interpretation='New training seeds; fixed bank and previously used '
        'evaluation worlds. No pooling or independent novelty clearance.',
        independent_result_review='pending')


def check(root=ROOT):
    """
    Verify the complete plan and exact progress-only bank intervention.
    """

    from scripts.progress_ablation_20261002 import ablated

    root = Path(root)
    planned = cells(root)
    if ablated(BANK, root) != audit.read(root / NOPROG):
        raise ValueError('Crafter progress bank is not the frozen ablation')
    from dataclasses import asdict
    from algos.ppo_crafter import Args

    for record in planned:
        args = dict(record['args'], out='unused-check-only')
        if asdict(Args(**args)) != args:
            raise ValueError('Crafter trainer argument schema changed')
    return dict(suite=SUITE, cells=len(planned), seeds=list(SEEDS),
                arms=list(ARMS), llm_calls=0,
                bank_sha256={name: hashlib.sha256(
                    (root / name).read_bytes().replace(b'\r\n', b'\n')
                ).hexdigest() for name in (BANK, NOPROG)})


def main():
    """
    Expose a read-only admission command for the parent bulk launcher.
    """

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('check', 'cells'))
    cli.add_argument('--root', type=Path, default=ROOT)
    args = cli.parse_args()
    result = check(args.root) if args.action == 'check' else cells(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
