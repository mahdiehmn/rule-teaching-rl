"""
Native target, immutable source and matched real trainer regression checks.

No teacher requests or scientific runs are made.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from algos import ppo_lesson_formats as old
from algos import ppo_plan_repair as trainer
from scripts import explanation_formats as formats
from scripts import plan_repair_targets as repair
from scripts.explanation_screen_panel import file_hash, restore
from tests.test_explanation_formats import (
    answer_for, fake_embed, make_panel)

REAL_SOURCE = (Path(__file__).resolve().parents[2] / 'vlm-rl-bench/results'
               '/vulcan_sync/data/explanation_lessons'
               '/explanation_formats_20260925_v1')


@pytest.fixture(scope='module')
def corpus():
    panel = make_panel()
    replies = {}
    for i, case in enumerate(panel['cases']):
        answer = answer_for(case)
        if i % 2 == 0:
            # Preserve admission while inserting a deliberate target error.
            answer['plan_next_direction'] = 'left'
        replies[case['case_id']] = dict(status='valid', answer=answer)
    raw = formats.freeze_bank(panel, replies, fake_embed)
    counts = tuple(len(raw[s]) for s in ('train', 'audit'))
    return raw, panel, counts


@pytest.fixture(scope='module')
def artifacts(corpus, tmp_path_factory):
    raw, panel, counts = corpus
    directory = tmp_path_factory.mktemp('repair_bank')
    paths = {}
    for key, value in (('raw_bank', raw), ('panel', panel)):
        path = directory / f'{key}.json'
        path.write_text(formats.canonical(value), encoding='utf-8')
        paths[key] = str(path)
        paths[f'{key}_sha256'] = file_hash(path)
    corrected = repair.build_corrected_bank(
        raw, panel, paths['raw_bank_sha256'], paths['panel_sha256'], counts)
    calibration = dict(bank_sha256=paths['raw_bank_sha256'],
                       frozen_before_rl=True,
                       calibration_cases=[r['case_id'] for r in raw['train']],
                       primary_scale={'plan': 0.23})
    for key, value in (('corrected_bank', corrected),
                       ('calibration', calibration)):
        path = directory / f'{key}.json'
        path.write_text(formats.canonical(value), encoding='utf-8')
        paths[key] = str(path)
        paths[f'{key}_sha256'] = file_hash(path)
    return paths, counts


def test_native_truth_matches_independent_direction_vectors(corpus):
    raw, panel, counts = corpus
    repair.validate_sources(raw, panel, counts)
    for case in panel['cases']:
        truth = repair.native_plan_truth(case)
        env = restore(case['state'])
        try:
            if truth['next_target'] == 'none':
                continue
            positions = [(x, y) for x in range(env.width)
                         for y in range(env.height)
                         if (obj := env.grid.get(x, y)) is not None
                         and obj.type == truth['next_target']]
            delta = np.asarray(positions[0]) - env.agent_pos
            forward, right = delta @ env.dir_vec, delta @ env.right_vec
            if abs(forward) >= abs(right):
                direction = 'ahead' if forward > 0 else 'behind'
            else:
                direction = 'right' if right > 0 else 'left'
            distance = np.abs(delta).sum()
            assert truth['next_direction'] == direction
            assert truth['next_distance'] == (
                'near' if distance <= 3 else
                'medium' if distance <= 7 else 'far')
        finally:
            env.close()


def test_plan_only_repair_keeps_all_cases_and_original_donors(corpus):
    raw, panel, counts = corpus
    before = deepcopy(raw)
    corrected = repair.build_corrected_bank(raw, panel, 'raw', 'panel', counts)
    report = repair.validate_derived_bank(corrected, raw, panel,
                                          'raw', 'panel', counts)
    assert raw == before
    assert report['splits']['train']['correction']['changed_cases'] > 0
    lookup = {r['case_id']: r for r in corrected['train']}
    for split in ('train', 'audit'):
        for original, row in zip(raw[split], corrected[split]):
            reconstructed = deepcopy(row)
            reconstructed['targets']['plan'] = original['targets']['plan']
            if split == 'train':
                donor = lookup[row['permuted']['donor_id']]
                assert row['episode'] != donor['episode']
                assert row['permuted']['targets']['plan'] == \
                    donor['targets']['plan']
                reconstructed['permuted']['targets']['plan'] = \
                    original['permuted']['targets']['plan']
            assert reconstructed == original
    marginals = report['splits']['train']['marginals']
    assert marginals['corrected'] == marginals['permuted_corrected']
    assert report['splits']['audit']['by_phase_action_pair']


@pytest.mark.parametrize('mutation', [
    'observation', 'positive_action', 'split', 'episode', 'phase',
    'truth', 'checks', 'donor', 'order', 'count'])
def test_source_mutations_are_rejected(corpus, mutation):
    raw, panel, counts = deepcopy(corpus)
    row = raw['train'][0]
    if mutation == 'observation':
        row['observation'][0][0][0] = 99
    elif mutation == 'positive_action':
        row[mutation] = (row[mutation] + 1) % 7
    elif mutation == 'split':
        row[mutation] = 'audit'
    elif mutation == 'episode':
        row[mutation] = ['changed', 0, 0]
    elif mutation == 'phase':
        row[mutation] = 'reach_target'
        if panel['cases'][0]['phase'] == 'reach_target':
            row[mutation] = 'seek_key'
    elif mutation == 'truth':
        row['checks']['plan_truth']['next_target'] = 'unknown'
    elif mutation == 'checks':
        row['checks']['plan']['next_target'] = not row['checks']['plan'][
            'next_target']
    elif mutation == 'donor':
        row['permuted']['donor_id'] = raw['audit'][0]['case_id']
    elif mutation == 'order':
        raw['train'].reverse()
    else:
        raw['train'].pop()
    with pytest.raises(ValueError):
        repair.build_corrected_bank(raw, panel, 'raw', 'panel', counts)


def test_native_crosscheck_rejects_bad_phase_and_bad_primary_truth(
        corpus, monkeypatch):
    raw, panel, counts = deepcopy(corpus)
    case = panel['cases'][0]
    case['phase'] = ('seek_key' if case['phase'] == 'reach_target'
                     else 'reach_target')
    raw['train'][0]['phase'] = case['phase']
    with pytest.raises(ValueError, match='phase'):
        repair.validate_sources(raw, panel, counts)
    raw, panel, counts = corpus
    monkeypatch.setattr(formats, 'plan_truth', lambda _case: {})
    with pytest.raises(ValueError, match='native simulator'):
        repair.validate_sources(raw, panel, counts)


@pytest.mark.parametrize('mutation', ['prose', 'plan', 'permuted', 'identity'])
def test_derived_mutations_are_rejected(corpus, mutation):
    raw, panel, counts = corpus
    corrected = repair.build_corrected_bank(raw, panel, 'raw', 'panel', counts)
    row = corrected['train'][0]
    if mutation == 'prose':
        row['future_plan'] = 'Unauthorized prose correction'
    elif mutation == 'identity':
        row['positive_action'] = 6
    elif mutation == 'plan':
        row['targets']['plan']['next_target'] = 'unknown'
    else:
        row['permuted']['targets']['plan']['next_target'] = 'unknown'
    with pytest.raises(ValueError, match='exact plan-only repair'):
        repair.validate_derived_bank(corrected, raw, panel, 'raw', 'panel',
                                      counts)


def args_for(paths, arm):
    bank_key = 'raw_bank' if arm in ('ppo', 'raw') else 'corrected_bank'
    return trainer.Args(
        repair_arm=arm, raw_bank=paths['raw_bank'],
        raw_bank_sha256=paths['raw_bank_sha256'],
        repair_panel=paths['panel'], repair_panel_sha256=paths['panel_sha256'],
        repair_calibration=paths['calibration'],
        repair_calibration_sha256=paths['calibration_sha256'],
        format_bank=paths[bank_key],
        format_bank_sha256=paths[f'{bank_key}_sha256'],
        lesson_format='plan',
        lesson_mode=('ppo' if arm == 'ppo' else 'detached'
                     if arm == 'detached' else 'explanation'),
        lesson_targets=('permuted' if arm == 'permuted_corrected'
                        else 'aligned'),
        lesson_scale=0 if arm == 'ppo' else 0.23,
        advisor_rng_isolation=True,
        seed=5, total_timesteps=512, num_steps=16, update_epochs=1,
        eval_interval=0, eval_episodes=1, lesson_batch=4, lesson_every=1,
        lesson_exposures=4, experiment_id='plan_repair_engineering')


def test_parent_arg_defaults_remain_exact():
    inherited = asdict(old.Args())
    actual = asdict(trainer.Args())
    assert {k: actual[k] for k in inherited} == inherited


@pytest.mark.parametrize('field,value', [
    ('guidance', True), ('query_budget', 1), ('bonus', 'none'),
    ('lesson_action_coef', 0.1), ('lesson_scale', 0.46),
    ('lesson_format', 'plain'), ('lesson_targets', 'permuted'),
    ('lesson_mode', 'combined'), ('format_bank_sha256', 'changed')])
def test_trainer_rejects_factor_and_hash_drift(artifacts, field, value):
    paths, counts = artifacts
    args = args_for(paths, 'raw')
    setattr(args, field, value)
    with pytest.raises(ValueError):
        trainer.validate_args(args, counts)


def test_all_arms_real_training_and_exact_legacy_raw_equivalence(
        artifacts, tmp_path, monkeypatch):
    paths, counts = artifacts
    validate = trainer.validate_args
    monkeypatch.setattr(trainer, 'validate_args',
                        lambda args: validate(args, counts))
    torch.set_num_threads(1)
    results = {}
    for arm in (*trainer.ARMS, 'legacy_raw'):
        directory = tmp_path / arm
        monkeypatch.setattr(ppo, '__file__', str(directory / 'algos/x.py'))
        args = args_for(paths, 'raw' if arm == 'legacy_raw' else arm)
        auxiliary = (old.FormatLessonAuxiliary() if arm == 'legacy_raw'
                     else trainer.PlanRepairAuxiliary())
        ppo.train(args, auxiliary=auxiliary)
        run = next((directory / 'results/runs').iterdir())
        finish = repair.read_json(run / 'lesson_finished.json')
        updates = [json.loads(line) for line in
                   (run / 'lesson_updates.jsonl').read_text().splitlines()]
        results[arm] = finish, updates
        assert finish['updates'] == 4
        assert finish['replay_exposures'] == 16
        assert finish['effective_action_exposures'] == 0
        assert finish['action_integral'] == 0
        assert len(updates) == 4
        if arm not in ('ppo', 'detached'):
            assert all(row['shared_grad_norm'] > 0 for row in updates)
        if arm != 'legacy_raw':
            assert 'native_corrected' in finish['audit_plan_metrics']
    assert len({r[0]['initial_policy_sha256'] for r in results.values()}) == 1
    assert len({json.dumps([u['ids'] for u in r[1]])
                for r in results.values()}) == 1
    assert results['ppo'][0]['final_policy_sha256'] == \
        results['detached'][0]['final_policy_sha256']
    assert results['raw'][0]['final_policy_sha256'] == \
        results['legacy_raw'][0]['final_policy_sha256']
    assert results['raw'][1] == results['legacy_raw'][1]
    assert results['corrected'][0]['final_policy_sha256'] != \
        results['raw'][0]['final_policy_sha256']


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason='Saved bank unavailable')
def test_actual_bank_repairs_without_removing_any_case(tmp_path):
    result = repair.derive(REAL_SOURCE, tmp_path / 'derived')
    report = repair.read_json(result['target_report'])
    assert report['splits']['train']['cases'] == 182
    assert report['splits']['audit']['cases'] == 58
    assert report['splits']['train']['correction']['changed_cases'] == 55
    assert report['splits']['audit']['correction']['changed_cases'] == 18
    assert report['splits']['train']['permutation']['changed_cases'] == 159
    assert result['lesson_scale'] == 0.22896900710982449
    assert Counter(report['splits']['train']['donors'].values()) == Counter(
        report['splits']['train']['case_ids'])
    with pytest.raises(FileExistsError):
        repair.derive(REAL_SOURCE, tmp_path / 'derived')
