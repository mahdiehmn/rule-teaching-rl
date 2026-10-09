"""Check optional early measurements through the real recurrent trainer."""

import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from monitoring.evaluation_diagnostics import (
    diagnostic_milestones, isolated_evaluation_rng,
)
from teachers.minigrid.rule_bank import file_sha256
from tests.test_advice_rng_isolation import run_training


def test_milestones_round_up_merge_and_keep_zero():
    assert diagnostic_milestones(
        '0,10000,25000,50000,100000,200000', 1024, 4882,
    ) == {0: [0], 10: [10000], 25: [25000], 49: [50000],
          98: [100000], 196: [200000]}
    assert diagnostic_milestones('1,1,1024', 1024, 1) == {1: [1, 1024]}
    assert diagnostic_milestones('', 1024, 1) == {}


@pytest.mark.parametrize('spec', ['-1', '1025', 'invalid'])
def test_invalid_milestones_fail_before_training(spec):
    with pytest.raises(ValueError):
        diagnostic_milestones(spec, 1024, 1)


def test_rng_restored_on_exception():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with isolated_evaluation_rng(19):
            random.random()
            np.random.random()
            torch.rand(5)
            raise RuntimeError('Injected evaluation failure')
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    assert np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('guidance', [False, True])
def test_extra_evaluations_leave_real_training_identical(
        tmp_path, monkeypatch, guidance):
    # Tiny offline runs check software, not scientific seed evidence.
    # The helper records real actions, observations, recurrent states,
    # gradients' resulting weights, Adam state and teacher counts.
    evaluating = False
    original_eval = ppo.greedy_eval
    original_step = ppo.optim.Adam.step
    original_advisor = ppo.make_advisor
    original_build_env = ppo.build_env

    def checked_eval(*args, **kwargs):
        nonlocal evaluating
        evaluating = True
        try:
            return original_eval(*args, **kwargs)
        finally:
            evaluating = False

    def checked_step(*args, **kwargs):
        assert not evaluating, 'Evaluation performed an optimizer step'
        return original_step(*args, **kwargs)

    def checked_advisor(*args, **kwargs):
        advisor = original_advisor(*args, **kwargs)
        original_advise = advisor.advise

        def checked_advise(*values, **options):
            assert not evaluating, 'Evaluation requested teacher advice'
            return original_advise(*values, **options)

        advisor.advise = checked_advise
        return advisor

    monkeypatch.setattr(ppo, 'greedy_eval', checked_eval)
    monkeypatch.setattr(ppo.optim.Adam, 'step', checked_step)
    monkeypatch.setattr(ppo, 'make_advisor', checked_advisor)
    # Cap only test evaluation episodes; training uses make_thunk.
    # Real observations, policy inference, resets and env steps remain.
    def short_eval_env(*args, **kwargs):
        return gym.wrappers.TimeLimit(
            original_build_env(*args, **kwargs), max_episode_steps=8)

    monkeypatch.setattr(ppo, 'build_env', short_eval_env)
    settings = dict(
        guidance=guidance, uniform_queries=False, query_budget=0,
        advisor='unlimited', advisor_rng_isolation=True,
        eval_interval=4, eval_episodes=1, eval_sampled=True)
    if guidance:
        bank = (Path(__file__).resolve().parents[1]
                / 'research/rule_banks/v3_20260928/blind_strict.json')
        settings.update(teacher='rule_bank', rule_bank=str(bank),
                        rule_bank_sha256=file_sha256(bank))
    baseline = run_training(
        tmp_path / 'baseline', monkeypatch, 9917500, **settings)
    measured = run_training(
        tmp_path / 'measured', monkeypatch, 9917500,
        diagnostic_eval_frames='0,1,129,512,1024', **settings)
    for key in ('initial', 'history', 'samples', 'minibatches',
                'permutations', 'distill_masks'):
        assert measured[key] == baseline[key], key
    assert measured['final'].keys() == baseline['final'].keys()
    for key in measured['final']:
        assert torch.equal(measured['final'][key], baseline['final'][key])

    # AUC inputs retain exactly the original points and outcomes.
    def regular(run):
        return [{k: v for k, v in row.items() if k != 'wall_time_sec'}
                for row in rows(run['path'] / 'evaluations.jsonl')]

    assert regular(measured) == regular(baseline)
    assert [row['global_step'] for row in regular(measured)] == [512, 1024]
    assert not (baseline['path'] / 'diagnostic_evaluations.jsonl').exists()
    extra = rows(measured['path'] / 'diagnostic_evaluations.jsonl')
    assert [row['global_step'] for row in extra] == [0, 128, 256, 512, 1024]
    assert [row['requested_frames'] for row in extra] == [
        [0], [1], [129], [512], [1024]]
    assert extra[0]['training_episodes'] == 0
    assert extra[0]['policy_sha256_before'] == measured['initial'].strip()
    assert all(row['teacher_on'] is False for row in extra)
    assert all(row['policy_sha256_before'] == row['policy_sha256_after']
               for row in extra)
    assert [row['reused_regular_evaluation'] for row in extra] == [
        False, False, False, True, True]
    assert [row['rng_isolated'] for row in extra] == [
        True, True, True, False, False]
    for row in extra[-2:]:
        matching = next(r for r in regular(measured)
                        if r['global_step'] == row['global_step'])
        assert row['sampled_success_rate'] == matching['sampled_success_rate']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_rng_restored():
    torch.empty(1, device='cuda')
    before = torch.cuda.get_rng_state_all()
    with isolated_evaluation_rng(23):
        for device in range(torch.cuda.device_count()):
            torch.rand(5, device=f'cuda:{device}')
    assert all(torch.equal(a, b)
               for a, b in zip(before, torch.cuda.get_rng_state_all()))
