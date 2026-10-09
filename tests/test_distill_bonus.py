"""
Verify bonus boundaries, teacher removal, and the four training arms.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algos import ppo_distill
from algos.distill_bonus import CountExploration, intrinsic_gae
from algos.intrinsic import CountBonus
from envs.registry import build_env
from monitoring.coverage import TrainingCoverage
from scripts.run_guidance_bonus_s3r3 import config


def test_factorial_configs_differ_only_in_interventions_and_seed():
    common = None
    seen = set()
    for index in range(20):
        row = asdict(config(index))
        seen.add((row.pop('guidance'), row.pop('bonus'), row.pop('seed')))
        if common is None:
            common = row
        assert row == common
    assert len(seen) == 20


def test_intrinsic_gae_has_hand_computed_continuing_targets():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    advantages, returns = intrinsic_gae(
        rewards, values, torch.tensor([4.0]), 0.5, 1.0)
    # The final target includes the bootstrap; lambda=1 then gives
    # exact discounted returns throughout this short sequence.
    torch.testing.assert_close(returns, torch.tensor([[3.0], [4.0]]))
    torch.testing.assert_close(advantages, returns - values)


def test_count_matches_existing_bonus_and_survives_rollout_boundary():
    args = ppo_distill.Args(bonus='count', norm_int_reward=False)
    enabled = CountExploration((1, 1, 1), 1, 'cpu', args)
    control = CountExploration((1, 1, 1), 1, 'cpu',
                               ppo_distill.Args())
    reference = CountBonus((1, 1, 1), 1, 'cpu')
    obs = torch.zeros(3, 1, 1, 1, 1)
    actions = torch.zeros(3, 1)
    starts = torch.tensor([[1.0], [0.0], [0.0]])
    for boundaries in (starts, torch.zeros_like(starts), starts):
        reward, stats = enabled.rollout(obs, obs, actions, boundaries)
        expected = reference.rollout_bonus(obs, obs, actions, boundaries)
        zero, control_stats = control.rollout(obs, obs, actions, boundaries)
        torch.testing.assert_close(reward, expected)
        assert torch.count_nonzero(zero) == 0
        assert stats['first_visit_frac'] == control_stats['first_visit_frac']
    np.testing.assert_allclose(reward[:, 0], [1, 1 / np.sqrt(2),
                                            1 / np.sqrt(3)])


def test_dual_head_preserves_original_policy_initialization():
    env = build_env('keycorridor_s3r3', seed=0, obs_mode='symbolic')
    specs = SimpleNamespace(single_observation_space=env.observation_space,
                            single_action_space=env.action_space)
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        old = ppo_distill.Agent(specs, symbolic=True, recurrent=True)
        torch.manual_seed(0)
        dual = ppo_distill.Agent(specs, symbolic=True, recurrent=True,
                                 dual_value=True)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, dual.state_dict()[name])
        obs, _ = env.reset(seed=0)
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        state = dual.initial_core_state(1, 'cpu')
        start = torch.ones(1)
        assert dual.get_value(x, state, start).shape == (1, 2)
        assert old.get_value(x, state, start).shape == (1, 1)
    finally:
        env.close()


def test_coverage_ignores_autoreset_and_preserves_completed_episodes():
    door = SimpleNamespace(type='door', is_locked=True)
    env = SimpleNamespace(
        width=2, height=1, agent_pos=(0, 0), carrying=None,
        grid=SimpleNamespace(get=lambda x, y: door if x == 1 else None))
    env.unwrapped = env
    coverage = TrainingCoverage([env])
    env.agent_pos = (1, 0)
    env.carrying = SimpleNamespace(type='key')
    door.is_locked = False
    coverage.step([False], [True])
    assert coverage.stats() == {
        'completed_episodes': 1, 'key_episode_fraction': 1.0,
        'unlock_episode_fraction': 1.0, 'mean_unique_positions': 2.0,
    }
    env.agent_pos = (0, 0)
    env.carrying = None
    door.is_locked = True
    coverage.step([True], [False])
    coverage.step([False], [True])
    assert coverage.stats()['completed_episodes'] == 2
    assert coverage.stats()['key_episode_fraction'] == 0.5


@pytest.mark.parametrize('guidance,bonus,dual', [
    (False, 'none', True), (True, 'none', True),
    (False, 'count', True), (True, 'count', True),
    (True, 'none', False),
])
def test_training_arm_smoke(tmp_path, monkeypatch, guidance, bonus, dual):
    """
    Exercise rollout, update, teacher cutoff, evaluation and saving.
    """

    # Redirect only the output root; real environments and training
    # code still run. This keeps test checkpoints out of research runs.
    monkeypatch.setattr(ppo_distill, '__file__',
                        str(tmp_path / 'algos' / 'ppo_distill.py'))
    original_agent = ppo_distill.Agent
    initial = {}

    def capture_initial(*args, **kwargs):
        agent = original_agent(*args, **kwargs)
        if agent.critic_int is not None:
            initial['critic_int.weight'] = (
                agent.critic_int.weight.detach().clone())
        return agent

    monkeypatch.setattr(ppo_distill, 'Agent', capture_initial)
    if not guidance:
        def forbidden_teacher(*args, **kwargs):
            raise AssertionError('no-guidance arm constructed a teacher')
        monkeypatch.setattr(ppo_distill, 'make_teacher', forbidden_teacher)
    torch.set_num_threads(1)
    args = ppo_distill.Args(
        task='keycorridor_s3r3', obs_mode='symbolic', recurrent=True,
        teacher='bot', guidance=guidance, bonus=bonus, dual_value=dual,
        coverage_log=True, cuda=False, total_timesteps=32,
        num_envs=2, num_steps=8, num_minibatches=1, update_epochs=1,
        eval_interval=2, eval_episodes=1, eval_sampled=False,
        distill_cutoff=0.5, gamma=0.999,
    )
    ppo_distill.train(args)
    run = next((tmp_path / 'results' / 'runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    latest = summary['latest']
    assert summary['status'] == 'completed'
    assert latest['distill_coef'] == 0.0
    assert (latest['teacher_total_queries'] > 0) == guidance
    assert np.isfinite(latest['value_loss'])
    assert np.isfinite(latest['intrinsic_value_loss'])
    assert 'eval_success_rate' in latest
    used_bonus = latest['exploration']['reward_mean_used'] > 0
    assert used_bonus == (bonus == 'count')
    saved = torch.load(run / 'agent.pt', weights_only=True)
    assert ('critic_int.weight' in saved) == dual
    if dual:
        changed = not torch.equal(saved['critic_int.weight'],
                                  initial['critic_int.weight'])
        assert changed == (bonus == 'count')


def test_invalid_bonus_configuration_fails_before_training():
    with pytest.raises(ValueError, match='requires --dual-value'):
        ppo_distill.train(ppo_distill.Args(bonus='count'))


def test_llm_general_rejects_query_interval_above_one():
    """
    llm_general keeps a per-episode history and judges an action
    'blocked' by comparing consecutive queried states, so skipping
    steps makes that comparison span the gap. The run must refuse
    rather than silently prompt on a trajectory that never happened.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, '-m', 'algos.ppo_distill',
         '--task', 'keycorridor_s3r3', '--teacher', 'llm_general',
         '--query-interval', '4', '--total-timesteps', '1024'],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert 'query_interval > 1 is unsafe for llm_general' in combined
