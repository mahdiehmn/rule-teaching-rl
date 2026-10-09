"""
Aliased labels must be consistent, not random.

The distinction carries the whole experiment. Noise redrawn on every
visit averages out under repetition, so a student that sees a state
often still learns its majority action. Aliasing is a fixed function of
unobservable state: the same hidden configuration always yields the same
wrong label, and no amount of revisiting resolves it. These tests pin
that property, the rate, and the fact that zero leaves every previous
experiment untouched.
"""

from dataclasses import replace

import numpy as np
import pytest

from envs.registry import build_env
from envs.state import extract_generic_state
from teachers.base import Advice, Cost
from teachers.controlled_advice import aliased_advice


@pytest.fixture
def env():
    made = build_env('doorkey_8x8', seed=0, obs_mode='symbolic',
                     agent_view_size=7)
    made.reset(seed=8_200_000)
    yield made.unwrapped
    made.close()


def advice_of(action=2):
    return Advice(action=action, confidence=1.0, cost=Cost(metadata={}))


def test_zero_rate_changes_nothing(env):
    original = advice_of()
    assert aliased_advice(original, env, 7, 0.0) is original


def test_full_rate_always_redirects_and_never_returns_the_teacher_action(env):
    out = aliased_advice(advice_of(2), env, 7, 1.0)
    assert out.action != 2
    assert 0 <= out.action < 7
    assert out.cost.metadata['aliased'] is True
    assert out.cost.metadata['teacher_action_before_alias'] == 2


def test_the_same_hidden_state_always_gives_the_same_label(env):
    """Consistency is what separates aliasing from noise."""
    first = aliased_advice(advice_of(2), env, 7, 1.0)
    for _ in range(20):
        assert aliased_advice(advice_of(2), env, 7, 1.0).action == first.action


def walk(env, episodes=40, steps=25, seed=0):
    """Visit many genuinely different states.

    A lazy walk is not enough: DoorKey starts the agent in a corner,
    and a policy that only turns and pushes forward stays pinned there
    for hundreds of steps, visiting two states. Fresh layouts plus
    uniform random actions reach a few hundred.
    """
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        env.reset(seed=8_200_000 + 37 * episode)
        for _ in range(steps):
            yield env
            _obs, _r, terminated, truncated, _info = env.step(
                int(rng.integers(env.action_space.n)))
            if terminated or truncated:
                break


def test_redirection_depends_on_state_not_on_the_call(env):
    """A fixed key would send every state to the same wrong action."""
    actions = {aliased_advice(advice_of(2), state, 7, 1.0).action
               for state in walk(env, episodes=8)}
    assert len(actions) > 1


@pytest.mark.parametrize('rate', [0.1, 0.3])
def test_measured_rate_tracks_the_requested_rate(env, rate):
    """The share of corrupted states should be the rate we asked for."""
    decided = {}
    for state in walk(env):
        out = aliased_advice(advice_of(2), state, 7, rate)
        decided[extract_generic_state(state)] = bool(
            out.cost.metadata.get('aliased'))
    observed = sum(decided.values()) / len(decided)
    assert len(decided) > 100, f'only {len(decided)} distinct states sampled'
    assert abs(observed - rate) < 0.06, f'{observed:.3f} versus {rate}'


def test_one_state_keeps_one_verdict_however_often_it_is_met(env):
    """Aliasing is a function of the state, so revisits never disagree."""
    verdicts = {}
    for state in walk(env, seed=1):
        key = extract_generic_state(state)
        out = aliased_advice(advice_of(2), state, 7, 0.3)
        seen = (bool(out.cost.metadata.get('aliased')), out.action)
        assert verdicts.setdefault(key, seen) == seen


def test_abstentions_pass_through_untouched(env):
    abstained = replace(advice_of(), action=None)
    assert aliased_advice(abstained, env, 7, 1.0) is abstained
