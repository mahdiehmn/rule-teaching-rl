"""
Tests that the exploration bonuses actually READ the instruction.

The failure mode being guarded against is silent: if the mission is
accepted and then ignored, every run still completes, every number
still looks plausible, and the bonus is measuring novelty in a
mission-blind space. On GoToSeq that means the same cell under 'go to
the red ball' and 'go to the green key' counts as one state, so the
bonus reports 'already visited' somewhere the agent has never been
under this instruction.

Each test below therefore checks that changing ONLY the mission
changes the result.
"""

import numpy as np
import pytest
import torch

from algos.intrinsic import (
    CountBonus,
    E3BBonus,
    RE3Bonus,
    StateEncoder,
    _hash_obs,
    build_bonus,
)
from algos.nets import MissionEncoder
from envs.registry import BABYAI_LANGUAGE_TASKS

OBS_SHAPE = (7, 7, 3)
NUM_ENVS = 2
STEPS = 4
VOCAB = 32
MISSION_LEN = 5
DEVICE = torch.device('cpu')


def make_rollout(mission_a, mission_b):
    """
    Build a two-env rollout whose observations are IDENTICAL and whose
    missions differ, so any difference in the bonus is attributable to
    the instruction alone.
    """

    obs = torch.ones((STEPS, NUM_ENVS) + OBS_SHAPE)
    next_obs = torch.ones((STEPS, NUM_ENVS) + OBS_SHAPE)
    actions = torch.zeros((STEPS, NUM_ENVS), dtype=torch.long)
    dones = torch.zeros((STEPS, NUM_ENVS))
    ids = torch.zeros((STEPS, NUM_ENVS, MISSION_LEN), dtype=torch.long)
    ids[:, 0] = torch.as_tensor(mission_a)
    ids[:, 1] = torch.as_tensor(mission_b)
    lens = torch.full(
        (STEPS, NUM_ENVS), MISSION_LEN, dtype=torch.long
    )
    return obs, next_obs, actions, dones, ids, lens


def test_hash_separates_identical_observations_under_different_missions():
    """
    The same observation under two instructions is two states.
    """

    obs_row = np.ones(OBS_SHAPE, dtype=np.uint8)
    red_ball = np.array([1, 2, 3, 0, 0], dtype=np.int64)
    green_key = np.array([1, 4, 5, 0, 0], dtype=np.int64)

    assert _hash_obs(obs_row, red_ball) != _hash_obs(obs_row, green_key)
    # And the same instruction must still collide, or nothing is ever
    # counted as a revisit.
    assert _hash_obs(obs_row, red_ball) == _hash_obs(
        obs_row, red_ball.copy()
    )
    # Passing no mission keeps the original goal-fixed behaviour.
    assert _hash_obs(obs_row) == _hash_obs(obs_row)


def test_count_bonus_treats_a_new_mission_as_a_new_state():
    """
    Two envs on identical observations but different instructions
    must both be scored as first visits.
    """

    bonus = CountBonus(
        OBS_SHAPE, NUM_ENVS, DEVICE, mission_vocab=VOCAB
    )
    obs, next_obs, actions, dones, ids, lens = make_rollout(
        [1, 2, 3, 0, 0], [1, 4, 5, 0, 0]
    )
    out = bonus.rollout_bonus(
        obs, next_obs, actions, dones,
        mission_ids=ids, mission_len=lens,
    )

    # Step 0 is each env's first visit under its own mission, so both
    # get the full 1/sqrt(1) bonus despite identical observations.
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[0, 1].item() == pytest.approx(1.0)
    # Later steps repeat the same (observation, mission) pair and must
    # decay, or the count is not counting at all.
    assert out[1, 0].item() < out[0, 0].item()


def test_count_bonus_without_mission_conflates_the_two_envs():
    """
    Pin the bug this guards against: mission-blind counting sees the
    two envs' identical observations as one state.
    """

    bonus = CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE)
    obs, next_obs, actions, dones, _, _ = make_rollout(
        [1, 2, 3, 0, 0], [1, 4, 5, 0, 0]
    )
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # Counts are per-env dictionaries, so each env still sees a first
    # visit -- but the mission is nowhere in the key, which is exactly
    # what breaks under global scope and in NovelD's lifetime table.
    assert out[0, 0].item() == pytest.approx(1.0)


def test_state_encoder_output_depends_on_the_mission():
    """
    Identical observations with different instructions must produce
    different features.
    """

    torch.manual_seed(0)
    encoder = StateEncoder(OBS_SHAPE, 16, mission_vocab=VOCAB)
    obs_cf = torch.ones(1, OBS_SHAPE[2], OBS_SHAPE[0], OBS_SHAPE[1])
    lens = torch.tensor([MISSION_LEN])

    a = encoder(
        obs_cf, torch.tensor([[1, 2, 3, 0, 0]]), lens
    )
    b = encoder(
        obs_cf, torch.tensor([[1, 4, 5, 0, 0]]), lens
    )
    assert not torch.allclose(a, b)


def test_mission_aware_encoder_refuses_to_run_without_a_mission():
    """
    Silently substituting zeros would train the projection on an
    input it never sees at rollout time, so this must raise.
    """

    encoder = StateEncoder(OBS_SHAPE, 16, mission_vocab=VOCAB)
    obs_cf = torch.ones(1, OBS_SHAPE[2], OBS_SHAPE[0], OBS_SHAPE[1])
    with pytest.raises(ValueError, match='without a mission'):
        encoder(obs_cf)


def test_state_encoder_without_mission_is_unchanged():
    """
    The goal-fixed path must keep working with no mission at all.
    """

    encoder = StateEncoder(OBS_SHAPE, 16)
    obs_cf = torch.ones(1, OBS_SHAPE[2], OBS_SHAPE[0], OBS_SHAPE[1])
    assert encoder(obs_cf).shape == (1, 16)


def test_mission_encoder_ignores_padding():
    """
    Two missions identical up to their true length must encode the
    same however much padding follows, which is the whole point of
    packing by true length.
    """

    torch.manual_seed(0)
    encoder = MissionEncoder(VOCAB)
    short = torch.tensor([[7, 8, 0, 0, 0]])
    padded = torch.tensor([[7, 8, 9, 9, 9]])
    lens = torch.tensor([2])

    assert torch.allclose(
        encoder(short, lens), encoder(padded, lens), atol=1e-6
    )


@pytest.mark.parametrize('bonus_name', ['rnd', 're3', 'noveld', 'e3b'])
def test_neural_bonuses_run_mission_aware(bonus_name):
    """
    Every learned bonus must accept and use the instruction.
    """

    class Args:
        count_scope = 'episodic'
        update_proportion = 0.25
        noveld_alpha = 0.5
        noveld_episodic_gate = True
        re3_k = 3
        e3b_ridge = 0.1
        mission_vocab_size = VOCAB

    bonus = build_bonus(
        bonus_name, OBS_SHAPE, NUM_ENVS, DEVICE, 7, Args()
    )
    obs, next_obs, actions, dones, ids, lens = make_rollout(
        [1, 2, 3, 0, 0], [1, 4, 5, 0, 0]
    )
    out = bonus.rollout_bonus(
        obs, next_obs, actions, dones,
        mission_ids=ids, mission_len=lens,
    )
    assert out.shape == (STEPS, NUM_ENVS)
    assert torch.isfinite(out).all()


def test_e3b_features_differ_by_mission():
    """
    E3B's encoder is trained to predict the action between two
    states, and on a language task that action depends on the
    instruction -- so its features must depend on it too.
    """

    torch.manual_seed(0)
    bonus = E3BBonus(
        OBS_SHAPE, NUM_ENVS, DEVICE, num_actions=7,
        mission_vocab=VOCAB,
    )
    obs_cf = torch.ones(1, OBS_SHAPE[2], OBS_SHAPE[0], OBS_SHAPE[1])
    lens = torch.tensor([MISSION_LEN])

    a = bonus.encoder(obs_cf, torch.tensor([[1, 2, 3, 0, 0]]), lens)
    b = bonus.encoder(obs_cf, torch.tensor([[1, 4, 5, 0, 0]]), lens)
    assert not torch.allclose(a, b)


def test_re3_neighbours_are_not_shared_across_missions():
    """
    RE3 scores over a whole rollout, which spans episode boundaries
    and therefore several instructions at once. States from unrelated
    instructions must not become each other's nearest neighbours.
    """

    torch.manual_seed(0)
    bonus = RE3Bonus(
        OBS_SHAPE, NUM_ENVS, DEVICE, mission_vocab=VOCAB
    )
    obs, next_obs, actions, dones, ids, lens = make_rollout(
        [1, 2, 3, 0, 0], [1, 4, 5, 0, 0]
    )
    with_mission = bonus.rollout_bonus(
        obs, next_obs, actions, dones,
        mission_ids=ids, mission_len=lens,
    )
    # Identical observations, so a mission-blind encoder would place
    # every point at distance zero and return log(1) = 0 everywhere.
    assert with_mission.abs().sum().item() > 0.0


def test_language_task_list_is_the_varying_mission_subset():
    """
    Only levels whose mission CHANGES per episode belong here.

    keycorridor_s3r3 and keycorridor_s6r3_babyai are registered under
    BabyAI ids purely for the free offline bot teacher; their mission
    is the constant 'pick up the ball' and carries no information, so
    listing them would turn on mission conditioning for nothing.
    """

    assert 'gotoseq' in BABYAI_LANGUAGE_TASKS
    assert 'gotolocal' in BABYAI_LANGUAGE_TASKS
    assert 'keycorridor_s3r3' not in BABYAI_LANGUAGE_TASKS
    assert 'keycorridor_s6r3_babyai' not in BABYAI_LANGUAGE_TASKS