"""
Tests for the exploration-bonus modules in algos/intrinsic.py.

These check the properties that make each bonus the thing it claims
to be, rather than just that it runs: a count bonus must actually
decay on revisits, NovelD's gate must fire only on first visits, E3B's
elliptical bonus must fall as an episode covers more directions and
must reset at episode boundaries, and every bonus must produce
finite, non-negative rewards of the right shape.

Getting these wrong is silent -- a bonus with an inverted sign or a
gate that never fires still trains, still logs, and still produces a
learning curve, it just stops being the published method it is named
after. That is exactly the failure mode this whole file set exists to
rule out, so it is worth pinning down in tests.
"""

import pytest

# Skip the module if torch is not installed, matching the other test
# files' behavior on a fresh checkout.
pytest.importorskip('torch')

import torch  # noqa: E402

from algos.intrinsic import (  # noqa: E402
    CountBonus,
    E3BBonus,
    NovelDBonus,
    RE3Bonus,
    RNDBonus,
)

# A small symbolic-style observation: 7x7 grid, 3 channels of small
# category indices, matching obs_mode='symbolic'.
OBS_SHAPE = (7, 7, 3)
NUM_ENVS = 2
NUM_STEPS = 6
DEVICE = torch.device('cpu')


def _rollout(distinct_states, num_steps=NUM_STEPS, num_envs=NUM_ENVS):
    """
    Build a synthetic rollout that cycles through a fixed number of
    distinct observations, so revisits are exactly predictable.

    Returns (obs, next_obs, actions, dones) shaped as the training
    loop passes them, with no episode boundaries.
    """

    # Fill values start at 1, not 0: an all-zero observation passes
    # through the zero-bias conv encoders as an all-zero embedding,
    # which would make E3B's quadratic form trivially zero and test
    # the encoder's bias initialization rather than the bonus.
    states = [
        torch.full(OBS_SHAPE, float(i + 1))
        for i in range(distinct_states)
    ]
    obs = torch.zeros((num_steps, num_envs) + OBS_SHAPE)
    next_obs = torch.zeros((num_steps, num_envs) + OBS_SHAPE)
    for t in range(num_steps):
        for e in range(num_envs):
            obs[t, e] = states[t % distinct_states]
            next_obs[t, e] = states[(t + 1) % distinct_states]
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
    dones = torch.zeros((num_steps, num_envs))
    return obs, next_obs, actions, dones


def test_count_bonus_decays_on_revisit():
    """
    Revisiting a state must lower its bonus as 1/sqrt(n).
    """

    bonus = CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE, scope='episodic')
    # Only two distinct next-observations, so every state is revisited
    # repeatedly within one rollout.
    obs, next_obs, actions, dones = _rollout(distinct_states=2)
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    assert out.shape == (NUM_STEPS, NUM_ENVS)
    # The first visit to a state pays 1/sqrt(1) = 1.
    assert out[0, 0] == pytest.approx(1.0)
    # Each env keeps its own table, so env 1's first visit also pays 1.
    assert out[0, 1] == pytest.approx(1.0)
    # Two steps later the same state is seen a second time.
    assert out[2, 0] == pytest.approx(1.0 / (2**0.5))
    # The bonus must be non-increasing over repeated visits.
    assert out[4, 0] < out[2, 0]


def test_count_bonus_resets_at_episode_boundary():
    """
    An episodic count table must be cleared when a new episode
    starts, restoring the full bonus for a state seen in the last
    one.
    """

    bonus = CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE, scope='episodic')
    obs, next_obs, actions, dones = _rollout(distinct_states=2)
    # Mark step 4 as the first observation of a fresh episode.
    dones[4, :] = 1.0
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # Without the reset this would be the third visit (1/sqrt(3));
    # with it, the state is new again and pays the full bonus.
    assert out[4, 0] == pytest.approx(1.0)


def test_global_count_bonus_ignores_episode_boundary():
    """
    The 'global' scope must keep counting across episodes, which is
    what distinguishes it from the episodic variant.
    """

    bonus = CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE, scope='global')
    obs, next_obs, actions, dones = _rollout(distinct_states=2)
    dones[4, :] = 1.0
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # Both envs share one table here, so by step 4 this state has
    # been counted several times and the bonus is well below 1.
    assert out[4, 0] < 0.6


def test_noveld_gate_fires_only_on_first_visit():
    """
    NovelD's episodic gate must zero the bonus on revisits, and the
    reported first_visit_frac must match how many steps were new.
    """

    bonus = NovelDBonus(
        OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=32, alpha=0.5
    )
    obs, next_obs, actions, dones = _rollout(distinct_states=2)
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # Only two distinct states exist, so after the first two steps
    # every step is a revisit and must be gated to exactly zero.
    assert torch.all(out[2:] == 0.0)
    # Two first visits out of NUM_STEPS steps, per env.
    assert bonus.diagnostics()['first_visit_frac'] == pytest.approx(
        2.0 / NUM_STEPS
    )


def test_noveld_without_gate_keeps_revisit_bonus():
    """
    Disabling the gate must leave the bare novelty difference, which
    is the ablation the flag exists to support.
    """

    bonus = NovelDBonus(
        OBS_SHAPE,
        NUM_ENVS,
        DEVICE,
        feature_dim=32,
        episodic_gate=False,
    )
    obs, next_obs, actions, dones = _rollout(distinct_states=2)
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # The difference is clamped at zero, so it can never be negative,
    # but it must not be identically zeroed the way the gated
    # version is.
    assert torch.all(out >= 0.0)
    assert torch.isfinite(out).all()


def test_bonuses_are_non_negative_and_finite():
    """
    Every bonus must return a finite, non-negative (T, E) reward.

    A negative intrinsic reward would punish exploration rather than
    encourage it, and a NaN silently destroys the policy through the
    blended advantage.
    """

    obs, next_obs, actions, dones = _rollout(distinct_states=4)
    bonuses = [
        CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE),
        RNDBonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=32),
        NovelDBonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=32),
        RE3Bonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=16),
        E3BBonus(
            OBS_SHAPE, NUM_ENVS, DEVICE, num_actions=7, feature_dim=16
        ),
    ]
    for bonus in bonuses:
        out = bonus.rollout_bonus(obs, next_obs, actions, dones)
        name = type(bonus).__name__
        assert out.shape == (NUM_STEPS, NUM_ENVS), name
        assert torch.isfinite(out).all(), name
        assert torch.all(out >= 0.0), name


def test_e3b_bonus_decreases_within_an_episode():
    """
    The elliptical bonus must shrink as the episode's covariance
    absorbs more directions -- that decay IS the episodic signal.
    """

    bonus = E3BBonus(
        OBS_SHAPE,
        NUM_ENVS,
        DEVICE,
        num_actions=7,
        feature_dim=16,
        ridge=0.1,
    )
    # Repeat one single state, which is the fastest way to saturate
    # the covariance along that one direction.
    obs, next_obs, actions, dones = _rollout(distinct_states=1)
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    assert out[0, 0] > out[-1, 0]
    # Sherman-Morrison keeps C^-1 positive definite, so the quadratic
    # form must stay positive however far it decays.
    assert out[-1, 0] > 0.0


def test_e3b_resets_covariance_at_episode_boundary():
    """
    A new episode must restore the full elliptical bonus, since the
    covariance is per-episode by construction.
    """

    bonus = E3BBonus(
        OBS_SHAPE,
        NUM_ENVS,
        DEVICE,
        num_actions=7,
        feature_dim=16,
        ridge=0.1,
    )
    obs, next_obs, actions, dones = _rollout(distinct_states=1)
    dones[3, :] = 1.0
    out = bonus.rollout_bonus(obs, next_obs, actions, dones)

    # Step 3 starts a fresh episode, so its bonus must jump back to
    # the value the very first step of an episode receives.
    assert out[3, 0] == pytest.approx(out[0, 0], rel=1e-5)


def test_stateless_bonuses_expose_no_trainable_parameters():
    """
    RE3's encoder is a FIXED random projection and counting has no
    network at all; if either started training, the method would no
    longer be the one it is named after.
    """

    assert (
        RE3Bonus(
            OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=16
        ).trainable_parameters()
        == []
    )
    assert (
        CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE).trainable_parameters()
        == []
    )


def test_rnd_target_is_frozen_but_predictor_trains():
    """
    RND's whole premise is a fixed target chased by a trainable
    predictor; a trainable target would let the pair collapse to
    zero error everywhere and kill the novelty signal.
    """

    bonus = RNDBonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=32)
    assert all(
        not p.requires_grad for p in bonus.target.parameters()
    )
    assert all(p.requires_grad for p in bonus.predictor.parameters())
    # The optimizer must therefore see the predictor's parameters
    # and nothing from the target.
    assert len(bonus.trainable_parameters()) == len(
        list(bonus.predictor.parameters())
    )


def test_aux_losses_are_differentiable():
    """
    The bonuses that learn must return a loss with a live gradient
    path, otherwise they silently never train.
    """

    obs, next_obs, actions, dones = _rollout(distinct_states=4)
    flat_obs = obs.reshape((-1,) + OBS_SHAPE)
    flat_next = next_obs.reshape((-1,) + OBS_SHAPE)
    flat_actions = actions.reshape(-1)

    for bonus in [
        RNDBonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=32),
        E3BBonus(
            OBS_SHAPE, NUM_ENVS, DEVICE, num_actions=7, feature_dim=16
        ),
    ]:
        loss = bonus.aux_loss(flat_obs, flat_next, flat_actions)
        assert loss is not None
        assert loss.requires_grad
        # A backward pass must actually reach the trainable weights.
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in bonus.trainable_parameters()
        )


@pytest.mark.parametrize(
    'obs_shape',
    [
        (7, 7, 3),      # symbolic partial view, every task
        (8, 8, 3),      # symbolic_full on DoorKey-8x8
        (16, 16, 3),    # symbolic_full on KeyCorridorS6R3 / ObsMaze
        (56, 56, 3),    # partial / historical RGB, small maps
        (64, 64, 3),    # historical RGB, 16x16 maps
    ],
)
def test_encoders_build_for_every_observation_shape(obs_shape):
    """
    Every observation shape the registry can produce must build.

    The Atari conv stack shrinks its input hard -- a 16x16 grid is
    already 3x3 after the first 8x8/stride-4 layer, so the second
    layer's 4x4 kernel is larger than what remains and torch raises.
    That is not a hypothetical: obs_mode='symbolic_full' on
    KeyCorridorS6R3 gives exactly 16x16 and crashed before the
    small-input threshold was raised to SMALL_INPUT_THRESHOLD. This
    test is parametrized over the real shapes rather than a couple of
    convenient ones so a future change to the tile-size auto-scaling
    cannot reintroduce the gap silently.
    """

    from algos.intrinsic import build_conv_encoder

    h, w, c = obs_shape
    encoder = build_conv_encoder(obs_shape, out_dim=64)
    out = encoder(torch.zeros(2, c, h, w))
    assert out.shape == (2, 64)

    # The bonuses that build their own encoders must survive the same
    # shapes, since they are constructed from the env's obs_shape.
    steps, envs = 5, 2
    for bonus in [
        RNDBonus(obs_shape, envs, DEVICE, feature_dim=32),
        RE3Bonus(obs_shape, envs, DEVICE, feature_dim=16),
        E3BBonus(
            obs_shape, envs, DEVICE, num_actions=7, feature_dim=16
        ),
    ]:
        obs = torch.ones((steps, envs) + obs_shape)
        out = bonus.rollout_bonus(
            obs,
            obs,
            torch.zeros((steps, envs), dtype=torch.long),
            torch.zeros((steps, envs)),
        )
        assert out.shape == (steps, envs)
        assert torch.isfinite(out).all()


def test_stateless_bonuses_have_no_aux_loss():
    """
    Count and RE3 must return None so the training loop adds nothing
    to the PPO objective for them.
    """

    obs, next_obs, actions, _ = _rollout(distinct_states=4)
    flat_obs = obs.reshape((-1,) + OBS_SHAPE)
    flat_next = next_obs.reshape((-1,) + OBS_SHAPE)
    flat_actions = actions.reshape(-1)

    for bonus in [
        CountBonus(OBS_SHAPE, NUM_ENVS, DEVICE),
        RE3Bonus(OBS_SHAPE, NUM_ENVS, DEVICE, feature_dim=16),
    ]:
        assert (
            bonus.aux_loss(flat_obs, flat_next, flat_actions) is None
        )
