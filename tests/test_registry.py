"""
Smoke tests for the env registry.

These validate the observation pipeline (milestone 4 in the README):
that every registered task builds, returns an RGB image to the agent,
and preserves its mission string in info. They are skipped
automatically if minigrid is not installed yet, so the test file is
safe to commit before `pip install -e .` has been run.
"""

import pytest

# Skip the whole module if the env stack is not installed, so a fresh
# checkout without dependencies does not show spurious failures.
pytest.importorskip('minigrid')

from envs.registry import TASKS, build_env


@pytest.mark.parametrize('task_id', list(TASKS))
def test_build_env_returns_image_obs(task_id):
    """
    Every task builds and hands the agent a 3-D image observation.
    """

    env = build_env(task_id, seed=0)
    try:
        obs, info = env.reset(seed=0)

        # The agent observation should be an RGB image:
        # height x width x 3 channels.
        assert obs.ndim == 3
        assert obs.shape[-1] == 3
    finally:
        env.close()


def test_default_partial_obs_keeps_baseline_shape():
    """
    The default 7x7 partial view stays at the old 56x56 RGB shape.
    """

    env = build_env('doorkey_8x8', seed=0)
    try:
        obs, _ = env.reset(seed=0)

        assert obs.shape == (56, 56, 3)
    finally:
        env.close()


def test_larger_partial_view_auto_scales_image():
    """
    A larger agent view auto-scales tiles to keep the image near 56px.
    """

    env = build_env('doorkey_8x8', seed=0, agent_view_size=9)
    try:
        obs, _ = env.reset(seed=0)

        assert obs.shape == (54, 54, 3)
    finally:
        env.close()


def test_full_rgb_obs_auto_scales_to_baseline_size():
    """
    Fully observable RGB mode scales an 8x8 map to 56x56 pixels.
    """

    env = build_env('doorkey_8x8', seed=0, obs_mode='fully_obs')
    try:
        obs, _ = env.reset(seed=0)

        assert obs.shape == (56, 56, 3)
    finally:
        env.close()


def test_explicit_obs_tile_size_is_honored():
    """
    Passing an explicit tile size disables auto-scaling.
    """

    env = build_env(
        'doorkey_8x8',
        seed=0,
        obs_mode='full',
        obs_tile_size=8,
    )
    try:
        obs, _ = env.reset(seed=0)

        assert obs.shape == (64, 64, 3)
    finally:
        env.close()


def test_mission_preserved_in_info():
    """
    The mission string survives the image-only wrapper via info.
    """

    env = build_env('gotoseq', seed=0)
    try:
        _, info = env.reset(seed=0)

        # KeepMissionWrapper should have republished the BabyAI
        # instruction here even though the agent obs is pixels only.
        assert 'mission' in info
        assert isinstance(info['mission'], str)
    finally:
        env.close()


def test_unknown_task_raises():
    """
    An unknown friendly id fails loudly rather than opaquely.
    """

    with pytest.raises(KeyError):
        build_env('does_not_exist')


def test_unknown_obs_mode_raises():
    """
    An unknown observation mode fails loudly.
    """

    with pytest.raises(ValueError):
        build_env('empty5x5', obs_mode='does_not_exist')
