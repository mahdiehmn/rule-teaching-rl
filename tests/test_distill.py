"""
Tests for the Stage-1 distillation pieces.

Covers the bot teacher (solves gotoseq end to end through the
Advice contract, abstains instead of crashing, explains itself),
the advice-softening math, the kickstarting coefficient schedule,
and the shared greedy evaluator's determinism.
"""

import numpy as np
import pytest

pytest.importorskip('minigrid')

from algos.ppo_distill import Args, distill_coef, soften_advice
from envs.registry import build_env
from envs.state import extract_generic_state
from monitoring.eval import greedy_eval
from teachers.base import Advice
from teachers.factory import make_teacher
from teachers.minigrid.llm_general import (
    build_generic_llm_prompt,
    render_ascii_map,
)
from teachers.minigrid.vlm_general import build_generic_vlm_prompt


def test_bot_teacher_solves_gotoseq_when_followed():
    """
    Following the bot's advice verbatim should solve gotoseq --
    this is the teacher-competence sanity check, run through the
    exact Advice interface the training loop uses.
    """

    env = build_env('gotoseq', seed=0, obs_mode='historical')
    u = env.unwrapped
    teacher = make_teacher('bot', 'BabyAI-GoToSeq-v0', seed=0)

    successes = []
    try:
        for episode in range(3):
            env.reset(seed=300 + episode)
            new_episode = True
            last_action = None
            total = 0.0
            for _ in range(u.max_steps):
                advice = teacher.recommend(
                    u,
                    {
                        'new_episode': new_episode,
                        'last_action': last_action,
                    },
                )
                new_episode = False
                assert advice.action is not None, (
                    'the bot should never abstain on its own '
                    'trajectory'
                )
                # The subgoal-stack explanation is the Stage-2
                # supervision signal; it must always be present.
                assert advice.explanation
                _, r, term, trunc, _ = env.step(advice.action)
                last_action = advice.action
                total += r
                if term or trunc:
                    break
            successes.append(1.0 if total > 0 else 0.0)
    finally:
        env.close()

    assert np.mean(successes) == 1.0, (
        f'bot should solve gotoseq; got {successes}'
    )


def test_bot_teacher_abstains_instead_of_crashing():
    """
    Under a random student the bot may occasionally fail to plan;
    the wrapper must convert that to an abstention Advice, never
    an exception, because a crash mid-rollout would kill a
    multi-hour training run.
    """

    env = build_env('gotoseq', seed=0, obs_mode='historical')
    u = env.unwrapped
    teacher = make_teacher('bot', 'BabyAI-GoToSeq-v0', seed=0)
    rng = np.random.default_rng(0)

    try:
        env.reset(seed=400)
        new_episode = True
        last_action = None
        for _ in range(200):
            advice = teacher.recommend(
                u,
                {
                    'new_episode': new_episode,
                    'last_action': last_action,
                },
            )
            new_episode = False
            # Whatever happened inside, the contract holds: an
            # Advice object with either a recommendation or an
            # explicit abstention.
            assert isinstance(advice, Advice)
            action = int(rng.integers(0, 7))
            _, _, term, trunc, _ = env.step(action)
            last_action = action
            if term or trunc:
                env.reset(seed=401)
                new_episode = True
                last_action = None
    finally:
        env.close()


def test_soften_advice_math():
    """
    Confidence 1.0 must give a one-hot target; lower confidence
    spreads the remainder uniformly; abstention gives None.
    """

    one_hot = soften_advice(Advice(action=2, confidence=1.0), 7)
    assert one_hot[2] == pytest.approx(1.0)
    assert one_hot.sum() == pytest.approx(1.0)

    soft = soften_advice(Advice(action=2, confidence=0.5), 5)
    # 0.5 direct mass + 0.5/5 residual on the recommended action.
    assert soft[2] == pytest.approx(0.5 + 0.1)
    assert soft[0] == pytest.approx(0.1)
    assert soft.sum() == pytest.approx(1.0)

    assert soften_advice(Advice(action=None), 7) is None
    assert soften_advice(None, 7) is None


def test_distill_coef_schedule_shape():
    """
    The schedule must decay linearly to the floor, hold it, and
    hard-drop to zero at the cutoff (LLM4Teach's verified shape).
    """

    args = Args()
    args.distill_coef_start = 10.0
    args.distill_coef_min = 0.1
    args.distill_fraction = 0.5
    args.distill_cutoff = 0.75
    n = 1000

    assert distill_coef(1, n, args) == pytest.approx(10.0, abs=0.1)
    # Midway through the decay window: halfway down.
    mid = distill_coef(n // 4, n, args)
    assert 4.0 < mid < 6.5
    # After the decay window but before the cutoff: at the floor.
    assert distill_coef(int(n * 0.6), n, args) == pytest.approx(
        0.1, abs=0.05
    )
    # At/after the cutoff: exactly zero (queries stop entirely).
    assert distill_coef(int(n * 0.75) + 1, n, args) == 0.0
    assert distill_coef(n, n, args) == 0.0


def test_extract_generic_state_finds_keycorridor_objects():
    """
    On a real KeyCorridorS6R3 layout, the generic extractor must
    find the target ball, the one locked door, and the key that
    matches its color -- without any KeyCorridor-specific code (the
    whole point of this extractor, unlike extract_doorkey_state).
    """

    env = build_env('keycorridor_s6r3', seed=0, obs_mode='historical')
    try:
        env.reset(seed=0)
        agent_x, agent_y, agent_dir, carrying, objects = (
            extract_generic_state(env.unwrapped)
        )

        assert 0 <= agent_x < env.unwrapped.width
        assert 0 <= agent_y < env.unwrapped.height
        assert agent_dir in (0, 1, 2, 3)
        assert carrying == '', 'agent starts empty-handed'

        kinds = {kind for kind, *_ in objects}
        assert 'ball' in kinds and 'door' in kinds and 'key' in kinds

        locked = [o for o in objects if o[0] == 'door' and o[4] == 'locked']
        assert len(locked) == 1, 'KeyCorridor has exactly one locked door'
        (_, lock_color, _, _, _) = locked[0]

        keys = [o for o in objects if o[0] == 'key']
        assert any(k[1] == lock_color for k in keys), (
            'the hidden key must match the locked door color'
        )
    finally:
        env.close()


def test_extract_generic_state_is_hashable_and_stable():
    """
    The extractor's output is used as a cache key, so it must be
    hashable, and calling it twice on an unchanged state must
    return an identical value (order-independent scan).
    """

    env = build_env('doorkey_8x8', seed=0, obs_mode='historical')
    try:
        env.reset(seed=0)
        a = extract_generic_state(env.unwrapped)
        b = extract_generic_state(env.unwrapped)
        hash(a)  # must not raise
        assert a == b
    finally:
        env.close()


def test_generic_vlm_prompt_reflects_door_states_and_mission():
    """
    The prompt text must surface the mission and every door's
    open/closed/locked state explicitly (the render alone can't be
    trusted for this distinction at low tile resolution).
    """

    objects = (
        ('key', 'red', 1, 1, ''),
        ('door', 'red', 3, 3, 'locked'),
        ('door', 'blue', 5, 5, 'open'),
        ('ball', 'red', 7, 7, ''),
    )
    prompt = build_generic_vlm_prompt(
        'MiniGrid-KeyCorridorS6R3-v0',
        'pick up the red ball',
        carrying='',
        objects=objects,
    )

    assert 'pick up the red ball' in prompt
    assert 'red door at (3, 3): locked' in prompt
    assert 'blue door at (5, 5): open' in prompt
    assert 'carrying: nothing' in prompt


def test_generic_vlm_prompt_reports_carried_item():
    """
    When the agent holds something, the prompt must say so (a
    carried object never appears in the rendered image).
    """

    prompt = build_generic_vlm_prompt(
        'MiniGrid-KeyCorridorS6R3-v0',
        'pick up the red ball',
        carrying='red key',
        objects=(),
    )
    assert 'carrying: red key' in prompt
    assert 'no doors visible' in prompt


def test_render_ascii_map_shows_walls_agent_and_objects():
    """
    The ASCII map is the text LLM teacher's only source of wall
    layout, so it must faithfully show the border walls, the agent
    arrow (with facing direction), and object markers at the right
    coordinates -- checked against the env's own grid.
    """

    env = build_env('keycorridor_s3r3', seed=0, obs_mode='historical')
    try:
        env.reset(seed=0)
        u = env.unwrapped
        ascii_map = render_ascii_map(u)
        rows = ascii_map.split('\n')

        # One text row per grid row, one char per cell.
        assert len(rows) == u.height
        assert all(len(row) == u.width for row in rows)

        # The outer border of every MiniGrid map is wall.
        assert set(rows[0]) == {'#'}
        assert set(rows[-1]) == {'#'}

        # The agent's cell carries a direction arrow.
        ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
        assert rows[ay][ax] in '><^v'

        # Every extracted object appears as its type marker at its
        # coordinates (unless the agent stands on it, which cannot
        # happen for keys/doors/balls).
        markers = {'door': 'D', 'key': 'k', 'ball': 'o', 'box': 'B'}
        _, _, _, _, objects = extract_generic_state(u)
        for kind, _color, x, y, _state in objects:
            if kind in markers:
                assert rows[y][x] == markers[kind]
    finally:
        env.close()


def test_generic_llm_prompt_contains_map_mission_and_status():
    """
    The text prompt must carry the mission verbatim, the ASCII map
    itself, every door's state with coordinates, the carried item,
    and the facing direction in words -- all the facts the DoorKey
    study showed a text LLM needs supplied explicitly.
    """

    objects = (
        ('ball', 'yellow', 5, 2, ''),
        ('door', 'purple', 3, 4, 'locked'),
        ('key', 'purple', 1, 1, ''),
    )
    fake_map = '#####\n#k>.#\n#####'
    prompt = build_generic_llm_prompt(
        'BabyAI-KeyCorridorS3R3-v0',
        'pick up the ball',
        agent_dir=0,
        carrying='',
        objects=objects,
        ascii_map=fake_map,
    )

    assert 'pick up the ball' in prompt
    assert fake_map in prompt
    assert 'purple door at (3, 4), locked' in prompt
    assert 'yellow ball at (5, 2)' in prompt
    assert 'facing east' in prompt
    assert 'carrying: nothing' in prompt


def test_generic_llm_prompt_reports_carried_item():
    """
    A carried object is not on the map, so the prompt's status
    block is the only place it can appear -- it must be there.
    """

    prompt = build_generic_llm_prompt(
        'BabyAI-GoToSeq-v0',
        'go to the red ball',
        agent_dir=3,
        carrying='purple key',
        objects=(),
        ascii_map='###\n#^#\n###',
    )
    assert 'carrying: purple key' in prompt
    assert 'facing north' in prompt
    assert 'no objects visible' in prompt


def test_greedy_eval_is_deterministic():
    """
    With a deterministic policy and fixed episode seeds, two eval
    calls must return identical stats -- this is what makes eval
    points along a training run paired measurements.
    """

    def go_forward(obs):
        return 2

    make_env = lambda: build_env(  # noqa: E731
        'empty5x5', seed=0, obs_mode='historical'
    )
    a = greedy_eval(go_forward, make_env, num_episodes=4, seed_base=7)
    b = greedy_eval(go_forward, make_env, num_episodes=4, seed_base=7)

    assert a == b
    assert set(a) == {'success_rate', 'mean_return', 'mean_length'}
