"""
Tests for the general grid navigator and the LLM teachers' episode
memory.

Everything here is offline: no API key, no network, no torch. The
point is that the parts of the new teachers that can be checked for
free are checked for free, so a paid evaluation run only ever tests
the model, never the plumbing.
"""

import pytest

from envs.registry import build_env
from teachers.minigrid import grid_nav
from teachers.minigrid.grid_nav import (
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    plan_first_action,
)
from teachers.minigrid.llm_general import format_history_block


@pytest.fixture
def keycorridor():
    """
    A freshly reset KeyCorridor S3R3 on seed 0.

    Its layout is the useful one for these tests: the agent starts in
    the corridor, and the key and the ball both sit behind doors that
    are still shut.
    """

    env = build_env('keycorridor_s3r3', seed=0, obs_mode='historical')
    env.reset(seed=0)
    yield env
    env.close()


def _walk(env, target, mode, max_steps=60):
    """
    Drive the env with the navigator until it reports 'ready'.

    Returns the number of actions taken, or None if the navigator
    never arrived within max_steps.
    """

    for step in range(max_steps):
        result = plan_first_action(env.unwrapped, target, mode)
        if result.status == 'ready':
            return step
        if result.status == 'unreachable':
            return None
        env.step(result.action)
    return None


def test_navigator_reaches_a_reachable_door(keycorridor):
    """
    The navigator must actually arrive, not just return plausible
    actions.

    This is the regression test for a bug where the BFS frontier
    propagated the action taken on each expansion instead of the
    first action of the route. Every returned action was then the
    LAST step of a path rather than the first, which made the agent
    turn in place forever while the planner kept insisting it was
    following a valid path.
    """

    # The grey door at (2, 3) is reachable from the start cell.
    steps = _walk(keycorridor, (2, 3), 'adjacent_facing')
    assert steps is not None, 'navigator never arrived at the door'
    # Optimal here is turn, forward, turn -- allow no slack, since
    # BFS is exact and any excess means the route was not shortest.
    assert steps == 3


def test_navigator_returns_shortest_first_action(keycorridor):
    """
    The first action of the route must be on a genuinely shortest
    path, checked against a brute-force ground truth.
    """

    u = keycorridor.unwrapped
    target, mode = (2, 3), 'adjacent_facing'
    result = plan_first_action(u, target, mode)
    assert result.status == 'path'

    # Ground truth: the true distance to a ready pose from each of
    # the three successors of the start. The action the planner
    # picked must lead to a strictly minimal one.
    start = (
        int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)
    )
    distances = {}
    for action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                   ACTION_FORWARD):
        pose = grid_nav._apply_move(u, start, action, set())
        distances[action] = _bfs_distance(u, pose, target, mode)

    best = min(distances.values())
    assert distances[result.action] == best


def _bfs_distance(unwrapped, start, target, mode):
    """
    Plain BFS distance in actions from `start` to any ready pose.

    Deliberately written differently from the implementation under
    test (it tracks depth rather than a first action), so a shared
    mistake is unlikely to cancel out.
    """

    from collections import deque

    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        pose, depth = queue.popleft()
        if grid_nav._is_ready(pose, target, mode):
            return depth
        for action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                       ACTION_FORWARD):
            nxt = grid_nav._apply_move(unwrapped, pose, action, set())
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return float('inf')


def test_target_behind_a_shut_door_is_unreachable(keycorridor):
    """
    A cell walled off by a closed door must report 'unreachable'
    rather than silently returning a bogus action.

    This is what lets the sub-goal teacher tell the model 'open that
    door first' instead of marching the agent into a wall.
    """

    # On seed 0 both the key at (1, 5) and the ball at (5, 5) start
    # behind doors that are still shut.
    for target in ((1, 5), (5, 5)):
        result = plan_first_action(keycorridor.unwrapped, target,
                                   'adjacent_facing')
        assert result.status == 'unreachable'
        assert result.action is None


def test_already_in_position_reports_ready(keycorridor):
    """
    Standing adjacent to and facing the target must report 'ready',
    which is the signal for the caller to interact rather than move.
    """

    env = keycorridor
    steps = _walk(env, (2, 3), 'adjacent_facing')
    assert steps is not None
    result = plan_first_action(env.unwrapped, (2, 3),
                               'adjacent_facing')
    assert result.status == 'ready'
    assert result.action is None


def test_closed_door_is_not_walkable(keycorridor):
    """
    Door walkability must follow the door's live state, since that
    is what makes a sub-goal become reachable after a toggle.
    """

    u = keycorridor.unwrapped
    door = u.grid.get(2, 3)
    assert door.type == 'door' and not door.is_open
    assert not grid_nav.cell_walkable(u, 2, 3)

    # Open it and the same cell becomes walkable.
    door.is_open = True
    assert grid_nav.cell_walkable(u, 2, 3)


def test_history_block_flags_repeated_blocked_action():
    """
    A blocked action must be reported as having had no effect, and a
    blocked MOST RECENT action must additionally raise the explicit
    warning -- that warning is the whole mechanism for breaking the
    wall-bumping loop.
    """

    block = format_history_block([(2, False), (2, True), (2, True)])
    assert 'NO EFFECT' in block
    assert 'WARNING' in block
    assert 'FORWARD' in block

    # An empty history must produce nothing at all, so the first
    # step of an episode reproduces the original stateless prompt.
    assert format_history_block([]) == ''

    # A successful last action must not raise the warning.
    ok = format_history_block([(2, True), (0, False)])
    assert 'WARNING' not in ok
