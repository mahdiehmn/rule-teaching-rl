"""
Tests for the pure-BFS door-only teacher (teachers/minigrid/door_bfs.py).

Offline, free, fast: no API key, no network, no torch.
"""

from envs.registry import build_env
from teachers.minigrid.door_bfs import DoorOnlyBFSTeacher


def _run_episode(task, seed, max_steps=120):
    """
    Drive one episode entirely with DoorOnlyBFSTeacher and return
    (success, steps).
    """

    env = build_env(task, seed=seed, obs_mode='historical')
    env.reset(seed=seed)
    u = env.unwrapped
    teacher = DoorOnlyBFSTeacher(env_id='MiniGrid-MultiRoom-N6-v0',
                                  seed=seed)

    for step in range(1, max_steps + 1):
        advice = teacher.recommend(u)
        assert advice.action is not None, (
            f'teacher abstained: {advice.cost.metadata}'
        )
        _obs, reward, terminated, truncated, _ = env.step(
            advice.action
        )
        if terminated or truncated:
            env.close()
            return reward > 0, step
    env.close()
    return False, max_steps


def test_solves_multiroom_across_seeds():
    """
    The teacher must actually reach the goal, not just run without
    crashing -- on several different procedurally generated layouts.
    """

    for seed in range(5):
        success, steps = _run_episode('multiroom_n6', seed)
        assert success, f'seed {seed} failed to solve in {steps} steps'


def test_opens_doors_along_the_way():
    """
    MultiRoom-N6 always has closed doors between rooms; a working
    solve on this task is itself evidence the teacher is opening
    them (walking is impossible otherwise), but assert the toggle
    action (5) actually appears in the trajectory as a direct check
    of that mechanism, not just the end-to-end outcome.
    """

    env = build_env('multiroom_n6', seed=0, obs_mode='historical')
    env.reset(seed=0)
    u = env.unwrapped
    teacher = DoorOnlyBFSTeacher(env_id='MiniGrid-MultiRoom-N6-v0',
                                  seed=0)

    actions_taken = []
    for _ in range(120):
        advice = teacher.recommend(u)
        actions_taken.append(advice.action)
        _obs, reward, terminated, truncated, _ = env.step(
            advice.action
        )
        if terminated or truncated:
            break
    env.close()

    TOGGLE = 5
    assert TOGGLE in actions_taken


def test_abstains_rather_than_crashing_when_locked():
    """
    This teacher has no key logic. On a task with a LOCKED door
    between the agent and the goal, it must abstain (action=None)
    rather than crash or silently walk into the locked door.
    """

    env = build_env('doorkey_8x8', seed=0, obs_mode='historical')
    env.reset(seed=0)
    u = env.unwrapped
    teacher = DoorOnlyBFSTeacher(env_id='MiniGrid-DoorKey-8x8-v0',
                                  seed=0)

    advice = teacher.recommend(u)
    # DoorKey-8x8's goal is always behind the locked door, so with
    # no key logic the relaxed route search cannot reach it either.
    assert advice.action is None
    env.close()
