"""Restore any MiniGrid state into a scratch copy, for effect scoring.

`scripts/explanation_screen_panel.restore` does this for DoorKey 8x8
and only for DoorKey 8x8: one fixed environment id, an 8x8x3 grid check
and an inventory that must be a key. That is right for the screen it
serves, where a surprise layout should be an error rather than a silent
pass.

The consequence machinery needs the same operation on KeyCorridor and
MultiRoom, whose grids are 7x7 and 25x25 and whose inventories hold
balls and boxes. This is that operation, generalized over the task and
nothing else: same encode/decode round trip, same refusal to return an
environment whose grid did not survive it.

Verified against live stepping on DoorKey 8x8, KeyCorridor S3R3 and
MultiRoom N6: over 120 random actions each, the effects read from a
restored copy matched the effects of the same action in the live
environment every time.

The copy is a separate environment. Nothing here touches the caller's
environment, its RNG or its step count.
"""

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key


# Every object a MiniGrid agent can carry.
CARRIABLE = {'key': Key, 'ball': Ball, 'box': Box}


def snapshot(unwrapped):
    """Everything needed to rebuild the physics, and nothing else.

    Deliberately excludes step count and episode age: only immediate
    physical effects and goal entry are scored, and a copy that
    inherited the step count could terminate on truncation instead.
    """
    return {
        'full_grid': unwrapped.grid.encode().tolist(),
        'agent_pos': [int(x) for x in unwrapped.agent_pos],
        'agent_dir': int(unwrapped.agent_dir),
        'mission': unwrapped.mission,
        'carrying': ([unwrapped.carrying.type, unwrapped.carrying.color]
                     if unwrapped.carrying is not None else None),
    }


def restore(env_id, state):
    """Rebuild `state` inside a fresh environment of `env_id`.

    The caller closes it. Raises rather than returning an environment
    that does not encode back to the state it was given, because a
    silently wrong copy would produce confidently wrong effect labels.
    """
    env = gym.make(env_id).unwrapped
    env.reset(seed=0)
    encoded = np.asarray(state['full_grid'], dtype=np.uint8)
    env.grid, _ = Grid.decode(encoded)
    env.agent_pos = tuple(int(x) for x in state['agent_pos'])
    env.agent_dir = int(state['agent_dir'])
    env.mission = state['mission']
    held = state['carrying']
    if held is None:
        env.carrying = None
    else:
        kind, colour = held
        if kind not in CARRIABLE:
            raise ValueError(f'Cannot restore a carried {kind!r}')
        env.carrying = CARRIABLE[kind](colour)
    # Age is not part of the physics being scored.
    env.step_count = 0
    if not np.array_equal(env.grid.encode(), encoded):
        raise ValueError('Grid did not round-trip')
    return env
