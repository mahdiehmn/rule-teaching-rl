"""
Topology-agnostic navigation over a live MiniGrid grid.

The general counterpart of teachers/minigrid/bfs_solver.py. That
solver is hard-coded to DoorKey: it infers the single dividing wall
from the door's x coordinate, which is meaningless on a layout like
KeyCorridor's grid of rooms. This module instead reads the actual
`env.grid`, so it plans correctly on any MiniGrid/BabyAI layout.

It deliberately does NOT decide *where* to go -- it only answers
'given that I want to reach this cell, what is the next action?'.
Choosing the target is the caller's job, which is exactly the split
teachers/minigrid/llm_subgoal.py exploits: a language model picks
the semantic sub-goal, and this module handles the geometry that
language models are demonstrably bad at (see
docs/progress_june_slides.md, slide 11a).

Everything here is offline, free, and deterministic.
"""

from collections import deque
from dataclasses import dataclass

# Direction unit vectors keyed by MiniGrid's dir convention:
#   0 = east (+x), 1 = south (+y), 2 = west (-x), 3 = north (-y).
DIR_DX = (1, 0, -1, 0)
DIR_DY = (0, 1, 0, -1)

# MiniGrid action ids, repeated here so this file reads standalone.
ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_DROP = 4
ACTION_TOGGLE = 5

# The only actions that can change the agent's (x, y, dir) pose, so
# the only ones worth expanding during the search.
_MOVE_ACTIONS = (ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD)

# How the agent must end up relative to the target cell.
#   'adjacent_facing' -- stand next to it, facing it. Required
#     before PICKUP or TOGGLE, and the definition of "go to X" in
#     BabyAI (an object's own cell cannot be entered).
#   'enter' -- stand ON the cell. Only meaningful for the goal
#     square and for plain floor.
APPROACH_MODES = ('adjacent_facing', 'enter')


@dataclass
class NavResult:
    """
    Outcome of one navigation query.

    Attributes
    ----------
    status: str
        'ready'       -- the agent already satisfies the approach
                         condition, so the caller should now take
                         its interaction action.
        'path'        -- `action` is the next step along a shortest
                         route to the target.
        'unreachable' -- no route exists under the current door
                         states, so the caller must pick a different
                         sub-goal.
    action: int or None
        The next action to take; None for 'ready' and
        'unreachable'.
    nodes_expanded: int
        Search effort, reported as Cost.compute_units so planner
        effort is comparable with other teachers.
    """

    status: str
    action: int | None
    nodes_expanded: int


def cell_walkable(unwrapped, x, y, extra_walkable=()):
    """
    Decide whether the agent can stand on cell (x, y).

    Empty floor and open doors are walkable. Walls, closed and
    locked doors, and any object still lying on the grid (key, ball,
    box) are not. Lava is treated as NOT walkable even though the
    engine permits stepping onto it, because doing so ends the
    episode in failure and a planner should never route through it.

    `extra_walkable` lets the caller whitelist specific cells -- used
    for the 'enter' mode, where the destination itself (e.g. the
    goal square) must be enterable even though it holds an object.
    """

    u = unwrapped
    if x < 0 or x >= u.width or y < 0 or y >= u.height:
        return False
    if (x, y) in extra_walkable:
        return True

    cell = u.grid.get(x, y)
    # An empty cell is always walkable.
    if cell is None:
        return True
    if cell.type == 'door':
        return bool(cell.is_open)
    if cell.type == 'goal':
        return True
    # Walls, lava, and loose objects all block movement.
    return False


def _apply_move(unwrapped, pose, action, extra_walkable):
    """
    Return the pose that results from taking `action` in `pose`.

    Mirrors MiniGrid's real dynamics: turning always succeeds, and
    walking into a blocked cell is a no-op that leaves the pose
    unchanged.
    """

    x, y, d = pose
    if action == ACTION_TURN_LEFT:
        return (x, y, (d - 1) % 4)
    if action == ACTION_TURN_RIGHT:
        return (x, y, (d + 1) % 4)

    nx, ny = x + DIR_DX[d], y + DIR_DY[d]
    if cell_walkable(unwrapped, nx, ny, extra_walkable):
        return (nx, ny, d)
    return (x, y, d)


def _is_ready(pose, target, mode):
    """
    Decide whether `pose` already satisfies the approach condition
    for `target` under `mode`.
    """

    x, y, d = pose
    if mode == 'enter':
        return (x, y) == target
    # 'adjacent_facing': the cell directly ahead is the target.
    return (x + DIR_DX[d], y + DIR_DY[d]) == target


def plan_first_action(unwrapped, target, mode='adjacent_facing'):
    """
    Breadth-first search for the next action toward `target`.

    Searches over (x, y, dir) poses, which is exhaustive because
    only turning and moving forward change the pose and every edge
    costs one action -- so the first action of the first path BFS
    reaches is on a genuinely shortest route.

    Parameters
    ----------
    unwrapped: MiniGridEnv
        The live env, read for its grid and agent pose. Never
        mutated.
    target: tuple of int
        The (x, y) cell to approach.
    mode: str
        One of APPROACH_MODES; see the module constant.

    Returns
    -------
    NavResult
    """

    if mode not in APPROACH_MODES:
        raise ValueError(
            f'unknown approach mode {mode!r}; '
            f'choose one of {APPROACH_MODES}'
        )

    u = unwrapped
    start = (
        int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)
    )
    target = (int(target[0]), int(target[1]))

    # In 'enter' mode the destination must itself be steppable, even
    # when an object sits on it; in 'adjacent_facing' mode we stop
    # one cell short, so it must stay blocked or BFS would walk
    # straight through it.
    extra = {target} if mode == 'enter' else set()

    # Already in position: the caller should interact, not move.
    if _is_ready(start, target, mode):
        return NavResult('ready', None, 0)

    # Standard BFS. Each queue entry carries the FIRST action taken
    # on the way to that pose, so reaching a ready pose immediately
    # yields the action to return without reconstructing the path.
    visited = {start}
    queue = deque()
    for action in _MOVE_ACTIONS:
        nxt = _apply_move(u, start, action, extra)
        if nxt in visited:
            continue
        visited.add(nxt)
        queue.append((nxt, action))

    nodes = 0
    while queue:
        pose, first_action = queue.popleft()
        nodes += 1
        if _is_ready(pose, target, mode):
            return NavResult('path', first_action, nodes)
        for action in _MOVE_ACTIONS:
            nxt = _apply_move(u, pose, action, extra)
            if nxt in visited:
                continue
            visited.add(nxt)
            # Carry the ORIGINAL first action forward, not the
            # action taken on this expansion: what the caller needs
            # is the first step of the route, not the last.
            queue.append((nxt, first_action))

    # The whole reachable component was searched without finding an
    # approach pose -- the target is walled off behind a door that is
    # still shut, so the caller needs a different sub-goal.
    return NavResult('unreachable', None, nodes)


def find_object(unwrapped, x, y):
    """
    Return the grid object at (x, y), or None if the cell is empty
    or out of bounds.

    Used to validate a language model's proposed sub-goal before
    committing to it -- e.g. refusing a 'pick up the key at (1, 5)'
    when nothing is there any more.
    """

    u = unwrapped
    if x < 0 or x >= u.width or y < 0 or y >= u.height:
        return None
    return u.grid.get(x, y)
