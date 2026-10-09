"""
Pure-BFS teacher for tasks where the only obstacle between the
agent and a known goal is closed (but never LOCKED) doors -- no
key-finding, no object reasoning. MultiRoom-N6 is exactly this
shape: six rooms connected by doors that only need a TOGGLE, and a
single fixed goal.

This is deliberately narrower than teachers/minigrid/llm_subgoal.py:
there, an LLM decides WHICH sub-goal to pursue because the correct
order (fetch this key before that door) takes real judgment. Here,
no judgment is needed at all -- the only decision is "which closed
door, if any, sits on the shortest route to the goal", which is
itself answerable by search. So the whole teacher is BFS twice over:
once with doors treated as passable to find the route and the first
door blocking it, and once (via grid_nav.plan_first_action) to walk
to whatever the immediate target is.

Do not use this on tasks with locked doors or where the goal itself
is not directly known (e.g. KeyCorridor, where the target object's
identity requires reading the mission and the key's location must
be discovered) -- it will silently ignore locks and has no concept
of fetching an object.
"""

import time
from collections import deque

from teachers.base import Advice, BaseTeacher, Cost
from teachers.minigrid.grid_nav import (
    ACTION_FORWARD,
    ACTION_TOGGLE,
    DIR_DX,
    DIR_DY,
    plan_first_action,
)


def _relaxed_walkable(unwrapped, x, y):
    """
    Like grid_nav.cell_walkable, but also treats a CLOSED, UNLOCKED
    door as walkable. Used only to find a ROUTE (including which
    doors sit on it); never to decide what the agent may physically
    step into right now.
    """

    u = unwrapped
    if x < 0 or x >= u.width or y < 0 or y >= u.height:
        return False
    cell = u.grid.get(x, y)
    if cell is None or cell.type == 'goal':
        return True
    if cell.type == 'door':
        # Locked doors are never relaxed: this teacher has no key
        # logic, so a locked door is a real dead end for it.
        return not cell.is_locked
    return False


def _shortest_route(unwrapped, start, target):
    """
    BFS from `start` (x, y, dir) to standing ON `target` (x, y),
    using the relaxed walkability above. Returns the sequence of
    (x, y) cells visited (excluding the start cell), or None if no
    route exists even with every unlocked door treated as open.
    """

    u = unwrapped
    if (start[0], start[1]) == target:
        return []

    visited = {(start[0], start[1])}
    # Each queue entry is (x, y, dir, path_so_far).
    queue = deque([(start[0], start[1], start[2], [])])
    while queue:
        x, y, d, path = queue.popleft()
        for dx, dy in zip(DIR_DX, DIR_DY):
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not _relaxed_walkable(u, nx, ny):
                continue
            new_path = path + [(nx, ny)]
            if (nx, ny) == target:
                return new_path
            visited.add((nx, ny))
            queue.append((nx, ny, d, new_path))
    return None


def _first_blocking_door(unwrapped, route):
    """
    Scan a route (list of (x, y) cells) and return the (x, y) of the
    first cell on it that is CURRENTLY a still-closed door, or None
    if the route is already fully walkable for real.
    """

    u = unwrapped
    for x, y in route:
        cell = u.grid.get(x, y)
        if cell is not None and cell.type == 'door' and not cell.is_open:
            return (x, y)
    return None


class DoorOnlyBFSTeacher(BaseTeacher):
    """
    Deterministic, learning-free teacher for door-only navigation
    tasks. Every recommend() call re-plans from scratch: find the
    shortest route to the goal treating closed doors as passable,
    open the first one that is still actually closed, then walk to
    the goal once none remain. No state carried between calls, no
    API cost, no randomness.

    Expected inputs
    ----------------
    recommend(state, context):
      - state: the unwrapped MiniGrid env.
      - context: unused: this teacher needs no mission text and no
        episode-boundary signal, since it carries no memory.
    """

    def __init__(self, env_id: str = 'MiniGrid-MultiRoom-N6-v0',
                 seed: int = 0):
        super().__init__(teacher_id=f'door_bfs:{env_id}', seed=seed)
        self.env_id = env_id

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return the next primitive action toward the goal, opening
        whatever closed door is next in the way.
        """

        t0 = time.perf_counter()
        u = state

        goal = None
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal = (x, y)
                    break
            if goal is not None:
                break
        if goal is None:
            return self._abstain(t0, 'no goal cell found on the grid')

        start = (
            int(u.agent_pos[0]), int(u.agent_pos[1]),
            int(u.agent_dir),
        )
        route = _shortest_route(u, start, goal)
        if route is None:
            return self._abstain(
                t0, f'no route to goal {goal} even treating every '
                'unlocked door as open'
            )

        door = _first_blocking_door(u, route)
        if door is not None:
            nav = plan_first_action(u, door, 'adjacent_facing')
            explanation = f'route passes through door at {door}'
            if nav.status == 'ready':
                # Facing the door already: open it.
                action = ACTION_TOGGLE
            elif nav.status == 'path':
                action = nav.action
            else:
                return self._abstain(
                    t0, f'door at {door} became unreachable'
                )
        else:
            nav = plan_first_action(u, goal, 'enter')
            explanation = f'route to goal {goal} is clear'
            if nav.status == 'path':
                action = nav.action
            elif nav.status == 'ready':
                # Already standing on the goal cell. There is
                # nothing left to interact with here (unlike a
                # door); the environment terminates the episode on
                # the step that moved the agent onto the goal, so
                # recommend() should not normally be called again
                # in this state. FORWARD is a harmless fallback if
                # it is.
                action = ACTION_FORWARD
            else:
                return self._abstain(t0, f'goal {goal} unreachable')

        return Advice(
            action=int(action),
            confidence=1.0,
            explanation=explanation,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                compute_units=len(route),
            ),
        )

    def _abstain(self, t0, reason):
        return Advice(
            action=None,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                metadata={'failed': True, 'error': reason},
            ),
        )
