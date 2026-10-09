"""
BFS-based optimal teacher for MiniGrid DoorKey environments.

Plans the shortest action sequence from the agent's current
(position, direction, has_key, door_open) state to a state in
which the next correct action is unambiguous. Returns the first
action of that sequence as Advice.

Why a BFS solver here
---------------------
On MountainCar we had two rule teachers: `push_with_velocity` (a
one-line heuristic) and `value_iteration` (the offline optimal
oracle). For DoorKey there isn't a one-line heuristic that
handles all (key, door, goal, agent) layouts -- you actually need
to plan around walls and switch sub-goals between get-key, open-
door, and reach-goal phases. BFS over the discrete (x, y, dir,
has_key, door_open) state space is the smallest planner that
does this correctly, and it's near-instant on a 5x5 grid.

This teacher is the analogue of `ValueIterationTeacher` on
MountainCar: an oracle upper bound that every other teacher
(future LLM teacher, future deliberately-weak rule teacher) gets
compared against.

Restrictions
------------
- Currently hard-coded to the DoorKey family. For other MiniGrid
  envs (Empty, MultiRoom, LavaGap, BabyAI levels) we'll write
  separate teachers later. The Teacher / Channel / Cost
  abstractions are env-agnostic; only this specific planner is
  task-specific.
- Wall layout is inferred from the env_id. DoorKey-5x5 always
  places its dividing wall at x=2, with the door at the y
  coordinate the env state already reports. Larger DoorKey
  variants put the wall elsewhere; we look this up below.
"""

import time
from collections import deque

from teachers.base import Advice, BaseTeacher, Cost


# Action ids in MiniGrid, copied here so the file is readable on
# its own without consulting the env module.
ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_DROP = 4
ACTION_TOGGLE = 5
ACTION_DONE = 6

# Direction unit vectors keyed by MiniGrid's dir convention:
#   0 = east  (+x)
#   1 = south (+y)
#   2 = west  (-x)
#   3 = north (-y)
DIR_DX = (1, 0, -1, 0)
DIR_DY = (0, 1, 0, -1)


class MiniGridBFSTeacher(BaseTeacher):
    """
    Optimal DoorKey teacher via breadth-first search.

    At every call:
      1. Decide the current sub-goal (key, door, or goal) based
         on `has_key` and `door_open` in the state tuple.
      2. BFS over (x, y, dir) from the agent's current cell to a
         set of "ready" states from which the right action is
         obvious (e.g. adjacent to and facing the key, holding
         the key, etc.).
      3. Return the first action of the shortest action sequence.

    Expected state
    --------------
    The 11-tuple delivered by MiniGridEnv:
      (agent_x, agent_y, agent_dir, has_key, door_open,
       key_x, key_y, door_x, door_y, goal_x, goal_y)

    Confidence
    ----------
    1.0 unconditionally. BFS finds the optimal path under the
    deterministic env dynamics; there is no genuine ambiguity to
    report.
    """

    # Answers depend on the state argument alone -- no plan, goal,
    # or history is carried between calls -- so a recorded answer
    # stays valid and advising/peekable.py may replay it instead of
    # paying for the same query twice.
    is_stateless = True

    # MiniGrid's action space: left, right, forward, pickup, drop,
    # toggle, done. `q_row` returns one entry per action, so this
    # has to match what the student's policy head emits.
    NUM_ACTIONS = 7

    def __init__(
        self,
        env_id: str = 'MiniGrid-DoorKey-5x5-v0',
        seed: int = 0,
    ):
        """
        Initialize the teacher with env-specific layout info.

        Parameters
        ----------
        env_id: str
            Gymnasium env id. Used to look up grid dimensions
            and where the dividing wall sits. Currently only
            the DoorKey family is supported.
        seed: int
            Unused (BFS is deterministic), accepted for signature
            consistency with other teachers.
        """

        super().__init__(teacher_id=f'bfs:{env_id}', seed=seed)

        # Parse the env id to figure out grid size and the
        # dividing wall's x coordinate. DoorKey places the wall
        # somewhere between x=2 and x=width-2; for the standard
        # 5x5 / 6x6 / 8x8 variants the wall is at the geometric
        # middle (or close to it). The env actually picks the
        # wall column randomly within a small range, but we
        # infer it from the door_x value in the state tuple at
        # `recommend` time, so this default just sets the grid
        # dims.
        if '5x5' in env_id:
            self.width, self.height = 5, 5
        elif '6x6' in env_id:
            self.width, self.height = 6, 6
        elif '8x8' in env_id:
            self.width, self.height = 8, 8
        elif '16x16' in env_id:
            self.width, self.height = 16, 16
        else:
            raise NotImplementedError(
                f'MiniGridBFSTeacher does not know the grid '
                f'dimensions for env_id={env_id!r}. Add a case '
                f'in __init__.'
            )

        # Memoized BFS results, keyed on the full state tuple.
        # Without this, `peek_action` followed by `recommend` at the
        # same state would run the search twice. The state tuple is
        # small and hashable and DoorKey's reachable state space is
        # tiny, so an unbounded dict is fine here.
        self._action_cache: dict[tuple, int] = {}

        # Optimal-action sets and per-action distances, memoized on
        # the same key. `_action_distances` runs a BFS per movement
        # action, so without this it would be the expensive path in
        # the rollout loop rather than a free one.
        self._optimal_cache: dict[tuple, tuple] = {}
        self._distance_cache: dict[tuple, list] = {}

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Compute the optimal next action for the given state.

        Parameters
        ----------
        state: tuple of 11 int
            (agent_x, agent_y, agent_dir, has_key, door_open,
             key_x, key_y, door_x, door_y, goal_x, goal_y).
        context: dict or None
            Unused.

        Returns
        -------
        Advice
            With `action` set to the BFS-optimal next action.
        """

        t0 = time.perf_counter()

        action, target, has_key, door_open = self._solve(state)

        elapsed = time.perf_counter() - t0

        return Advice(
            action=int(action),
            confidence=1.0,
            explanation=(
                f'BFS-optimal next action toward {target} '
                f'(has_key={has_key}, door_open={door_open})'
            ),
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=elapsed,
                compute_units=1,  # one BFS expansion per call
                metadata={'phase_target': target},
            ),
        )

    def peek_action(self, state) -> int:
        """
        Return the action this teacher would recommend, without
        producing an Advice and without charging for it.

        This is the marker the advice-budgeting strategies look for
        (see `advising/strategies.peek_action`). A teacher that
        implements it can be compared against the student's intended
        action BEFORE any budget is committed -- which is the
        situation Torrey & Taylor (2013) assume throughout, since
        their teacher is a Q-table. A teacher that cannot, such as
        an LLM, forces that comparison to happen only after a query
        has been paid for.

        The BFS search is not literally free the way a table lookup
        is, but it costs no money and no API latency, which are the
        resources the budgets track. Results are memoized so peeking
        and then asking at the same state searches only once.

        Parameters
        ----------
        state: tuple of 11 int
            Same state tuple `recommend` takes.

        Returns
        -------
        int
            The BFS-optimal next action.
        """

        action, _, _, _ = self._solve(state)
        return int(action)

    def optimal_actions(self, state) -> tuple:
        """
        Every action that is optimal at `state`, free of charge.

        `peek_action` returns one of these. This returns all of
        them, which is what a mistake test needs: a student choosing
        a different member of a tie is not making a mistake, and
        counting it as one spends budget correcting a student that
        was exactly as right as the teacher.

        Parameters
        ----------
        state: tuple of 11 int
            Same state tuple `recommend` takes.

        Returns
        -------
        tuple[int, ...]
            Sorted optimal actions, never empty.
        """

        actions, _ = self._solve_optimal_set(state)
        return actions

    def q_row(self, state):
        """
        Negated distance-to-completion for every action, which is
        Torrey & Taylor's Q-function for this task.

        Their state importance is
        `I(s) = max_a Q(s,a) - min_a Q(s,a)`, computed on a
        CONVERGED teacher. On a shortest-path task the converged
        Q-value of an action is just the negative number of steps
        that remain after taking it, so this is not an analogy --
        it is the paper's quantity, exactly, available for free from
        a search the teacher already runs.

        That matters because everything else this project computes
        as "importance" is student-side: policy entropy and its
        relatives measure whether the STUDENT is unsure, which is
        Clouse's (1996) reading. Torrey's claim is about the
        teacher's view -- whether picking wrong at this state is
        EXPENSIVE -- and the two genuinely disagree. At a junction
        where a wrong turn costs forty steps, a trained student is
        confident and the student-side signal reads low while this
        one reads high.

        Returns
        -------
        numpy.ndarray of shape (7,)
            Negated remaining distance per MiniGrid action id.
        """

        import numpy as np

        distances = self._action_distances(state)
        return -np.asarray(distances, dtype=np.float64)

    def _solve_optimal_set(self, state):
        """
        The optimal action set and the distance to completion at
        `state`, memoized alongside the single-action solve.

        Returns
        -------
        (actions, distance): tuple[tuple[int, ...], int]
        """

        key = tuple(int(v) for v in state)
        cached = self._optimal_cache.get(key)
        if cached is not None:
            return cached

        (
            agent_x, agent_y, agent_dir, has_key, door_open,
            key_x, key_y, door_x, door_y, goal_x, goal_y,
        ) = state

        target, interact_action, need_to_walk_in = self._phase(
            has_key, door_open, key_x, key_y, door_x, door_y,
            goal_x, goal_y,
        )

        actions, distance = self._bfs_optimal_actions(
            start=(agent_x, agent_y, agent_dir),
            target=target,
            need_to_walk_in=need_to_walk_in,
            door_x=door_x,
            door_y=door_y,
            door_open=door_open,
        )

        if actions is None:
            # Already in a ready state: the phase's interaction is
            # the single optimal move (walking onto the goal counts
            # as the interaction for the final phase).
            only = (
                ACTION_FORWARD
                if interact_action is None
                else interact_action
            )
            result = ((int(only),), 0)
        elif not actions:
            # Unreachable under the current door state. Fall back to
            # the single-action solve so the teacher still advises
            # something rather than abstaining mid-episode.
            single, _, _, _ = self._solve(state)
            result = ((int(single),), -1)
        else:
            result = (tuple(sorted(int(a) for a in actions)), distance)

        self._optimal_cache[key] = result
        return result

    def _action_distances(self, state):
        """
        Steps remaining after each MiniGrid action, used to build
        `q_row`.

        Pose-changing actions cost one step plus whatever remains
        from the pose they lead to. Interaction actions complete the
        phase when the agent is ready and are otherwise a wasted
        step. Unreachable states are given a large finite penalty
        rather than infinity, so the max-minus-min in Torrey's
        formula stays a real number.
        """

        key = tuple(int(v) for v in state)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        (
            agent_x, agent_y, agent_dir, has_key, door_open,
            key_x, key_y, door_x, door_y, goal_x, goal_y,
        ) = state

        target, interact_action, need_to_walk_in = self._phase(
            has_key, door_open, key_x, key_y, door_x, door_y,
            goal_x, goal_y,
        )
        pose = (agent_x, agent_y, agent_dir)
        ready_here = self._is_ready(pose, target, need_to_walk_in)

        # Distance ceiling for moves that lead nowhere useful. Twice
        # the grid's cell count comfortably exceeds any real path.
        unreachable = 2 * self.width * self.height

        distances = [float(unreachable)] * self.NUM_ACTIONS

        for action in (
            ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD
        ):
            nxt = self._apply_action(
                pose, action, door_x, door_y, door_open
            )
            if self._is_ready(nxt, target, need_to_walk_in):
                distances[action] = 1.0
                continue
            _, d = self._bfs_optimal_actions(
                start=nxt,
                target=target,
                need_to_walk_in=need_to_walk_in,
                door_x=door_x,
                door_y=door_y,
                door_open=door_open,
            )
            distances[action] = (
                float(unreachable) if d < 0 else float(1 + d)
            )

        # The interaction that completes the current phase is the
        # best possible move when the agent is already in position,
        # and a wasted step otherwise.
        if ready_here and interact_action is not None:
            distances[interact_action] = 0.0

        self._distance_cache[key] = distances
        return distances

    def _phase(
        self, has_key, door_open, key_x, key_y, door_x, door_y,
        goal_x, goal_y,
    ):
        """
        Current sub-goal: which cell to head for, which interaction
        completes it, and whether the agent must stand on it.

        Extracted so `_solve`, `_solve_optimal_set` and
        `_action_distances` cannot drift apart on the phase rules.
        """

        if not has_key:
            return (key_x, key_y), ACTION_PICKUP, False
        if not door_open:
            return (door_x, door_y), ACTION_TOGGLE, False
        return (goal_x, goal_y), None, True

    def _solve(self, state):
        """
        Compute the BFS-optimal next action for `state`, memoized.

        Shared by `recommend` and `peek_action` so the two can never
        disagree. Returns the action plus the sub-goal context the
        Advice explanation needs.

        Returns
        -------
        (action, target, has_key, door_open)
        """

        key = tuple(int(v) for v in state)
        cached = self._action_cache.get(key)

        # Unpack the env state tuple.
        (
            agent_x,
            agent_y,
            agent_dir,
            has_key,
            door_open,
            key_x,
            key_y,
            door_x,
            door_y,
            goal_x,
            goal_y,
        ) = state

        # Identify the current sub-goal phase. This drives both
        # the BFS target set and the interaction action to take
        # once we are adjacent + facing the target.
        if not has_key:
            target = (key_x, key_y)
            interact_action = ACTION_PICKUP
            need_to_walk_in = False
        elif not door_open:
            target = (door_x, door_y)
            interact_action = ACTION_TOGGLE
            need_to_walk_in = False
        else:
            target = (goal_x, goal_y)
            interact_action = None
            need_to_walk_in = True

        # A memoized result still needs the sub-goal context above,
        # which is cheap to recompute, but skips the search itself.
        if cached is not None:
            return cached, target, has_key, door_open

        # Run BFS over (x, y, dir) from agent's current state to
        # any "ready" state. Ready means either:
        #   (a) need_to_walk_in: the agent stands ON the target
        #       cell (walk-onto goal),
        #   (b) otherwise: the agent stands adjacent to the
        #       target and is facing it (pickup or toggle).
        action = self._bfs_first_action(
            start=(agent_x, agent_y, agent_dir),
            target=target,
            need_to_walk_in=need_to_walk_in,
            door_x=door_x,
            door_y=door_y,
            door_open=door_open,
        )

        # If the agent is already in a ready state, BFS returns
        # None for the first action -- we take the interaction
        # action directly.
        if action is None:
            action = (
                ACTION_FORWARD
                if interact_action is None
                else interact_action
            )

        action = int(action)
        self._action_cache[key] = action
        return action, target, has_key, door_open

    def _bfs_optimal_actions(
        self,
        start,
        target,
        need_to_walk_in,
        door_x,
        door_y,
        door_open,
    ):
        """
        BFS from `start` to a ready state, returning EVERY first
        action that begins some shortest path, plus that path's
        length.

        `_bfs_first_action` returns one of these arbitrarily, which
        is all a teacher needs to act. It is not all a *grader*
        needs. Gridworlds are full of ties -- two shortest routes
        around an obstacle, or a turn that can be taken left or
        right with equal cost -- and scoring the student against one
        arbitrarily chosen member of the tie counts it wrong for
        picking an equally optimal action. Measured on a policy at
        success 1.00, that mislabels a third of its states.

        The search is level-synchronous rather than a plain queue
        walk because a pose can be reached at the same depth by more
        than one opening action, and those have to be merged BEFORE
        the pose is expanded. A standard BFS marks the pose visited
        on first arrival and silently discards the second route,
        which is exactly the tie we are trying to detect.

        Returns
        -------
        (actions, distance): tuple[frozenset[int], int] | tuple[None, 0]
            All optimal first actions and the number of steps to
            reach a ready state, or (None, 0) when `start` is
            already ready and no movement is needed.
        """

        if self._is_ready(start, target, need_to_walk_in):
            return None, 0

        # Only these three change the agent's pose; pickup, drop and
        # toggle never appear on an optimal navigation path.
        moves = (ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD)

        # Poses settled at a strictly shallower depth. Reaching one
        # again is a longer route and is discarded.
        visited = {start}
        # Current level: pose -> the set of opening actions that
        # reach it in exactly `depth` steps.
        frontier = {start: frozenset()}
        depth = 0

        while frontier:
            depth += 1
            next_level = {}

            for pose, openings in frontier.items():
                for action in moves:
                    nxt = self._apply_action(
                        pose, action, door_x, door_y, door_open
                    )
                    if nxt in visited:
                        continue
                    # At depth 1 the action being taken IS the
                    # opening action; deeper, the opening actions
                    # are inherited from the pose we came from.
                    inherited = (
                        frozenset({action})
                        if depth == 1
                        else openings
                    )
                    next_level[nxt] = (
                        next_level.get(nxt, frozenset()) | inherited
                    )

            # Check readiness only after the whole level is built,
            # so every route into a ready pose has been merged.
            ready = [
                pose
                for pose in next_level
                if self._is_ready(pose, target, need_to_walk_in)
            ]
            if ready:
                actions = frozenset()
                for pose in ready:
                    actions |= next_level[pose]
                return actions, depth

            visited |= set(next_level)
            frontier = next_level

        # Target unreachable under the current door state.
        return frozenset(), -1

    def _bfs_first_action(
        self,
        start,
        target,
        need_to_walk_in,
        door_x,
        door_y,
        door_open,
    ):
        """
        BFS from `start` to a ready state. Returns the first
        action of the shortest path, or None if `start` is itself
        a ready state.

        Parameters
        ----------
        start: tuple (x, y, dir)
            Agent's current pose.
        target: tuple (x, y)
            The (x, y) cell the agent is heading toward (key,
            door, or goal).
        need_to_walk_in: bool
            True for the goal phase (agent must end ON the
            target cell). False for the key/door phases (agent
            must end adjacent + facing the target cell).
        door_x, door_y: int
            Position of the door, used to decide whether the
            door cell is walkable from this state.
        door_open: int
            1 if the door is currently open. The door cell is
            walkable iff door is open.

        Returns
        -------
        first_action: int or None
            Action 0..5 to take next. None if start is already
            a ready state.
        """

        # Quick check: are we already ready? Avoid running BFS
        # for the no-op case.
        if self._is_ready(start, target, need_to_walk_in):
            return None

        # Standard BFS. The frontier holds (pose, first_action)
        # pairs: the first action taken at the very first
        # expansion step from `start`. Subsequent expansions
        # carry that first_action forward so when we hit a
        # ready state, we know exactly which action to return.
        visited = {start}
        queue = deque()

        # Seed the queue with successors of `start`, tagging
        # each with the action that produced it.
        for action in (
            ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD
        ):
            nxt = self._apply_action(
                start, action, door_x, door_y, door_open
            )
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, action))

        while queue:
            pose, first_action = queue.popleft()

            if self._is_ready(pose, target, need_to_walk_in):
                return first_action

            # Expand neighbors. Only the three movement-ish
            # actions matter for BFS; pickup/drop/toggle don't
            # change pose and never appear on an optimal
            # navigation path.
            for action in (
                ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD
            ):
                nxt = self._apply_action(
                    pose, action, door_x, door_y, door_open
                )
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_action))

        # Should never reach here on a solvable DoorKey layout.
        # Fall back to a benign action so we don't crash the run.
        return ACTION_FORWARD

    def _is_ready(self, pose, target, need_to_walk_in):
        """
        Decide whether `pose` is a state from which the right
        action is immediately the interaction action.
        """

        x, y, d = pose
        tx, ty = target

        if need_to_walk_in:
            # Goal phase: ready iff agent is ON the goal cell.
            # The BFS itself won't produce "agent steps onto
            # goal" because the goal isn't walkable mid-search;
            # instead we treat "adjacent + facing" as ready and
            # the caller takes ACTION_FORWARD to step in.
            if (x, y) == (tx, ty):
                return True
            # Or adjacent + facing the goal: take forward to
            # step onto the goal cell.
            return self._adjacent_facing(pose, target)

        # Key / door phase: adjacent + facing the target cell.
        return self._adjacent_facing(pose, target)

    def _adjacent_facing(self, pose, target):
        """
        True if `pose` is exactly one cell behind `target` and
        oriented to face the target.
        """

        x, y, d = pose
        tx, ty = target
        return (x + DIR_DX[d], y + DIR_DY[d]) == (tx, ty)

    def _apply_action(
        self, pose, action, door_x, door_y, door_open
    ):
        """
        Compute the (x, y, dir) that results from taking
        `action` in `pose`. Movement that would walk into a
        wall (or the closed door) is treated as a no-op, which
        matches MiniGrid's real env behaviour.
        """

        x, y, d = pose

        if action == ACTION_TURN_LEFT:
            return (x, y, (d - 1) % 4)
        if action == ACTION_TURN_RIGHT:
            return (x, y, (d + 1) % 4)
        if action == ACTION_FORWARD:
            nx = x + DIR_DX[d]
            ny = y + DIR_DY[d]
            if self._is_walkable(
                nx, ny, door_x, door_y, door_open
            ):
                return (nx, ny, d)
            # Walking into a wall is a no-op in MiniGrid: pose
            # unchanged. BFS-wise this is fine; the unchanged
            # pose is already visited so we won't expand it.
            return (x, y, d)

        # pickup/drop/toggle don't change pose. Returning the
        # same pose means BFS won't make progress through them,
        # which is what we want (we never need to interact
        # *during* navigation).
        return (x, y, d)

    def _is_walkable(self, x, y, door_x, door_y, door_open):
        """
        True if cell (x, y) can be stepped into.

        Outer-wall cells (x in {0, width-1}, y in {0, height-1})
        are never walkable. Cells on the dividing wall (any
        cell with the same x as the door but a different y) are
        never walkable. The door cell itself is walkable iff
        the door is open.

        We infer the dividing-wall x from the door's x (the env
        guarantees the door sits in a vertical wall, so door_x
        IS the wall column).
        """

        # Outside the grid: not walkable.
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        # Outer wall ring.
        if x == 0 or x == self.width - 1:
            return False
        if y == 0 or y == self.height - 1:
            return False
        # Dividing wall: cells with x equal to door_x but y not
        # equal to door_y. door_y is the one and only opening.
        if x == door_x and y != door_y:
            return False
        # The door cell itself: walkable only when open.
        if x == door_x and y == door_y:
            return door_open == 1
        return True
