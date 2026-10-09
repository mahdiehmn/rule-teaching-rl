"""
Sub-goal LLM teacher: the model plans, a BFS planner walks.

The third general teacher, alongside teachers/minigrid/llm_general.py
(text, per-step primitive actions) and vlm_general.py (vision,
per-step primitive actions). Those two ask a language model for one
low-level action -- turn/forward/pickup -- at every single step. This
one asks a different, much better-posed question:

    'What should the agent go and do next?'

and answers it as a SUB-GOAL: a target cell plus an interaction
('go to the purple key at (1, 5) and pick it up'). The route to that
cell is then computed by teachers/minigrid/grid_nav.py, an exact BFS
over the live grid. The model is re-queried only when the sub-goal is
achieved, becomes impossible, or turns out to be stale.

Why split it this way
---------------------
1. It plays to the model's strength and away from its weakness. The
   DoorKey study (docs/progress_june_slides.md, slide 11a) found the
   failure mode of the API teachers is spatial grounding -- which
   cell is in front, which way to turn -- not task understanding.
   Choosing 'fetch the key that opens that door' is semantics;
   turning left twice is geometry. BFS never mis-grounds.
2. It costs 10-50x less. One call per sub-goal instead of one call
   per primitive action.
3. It cannot oscillate. A shortest path is monotone progress by
   construction, so the turn-back-and-forth failure the per-step
   teachers need prompt warnings and action history to avoid is
   simply unreachable here.
4. Its output is already the unit Stage 2 measures. The
   embedding / vector-quantization / probe work
   (docs/clarification_explanation_vs_conditioning.md) all treat the
   SUB-GOAL as the thing that should transfer. This teacher emits it
   directly and discretely, instead of it having to be recovered
   from verbose free text.

Like the other API teachers this needs internet, so it runs locally
rather than on cluster compute nodes.
"""

import json
import time

from envs.state import extract_generic_state
from teachers.aleph import create_teacher_response
from teachers.base import (
    Advice,
    BaseTeacher,
    Cost,
    build_openai_client,
    call_with_cold_start_retry,
)
from teachers.minigrid.grid_nav import (
    ACTION_DROP,
    ACTION_PICKUP,
    ACTION_TOGGLE,
    find_object,
    plan_first_action,
)
from teachers.minigrid.llm_general import render_ascii_map
from teachers.minigrid.llm_prompts import estimate_dollars

# The interactions a sub-goal can ask for, mapped to the primitive
# action to take once the agent is in position. 'goto' and 'enter'
# map to None: arriving IS the whole sub-goal, so on arrival we
# immediately ask for the next one.
INTERACTION_ACTIONS = {
    'goto': None,
    'enter': None,
    'pickup': ACTION_PICKUP,
    'toggle': ACTION_TOGGLE,
    'drop': ACTION_DROP,
}

# How the agent must approach the target for each interaction.
# Everything except 'enter' stops one cell short and faces the
# target, because MiniGrid forbids standing on an object's cell.
INTERACTION_APPROACH = {
    'goto': 'adjacent_facing',
    'enter': 'enter',
    'pickup': 'adjacent_facing',
    'toggle': 'adjacent_facing',
    'drop': 'adjacent_facing',
}

# Object types the agent is allowed to pick up.
_CARRIABLE = ('key', 'ball', 'box')

# Structured-output schema. Mirrors RESPONSE_SCHEMA in
# teachers/minigrid/llm.py, but returns a sub-goal rather than a
# primitive action. Every property is listed in `required` and
# additionalProperties is False, as OpenAI's strict json_schema mode
# demands.
SUBGOAL_SCHEMA = {
    'format': {
        'type': 'json_schema',
        'name': 'minigrid_subgoal',
        'schema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'reasoning': {'type': 'string'},
                'target_x': {'type': 'integer'},
                'target_y': {'type': 'integer'},
                'interaction': {
                    'type': 'string',
                    'enum': list(INTERACTION_ACTIONS),
                },
                'confidence': {'type': 'number'},
            },
            'required': [
                'reasoning',
                'target_x',
                'target_y',
                'interaction',
                'confidence',
            ],
        },
    }
}


def build_subgoal_prompt(
    env_id, mission, facing, carrying, objects, ascii_map,
    subgoal_history, note,
):
    """
    Build the sub-goal prompt.

    Deliberately shorter than the per-step prompt in llm_general.py:
    every rule about turning, facing, and walking into walls is gone,
    because the model no longer chooses movement -- a planner does.
    What remains is the task logic (keys unlock same-colored doors,
    the mission names the target) plus a record of the sub-goals
    already attempted this episode and why the last one ended.

    Parameters
    ----------
    env_id, mission, facing, carrying, objects, ascii_map
        Same meanings as in llm_general.build_generic_llm_prompt.
    subgoal_history: list
        (target, interaction, outcome) triples already attempted
        this episode, oldest first.
    note: str
        Why the previous sub-goal ended, when it ended badly (e.g.
        unreachable). '' when there is nothing to report.
    """

    object_lines = []
    for kind, color, x, y, door_state in objects:
        suffix = f', {door_state}' if door_state else ''
        object_lines.append(f'- {color} {kind} at ({x}, {y}){suffix}')
    object_block = (
        '\n'.join(object_lines) if object_lines
        else '- (no objects on the map)'
    )

    # Replay the episode's sub-goal attempts so the model does not
    # re-propose something it has already achieved or already failed.
    if subgoal_history:
        lines = [
            f'- {interaction} at ({t[0]}, {t[1]}) -> {outcome}'
            for t, interaction, outcome in subgoal_history
        ]
        history_block = (
            '\nSUB-GOALS ALREADY ATTEMPTED THIS EPISODE '
            '(oldest first):\n' + '\n'.join(lines) + '\n'
        )
    else:
        history_block = ''

    note_block = f'\nIMPORTANT: {note}\n' if note else ''

    return (
        'You are the planner for an agent in a MiniGrid environment '
        f'({env_id}). Mission: "{mission}"\n'
        '\n'
        'You do NOT control movement. A shortest-path planner walks '
        'the agent wherever you send it and handles all turning and '
        'facing. Your only job is to choose the next SUB-GOAL: '
        'which cell to go to, and what to do on arrival.\n'
        '\n'
        'Coordinate system: x grows EAST (right), y grows SOUTH '
        '(down); (0, 0) is the top-left corner.\n'
        '\n'
        'MAP of the whole grid (one character per cell):\n'
        '- # = wall, . = empty floor\n'
        '- D = door, k = key, o = ball, B = box, G = goal, '
        'L = lava\n'
        '- The arrow (> v < ^) is the agent.\n'
        '\n'
        f'{ascii_map}\n'
        '\n'
        'Objects (colors and door states, matching the map by '
        'coordinates):\n'
        f'{object_block}\n'
        '\n'
        'STATUS (trust these lines):\n'
        f'- The agent is at the arrow, facing {facing}.\n'
        f'- The agent is carrying: {carrying or "nothing"}. A '
        'carried object is NOT on the map.\n'
        f'{history_block}'
        f'{note_block}'
        '\n'
        'Task rules:\n'
        '- A LOCKED door needs a key of the SAME color. Fetch that '
        'key first, then toggle the door.\n'
        '- A CLOSED but unlocked door just needs a toggle.\n'
        '- The agent can carry only one object at a time; drop what '
        'it holds before picking up something else.\n'
        '- A key is a TOOL for a matching locked door, never the '
        'mission target unless the mission says "key".\n'
        '- If the mission target is behind a shut door, your '
        'sub-goal must be that door (or its key), not the target: '
        'the planner cannot walk through it.\n'
        '\n'
        'Choose ONE sub-goal, as a target cell plus an '
        'interaction:\n'
        '- "goto": stand next to the cell, facing it. Use this to '
        'satisfy a "go to X" mission.\n'
        '- "enter": stand ON the cell. Use this for the goal '
        'square (G) or a plain floor cell.\n'
        '- "pickup": stand next to the object and pick it up.\n'
        '- "toggle": stand next to the door and open it.\n'
        '- "drop": stand next to the cell and drop what is held.\n'
        '\n'
        'Give the target as the coordinates of the OBJECT or CELL '
        'itself, not of a cell beside it -- the planner works out '
        'where to stand.\n'
        '\n'
        'What is the next sub-goal?'
    )


class MiniGridSubgoalLLMTeacher(BaseTeacher):
    """
    Language-model planner plus exact BFS navigator.

    Expected inputs
    ---------------
    recommend(state, context):
      - state: the unwrapped MiniGrid env, same call-site shape as
        the bot, llm_general, and vlm_general teachers, so training
        loops treat all of them identically.
      - context['mission']: the episode's mission string.
      - context['new_episode']: True on the first call of a new
        episode, so the sub-goal and its history are cleared.

    Every Advice carries the active sub-goal in
    `cost.metadata['subgoal']` as a (x, y, interaction) triple -- a
    ready-made discrete label for the Stage-2 probe and codebook
    work, which otherwise has to recover the sub-goal from free
    text.
    """

    def __init__(
        self,
        env_id: str = 'BabyAI-KeyCorridorS3R3-v0',
        model: str = 'gpt-4.1-mini',
        strict: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        seed: int = 0,
        max_queries_per_step: int = 3,
        stuck_limit: int = 4,
        verbose: bool = False,
    ):
        """
        Initialize the teacher and validate API access eagerly, so a
        missing key fails before any run starts rather than mid-way.

        `max_queries_per_step` bounds how many sub-goals may be
        proposed while producing a single action, so a model that
        keeps naming unreachable targets costs a bounded amount and
        terminates. `stuck_limit` forces a re-plan after that many
        consecutive steps with no state change at all -- the
        backstop against any residual no-progress loop. `verbose`
        prints each proposed sub-goal and why it was accepted or
        rejected, which is how a failing run is diagnosed cheaply.
        """

        super().__init__(
            teacher_id=f'minigrid_llm_subgoal:{model}', seed=seed
        )

        self.env_id = env_id
        self.model = model
        self.strict = strict
        self.max_queries_per_step = max_queries_per_step
        self.stuck_limit = stuck_limit
        self.verbose = verbose

        # Builds either an OpenAI client or a free Aleph (Vulcan)
        # client depending on the LLM_PROVIDER env var -- see
        # teachers/base.py for the switch. A missing key fails here,
        # before any run starts, not mid-episode.
        self.client = build_openai_client(
            timeout=timeout, max_retries=max_retries
        )

        # Active sub-goal as (target, interaction, reasoning,
        # confidence), plus the episode's attempt log and the reason
        # the last attempt ended.
        self._subgoal = None
        self._history = []
        self._note = ''
        self._prev_state_key = None
        self._stuck_steps = 0

        self.num_api_calls = 0
        self.num_failures = 0
        self.num_replans = 0

    def reset_episode(self):
        """
        Clear all per-episode planning state.
        """

        self._subgoal = None
        self._history = []
        self._note = ''
        self._prev_state_key = None
        self._stuck_steps = 0

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return the next primitive action.

        Queries the model only when there is no usable sub-goal;
        otherwise this is a pure, free BFS step toward the sub-goal
        already in hand.
        """

        t0 = time.perf_counter()

        mission = ''
        new_episode = False
        if context is not None:
            mission = str(context.get('mission', ''))
            new_episode = bool(context.get('new_episode', False))
        if new_episode:
            self.reset_episode()

        generic = extract_generic_state(state)
        state_key = generic

        # Backstop: if nothing at all has changed for several steps,
        # whatever we are doing is not working. Throw the sub-goal
        # away and make the model pick something else.
        if state_key == self._prev_state_key:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._prev_state_key = state_key

        if self._stuck_steps >= self.stuck_limit and self._subgoal:
            self._finish_subgoal('made no progress')
            self._note = (
                'The previous sub-goal produced no change for '
                f'{self._stuck_steps} steps. Choose a DIFFERENT one.'
            )
            self._stuck_steps = 0

        # Accumulated across every model call made while producing
        # this one action, so the reported cost is per-step honest.
        spend = {'dollars': 0.0, 'tin': 0, 'tout': 0, 'nodes': 0}
        queries = 0

        while True:
            # Get a sub-goal if we do not have a usable one.
            if self._subgoal is None:
                if queries >= self.max_queries_per_step:
                    return self._abstain(
                        t0, spend,
                        'no reachable sub-goal after '
                        f'{queries} proposals',
                    )
                queries += 1
                ok = self._query_subgoal(
                    state, generic, mission, spend, t0
                )
                if ok is not None:
                    # Hard API failure in non-strict mode.
                    return ok

            target, interaction, reasoning, confidence = self._subgoal
            if self.verbose:
                print(
                    f'    proposed: {interaction} at {target} '
                    f'(conf {confidence:.2f}) -- {reasoning}'
                )

            # Reject sub-goals that no longer make sense before
            # spending a BFS on them.
            problem = self._validate(state, target, interaction)
            if problem:
                if self.verbose:
                    print(f'    REJECTED (invalid): {problem}')
                self._finish_subgoal(problem)
                self._note = (
                    f'Your last sub-goal was rejected: {problem}.'
                )
                continue

            nav = plan_first_action(
                state, target, INTERACTION_APPROACH[interaction]
            )
            spend['nodes'] += nav.nodes_expanded

            if nav.status == 'unreachable':
                if self.verbose:
                    print('    REJECTED (unreachable): '
                          f'{target} is walled off right now')
                self._finish_subgoal('unreachable')
                self._note = (
                    f'Cell {target} could not be reached with the '
                    'doors in their current state. Open the door '
                    'blocking it (or fetch its key) first.'
                )
                continue

            if nav.status == 'path':
                # Still walking: hand back the planner's step.
                return self._advice(
                    nav.action, confidence, reasoning, target,
                    interaction, 'navigating', spend, t0,
                )

            # nav.status == 'ready': in position at last.
            action = INTERACTION_ACTIONS[interaction]
            if action is None:
                # Arriving was the entire sub-goal, so it is done and
                # we ask for the next one on this same step.
                self._finish_subgoal('reached')
                self._note = ''
                continue

            # Issue the interaction, then retire the sub-goal so the
            # next step re-plans from the resulting state.
            self._finish_subgoal(f'{interaction} issued')
            self._note = ''
            return self._advice(
                action, confidence, reasoning, target, interaction,
                'interacting', spend, t0,
            )

    def _query_subgoal(self, state, generic, mission, spend, t0):
        """
        Ask the model for one sub-goal and store it.

        Returns None on success, or an abstaining Advice when the API
        call fails and `strict` is off (in strict mode it raises).
        """

        _ax, _ay, agent_dir, carrying, objects = generic
        facing = ('east', 'south', 'west', 'north')[agent_dir]

        prompt = build_subgoal_prompt(
            self.env_id, mission, facing, carrying, objects,
            render_ascii_map(state), self._history, self._note,
        )

        try:
            self.num_api_calls += 1
            response = call_with_cold_start_retry(lambda: (
                create_teacher_response(
                    self.client,
                    model=self.model, input=prompt, text=SUBGOAL_SCHEMA
                )
            ))
            data = json.loads(response.output_text)
            # Kept inside this try block: a well-formed JSON
            # document can still hold garbage values (an absurdly
            # large 'confidence', a missing key), and letting that
            # raise here would crash the whole job instead of
            # failing just this one call.
            subgoal = (
                (int(data['target_x']), int(data['target_y'])),
                str(data['interaction']),
                str(data['reasoning']),
                max(0.0, min(1.0, float(data['confidence']))),
            )
        except Exception as exc:
            self.num_failures += 1
            if self.strict:
                raise RuntimeError(
                    f'MiniGridSubgoalLLMTeacher API call or '
                    f'response parsing failed mission={mission!r} '
                    f'(model={self.model!r}): '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
            return self._abstain(
                t0, spend, f'{type(exc).__name__}: {exc}'
            )

        usage = getattr(response, 'usage', None)
        tokens_in = int(usage.input_tokens) if usage else 0
        tokens_out = int(usage.output_tokens) if usage else 0
        spend['tin'] += tokens_in
        spend['tout'] += tokens_out
        spend['dollars'] += estimate_dollars(
            self.model, tokens_in, tokens_out
        )

        self._subgoal = subgoal
        self.num_replans += 1
        return None

    def _validate(self, unwrapped, target, interaction):
        """
        Check that a proposed sub-goal is still coherent.

        Returns a short reason string when it is not, or '' when it
        is fine. Catching these here rather than in the prompt means
        a hallucinated coordinate costs one extra model call instead
        of an entire wasted episode.
        """

        x, y = target
        u = unwrapped
        if not (0 <= x < u.width and 0 <= y < u.height):
            return f'({x}, {y}) is outside the {u.width}x{u.height} grid'

        cell = find_object(u, x, y)
        kind = cell.type if cell is not None else 'empty floor'

        if kind == 'wall':
            return f'({x}, {y}) is a wall'

        if interaction == 'pickup':
            if kind not in _CARRIABLE:
                return f'there is no pickable object at ({x}, {y})'
            if u.carrying is not None:
                return (
                    'the agent already carries '
                    f'{u.carrying.color} {u.carrying.type}; drop it '
                    'first'
                )
        elif interaction == 'toggle':
            if kind != 'door':
                return f'there is no door at ({x}, {y})'
            if cell.is_open:
                return f'the door at ({x}, {y}) is already open'
        elif interaction == 'drop':
            if u.carrying is None:
                return 'the agent is not carrying anything to drop'
        elif interaction == 'enter':
            if kind not in ('empty floor', 'goal'):
                return (
                    f'({x}, {y}) holds a {kind}, which cannot be '
                    'stood on; use "goto" instead'
                )

        return ''

    def _finish_subgoal(self, outcome):
        """
        Retire the active sub-goal, recording how it ended so the
        next prompt can show the model its own track record.
        """

        if self._subgoal is not None:
            target, interaction, _reasoning, _conf = self._subgoal
            self._history.append((target, interaction, outcome))
        self._subgoal = None

    def _advice(
        self, action, confidence, reasoning, target, interaction,
        stage, spend, t0,
    ):
        """
        Package one primitive action as Advice, tagged with the
        sub-goal that produced it.
        """

        return Advice(
            action=int(action),
            confidence=confidence,
            explanation=(
                f'sub-goal: {interaction} at {target} -- {reasoning}'
            ),
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                dollars=spend['dollars'],
                tokens_in=spend['tin'],
                tokens_out=spend['tout'],
                compute_units=spend['nodes'],
                model=self.model,
                metadata={
                    'subgoal': (target[0], target[1], interaction),
                    'stage': stage,
                },
            ),
        )

    def _abstain(self, t0, spend, reason):
        """
        Build an abstaining Advice for the non-strict failure path
        and the exhausted-proposals path.
        """

        return Advice(
            action=None,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                dollars=spend['dollars'],
                tokens_in=spend['tin'],
                tokens_out=spend['tout'],
                compute_units=spend['nodes'],
                model=self.model,
                metadata={'failed': True, 'error': reason},
            ),
        )

    def stats(self) -> dict:
        """
        Summary counters for end-of-run reporting.
        """

        return {
            'model': self.model,
            'env_id': self.env_id,
            'num_api_calls': self.num_api_calls,
            'num_replans': self.num_replans,
            'num_failures': self.num_failures,
        }
