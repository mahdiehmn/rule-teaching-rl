"""
General-purpose text-only LLM teacher for any MiniGrid/BabyAI task.

The text sibling of teachers/minigrid/vlm_general.py, and the
generalization of the frozen, DoorKey-only teachers/minigrid/llm.py.
Instead of showing the model a rendered image (vlm_general) or a
hand-written description of one fixed topology (llm.py), this
teacher describes the CURRENT layout as text: an ASCII map of the
whole grid (walls, doors, objects, the agent and its facing
direction) plus a list of every object with its color, position,
and door state. The model never sees pixels.

Why this teacher exists alongside vlm_general: the DoorKey
experiments found (docs/progress_june_slides.md, slide 11a) that
the VLM's failure mode is PERCEPTION -- reading egocentric spatial
relations out of the rendered image -- while a text LLM given the
same facts as text reasons fine. This teacher removes exactly that
failure mode for the general tasks, so comparing it against
vlm_general on the same task isolates "can the model plan" from
"can the model see": same model, same task, same channel, only the
input representation differs.

Shares the structured-output schema and pricing table with the
other API teachers by import (no duplication), and mirrors
vlm_general's caching (by generic symbolic state + mission) and
strict/non-strict failure handling exactly.
"""

import json
import time
from collections import deque

from envs.state import extract_generic_state
from teachers.aleph import create_teacher_response
from teachers.base import (
    Advice,
    BaseTeacher,
    Cost,
    build_openai_client,
    call_with_cold_start_retry,
)
from teachers.minigrid.llm import RESPONSE_SCHEMA
from teachers.minigrid.llm_prompts import estimate_dollars
from teachers.reliability import failure_details, redact

# Direction names and arrow characters indexed by MiniGrid's dir
# convention: 0=east, 1=south, 2=west, 3=north.
_DIR_NAMES = ('east', 'south', 'west', 'north')
_DIR_ARROWS = ('>', 'v', '<', '^')

# One map character per world-object type. Colors and door states
# are NOT encoded in the map (one char cannot carry them); they are
# listed per object, with coordinates, in the prompt's object list,
# and the coordinates tie the two together.
_TYPE_CHARS = {
    'wall': '#',
    'door': 'D',
    'key': 'k',
    'ball': 'o',
    'box': 'B',
    'goal': 'G',
    'lava': 'L',
}

# Action names for the recent-history block, so the model reads back
# what it did in words rather than as bare integers.
#
# All SEVEN MiniGrid actions, including DONE. The student's actor
# samples from Discrete(7), so action 6 is a perfectly ordinary thing
# for it to execute -- it was invisible in teacher-alone evaluation only
# because the teacher's recommendation schema never proposes it. Leaving
# DONE out made the history call it an unknown action and, worse, made
# the blocked-action warning index past the end of this tuple and crash
# the next consultation. Use len(_ACTION_NAMES) rather than a literal
# so the two stay in step.
_ACTION_NAMES = (
    'TURN LEFT', 'TURN RIGHT', 'FORWARD', 'PICKUP', 'DROP', 'TOGGLE',
    'DONE',
)


def _action_name(action):
    """
    Name an action, tolerating ids outside the known vocabulary.

    Formatting happens before the API call's exception handler, so an
    unchecked index here takes down the whole consultation. Anything
    unrecognized degrades to a readable placeholder instead.
    """

    if isinstance(action, int) and 0 <= action < len(_ACTION_NAMES):
        return _ACTION_NAMES[action]
    return f'UNKNOWN ACTION {action}'


def render_ascii_map(unwrapped):
    """
    Render the env's full grid as an ASCII map string.

    One character per cell, row by row from north (top) to south
    (bottom); x grows east. The agent's cell is drawn as an arrow
    showing its facing direction, overriding whatever it stands on.
    Kept module-level and client-free so it is testable offline.
    """

    u = unwrapped
    rows = []
    for y in range(u.height):
        row = []
        for x in range(u.width):
            cell = u.grid.get(x, y)
            if cell is None:
                row.append('.')
            else:
                row.append(_TYPE_CHARS.get(cell.type, '?'))
        rows.append(row)

    # Draw the agent last so it is always visible, with its facing
    # direction encoded in the arrow character.
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    rows[ay][ax] = _DIR_ARROWS[int(u.agent_dir)]

    return '\n'.join(''.join(row) for row in rows)


def input_token_ceiling():
    """
    Maximum input tokens per request, or 0 for no ceiling.

    Companion to the output ceiling. Both come from the environment for
    the same reason: a budgeted trainer resolves its reservation after
    its teachers already exist.
    """

    import os

    try:
        return max(0, int(os.getenv('LLM_MAX_INPUT_TOKENS', '') or 0))
    except ValueError:
        return 0


def estimated_tokens(text):
    """
    Bound text tokens by UTF-8 bytes for the OpenAI byte-BPE tokenizer.

    Dividing character count by four undercounts maps and coordinates.
    One token per byte is conservative for text; callers must also
    include the schema and allow for request framing.
    """

    return len(text.encode('utf-8'))


def output_token_ceiling():
    """
    Maximum output tokens per request, or 0 for no ceiling.

    Sourced from the environment so every OpenAI-backed teacher picks
    it up without a parameter threaded through the factory, and read at
    call time because a budgeted trainer resolves its reservation after
    its teachers already exist. Unset reproduces the previous
    behaviour: no ceiling is sent at all.
    """

    import os

    try:
        return max(0, int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '') or 0))
    except ValueError:
        return 0


def format_history_block(history):
    """
    Render recent (action, blocked) pairs as a prompt section.

    Without this the teacher is stateless: every call is an
    independent snapshot, so the model cannot tell that the action it
    just recommended did nothing. That is the mechanism behind the
    classic oscillation and wall-bumping failures, and the reason the
    prompt used to carry the (unenforceable) instruction 'do not turn
    back and forth in place'.

    `blocked` means the action left the symbolic state completely
    unchanged -- walking into a wall, toggling nothing, picking up
    empty air. Repeats of a blocked action are called out explicitly
    because that is the pattern the model must break.

    Returns '' for an empty history so the first step of an episode
    reads exactly like the original stateless prompt.
    """

    if not history:
        return ''

    lines = []
    for age, (action, blocked) in enumerate(reversed(history), 1):
        # Three states, not two. `None` means the action ran but its
        # individual effect was never observed -- several actions
        # elapsed between consultations, or it was the first query of
        # the episode. Printing 'state changed' there would assert a
        # success we did not measure.
        if blocked is None:
            outcome = 'ran; effect not recorded'
        elif blocked:
            outcome = 'NO EFFECT -- you were blocked'
        else:
            outcome = 'state changed'
        name = _action_name(action)
        lines.append(
            f'- {age} step(s) ago: action {action} ({name}) '
            f'-> {outcome}'
        )
    # Oldest first reads like a narrative of how we got here.
    lines.reverse()

    # Warn loudly when the most recent action achieved nothing, since
    # repeating it is guaranteed to achieve nothing again.
    warning = ''
    last_action, last_blocked = history[-1]
    if last_blocked:
        warning = (
            f'\nWARNING: action {last_action} '
            f'({_action_name(last_action)}) just had NO EFFECT. '
            'Repeating it will fail again -- the way is blocked or '
            'there is nothing there. Turn, or route around it.'
        )

    return (
        '\nRECENT HISTORY (what you already tried this episode):\n'
        + '\n'.join(lines)
        + warning
        + '\n'
    )


def build_generic_llm_prompt(
    env_id, mission, agent_dir, carrying, objects, ascii_map,
    history_block='',
):
    """
    Build the full text prompt for the general LLM teacher.

    Unlike the DoorKey prompts in llm_prompts.py, nothing here
    assumes a fixed topology: the ASCII map carries the actual wall
    layout, and the object list enumerates whatever this episode's
    layout contains. All spatial facts the DoorKey VLM needed the
    image for are given as text instead.

    Parameters
    ----------
    env_id: str
        Gym id, for the prompt header only.
    mission: str
        The episode's mission string; often the only place the
        target object's identity appears.
    agent_dir: int
        MiniGrid facing direction (0=east, 1=south, 2=west,
        3=north), stated in words as well as by the map arrow.
    carrying: str
        What the agent holds right now (e.g. 'red key'), or '' if
        empty-handed. Carried objects appear nowhere on the map.
    objects: tuple
        The `objects` field from envs.state.extract_generic_state:
        (kind, color, x, y, door_state) tuples.
    ascii_map: str
        Output of render_ascii_map for the current state.
    history_block: str
        Output of format_history_block; '' at the first step of an
        episode, which reproduces the original stateless prompt
        exactly.
    """

    object_lines = []
    for kind, color, x, y, door_state in objects:
        state_suffix = f', {door_state}' if door_state else ''
        object_lines.append(
            f'- {color} {kind} at ({x}, {y}){state_suffix}'
        )
    object_block = (
        '\n'.join(object_lines)
        if object_lines
        else '- (no objects visible)'
    )
    carrying_str = carrying if carrying else 'nothing'
    facing = _DIR_NAMES[agent_dir]

    return (
        'You are coaching an agent in a MiniGrid environment '
        f'({env_id}). Mission: "{mission}"\n'
        '\n'
        'Coordinate system: x grows EAST (right), y grows SOUTH '
        '(down). Position (0, 0) is the top-left corner. The map '
        'may contain several rooms connected by doors and '
        'corridors.\n'
        '\n'
        'MAP of the whole grid (one character per cell):\n'
        '- # = wall, . = empty floor\n'
        '- D = door, k = key, o = ball, B = box, G = goal, '
        'L = lava\n'
        '- The arrow (> v < ^) is the agent, pointing the way it '
        'faces. The cell directly in front is the adjacent cell '
        'the arrow points at.\n'
        '\n'
        f'{ascii_map}\n'
        '\n'
        'Objects (colors and door states, matching the map by '
        'coordinates):\n'
        f'{object_block}\n'
        '\n'
        'STATUS (trust these lines):\n'
        f'- You are at the arrow, facing {facing}.\n'
        f'- You are carrying: {carrying_str}. A carried object is '
        'NOT on the map, so do not try to pick it up again.\n'
        f'{history_block}'
        '\n'
        'Rules:\n'
        '- You can only move onto empty floor, open doors, or the '
        'goal -- never through a wall (#) or a closed/locked '
        'door.\n'
        '- A LOCKED door can only be opened by facing it while '
        'carrying a key of the SAME color, then using action 5 '
        '(TOGGLE); this both unlocks and opens it in one action.\n'
        '- A CLOSED (unlocked) door opens with action 5 (TOGGLE) '
        'while facing it, no key needed.\n'
        '- Pick up an object with action 3 (PICKUP) while facing '
        'it and holding nothing (or after dropping what you hold '
        'with action 4).\n'
        '- The mission tells you the target; a key is a TOOL for a '
        'matching-colored locked door, never the mission target '
        'unless the mission itself says "key".\n'
        '- "Go to X" is satisfied by standing on the cell directly '
        'ADJACENT to X, facing it -- you cannot stand on an '
        'object\'s own cell.\n'
        '\n'
        'Plan a route on the map toward whatever the mission still '
        'requires (find/collect a key if a relevant door is '
        'locked, open that door, then reach the target). Walls '
        'often force a detour: moving AWAY from the target to get '
        'around a wall is correct. Do not turn back and forth in '
        'place.\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place (counter-clockwise).\n'
        '- 1: turn RIGHT in place (clockwise).\n'
        '- 2: move FORWARD one cell in the direction the arrow '
        'points.\n'
        '- 3: PICK UP the object directly in front.\n'
        '- 4: DROP the carried object in front.\n'
        '- 5: TOGGLE the door directly in front (see rules '
        'above).\n'
        '\n'
        'What ONE action should the agent take next?'
    )


class MiniGridGeneralLLMTeacher(BaseTeacher):
    """
    Text-only language-model teacher for arbitrary MiniGrid/BabyAI
    layouts.

    Expected inputs
    ---------------
    recommend(state, context):
      - state: the unwrapped MiniGrid env. Used both for the cache
        key (via envs.state.extract_generic_state) and to render
        the ASCII map -- the same call-site shape as the bot and
        vlm_general teachers, so training loops treat all three
        identically.
      - context['mission']: the episode's mission string.
      - context['new_episode']: True on the first call of a new
        episode, so the per-episode action history is cleared.
        Callers that never pass it get a history that leaks across
        episode boundaries, so pass it (or call reset_episode()).

    Episode memory
    --------------
    The teacher remembers the last `history_len` actions it advised
    and whether each one changed the symbolic state at all. This is
    what makes it possible to escape a dead end, and it also closes a
    real bug: the cache used to be keyed on the symbolic state alone,
    so an action that left the state unchanged (walking into a wall)
    produced a cache hit returning that same action forever -- a
    silent, free, deterministic loop until the episode truncated.
    Folding the recent history into the key means a repeat of the
    same state after a failed action is a genuinely different
    situation, and gets a fresh answer.
    """

    def __init__(
        self,
        env_id: str = 'BabyAI-GoToSeq-v0',
        model: str = 'gpt-4.1-mini',
        strict: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        seed: int = 0,
        history_len: int = 6,
    ):
        """
        Initialize the teacher and validate API access eagerly, so
        a missing key fails before any training starts rather than
        mid-run.
        """

        super().__init__(
            teacher_id=f'minigrid_llm_general:{model}', seed=seed
        )

        self.env_id = env_id
        self.model = model
        self.strict = strict
        self.history_len = history_len
        # Opt-in experiment; existing teacher prompts remain unchanged.
        self.structured_explanations = False
        # '' (unchanged prompt), 'cite' or 'cite_view': see
        # teachers/evidence.py. Opt-in, like structured_explanations.
        self.evidence_citations = ''

        # Per-episode memory: recent (action, blocked) pairs, the
        # previous symbolic state used to detect no-op actions, and
        # the action awaiting its outcome.
        self._history = deque(maxlen=history_len)
        self._prev_state_key = None
        self._pending_action = None

        # Builds either an OpenAI client or a free Aleph (Vulcan)
        # client depending on the LLM_PROVIDER env var -- see
        # teachers/base.py for the switch. A missing key fails here,
        # before any run starts, not mid-episode.
        self.client = build_openai_client(
            timeout=timeout, max_retries=max_retries
        )

        self._cache: dict[tuple, Advice] = {}
        self.num_api_calls = 0
        self.num_cache_hits = 0
        self.num_failures = 0

    def reset_episode(self):
        """
        Forget the current episode's action history.

        Called automatically when context['new_episode'] is set;
        exposed publicly for callers that track episode boundaries
        themselves.
        """

        self._history.clear()
        self._prev_state_key = None
        self._pending_action = None

    def _observe_outcome(self, state_key, executed=None):
        """
        Record what the STUDENT actually did since the last query, and
        append it to the history.

        `executed` is the list of actions the student really executed
        since the previous query, oldest first. When it is None the
        teacher falls back to the action it advised, which is correct
        only when the caller executes the teacher's advice verbatim --
        i.e. teacher-alone evaluation.

        Why this matters: during distillation the student picks its
        own action, so recording the advised action makes the prompt
        history a fiction, and every later rationale is conditioned on
        a trajectory that never happened. The history is also folded
        into the cache key, so a wrong history additionally corrupts
        cache identity.

        An action is recorded as `blocked` when the symbolic state is
        byte-for-byte identical to the one it was issued in. Because
        that state carries position, facing, what is carried, and
        every object's position and door state, an unchanged state
        means the action provably accomplished nothing -- a wall
        bump, a toggle of thin air, a pickup with nothing in front.
        That comparison is only valid across a SINGLE action with a
        previous state to compare against. Every other entry is recorded
        with an outcome of `None`, meaning the action ran and its effect
        was never observed -- several actions elapsed between
        consultations, or this was the episode's first query. `None` is
        deliberately distinct from `False`: claiming the state changed
        would assert a measurement that was never made, which is the
        same error in the opposite direction as claiming it was blocked.
        """

        if executed is not None:
            # `[]` means "the student executed nothing since the last
            # consultation" -- two queries with no environment step
            # between them. That is NOT the same as `None`, which means
            # the caller did not say, and only then may we assume our
            # own advice was carried out.
            for i, taken in enumerate(executed):
                last = i == len(executed) - 1
                if last and len(executed) == 1                         and self._prev_state_key is not None:
                    # The only case supporting a real judgement: one
                    # action, and a previous state to compare against.
                    outcome = state_key == self._prev_state_key
                else:
                    # Unknown: several actions elapsed with no
                    # intermediate states, or this is the first query
                    # of the episode so there is nothing to compare.
                    # Claiming the state changed would be as unfounded
                    # as claiming it did not.
                    outcome = None
                self._history.append((int(taken), outcome))
        elif self._pending_action is not None:
            blocked = state_key == self._prev_state_key
            self._history.append((self._pending_action, blocked))
        self._prev_state_key = state_key
        self._pending_action = None

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return advice for the env's current state, described as
        text (ASCII map + object list + recent action history),
        hitting the cache when the same state recurs after the same
        recent history.
        """

        t0 = time.perf_counter()

        mission = ''
        new_episode = False
        executed = None
        if context is not None:
            mission = str(context.get('mission', ''))
            new_episode = bool(context.get('new_episode', False))
            # What the student ACTUALLY executed since the previous
            # query. Accepts a list (query_interval > 1) or a single
            # action. Absent means "caller executed our advice", which
            # is true for teacher-alone evaluation only.
            executed = context.get('executed_actions')
            if executed is None:
                single = context.get('last_action')
                # Absence stays None so the teacher-alone fallback
                # survives; an explicit empty list stays empty.
                executed = None if single is None else [single]
            else:
                executed = list(executed)

        if new_episode:
            self.reset_episode()

        agent_x, agent_y, agent_dir, carrying, objects = (
            extract_generic_state(state)
        )
        state_key = (
            agent_x, agent_y, agent_dir, carrying, objects,
        )
        # Judge the last action before building this step's prompt,
        # so the history the model reads includes what just happened.
        self._observe_outcome(state_key, executed)
        history_block = format_history_block(self._history)

        # The history is part of the key, not just the prompt: the
        # same state reached after a different recent history is a
        # different situation and must not reuse the old answer.
        cache_key = (
            self.model, mission, state_key, tuple(self._history),
        )
        reference_action = (context or {}).get('reference_action')
        if self.structured_explanations:
            cache_key += ('structured-v1', reference_action)
        if self.evidence_citations:
            cache_key += ('evidence-v1', self.evidence_citations)

        if cache_key in self._cache:
            self.num_cache_hits += 1
            cached = self._cache[cache_key]
            self._pending_action = cached.action
            return Advice(
                action=cached.action,
                forbidden_actions=cached.forbidden_actions,
                confidence=cached.confidence,
                explanation=cached.explanation,
                teacher_id=self.teacher_id,
                call_id=self._next_call_id(),
                cost=Cost(
                    wall_time_s=time.perf_counter() - t0,
                    model=self.model,
                    metadata={'cache_hit': True, 'outcome': 'cache_hit',
                              'subgoal': cached.cost.metadata.get('subgoal'),
                              'evidence_object':
                                  cached.cost.metadata.get('evidence_object'),
                              'evidence_color':
                                  cached.cost.metadata.get('evidence_color'),
                              'evidence_cell':
                                  cached.cost.metadata.get('evidence_cell'),
                              'teacher_action': cached.action,
                              'attempt_count': 0, 'usage_known': True,
                              'cost_known': True},
                ),
            )

        prompt = build_generic_llm_prompt(
            self.env_id,
            mission,
            agent_dir,
            carrying,
            objects,
            render_ascii_map(state),
            history_block=history_block,
        )

        schema = RESPONSE_SCHEMA
        if self.evidence_citations:
            from teachers.evidence import evidence_prompt, evidence_schema
            schema = evidence_schema(schema)
            prompt += evidence_prompt(
                state, render_ascii_map(state),
                show_view=self.evidence_citations == 'cite_view')
        if self.structured_explanations:
            from teachers.controlled_advice import rationale_schema
            schema = rationale_schema(schema, reference_action)
            prompt += ('\nAlso classify the immediate subgoal using the '
                       'subgoal enum in the response schema.')
            if reference_action is not None:
                prompt += (f'\nA deterministic planner supplies action '
                           f'{reference_action}. Explain that action in '
                           'this state and return it unchanged.')

        # Refuse a prompt the reservation did not cover, BEFORE sending
        # it. Reserving against an input ceiling that nothing checks
        # would make the ceiling a comment rather than a bound. The
        # margin covers schema overhead and the tokenizer being less
        # efficient than the estimate on ASCII maps.
        ceiling = input_token_ceiling()
        if ceiling:
            estimate = (estimated_tokens(prompt)
                        + estimated_tokens(json.dumps(schema))
                        + 1024)
            if estimate > ceiling:
                self.num_failures += 1
                return self._abstain(
                    t0,
                    reason=(f'prompt estimated at {estimate} tokens '
                            f'exceeds the budgeted input ceiling of '
                            f'{ceiling}; not sending an unreserved '
                            f'request'),
                    metadata={'outcome': 'input_bound', 'attempt_count': 0,
                              'usage_known': True, 'cost_known': True},
                )

        response = None
        attempt_count = 0
        attempt_errors = []
        response_cost = Cost(model=self.model)
        try:
            self.num_api_calls += 1
            # An unbounded output length on a reasoning model is an
            # unbounded bill: these calls spend most of their tokens on
            # hidden reasoning, and nothing in the request stopped that
            # from being ten times what was observed. The ceiling is
            # passed only when set, so unbudgeted runs keep their
            # previous behaviour exactly.
            ceiling = output_token_ceiling()
            extra = ({'max_output_tokens': ceiling} if ceiling else {})
            def request():
                nonlocal attempt_count
                attempt_count += 1
                try:
                    return create_teacher_response(
                        self.client, model=self.model, input=prompt,
                        text=schema, **extra)
                except Exception as exc:
                    details = failure_details(exc)
                    # Journal types/status, never provider error bodies.
                    details.pop('message', None)
                    details['request_id'] = getattr(exc, 'request_id', None)
                    attempt_errors.append(details)
                    raise

            response = call_with_cold_start_retry(request)
            # Capture usage BEFORE parsing. Incomplete HTTP-200 replies
            # can have billed reasoning tokens and no visible JSON.
            usage = getattr(response, 'usage', None)
            response_cost.tokens_in = int(
                getattr(usage, 'input_tokens', 0) or 0)
            response_cost.tokens_out = int(
                getattr(usage, 'output_tokens', 0) or 0)
            tier = getattr(response, 'service_tier', None)
            response_cost.metadata = {
                'cache_hit': False, 'outcome': 'valid',
                'response_id': getattr(response, 'id', None),
                'request_id': getattr(response, '_request_id', None),
                'response_model': getattr(response, 'model', None),
                'response_status': getattr(response, 'status', None),
                'service_tier': tier,
                'usage_known': usage is not None,
                'cost_known': usage is not None and tier in (None, 'default'),
            }
            if tier in (None, 'default'):
                response_cost.dollars = estimate_dollars(
                    self.model, response_cost.tokens_in,
                    response_cost.tokens_out)
            else:
                # Unexpected-tier prices are unknown, not Standard.
                # Preserve tokens, mark the cost unknown and stop the
                # budgeted caller after it has persisted this evidence.
                response_cost.metadata['budget_violation'] = 'service_tier'
                raise ValueError('Response did not use Standard service')
            status = getattr(response, 'status', None)
            if status is not None and status != 'completed':
                details = getattr(response, 'incomplete_details', None)
                response_cost.metadata['outcome'] = 'incomplete_response'
                response_cost.metadata['incomplete_reason'] = getattr(
                    details, 'reason', None)
                raise ValueError('Response was not completed')
            data = json.loads(response.output_text)
            from jsonschema import Draft202012Validator
            Draft202012Validator(schema['format']['schema']).validate(
                data)
            # A model can return a well-formed JSON document whose
            # values are still garbage (an out-of-range or absurdly
            # large 'confidence', a missing key) -- parsing that is
            # kept inside this try block so a single malformed
            # response fails just this one seed, the same as a
            # network error, instead of an uncaught exception here
            # crashing the whole job and losing every remaining seed.
            action = int(data['action'])
            confidence = max(
                0.0, min(1.0, float(data['confidence']))
            )
            reasoning = str(data['reasoning'])
            if self.structured_explanations:
                response_cost.metadata['subgoal'] = data['subgoal']
            if self.evidence_citations:
                response_cost.metadata['evidence_object'] = str(
                    data['evidence_object'])
                response_cost.metadata['evidence_color'] = str(
                    data['evidence_color'])
                response_cost.metadata['evidence_cell'] = (
                    None if data['evidence_object'] == 'none'
                    else [int(data['evidence_x']), int(data['evidence_y'])])
            forbidden_raw = data.get('forbidden_actions') or []
            forbidden = tuple(int(a) for a in forbidden_raw)
        except Exception as exc:
            self.num_failures += 1
            metadata = response_cost.metadata
            metadata.update(attempt_count=attempt_count,
                            attempt_errors=attempt_errors)
            if response is None:
                details = failure_details(exc)
                metadata.update(
                    outcome=('transport_failure' if details['kind'] ==
                             'transient' else 'request_failure'),
                    usage_known=False, cost_known=False,
                    http_status=details['http_status'])
            elif metadata.get('budget_violation'):
                metadata['outcome'] = 'budget_violation'
            elif metadata.get('outcome') != 'incomplete_response':
                metadata['outcome'] = 'schema_failure'
            metadata['error_type'] = type(exc).__name__
            abstention = self._abstain(
                t0, reason=type(exc).__name__, cost=response_cost)
            if self.strict:
                error = RuntimeError(
                    f'MiniGridGeneralLLMTeacher API call or '
                    f'response parsing failed mission={mission!r} '
                    f'(model={self.model!r}): '
                    f'{type(exc).__name__}: {exc}'
                )
                error.advice = abstention
                raise error from exc
            return abstention

        response_cost.wall_time_s = time.perf_counter() - t0
        response_cost.metadata.update(attempt_count=attempt_count,
                                      attempt_errors=attempt_errors,
                                      teacher_action=action)

        advice = Advice(
            action=action,
            forbidden_actions=forbidden,
            confidence=confidence,
            explanation=reasoning,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=response_cost,
        )
        self._cache[cache_key] = advice
        # Remember what we advised so the next call can judge whether
        # it actually did anything.
        self._pending_action = action
        return advice

    def _abstain(self, t0, reason, cost=None, metadata=None):
        """
        Build an abstaining Advice for the non-strict failure path.
        """

        cost = cost if cost is not None else Cost(model=self.model)
        cost.wall_time_s = time.perf_counter() - t0
        cost.metadata.update(metadata or {})
        cost.metadata.update(failed=True, error=redact(reason))
        return Advice(
            action=None,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=cost,
        )

    def stats(self) -> dict:
        """
        Summary counters for end-of-run reporting.
        """

        total = self.num_api_calls + self.num_cache_hits
        return {
            'model': self.model,
            'env_id': self.env_id,
            'num_api_calls': self.num_api_calls,
            'num_cache_hits': self.num_cache_hits,
            'num_failures': self.num_failures,
            'total_queries': total,
            'cache_hit_rate': (
                self.num_cache_hits / total if total > 0 else 0.0
            ),
        }
