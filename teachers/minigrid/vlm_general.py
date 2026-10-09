"""
General-purpose VLM teacher for any MiniGrid/BabyAI task.

teachers/minigrid/vlm.py's MiniGridVLMTeacher is deliberately kept
frozen and DoorKey-specific: its prompt hard-codes a "single
dividing wall, one key, one door, one goal" topology, and its state
handling unpacks a fixed 11-tuple by position (state_key[3] for
has_key, state_key[4] for door_open).

This is the general-purpose sibling for tasks whose room structure
is NOT known in advance -- KeyCorridorS6R3 is the motivating case:
an R-row grid of rooms, a key hidden in one of them, one locked
door, a target object behind it. Rather than hand-writing a
navigation procedure over an unknown room graph (fragile, and a
much harder text-prompt-engineering problem than DoorKey's fixed
two-room split -- effectively reimplementing pathfinding in prose),
this teacher shows the model the rendered full map and general
MiniGrid rules, and lets its vision do the spatial reasoning DoorKey
prompts previously did by hand.

Because it makes no topology assumptions, this teacher also works
on doorkey_* and gotoseq -- it is simply a more general (and, on
DoorKey specifically, presumably weaker, since it lacks the tuned
step-by-step procedure) alternative to the DoorKey-specific teacher,
not a replacement for it.

Shares the DoorKey VLM teacher's structured-output schema, pricing
table, and image encoder by import (no duplication), and mirrors its
caching (by generic symbolic state + mission) and strict/non-strict
failure handling.
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
from teachers.minigrid.llm import RESPONSE_SCHEMA
from teachers.minigrid.llm_prompts import estimate_dollars
from teachers.minigrid.vlm import encode_png_b64


def build_generic_vlm_prompt(env_id, mission, carrying, objects):
    """
    Build the text that accompanies the rendered map image.

    Unlike the DoorKey prompt, this does not describe a fixed
    two-room topology or hand a step-by-step navigation procedure:
    the model is told the general MiniGrid visual vocabulary and the
    mission, shown the image, and left to plan the route itself --
    the room layout is whatever the image shows.

    Parameters
    ----------
    env_id: str
        Gym id, for the prompt header only.
    mission: str
        The episode's mission string (e.g. "pick up the red ball");
        this is often the ONLY place the target object's identity
        appears, since the image alone doesn't label objects.
    carrying: str
        What the agent holds right now (e.g. 'red key'), or '' if
        empty-handed. Supplied as text because a carried object is
        not drawn in the render.
    objects: tuple
        The `objects` field from envs.state.extract_generic_state:
        (kind, color, x, y, door_state) tuples. Doors get an
        explicit open/closed/locked line, since that distinction can
        be hard to read reliably from a small rendered tile.
    """

    door_lines = [
        f'- {color} door at ({x}, {y}): {state}'
        for (kind, color, x, y, state) in objects
        if kind == 'door'
    ]
    door_block = (
        '\n'.join(door_lines) if door_lines else '- (no doors visible)'
    )
    carrying_str = carrying if carrying else 'nothing'

    return (
        'You are coaching an agent in a MiniGrid environment '
        f'({env_id}). Mission: "{mission}"\n'
        '\n'
        'The attached image is a top-down view of the WHOLE map, '
        'which may contain several rooms connected by doors and '
        'corridors.\n'
        'How to read it:\n'
        '- The red/pointed TRIANGLE is the agent; it points in the '
        'direction it is currently facing. The cell directly in '
        'front is the one the triangle points at.\n'
        '- A KEY-shaped icon is a key on the floor, colored to match '
        'the door it opens (gone once picked up).\n'
        '- A BALL (circle) or BOX is a colored object that can be '
        'picked up.\n'
        '- A colored rectangle set into a wall is a door.\n'
        '- Grey cells are walls; darker cells are empty floor. You '
        'can only move onto floor, open doors, or pick-up-able '
        'objects -- never through a wall or a closed/locked door.\n'
        '\n'
        'STATUS (not always reliable from the image -- trust these '
        'lines for these facts):\n'
        f'- You are carrying: {carrying_str}. A carried object is '
        'INVISIBLE in the image, so do not try to pick it up again.\n'
        f'- Known doors:\n{door_block}\n'
        '\n'
        'Rules:\n'
        '- A LOCKED door can only be opened by facing it while '
        'carrying a key of the SAME color, then using action 5 '
        '(TOGGLE); this both unlocks and opens it in one action.\n'
        '- A CLOSED (unlocked) door opens with action 5 (TOGGLE) '
        'while facing it, no key needed.\n'
        '- Pick up an object with action 3 (PICKUP) while facing '
        'it and holding nothing (or after dropping what you hold '
        'with action 4).\n'
        '- The mission tells you the target object\'s color and '
        'type; a key is a TOOL for a matching-colored locked door, '
        'never the mission target unless the mission itself says '
        '"key".\n'
        '\n'
        'Plan a route in the image toward whatever the mission '
        'still requires (find/collect a key if a relevant door is '
        'locked, open that door, then reach and pick up the target), '
        'then choose ONE next action.\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place (counter-clockwise).\n'
        '- 1: turn RIGHT in place (clockwise).\n'
        '- 2: move FORWARD one cell the triangle points at.\n'
        '- 3: PICK UP the object directly in front.\n'
        '- 4: DROP the carried object in front.\n'
        '- 5: TOGGLE the door directly in front (see rules above).\n'
        '\n'
        'Look at the image and output what ONE action the agent '
        'should take next.'
    )


class MiniGridGeneralVLMTeacher(BaseTeacher):
    """
    Vision-language teacher for arbitrary MiniGrid/BabyAI layouts.

    Expected inputs
    ---------------
    recommend(state, context):
      - state: the unwrapped MiniGrid env (used to build the
        topology-agnostic cache key via
        envs.state.extract_generic_state -- passing the env rather
        than a pre-extracted tuple keeps the call site identical to
        the bot teacher's and avoids computing the state twice).
      - context['image']: an (H, W, 3) uint8 RGB array of the full
        rendered map.
      - context['mission']: the episode's mission string.
    """

    def __init__(
        self,
        env_id: str = 'MiniGrid-KeyCorridorS6R3-v0',
        model: str = 'gpt-4.1-mini',
        strict: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        seed: int = 0,
    ):
        """
        Initialize the teacher and validate API access eagerly, so
        a missing key fails before any training starts rather than
        mid-run.
        """

        super().__init__(
            teacher_id=f'minigrid_vlm_general:{model}', seed=seed
        )

        self.env_id = env_id
        self.model = model
        self.strict = strict

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

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return advice for the rendered map in `context['image']`.

        `state` is the unwrapped MiniGrid env; extract_generic_state
        derives the cache key and the text status hints from it.
        """

        t0 = time.perf_counter()

        image = None if context is None else context.get('image')
        if image is None:
            if self.strict:
                raise ValueError(
                    'MiniGridGeneralVLMTeacher.recommend requires '
                    "context['image'] (the rendered map)."
                )
            return self._abstain(t0, reason='no_image')

        mission = ''
        if context is not None and 'mission' in context:
            mission = str(context['mission'])

        agent_x, agent_y, agent_dir, carrying, objects = (
            extract_generic_state(state)
        )
        cache_key = (
            self.model, mission, agent_x, agent_y, agent_dir,
            carrying, objects,
        )

        if cache_key in self._cache:
            self.num_cache_hits += 1
            cached = self._cache[cache_key]
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
                    metadata={'cache_hit': True},
                ),
            )

        b64 = encode_png_b64(image)
        prompt = build_generic_vlm_prompt(
            self.env_id, mission, carrying, objects
        )

        try:
            self.num_api_calls += 1
            response = call_with_cold_start_retry(lambda: (
                create_teacher_response(
                    self.client,
                    model=self.model,
                    input=[
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'input_text', 'text': prompt},
                                {
                                    'type': 'input_image',
                                    'image_url': (
                                        f'data:image/png;base64,{b64}'
                                    ),
                                    'detail': 'high',
                                },
                            ],
                        }
                    ],
                    text=RESPONSE_SCHEMA,
                )
            ))
            data = json.loads(response.output_text)
            # Kept inside this try block: a well-formed JSON
            # document can still hold garbage values (an absurdly
            # large 'confidence', a missing key), and letting that
            # raise here would crash the whole job instead of
            # failing just this one call.
            action = int(data['action'])
            confidence = max(
                0.0, min(1.0, float(data['confidence']))
            )
            reasoning = str(data['reasoning'])
            forbidden_raw = data.get('forbidden_actions') or []
            forbidden = tuple(int(a) for a in forbidden_raw)
        except Exception as exc:
            self.num_failures += 1
            if self.strict:
                raise RuntimeError(
                    f'MiniGridGeneralVLMTeacher API call or '
                    f'response parsing failed mission={mission!r} '
                    f'(model={self.model!r}): '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
            return self._abstain(
                t0, reason=f'{type(exc).__name__}: {exc}'
            )

        usage = getattr(response, 'usage', None)
        tokens_in = int(usage.input_tokens) if usage else 0
        tokens_out = int(usage.output_tokens) if usage else 0
        dollars = estimate_dollars(self.model, tokens_in, tokens_out)

        advice = Advice(
            action=action,
            forbidden_actions=forbidden,
            confidence=confidence,
            explanation=reasoning,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                dollars=dollars,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self.model,
                metadata={'cache_hit': False},
            ),
        )
        self._cache[cache_key] = advice
        return advice

    def _abstain(self, t0, reason):
        """
        Build an abstaining Advice for the non-strict failure path.
        """

        return Advice(
            action=None,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                model=self.model,
                metadata={'failed': True, 'error': reason},
            ),
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
