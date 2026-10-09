"""
VLM (vision-language) teacher for MiniGrid.

The image-based counterpart of MiniGridLLMTeacher. Instead of a text
description of the symbolic state, this teacher is shown the *rendered
full map* of the environment and asked, in natural language, which
action the agent should take. It calls a vision-capable OpenAI model
through the Responses API with the image attached, and returns the
same structured Advice every other teacher produces, so the override
mechanism is unchanged.

Why the full map (not the agent's partial view)
-----------------------------------------------
A teacher should know at least as much as the planner it stands in
for. The BFS oracle sees the entire symbolic state; to be a fair
peer, the VLM is shown the entire rendered board (keys, doors, goal,
walls, and the agent as a colored triangle). Feeding it the agent's
blacked-out 7x7 egocentric view would make it a poor teacher that
often cannot see the goal. The agent still trains on its partial
pixels; only the teacher gets the full picture.

Cache key
---------
Identical symbolic states render to identical maps and therefore
deserve identical advice, so we cache by (model, prompt_id, mission,
state_tuple) exactly like the LLM teacher -- the image itself is a
deterministic function of that key, so there is no need to hash
pixels. A repeated (state, mission) pair is a free cache hit; every
new layout costs one vision API call.

Failure handling mirrors the LLM teacher: strict by default (raise on
API/parse failure), or abstain when strict=False.
"""

import base64
import io
import json
import time

from PIL import Image

from teachers.aleph import create_teacher_response
from teachers.base import (
    Advice,
    BaseTeacher,
    Cost,
    build_openai_client,
    call_with_cold_start_retry,
)
# Reuse the LLM teacher's structured-output schema (reasoning-first,
# action enum) and the shared pricing helper so cost accounting and
# response parsing stay identical across the two API teachers.
from teachers.minigrid.llm import RESPONSE_SCHEMA
from teachers.minigrid.llm_prompts import estimate_dollars


def encode_png_b64(rgb_array):
    """
    Encode an (H, W, 3) uint8 RGB array as a base64 PNG string.

    The Responses API accepts inline images as a data URL; this
    produces the base64 payload that goes after the data-URL prefix.
    Kept module-level so it can be tested without an API key.
    """

    img = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def build_vlm_prompt(env_id, mission, has_key, door_open):
    """
    Build the text that accompanies the rendered map image.

    The model reads positions, walls, and facing from the image, but
    two state bits are NOT reliably visible in a MiniGrid render and
    are supplied here as text: whether the agent is carrying the key
    (a carried key is not drawn at all) and whether the door is open.
    Without the carry bit the VLM loops forever trying to pick up a
    key it already holds. Kept module-level and client-free so it is
    testable offline.
    """

    carry_str = 'YES' if has_key else 'NO'
    door_str = 'OPEN' if door_open else 'CLOSED'

    return (
        'You are coaching an agent in a MiniGrid environment '
        f'({env_id}). Mission: "{mission}"\n'
        '\n'
        'The attached image is a top-down view of the WHOLE grid.\n'
        'How to read it:\n'
        '- The red/pointed TRIANGLE is the agent; it points in the '
        'direction the agent is currently facing. The cell directly '
        'in front is the one the triangle points at.\n'
        '- A yellow KEY shape is a key on the floor (gone once the '
        'agent is carrying it).\n'
        '- A colored SQUARE with a small line is a door; an OPEN '
        'door shows a gap, a CLOSED door is a solid square.\n'
        '- A solid GREEN square is the goal tile.\n'
        '- Grey cells are walls; black/dark cells are empty floor. '
        'You can only move onto floor, the goal, or an open door.\n'
        '\n'
        'STATUS (NOT shown in the image -- trust these values, not '
        'the picture, for these two facts):\n'
        f'- Are you carrying the key? {carry_str}. A carried key is '
        'INVISIBLE in the image, so once this says YES, do not try '
        'to pick up a key again.\n'
        f'- Is the door open? {door_str}.\n'
        '\n'
        'Follow this procedure exactly:\n'
        '\n'
        'STEP 1 - Pick your TARGET (first matching rule):\n'
        '- If you are NOT carrying the key (STATUS above): '
        'TARGET = the key.\n'
        '- Else if the door is CLOSED (STATUS above): '
        'TARGET = the door.\n'
        '- Else (carrying the key AND door open): TARGET = the '
        'green goal. Completely IGNORE the key and door now; only '
        'the goal matters.\n'
        '\n'
        'STEP 2 - Choose the action:\n'
        '- If the TARGET is the cell directly in front of the '
        'triangle: key -> 3 (PICK UP), closed door -> 5 (TOGGLE), '
        'goal -> 2 (FORWARD).\n'
        '- Otherwise WALK toward the target. Move FORWARD only into '
        'floor / open door / goal cells, never into a grey wall or '
        'a closed door. To get next to the target you often must go '
        'AROUND a wall: it is correct to move to an open cell that '
        'is NOT directly toward the target when a wall blocks the '
        'direct route. Do not turn back and forth in place.\n'
        '- If the cell in front is open and moving there makes '
        'progress toward the target, take action 2 (FORWARD). '
        'Otherwise TURN one step (0 LEFT or 1 RIGHT) toward an open '
        'route that leads around the wall to the target.\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place (counter-clockwise).\n'
        '- 1: turn RIGHT in place (clockwise).\n'
        '- 2: move FORWARD one cell the triangle points at.\n'
        '- 3: PICK UP the object directly in front.\n'
        '- 4: DROP the carried object in front.\n'
        '- 5: TOGGLE the object in front (opens the door when '
        'holding the key and facing it).\n'
        '\n'
        'Look at the image and output what ONE action the agent '
        'should take next.'
    )


class MiniGridVLMTeacher(BaseTeacher):
    """
    MiniGrid teacher backed by a vision-language model.

    Expected inputs
    ---------------
    recommend(state, context):
      - state: the symbolic state tuple, used ONLY as the cache key
        (identical states render identical maps). On DoorKey this is
        the 11-tuple from envs.state.extract_doorkey_state.
      - context['image']: an (H, W, 3) uint8 RGB array of the full
        rendered map -- this is what the model actually reasons over.
      - context['mission']: optional language goal string.

    The model must be vision-capable (e.g. gpt-4o, gpt-4o-mini).
    """

    # Answers depend on the state argument alone -- no plan, goal,
    # or history is carried between calls -- so a recorded answer
    # stays valid and advising/peekable.py may replay it instead of
    # paying for the same query twice.
    is_stateless = True

    def __init__(
        self,
        env_id: str = 'MiniGrid-DoorKey-5x5-v0',
        model: str = 'gpt-4.1-mini',
        prompt_id: str = 'default',
        strict: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        seed: int = 0,
    ):
        """
        Initialize the teacher and validate API access eagerly.

        Parameters mirror MiniGridLLMTeacher; `model` must be a
        vision-capable model. A missing OPENAI_API_KEY raises here,
        before any training starts, rather than mid-run.
        """

        super().__init__(
            teacher_id=f'minigrid_vlm:{model}:{prompt_id}',
            seed=seed,
        )

        self.env_id = env_id
        self.model = model
        self.prompt_id = prompt_id
        self.strict = strict

        # Builds either an OpenAI client or a free Aleph (Vulcan)
        # client depending on the LLM_PROVIDER env var -- see
        # teachers/base.py for the switch. A missing key fails here,
        # before any run starts, not mid-episode.
        self.client = build_openai_client(
            timeout=timeout,
            max_retries=max_retries,
        )

        # Cache and run-level counters, identical in spirit to the
        # LLM teacher's.
        self._cache: dict[tuple, Advice] = {}
        self.num_api_calls = 0
        self.num_cache_hits = 0
        self.num_failures = 0

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return advice for the rendered map in `context['image']`,
        hitting the cache when the symbolic state repeats.
        """

        t0 = time.perf_counter()

        # The image is mandatory for a vision teacher.
        image = None if context is None else context.get('image')
        if image is None:
            if self.strict:
                raise ValueError(
                    'MiniGridVLMTeacher.recommend requires '
                    "context['image'] (the rendered map)."
                )
            return self._abstain(t0, reason='no_image')

        # Normalize the state to a hashable tuple for the cache key.
        state_key = tuple(int(s) for s in state)
        mission = ''
        if context is not None and 'mission' in context:
            mission = str(context['mission'])
        cache_key = (self.model, self.prompt_id, mission, state_key)

        if cache_key in self._cache:
            self.num_cache_hits += 1
            cached = self._cache[cache_key]
            elapsed = time.perf_counter() - t0
            return Advice(
                action=cached.action,
                forbidden_actions=cached.forbidden_actions,
                confidence=cached.confidence,
                explanation=cached.explanation,
                teacher_id=self.teacher_id,
                call_id=self._next_call_id(),
                cost=Cost(
                    wall_time_s=elapsed,
                    model=self.model,
                    metadata={'cache_hit': True},
                ),
            )

        # Cache miss: encode the image, build the prompt, and call
        # the vision model. The carry-key and door-open bits (state
        # indices 3 and 4) are passed as text because they are not
        # reliably visible in the render.
        b64 = encode_png_b64(image)
        prompt = build_vlm_prompt(
            self.env_id, mission, state_key[3], state_key[4]
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
                                    # Request maximum image detail;
                                    # the agent triangle and door are
                                    # tiny, so finer perception is
                                    # worth the extra image tokens.
                                    'detail': 'high',
                                },
                            ],
                        }
                    ],
                    text=RESPONSE_SCHEMA,
                )
            ))
            data = json.loads(response.output_text)
            # Parse the structured fields (same schema as the LLM
            # teacher). Confidence is clamped on our side. Kept
            # inside this try block: a well-formed JSON document
            # can still hold garbage values (an absurdly large
            # 'confidence', a missing key), and letting that raise
            # here would crash the whole job instead of failing
            # just this one call.
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
                    f'MiniGridVLMTeacher API call or response '
                    f'parsing failed at state={state_key} '
                    f'mission={mission!r} (model={self.model!r}): '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
            return self._abstain(
                t0, reason=f'{type(exc).__name__}: {exc}'
            )

        # Cost from token usage (vision input tokens are included in
        # input_tokens, so image cost is captured here).
        usage = getattr(response, 'usage', None)
        tokens_in = int(usage.input_tokens) if usage else 0
        tokens_out = int(usage.output_tokens) if usage else 0
        dollars = estimate_dollars(self.model, tokens_in, tokens_out)
        elapsed = time.perf_counter() - t0

        advice = Advice(
            action=action,
            forbidden_actions=forbidden,
            confidence=confidence,
            explanation=reasoning,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=elapsed,
                dollars=dollars,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self.model,
                metadata={'cache_hit': False},
            ),
        )

        # Cache only successful advice.
        self._cache[cache_key] = advice
        return advice

    def _abstain(self, t0, reason):
        """
        Build an abstaining Advice (action=None) for the non-strict
        failure path, tagged with why it abstained.
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
