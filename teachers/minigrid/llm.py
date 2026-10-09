"""
LLM teacher for MiniGrid.

Calls the OpenAI Responses API with a MiniGrid-specific prompt
and a JSON schema that forces structured output. Returns an
Advice populated with the chosen action, self-reported
confidence, reasoning, and (optionally) forbidden actions.
Every call is fully costed.

Cache key
---------
Unlike the MountainCar LLM teacher, which cached by
`(prompt_id, model, pos_bin, vel_bin)`, MiniGrid caches by the
*full* 11-tuple state plus the mission string. Two reasons:

1. MiniGrid is procedurally generated -- each episode has a
   different layout, so two visits with the same agent
   (x, y, dir) but different key/door positions are genuinely
   different states that demand different advice. The state
   tuple already includes those positions.
2. BabyAI variants carry a language mission that changes the
   semantics of "the goal" -- the cache must distinguish them.

Caching by the exact (mission, state) pair means a repeated
state-mission combo across episodes is a cache hit (one API
call's cost), but every new layout costs one fresh API call.

Failure handling: strict by default. If the API call fails or
the response is malformed, the exception is raised so the
experiment stops rather than silently degrading. Pass
`strict=False` to fall back to an abstaining Advice (action=
None) instead.
"""

import json
import time

from teachers.aleph import create_teacher_response
from teachers.base import (
    Advice,
    BaseTeacher,
    Cost,
    build_openai_client,
    call_with_cold_start_retry,
)
from teachers.minigrid.llm_prompts import (
    PROMPTS,
    estimate_dollars,
)


# JSON schema used for structured responses.
#
# Field ORDER matters here. Structured outputs are generated
# field-by-field in declaration order, and a model picking its
# action FIRST has not yet produced any chain-of-thought to
# constrain that choice -- it commits, then writes a reasoning
# trace that may or may not agree with the action it just
# emitted. We observed this directly on gpt-4.1-mini: reasoning
# said "move forward (action 2)" while action came out as 0.
#
# Putting `reasoning` first forces the model to think before it
# picks, so the action field is constrained by the reasoning
# the model just generated. This is a well-known structured-
# output pattern for smaller models; reasoning-first schemas
# materially improve correctness on spatial / planning tasks.
#
# We keep the required-fields list aligned with the property
# order so the schema reads consistently.
RESPONSE_SCHEMA = {
    'format': {
        'type': 'json_schema',
        'name': 'minigrid_advice',
        'schema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'reasoning': {
                    'type': 'string',
                },
                'action': {
                    'type': 'integer',
                    'enum': [0, 1, 2, 3, 4, 5],
                },
                'confidence': {
                    'type': 'number',
                },
                'forbidden_actions': {
                    'anyOf': [
                        {
                            'type': 'array',
                            'items': {
                                'type': 'integer',
                                'enum': [0, 1, 2, 3, 4, 5],
                            },
                        },
                        {'type': 'null'},
                    ],
                },
            },
            'required': [
                'reasoning',
                'action',
                'confidence',
                'forbidden_actions',
            ],
        },
    }
}


class MiniGridLLMTeacher(BaseTeacher):
    """
    MiniGrid teacher backed by an LLM API.

    Expected state
    --------------
    The 11-tuple from MiniGridEnv: (agent_x, agent_y,
    agent_dir, has_key, door_open, key_x, key_y, door_x,
    door_y, goal_x, goal_y).

    Context
    -------
    The teacher reads the per-episode mission string from
    `context['mission']` when provided. The with-teacher
    controller is responsible for passing the mission in,
    because the mission is a per-episode property that the
    env's get_env_info reply only carries after the first
    env_start. If `context` is None or lacks a mission, the
    teacher falls back to an empty mission string.
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
        reasoning_effort: str = '',
        strict: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        seed: int = 0,
    ):
        """
        Initialize the teacher and validate API access eagerly.

        Parameters
        ----------
        env_id: str
            Used for logging/cache only; the env_id appears in
            the prompt header.
        model: str
            OpenAI model name. Default gpt-4.1-mini -- on the
            MountainCar probe it was the cheapest model that
            passed all 7 swing-back states; we start with the
            same on MiniGrid until we run an analogous probe.
        prompt_id: str
            Key into `llm_prompts.PROMPTS`.
        reasoning_effort: str
            Reasoning budget for the gpt-5 / o-series models:
            'minimal', 'low', 'medium' or 'high'. Empty (the
            default) sends no `reasoning` field at all, which is
            REQUIRED for the gpt-4.x models -- they reject the
            parameter -- and leaves the reasoning models on their
            own default (medium).

            This is the single biggest cost lever for a paid
            teacher, because reasoning tokens bill as OUTPUT
            tokens. Measured on doorkey_8x8 with gpt-5-mini and no
            setting: ~650 output tokens per call against
            gpt-4.1-mini's ~128, i.e. ~520 hidden reasoning tokens
            and 3.7x the per-call price. 'minimal' should recover
            most of that -- but see
            docs/teacher_competence_2026-07-24.md, which recorded
            gpt-4.1-mini thrashing where gpt-5-mini succeeded. If
            that competence gap IS the reasoning, suppressing it
            buys back the cost and the failure with it. Probe with
            scripts/eval_teacher.py before trusting it in a sweep.
        strict: bool
            If True, API failures and malformed responses raise.
            If False, the teacher returns an abstaining Advice
            and increments a failure counter.
        timeout: float
            Per-call timeout passed to the OpenAI client.
        max_retries: int
            Per-call retry count passed to the OpenAI client.
        seed: int
            RNG seed propagated to BaseTeacher; the API itself
            is not seeded by us.
        """

        suffix = f':r-{reasoning_effort}' if reasoning_effort else ''
        super().__init__(
            teacher_id=f'minigrid_llm:{model}:{prompt_id}{suffix}',
            seed=seed,
        )

        # Validate prompt_id eagerly so a typo fails before the
        # first run rather than mid-training.
        if prompt_id not in PROMPTS:
            raise ValueError(
                f'Unknown prompt_id={prompt_id!r}. '
                f'Available: {sorted(PROMPTS)}.'
            )

        if reasoning_effort and reasoning_effort not in (
            'minimal', 'low', 'medium', 'high'
        ):
            raise ValueError(
                f'Unknown reasoning_effort={reasoning_effort!r}. '
                "Available: 'minimal', 'low', 'medium', 'high', or "
                "'' to send no reasoning field."
            )

        self.env_id = env_id
        self.model = model
        self.prompt_id = prompt_id
        self.reasoning_effort = reasoning_effort
        self.strict = strict

        # Builds either an OpenAI client or a free Aleph (Vulcan)
        # client depending on the LLM_PROVIDER env var -- see
        # teachers/base.py for the switch. A missing key fails here,
        # before any training starts, not mid-run.
        self.client = build_openai_client(
            timeout=timeout,
            max_retries=max_retries,
        )

        # Cache: (prompt_id, model, mission, state_tuple) ->
        # Advice. Only successful Advice objects are cached;
        # failures must not poison future lookups.
        self._cache: dict[tuple, Advice] = {}

        # Aggregate counters for end-of-run reporting. Per-call
        # cost data lives on individual Advice objects; these
        # counters are the run-level summary.
        self.num_api_calls = 0
        self.num_cache_hits = 0
        self.num_failures = 0

    def recommend(
        self, state, context: dict | None = None
    ) -> Advice:
        """
        Return advice for `state`, hitting the cache when
        possible and otherwise calling the API.

        Parameters
        ----------
        state: tuple of 11 int
            See class docstring.
        context: dict or None
            Optional. The controller may include 'mission'
            (str) so the prompt can reference the episode's
            language goal.
        """

        t0 = time.perf_counter()

        # Normalize state to a plain tuple of ints for hashing.
        # Numpy scalars or list-typed states would otherwise
        # break dict-key equality across episodes.
        state_key = tuple(int(s) for s in state)

        mission = ''
        if context is not None and 'mission' in context:
            mission = str(context['mission'])

        cache_key = (
            self.prompt_id, self.model, self.reasoning_effort,
            mission, state_key,
        )

        if cache_key in self._cache:
            # Return a fresh Advice that copies the cached
            # decision but reports zero-cost (we did no work
            # other than the lookup wall time).
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
                    metadata={
                        'cache_hit': True,
                        'prompt_id': self.prompt_id,
                    },
                ),
            )

        # Cache miss: build the prompt and call the API.
        prompt = PROMPTS[self.prompt_id](
            self.env_id, state_key, mission
        )

        try:
            self.num_api_calls += 1
            call_kwargs = {
                'model': self.model,
                'input': prompt,
                'text': RESPONSE_SCHEMA,
            }
            if self.reasoning_effort:
                call_kwargs['reasoning'] = {
                    'effort': self.reasoning_effort
                }
            response = call_with_cold_start_retry(lambda: (
                create_teacher_response(self.client, **call_kwargs)
            ))
            data = json.loads(response.output_text)
            # Extract structured fields. Confidence is clamped to
            # [0, 1] on this side because OpenAI's strict mode
            # disallows numeric min/max constraints in the schema.
            # Kept inside this try block because a well-formed JSON
            # document can still hold garbage values (an absurdly
            # large 'confidence', a missing key) -- letting that
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
                    f'MiniGridLLMTeacher API call or response '
                    f'parsing failed at state={state_key} '
                    f'mission={mission!r} (model={self.model!r}, '
                    f'prompt={self.prompt_id!r}): '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
            elapsed = time.perf_counter() - t0
            return Advice(
                action=None,
                teacher_id=self.teacher_id,
                call_id=self._next_call_id(),
                cost=Cost(
                    wall_time_s=elapsed,
                    model=self.model,
                    metadata={
                        'failed': True,
                        'error': f'{type(exc).__name__}: {exc}',
                        'prompt_id': self.prompt_id,
                    },
                ),
            )

        # Cost from token usage.
        usage = getattr(response, 'usage', None)
        tokens_in = (
            int(usage.input_tokens) if usage is not None else 0
        )
        tokens_out = (
            int(usage.output_tokens) if usage is not None else 0
        )
        dollars = estimate_dollars(
            self.model, tokens_in, tokens_out
        )
        elapsed = time.perf_counter() - t0

        cost = Cost(
            wall_time_s=elapsed,
            dollars=dollars,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            metadata={
                'cache_hit': False,
                'prompt_id': self.prompt_id,
                'raw_response': response.output_text,
            },
        )

        advice = Advice(
            action=action,
            forbidden_actions=forbidden,
            confidence=confidence,
            explanation=reasoning,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=cost,
        )

        # Cache only on success. Failures must not be cached
        # because they would poison every subsequent visit.
        self._cache[cache_key] = advice
        return advice

    def stats(self) -> dict:
        """
        Summary counters for end-of-run reporting.
        """

        total = self.num_api_calls + self.num_cache_hits
        return {
            'model': self.model,
            'prompt_id': self.prompt_id,
            'env_id': self.env_id,
            'num_api_calls': self.num_api_calls,
            'num_cache_hits': self.num_cache_hits,
            'num_failures': self.num_failures,
            'total_queries': total,
            'cache_hit_rate': (
                self.num_cache_hits / total if total > 0 else 0.0
            ),
        }
