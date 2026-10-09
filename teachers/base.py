"""
Base classes for all teachers.

Every teacher in the project — rule-based, planning-based, oracle,
LLM, or anything added later — implements the BaseTeacher contract
defined here and returns a single Advice object per call.

The two dataclasses (Advice, Cost) are the shared vocabulary that
controllers, channels, and logging code use to interpret a teacher's
output regardless of where the advice came from. Adding a new teacher
type later does not require changing this file or any consumer; new
fields can be added to Advice or Cost as optional attributes with
safe defaults.

Layout
------
- Cost: per-call cost record. Every teacher fills in whichever fields
  are meaningful for it and leaves the rest at zero so cost
  comparisons across teacher types are apples-to-apples.
- Advice: the structured output of a single teacher call. Supports
  positive advising (an action), negative advising (forbidden
  actions), abstention (no opinion), confidence, and explanation.
- BaseTeacher: abstract class subclassed by every concrete teacher.
  Intentionally environment-agnostic — `state` and `context` are
  untyped so a FrozenLake teacher and a Mountain Car teacher can both
  inherit from the same base without one being forced to accept the
  other's state shape.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Use the operating system's certificate store (the Windows store)
# for TLS, so OpenAI API calls succeed behind networks whose root
# certificate Python's bundled certifi does not include -- the
# Python equivalent of git's schannel backend. Every OpenAI-backed
# teacher (llm.py, vlm.py, llm_general.py, vlm_general.py,
# llm_subgoal.py) subclasses BaseTeacher, so patching SSL once here,
# before any of them opens a connection, fixes all of them instead
# of repeating the patch in each file. A no-op for offline teachers
# (oracle, bot). Guarded so it does nothing if the optional package
# is absent.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


def build_openai_client(timeout: float = 300.0, max_retries: int = 2):
    # NOTE: LLM_MAX_SDK_RETRIES overrides `max_retries` below. A
    # budgeted run sets it to 0, because the client's own retries
    # multiply the outer cold-start cap: with both at their defaults one
    # consultation can reach the provider nine times, and a reservation
    # built on the outer cap alone would be short by 3x.

    """
    Build an OpenAI-compatible client, switching between the real
    OpenAI API and the University of Alberta's free Aleph Inference
    Gateway (hosted on Vulcan, OpenAI-compatible at
    https://inference.vulcan.alliancecan.ca/v1) based on the
    LLM_PROVIDER env var. Every OpenAI-backed teacher (llm.py,
    vlm.py, llm_general.py, vlm_general.py, llm_subgoal.py) calls
    this instead of constructing an OpenAI client directly, so the
    provider switch lives in one place.

    `timeout` is how long ONE request is allowed to hang before the
    client gives up on it -- a different number from
    call_with_cold_start_retry's max_wait_s, which is the total
    budget across repeated attempts. Raised from 60s to 300s
    (2026-08): a cold-starting or just genuinely slow model can take
    longer than 60s to answer a single request even while making
    real progress, so a short per-attempt timeout was cutting
    attempts off and forcing a retry instead of just waiting a
    little longer on the one already in flight.

    LLM_PROVIDER=aleph routes through Aleph using exported TYK_KEY,
    falling back to ALEPH_API_KEY for older setups. Source
    ~/.aleph_tyk.env and export TYK_KEY before launching Python.
    Aleph hosts open models (Command R,
    Qwen, Gemma, GPT-OSS, ...) for free -- request a key from
    research.support+aleph@ualberta.ca. Model NAMES are
    provider-specific: --model must name an Aleph model on this
    path (e.g. 'command-r-7b'), never an OpenAI one, and vice
    versa.

    Default (LLM_PROVIDER unset, or set to anything else) is
    OpenAI, unchanged from before this switch existed.
    """

    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()

    if provider == 'aleph':
        from teachers.aleph import AlephClient

        # Prefer the gateway credential confirmed by support. A stale
        # ALEPH_API_KEY in .env must not shadow an exported TYK_KEY.
        api_key = os.getenv('TYK_KEY') or os.getenv('ALEPH_API_KEY')
        api_key = api_key.strip() if api_key else ''
        if not api_key:
            raise RuntimeError(
                'LLM_PROVIDER=aleph needs a gateway key. Run '
                'source ~/.aleph_tyk.env; export TYK_KEY, or set '
                'ALEPH_API_KEY to that same gateway key.'
            )
        # Overridable in case Aleph's hostname changes; the current
        # one is confirmed OpenAI-compatible at /v1/chat/completions.
        base_url = os.getenv(
            'ALEPH_BASE_URL',
            'https://inference.vulcan.alliancecan.ca/v1',
        )
        return AlephClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            # Avoid rapid SDK retries underneath our paced backoff.
            max_retries=0,
        )

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError(
            'OPENAI_API_KEY not found in environment. Set it in a '
            '.env file or export it before running (or set '
            'LLM_PROVIDER=aleph plus ALEPH_API_KEY to use the '
            'free Vulcan gateway instead).'
        )
    override = os.getenv('LLM_MAX_SDK_RETRIES', '')
    if override:
        try:
            max_retries = max(0, int(override))
        except ValueError:
            pass
    return OpenAI(
        api_key=api_key, timeout=timeout, max_retries=max_retries
    )


def _is_retriable(exc: Exception) -> bool:
    """
    Decide whether `exc` is a transient/server-side failure worth
    retrying, using the openai SDK's actual exception hierarchy
    instead of matching specific error message strings.

    Matching message text is fragile by construction: every time
    Aleph's backend has surfaced a new failure shape this project
    has had to add another string to look for (first 'scaled_to_zero'
    / 'model_starting', then a bare client timeout, then, confirmed
    2026-08, a plain 'APIConnectionError: Connection error' that
    slipped through all of those checks and was never retried at
    all). Checking the exception TYPE instead covers every case in
    that category at once, including ones never explicitly seen yet:

    - APIConnectionError (network/DNS/connection-reset failures) and
      its subclass APITimeoutError (client-side request timeout) --
      both mean the request never got a real answer from the server.
    - RateLimitError (HTTP 429) -- transient by definition.
    - Any APIStatusError with a 5xx status code (InternalServerError
      and friends, including the explicit 503 'scaled to zero' cold
      start) -- server-side failures, not something retrying the
      identical request differently would avoid.

    Deliberately NOT retried: 4xx APIStatusErrors (BadRequestError,
    AuthenticationError, NotFoundError, ...) -- these mean the
    request itself is wrong (bad model name, bad API key, malformed
    schema) and an identical retry will fail identically forever, so
    surfacing it immediately is more useful than hiding it behind a
    multi-hour wait. Same for anything that is not an openai SDK
    error at all (e.g. a KeyError from parsing a malformed response)
    -- retrying will not produce a different response body to parse.
    """

    import openai

    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


def call_with_cold_start_retry(
    fn, max_wait_s: float = 7200.0, poll_interval_s: float = 30.0,
):
    """
    Retry transient API failures with a bounded, increasing delay.

    Aleph allows at most five attempts, delays of at least 30 seconds
    (then 60, then 120), and a 600-second retry eligibility window.
    A longer numeric Retry-After is honored if it fits that window.
    The window does not interrupt a request already in flight; the
    client's per-request timeout remains a separate limit.

    Authentication, schema and truncated-answer errors are not retried.
    Optional context-local observation records each request attempt for
    evaluation journals, without serializing prompts or credentials.
    Other providers retain their existing max_wait_s/poll defaults.
    """

    import os
    import time

    from teachers.reliability import failure_details, record_attempt

    # A paid run reserves for a fixed number of attempts per call, so
    # the retry loop must actually stop at that number. Without a cap
    # the non-Aleph path retries until `max_wait_s` elapses -- bounded
    # by time, not by attempts -- and any per-call cost bound built on
    # an attempt count would be fiction. Unset keeps the old behaviour.
    try:
        attempt_cap = int(os.getenv('LLM_MAX_ATTEMPTS', '') or 0)
    except ValueError:
        attempt_cap = 0

    aleph = os.getenv('LLM_PROVIDER', '').lower() == 'aleph'
    # Aleph is shared: bound retries as well as elapsed time, and
    # increase delays instead of repeatedly polling a busy model.
    if aleph:
        max_wait_s = min(max_wait_s, 600.0)
    deadline = time.monotonic() + max_wait_s
    attempts = 0
    while True:
        attempts += 1
        started = time.monotonic()
        record_attempt('request_start', attempt=attempts)
        try:
            reply = fn()
        except Exception as exc:
            retry = (_is_retriable(exc)
                     and time.monotonic() < deadline)
            if attempt_cap and attempts >= attempt_cap:
                retry = False
            delay = poll_interval_s
            if aleph:
                retry = retry and attempts < 5
                delay = min(max(30.0, poll_interval_s) *
                            2 ** (attempts - 1), 120.0)
                response = getattr(exc, 'response', None)
                retry_after = (
                    response.headers.get('Retry-After', '')
                    if response is not None else ''
                )
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
                if delay >= deadline - time.monotonic():
                    retry = False
            details = failure_details(exc)
            # Messages can contain prompts or credentials. Retry logs
            # use only types/status; the evaluator redacts seed errors.
            details.pop('message')
            record_attempt(
                'request_end', attempt=attempts, outcome='error',
                elapsed_s=time.monotonic() - started,
                retry_delay_s=delay if retry else None, **details,
            )
            if not retry:
                raise
            time.sleep(delay)
        else:
            usage = getattr(reply, 'usage', None)
            record_attempt(
                'request_end', attempt=attempts, outcome='ok',
                elapsed_s=time.monotonic() - started,
                tokens_in=getattr(usage, 'input_tokens', None),
                tokens_out=getattr(usage, 'output_tokens', None),
            )
            return reply


@dataclass
class Cost:
    """
    Per-call cost record.

    Every teacher fills in whichever fields are meaningful for it. A
    rule-based teacher populates `wall_time_s` and `compute_units`
    and leaves the LLM-specific fields at zero. An LLM teacher
    populates `wall_time_s`, `dollars`, `tokens_in`, `tokens_out`,
    and `model`. The shared schema lets analysis code compare cost
    across teacher types without special-casing each one.

    Attributes
    ----------
    wall_time_s: float
        Wall-clock time spent producing the advice.
    dollars: float
        Monetary cost in USD. Zero for local teachers.
    tokens_in, tokens_out: int
        LLM token counts. Zero for non-LLM teachers.
    compute_units: int
        Unit-free measure of computational work for non-LLM
        teachers — for example, the number of BFS nodes expanded,
        value-iteration sweeps, or N-step rollouts. Makes "effort"
        comparable across rule and planning teachers.
    model: str | None
        Identifier of the model that produced the advice, when
        applicable (e.g. 'gpt-4o-mini'). None for non-LLM teachers.
    metadata: dict
        Free-form bag for teacher-specific cost or provenance fields
        not covered above (prompt version, raw LLM response, etc.).
    """

    wall_time_s: float = 0.0
    dollars: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    compute_units: int = 0
    model: str | None = None
    metadata: dict = field(default_factory=dict)

    def __add__(self, other: 'Cost') -> 'Cost':
        """
        Sum two Cost records field by field.

        Useful for aggregating cost across all calls in an episode or
        a full training run. The `model` field is preserved from the
        left operand if both are equal and dropped otherwise, since
        a sum of costs across multiple models has no single model
        identifier. The `metadata` dict is shallow-merged.
        """

        if not isinstance(other, Cost):
            return NotImplemented

        # Decide which model identifier survives the sum. If both
        # operands report the same model, keep it; otherwise None,
        # since the sum no longer refers to one specific model.
        if self.model == other.model:
            merged_model = self.model
        else:
            merged_model = None

        return Cost(
            wall_time_s=self.wall_time_s + other.wall_time_s,
            dollars=self.dollars + other.dollars,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            compute_units=self.compute_units + other.compute_units,
            model=merged_model,
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class Advice:
    """
    A single piece of advice from a teacher.

    The same object supports three forms of advice so the controller
    code does not need to dispatch on teacher type:

    - Positive advising: set `action` to the recommended action.
    - Negative advising: set `forbidden_actions` to actions the
      teacher believes the agent should avoid.
    - Abstention: leave `action` as None and `forbidden_actions`
      empty when the teacher has no opinion at this state.

    Positive and negative advising can be combined in one Advice —
    `action=2, forbidden_actions=(0,)` means "do 2; if you override
    me, do not do 0".

    Attributes
    ----------
    action: int | None
        Recommended action, or None if the teacher abstains or only
        gives negative advice.
    forbidden_actions: tuple[int, ...]
        Actions the teacher believes the agent should not take. May
        be empty even when `action` is set.
    confidence: float
        Confidence in the recommendation, in [0, 1]. Deterministic
        rule teachers may use 1.0 unconditionally. LLM teachers
        should reflect their actual uncertainty.
    explanation: str | None
        Free-form explanation. Rule teachers may leave this None;
        LLM teachers fill it whenever the prompt requested
        reasoning.
    teacher_id: str
        Stable identifier for the teacher that produced this
        advice. Used by logs to group advice by source.
    call_id: str
        Unique identifier for this specific advice. Makes individual
        pieces of advice traceable so a controller can link
        downstream agent behavior back to the call that triggered
        it.
    cost: Cost
        Per-call cost record (see Cost docstring).
    """

    action: int | None = None
    forbidden_actions: tuple[int, ...] = ()
    confidence: float = 1.0
    explanation: str | None = None
    teacher_id: str = ''
    call_id: str = ''
    cost: Cost = field(default_factory=Cost)


class BaseTeacher(ABC):
    """
    Abstract base class for all teachers.

    Subclasses implement `recommend` and decide for themselves what
    shape `state` and `context` take. The base class is deliberately
    environment-agnostic: a FrozenLake teacher might accept an int
    state and a world map in `context`; a Mountain Car teacher
    accepts a (position, velocity) tuple and observation bounds; an
    LLM teacher accepts both, plus possibly rendered images. Each
    subclass documents its own expected inputs.

    The `context` argument is intentionally a free-form dict. It is
    where 'student transparency' experiments live: include or
    exclude the agent's Q-values, recent trajectory, current epsilon
    in `context` to vary what the teacher sees about the student,
    and observe how its advice changes.

    Subclasses are responsible for populating `teacher_id`,
    `call_id`, and `cost` on every Advice they return so that logs
    are always complete.

    Attributes
    ----------
    teacher_id: str
        Stable identifier propagated onto every Advice this teacher
        produces.
    seed: int
        RNG seed used by subclasses that need randomness (random
        tie-breaking, sampling, noisy oracles).
    is_stateless: bool
        Whether this teacher's answer depends on nothing but the
        state it is given. A subclass must opt in.

        This is not documentation -- it gates a correctness
        assumption elsewhere. `advising/peekable.py` avoids paying
        for a repeated query by replaying the answer it recorded at
        that state, which is only sound if the same state always
        earns the same answer. A teacher carrying a plan or a
        sub-goal across steps breaks that: the agent can stand in
        the same cell facing the same way while the active sub-goal
        is 'fetch the key' in one episode and 'carry the key to the
        door' in another, and the live teacher rightly says
        different things where the recording says one.

        The default is False so that a teacher which has not
        considered the question is refused rather than silently
        cached. Set it True only after checking the subclass holds
        no cross-step state.
    """

    # Conservative default: assume a teacher remembers something
    # until it declares otherwise.
    is_stateless: bool = False

    def __init__(self, teacher_id: str, seed: int = 0):
        """
        Initialize the teacher.

        Parameters
        ----------
        teacher_id: str
            Stable identifier for this teacher. Stored on every
            Advice the teacher produces and used by logging code to
            group advice by source.
        seed: int
            RNG seed for any subclass that needs randomness.
        """

        self.teacher_id = teacher_id
        self.seed = seed

        # Monotonic counter used to build per-call ids. Each call to
        # `_next_call_id` advances the counter so call ids are unique
        # within a teacher instance regardless of how the teacher is
        # used.
        self._call_counter = 0

    @abstractmethod
    def recommend(
        self, state: Any, context: dict | None = None
    ) -> Advice:
        """
        Produce a single piece of advice for `state`.

        Parameters
        ----------
        state: Any
            Whatever state representation the subclass expects. The
            base class places no constraints on this so subclasses
            can choose the representation most natural for their
            environment (int, tuple, array, embedding, etc.).
        context: dict | None
            Free-form auxiliary information the teacher may use.
            Common keys (subclass-dependent): 'q_values',
            'recent_states', 'epsilon', 'env_bounds', 'image'.

        Returns
        -------
        Advice
            Structured advice. Subclasses must populate at minimum
            `teacher_id`, `call_id`, and `cost`. They populate
            `action`, `forbidden_actions`, `confidence`, and
            `explanation` as appropriate to their role.
        """

    def _next_call_id(self) -> str:
        """
        Generate the next unique call id for this teacher.

        Returns
        -------
        call_id: str
            String of the form '<teacher_id>-NNNNNN' where NNNNNN
            is a zero-padded six-digit counter.
        """

        self._call_counter += 1
        return f'{self.teacher_id}-{self._call_counter:06d}'
