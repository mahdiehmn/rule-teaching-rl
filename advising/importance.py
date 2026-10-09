"""
State-importance functions, I(s).

Torrey & Taylor define a state's importance as the spread of
Q-values available at it:

    I(s) = max_a Q(s, a) - min_a Q(s, a)

The intuition is that when every action at `s` has the same value it
does not matter which one the student takes, so advice there is
wasted; when the spread is large, one action wins the episode and
another loses it, so advice there is worth paying for. The measure
originates with Clouse (1996), who applied it to the *learner's* own
Q-function as a confidence signal; Torrey & Taylor's contribution was
to compute it on the *teacher's* fully-learned Q-function instead,
where it reads as genuine state importance rather than learner
uncertainty.

Which of those two readings we can use depends on the teacher.

- A teacher with a Q-function (`value_iteration` here) supports the
  paper's exact formulation. Use `source='teacher_q'`. This is the
  replication-faithful setting.
- An LLM or VLM teacher has no Q-function and no free way to produce
  one. The only importance signal available before paying for a query
  is the *student's* own Q-spread -- Clouse's original reading. Use
  `source='student_q'`. It is a weaker signal (the student is still
  learning, so early on its Q-values are near-uniform everywhere),
  but it is free, it works with any teacher, and it is the honest
  substitute.

A hard constraint runs through this whole module: **importance must
be computable without consulting the teacher.** Its entire job is to
gate the decision to spend a query, so anything that costs a query to
evaluate is useless here. That rules out the LLM's self-reported
confidence as an ask-gate signal; it can only ever inform the
deliver gate, after the money is already spent. It is exposed here
anyway, clearly labelled, because that asymmetry is a result worth
measuring rather than a limitation to hide.

Thresholds and scale
--------------------
The raw scale of I(s) differs per teacher -- the paper notes that
"each teacher has its own range for I(s)" and sweeps ten thresholds
across whatever that range turns out to be. Rather than ask the user
to know the range in advance, `Normalizer` rescales any importance
function to roughly [0, 1] from a running min/max, so a threshold of
`t=0.5` means the same thing ("the more important half of states")
regardless of teacher. Pass `normalize=False` to work in raw units.
"""

from typing import Any, Callable

import numpy as np

# Importance functions all share this signature: they take the raw
# state and the free-form context the controller assembled, and
# return a non-negative float. Returning 0.0 is the standard way to
# say "no signal available here".
ImportanceFn = Callable[[Any, dict], float]


# Names accepted by `make_importance`, exposed so controllers can
# offer them as CLI choices without duplicating the list.
IMPORTANCE_SOURCES = (
    'none',
    'student_q',
    'teacher_q',
    'confidence',
)


def constant_importance(state: Any, context: dict) -> float:
    """
    Report every state as equally unimportant.

    With this function installed, any threshold of 0 passes at every
    state, which collapses importance advising into early advising
    and mistake correcting into its threshold-free form. The paper
    notes this equivalence explicitly (Sections 3.2 and 3.3), and it
    is the cleanest way to isolate the contribution of importance
    from the contribution of the rest of a strategy.
    """

    return 0.0


def student_q_importance(state: Any, context: dict) -> float:
    """
    Q-spread of the *student's* current Q-values at this state.

    Reads `context['q_values']`, which the controller fills with the
    student's Q-row for the state it is about to act in. Free to
    compute and available with any teacher, which is what makes it
    the workable choice for an LLM teacher.

    Returns 0.0 when the context carries no Q-values, so that a
    controller that has not been wired up yet degrades to
    importance-free behaviour instead of crashing.
    """

    q_values = context.get('q_values')
    if q_values is None:
        return 0.0

    q_row = np.asarray(q_values, dtype=np.float64).ravel()
    if q_row.size == 0:
        return 0.0

    return float(q_row.max() - q_row.min())


def advice_confidence_importance(state: Any, context: dict) -> float:
    """
    Importance read from a teacher's self-reported confidence.

    **Only usable in a deliver gate.** The value lives on the Advice
    object, so obtaining it means the query has already been paid
    for. Included so that experiments can ask "would confidence have
    been a better gate than student Q-spread, if it were free?" --
    a question worth answering even though the answer cannot be
    turned into a query-budget saving.

    Reads `context['confidence']`, which the deliver path fills in
    from the Advice; returns 0.0 when absent.
    """

    confidence = context.get('confidence')
    if confidence is None:
        return 0.0
    return float(confidence)


def make_teacher_q_importance(teacher: Any) -> ImportanceFn:
    """
    Build Torrey & Taylor's exact I(s) from a teacher's Q-function.

    Parameters
    ----------
    teacher: Any
        A teacher exposing `q_row(state) -> array-like | None`, the
        vector of Q-values it holds for `state`. `ValueIterationTeacher`
        implements this; teachers without a value function do not and
        cannot be used with this source.

    Returns
    -------
    ImportanceFn
        Closure computing `max_a Q_T(s,a) - min_a Q_T(s,a)`.

    Raises
    ------
    TypeError
        If the teacher has no `q_row` method. Failing here, at setup
        time, is much easier to diagnose than silently reporting
        zero importance for a whole run.
    """

    if not hasattr(teacher, 'q_row'):
        raise TypeError(
            f'Teacher {type(teacher).__name__} has no q_row() '
            f'method, so teacher-side importance cannot be '
            f'computed. Use --importance-source student_q with '
            f'this teacher.'
        )

    def teacher_q_importance(state: Any, context: dict) -> float:
        """
        Q-spread of the teacher's converged Q-values at this state.
        """

        q_row = teacher.q_row(state)
        if q_row is None:
            return 0.0

        row = np.asarray(q_row, dtype=np.float64).ravel()
        if row.size == 0:
            return 0.0

        return float(row.max() - row.min())

    return teacher_q_importance


class Normalizer:
    """
    Rescale an importance function to roughly [0, 1] on the fly.

    Keeps a running minimum and maximum of the raw values seen so
    far and maps each new value affinely into [0, 1]. This makes a
    threshold `t` mean the same thing across teachers whose raw
    I(s) ranges differ by orders of magnitude -- the paper handles
    the same problem by sweeping ten thresholds "uniformly
    distributed across that teacher's I(s) range", which requires
    knowing the range up front. A running estimate gets the same
    effect without the prior sweep.

    During the first `warmup` calls the running range is too thin to
    rank states meaningfully, so the normalizer reports 1.0 --
    "maximally important" -- which lets budget flow freely at the
    very start of training. That is a deliberate choice, not a
    fallback: it matches the paper's finding that early advice is
    the strong baseline, so a strategy that has not yet calibrated
    its importance estimates behaves like early advising rather
    than stalling.

    Attributes
    ----------
    fn: ImportanceFn
        The raw importance function being wrapped.
    warmup: int
        Number of initial calls during which 1.0 is returned.
    lo, hi: float
        Running range of raw values observed.
    """

    def __init__(self, fn: ImportanceFn, warmup: int = 100):
        """
        Parameters
        ----------
        fn: ImportanceFn
            Raw importance function to wrap.
        warmup: int
            Calls to observe before trusting the running range.
        """

        self.fn = fn
        self.warmup = warmup
        self.lo = float('inf')
        self.hi = float('-inf')
        self.num_calls = 0

    def __call__(self, state: Any, context: dict) -> float:
        """
        Return the wrapped function's value rescaled to [0, 1].
        """

        raw = float(self.fn(state, context))
        self.num_calls += 1

        # Widen the running range before rescaling, so the current
        # value is always inside the range it is measured against.
        self.lo = min(self.lo, raw)
        self.hi = max(self.hi, raw)

        if self.num_calls <= self.warmup:
            return 1.0

        # A degenerate range means every state seen so far scored
        # identically, so there is nothing to rank; treat them all
        # as important rather than all as unimportant, for the same
        # reason the warmup does.
        span = self.hi - self.lo
        if span <= 0.0:
            return 1.0

        return float((raw - self.lo) / span)

    @property
    def is_warm(self) -> bool:
        """
        Whether enough values have been seen to rank meaningfully.

        During warmup every state is reported as maximally
        important, so a controller reading the resulting delivery
        rate would see 100% and react to an artefact of the warmup
        rather than to the strategy. `BaseAdvisor` reads this to
        stay quiet until the signal is real.
        """

        return self.num_calls > self.warmup

    def stats(self) -> dict:
        """
        Observed raw range, for reporting alongside the threshold so
        a recorded run can be re-read in raw units later.
        """

        return {
            'importance_raw_min': (
                None if self.lo == float('inf') else self.lo
            ),
            'importance_raw_max': (
                None if self.hi == float('-inf') else self.hi
            ),
            'importance_calls': self.num_calls,
        }


class QuantileNormalizer:
    """
    Rescale an importance function to its quantile within a sliding
    window of recent values.

    Preferred over `Normalizer` (min-max) for two reasons.

    **Interpretability.** A min-max rescale maps a value to its
    position between the smallest and largest ever seen, which is
    not a percentile. If one freak state scores far above the rest,
    everything else is squashed toward zero and a threshold of 0.3
    silently rejects almost everything. Under a quantile, `t = 0.8`
    means exactly "the most important 20% of recent states", in
    every environment and for every importance signal.

    **Distribution shift.** Importance is not stationary here. Early
    in training a policy is near-uniform, so a policy-uncertainty
    signal is high nearly everywhere; late in training it is low
    nearly everywhere. A fixed cut on raw values therefore spends
    the entire budget in the first few thousand steps and then
    filters nothing -- which is early advising wearing importance
    advising's clothes. Ranking against a sliding window of RECENT
    values keeps the cut meaningful as the distribution moves.

    Attributes
    ----------
    fn: ImportanceFn
        The raw importance function being wrapped.
    window: int
        How many recent raw values to rank against. Larger is
        smoother but slower to track a shifting distribution.
    warmup: int
        Calls to observe before the ranking is trusted; 1.0 is
        returned until then, for the same reason as in `Normalizer`.
    """

    def __init__(
        self,
        fn: ImportanceFn,
        window: int = 2000,
        warmup: int = 100,
    ):
        """
        Parameters
        ----------
        fn: ImportanceFn
            Raw importance function to wrap.
        window: int
            Sliding-window size for the rank computation.
        warmup: int
            Calls before the ranking is trusted.
        """

        from collections import deque

        self.fn = fn
        self.window = window
        self.warmup = warmup
        self.recent: deque = deque(maxlen=window)
        self.num_calls = 0

        # Raw range is tracked purely for reporting, so a recorded
        # run can still be read back in the signal's own units.
        self.lo = float('inf')
        self.hi = float('-inf')

    def __call__(self, state: Any, context: dict) -> float:
        """
        Return the fraction of recent values this one exceeds.
        """

        raw = float(self.fn(state, context))
        self.num_calls += 1
        self.lo = min(self.lo, raw)
        self.hi = max(self.hi, raw)

        if self.num_calls <= self.warmup:
            # Not enough history to rank against. Pass everything,
            # matching `Normalizer` -- an uncalibrated strategy
            # should behave like early advising, not stall.
            self.recent.append(raw)
            return 1.0

        # Fraction of the window strictly below this value. Computed
        # before appending so a value is never ranked against
        # itself.
        window = self.recent
        if not window:
            # Reachable when warmup is 0: there is no history to
            # rank against yet, so pass the state through for the
            # same reason the warmup does.
            window.append(raw)
            return 1.0

        # Mid-rank, not the fraction strictly below. Counting only
        # strictly-smaller values is catastrophic when the signal
        # has ties, and this signal is full of them: early in
        # training a policy is uniform, so normalized entropy is
        # exactly 1.0 at nearly every state. A window of identical
        # values then scores every new identical value at quantile
        # 0.0 -- the most important states in the run are reported
        # as the least important, and no threshold above zero ever
        # passes anything.
        #
        # Splitting the tied mass puts a constant signal at 0.5, so
        # a threshold either side of it behaves sensibly and the
        # pacing controller has a direction to move in.
        below = 0
        equal = 0
        for v in window:
            if v < raw:
                below += 1
            elif v == raw:
                equal += 1
        quantile = (below + 0.5 * equal) / len(window)

        window.append(raw)
        return float(quantile)

    @property
    def is_warm(self) -> bool:
        """
        Whether enough values have been seen to rank meaningfully.

        During warmup every state is reported as maximally
        important, so a controller reading the resulting delivery
        rate would see 100% and react to an artefact of the warmup
        rather than to the strategy. `BaseAdvisor` reads this to
        stay quiet until the signal is real.
        """

        return self.num_calls > self.warmup

    def stats(self) -> dict:
        """
        Observed raw range and window occupancy, so a threshold can
        be translated back into the signal's own units later.
        """

        return {
            'importance_normalizer': 'quantile',
            'importance_raw_min': (
                None if self.lo == float('inf') else self.lo
            ),
            'importance_raw_max': (
                None if self.hi == float('-inf') else self.hi
            ),
            'importance_calls': self.num_calls,
            'importance_window': len(self.recent),
        }


def make_importance(
    source: str,
    teacher: Any = None,
    normalize: bool = True,
    warmup: int = 100,
    normalizer: str = 'quantile',
    window: int = 2000,
) -> ImportanceFn:
    """
    Build an importance function by name.

    Parameters
    ----------
    source: str
        One of IMPORTANCE_SOURCES. 'none' gives constant zero
        importance, 'student_q' the free student-side spread,
        'teacher_q' the paper's teacher-side spread, and
        'confidence' the deliver-gate-only teacher self-report.
    teacher: Any
        Required for 'teacher_q'; ignored otherwise.
    normalize: bool
        Wrap the result so thresholds live in [0, 1]. Turned off
        when a run wants raw units.
    warmup: int
        Forwarded to the normalizer.
    normalizer: str
        'quantile' (default) makes `t` a true percentile and is
        robust to outliers and to the way importance drifts as the
        student learns. 'minmax' is the simpler affine rescale;
        keep it only to reproduce runs recorded before quantile
        normalization existed.
    window: int
        Sliding-window size for the quantile normalizer.

    Returns
    -------
    ImportanceFn
        The importance function, possibly wrapped.
    """

    if source not in IMPORTANCE_SOURCES:
        raise ValueError(
            f'Unknown importance source {source!r}; expected one '
            f'of {IMPORTANCE_SOURCES}.'
        )

    if source == 'none':
        # Constant importance carries no scale, so normalizing it
        # would only add a pointless warmup phase.
        return constant_importance

    if source == 'student_q':
        fn: ImportanceFn = student_q_importance
    elif source == 'confidence':
        # Confidence is already reported in [0, 1] by construction,
        # so it needs no rescaling.
        return advice_confidence_importance
    else:
        if teacher is None:
            raise ValueError(
                'teacher_q importance requires a teacher instance.'
            )
        fn = make_teacher_q_importance(teacher)

    if not normalize:
        return fn

    if normalizer == 'quantile':
        return QuantileNormalizer(fn, window=window, warmup=warmup)
    if normalizer == 'minmax':
        return Normalizer(fn, warmup=warmup)

    raise ValueError(
        f'Unknown normalizer {normalizer!r}; expected '
        f"'quantile' or 'minmax'."
    )
