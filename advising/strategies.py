"""
The advice-budgeting strategies.

Four of the five reproduce Torrey & Taylor (AAMAS 2013) directly:
early advising, importance advising, mistake correcting, and
predictive advising. The fifth, `SurrogateAdvisor`, is the version of
predictive advising that an expensive teacher forces on us and has no
counterpart in the paper.

The free-lookup distinction
---------------------------
Two of the paper's strategies need to know the teacher's action
BEFORE deciding whether to spend budget: mistake correcting compares
it against what the student intends, and predictive advising compares
it against what the student is predicted to intend. For a teacher
that is a Q-table, that lookup is free, which is why the paper can
treat those comparisons as costless. For an LLM it is a paid API
call, and no amount of restructuring changes that.

Rather than fork the code, the strategies here probe the teacher for
an optional `peek_action(state)` method, which a teacher implements
if and only if it can name its action for free:

- **Teacher offers `peek_action`** (e.g. `ValueIterationTeacher`):
  the comparison happens in the ask gate, budget is spent only where
  it can change the student's behaviour, and the result matches the
  paper exactly.
- **Teacher does not** (e.g. `LLMTeacher`): the comparison moves to
  the deliver gate. The query is paid for either way and the advice
  is discarded when the student was already going to do the right
  thing. Student attention is still saved; money is not.

That asymmetry is not a shortcoming of the implementation, it is the
finding. `BaseAdvisor.stats()['waste_rate']` measures its size: the
fraction of paid queries thrown away. Running the same strategy
against a free teacher and a paid one and comparing waste rates is
the experiment.
"""

from typing import Any

from advising.base import BaseAdvisor
from advising.predictor import BasePredictor, make_predictor
from teachers.base import Advice, Cost

# Names accepted by `make_advisor`; exposed for CLI choices.
ADVISOR_NAMES = (
    'unlimited',
    'early',
    'importance',
    'mistake',
    'predictive',
    'surrogate',
)


def peek_action(teacher: Any, state: Any) -> int | None:
    """
    Ask a teacher for its action without paying for it.

    Returns the teacher's action when the teacher advertises a free
    `peek_action(state)` lookup, and None otherwise -- None meaning
    "this teacher cannot be consulted for free", not "this teacher
    has no opinion".

    Keeping this as a module-level helper rather than a method means
    the strategies never need to know which concrete teacher they
    are talking to; they only need to know whether a free lookup
    exists.
    """

    fn = getattr(teacher, 'peek_action', None)
    if fn is None:
        return None
    return fn(state)


class EarlyAdvisor(BaseAdvisor):
    """
    Torrey & Taylor's baseline: advise at the first `n` states the
    student encounters, then stop.

    Deliberately trivial -- the base class's budget check is the
    entire strategy. It is the baseline the paper's other three
    algorithms have to beat, and on Pac-Man with low-asymptote
    students it turned out to be surprisingly hard to beat, so it is
    worth carrying as a real arm rather than assuming it is weak.

    Note that this repo's existing `--teacher-until-step` /
    `--teacher-until-episode` flags are also forms of early advising,
    but they cap advice *per episode* or *by episode index* rather
    than in total. They compose with this strategy rather than
    duplicating it: the schedule decides when the teacher is present
    at all, and the budget decides how much it may say.
    """

    def __init__(self, **kwargs):
        """
        Initialize with the strategy name fixed to 'early'.
        """

        kwargs.setdefault('name', 'early')
        super().__init__(**kwargs)

    def _gate_ask(self, state, student_action, importance, context):
        """
        Always ask. Only the budget stops this strategy.
        """

        return True, 'ask'


class ImportanceAdvisor(BaseAdvisor):
    """
    Spend advice only at states whose importance clears a threshold.

    Implements the paper's Section 3.2. With `threshold=0` and a
    non-negative importance function this is provably identical to
    early advising, an equivalence the paper calls out and which the
    test suite checks.

    The importance function comes from `advising/importance.py` and
    must be free to evaluate, since it runs before any query is paid
    for. See that module for what "free" rules out.
    """

    def __init__(self, **kwargs):
        """
        Initialize with the strategy name fixed to 'importance'.
        """

        kwargs.setdefault('name', 'importance')
        super().__init__(**kwargs)

    def _gate_ask(self, state, student_action, importance, context):
        """
        Ask only at states at or above the importance threshold.
        """

        if importance < self.threshold:
            return False, 'below_threshold'
        return True, 'ask'


class MistakeCorrectingAdvisor(BaseAdvisor):
    """
    Spend advice only where the student is about to act differently
    from the teacher.

    Implements the paper's Section 3.3, which treats this as an
    upper bound on what advice can achieve, since advice at a state
    where the student already intends the right action changes
    nothing.

    The paper needs the student to *announce* its intended action,
    which it flags as extra communication that may not be available.
    We get the announcement for free: this advisor runs inside the
    student's own control loop, so the intended action is simply a
    local variable. What we do not get for free is the other half of
    the comparison -- the teacher's action -- which is the exact
    mirror image of the paper's difficulty. See the module docstring.

    Where the comparison happens is decided per teacher:

    - Free-lookup teacher: in the ask gate, so no budget of either
      kind is spent on non-mistakes. This is the paper's behaviour.
    - Paid teacher: in the deliver gate, so the query is spent and
      the advice discarded. `waste_rate` reports how often.
    """

    def __init__(self, **kwargs):
        """
        Initialize with the strategy name fixed to 'mistake'.
        """

        kwargs.setdefault('name', 'mistake')
        super().__init__(**kwargs)

    def _gate_ask(self, state, student_action, importance, context):
        """
        Apply the importance threshold, then -- only if the teacher
        can be consulted for free -- skip states where the student
        already agrees with the teacher.
        """

        if importance < self.threshold:
            return False, 'below_threshold'

        # A free lookup lets the mistake check happen before any
        # budget is committed. Teachers that cannot answer for free
        # return None here and the check is deferred.
        teacher_action = peek_action(self.teacher, state)
        if teacher_action is None or student_action is None:
            return True, 'ask'

        if not self.is_mistake(
            state, student_action, teacher_action, context
        ):
            return False, 'not_a_mistake'

        return True, 'ask'

    def _gate_deliver(
        self, state, student_action, advice, importance, context
    ):
        """
        Withhold advice that merely confirms what the student was
        already going to do.

        For a free-lookup teacher this gate is redundant -- the ask
        gate already filtered those states -- and harmlessly passes
        everything. For a paid teacher it is the only place the
        check can happen, and every rejection here is a query that
        was paid for and thrown away.
        """

        if student_action is None or advice.action is None:
            return True, 'deliver'

        if not self.is_mistake(
            state, student_action, advice.action, context
        ):
            return False, 'not_a_mistake'

        return True, 'deliver'


class PredictiveAdvisor(BaseAdvisor):
    """
    Mistake correcting without the student's announcement: predict
    what the student will do, and advise only where the prediction
    disagrees with the teacher.

    Implements the paper's Section 3.4. The teacher trains a
    classifier on the student's observed behaviour and substitutes
    its prediction for the announcement that mistake correcting
    requires.

    **Read this before interpreting results from this strategy.** In
    the paper's framing, predictive advising is a practical
    substitute for mistake correcting, which needs communication the
    teacher may not have. In our setting that communication is free
    -- the advisor sits inside the student's loop and can read its
    intended action directly. So predictive advising here is
    strictly worse than mistake correcting by construction, and it
    is carried for one reason only: to reproduce the paper's
    comparison and check that our implementation reproduces its
    ordering. `SurrogateAdvisor` is the strategy that solves the
    scarcity our setting actually has.

    To keep the replication honest, this advisor **ignores** the
    student action it is handed and uses only its predictor, exactly
    as a teacher without announcements would have to.
    """

    def __init__(
        self,
        predictor: BasePredictor | None = None,
        num_actions: int = 3,
        predictor_kind: str = 'count',
        key_fn=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        predictor: BasePredictor | None
            A pre-built predictor of the STUDENT's action. Built
            from `predictor_kind` when None.
        num_actions: int
            Action-space size, used when building the predictor.
        predictor_kind: str
            'count' or 'linear'; see advising/predictor.py.
        key_fn: callable | None
            State-key function for the count predictor. Pass the
            student's own discretizer when available.
        **kwargs
            Forwarded to BaseAdvisor.
        """

        kwargs.setdefault('name', 'predictive')
        super().__init__(**kwargs)

        self.predictor = predictor or make_predictor(
            predictor_kind, num_actions, key_fn=key_fn
        )

        # Remembered between the ask gate and the deliver gate so
        # both can reason about the same prediction, and so the
        # trace records what was predicted.
        self._last_prediction: int | None = None

    def observe(self, state, student_action=None, advice=None):
        """
        Train on the student's intended action.

        The controller calls this on every step, advised or not,
        because watching the student is free -- the paper's teacher
        observes every state and action and budgets only its advice.
        """

        if student_action is not None:
            self.predictor.observe(state, int(student_action))

    def note_episode_end(self, episode):
        """
        Refit the predictor between episodes, per the paper's
        protocol.
        """

        super().note_episode_end(episode)
        self.predictor.fit()

    def _gate_ask(self, state, student_action, importance, context):
        """
        Ask where the predicted student action disagrees with the
        teacher's -- or where there is no prediction yet, since an
        unknown state is a state we cannot rule out as a mistake.
        """

        if importance < self.threshold:
            return False, 'below_threshold'

        # Deliberately ignore the true student action: the point of
        # this strategy is to work without it.
        predicted = self.predictor.predict(state)
        self._last_prediction = predicted

        if predicted is None:
            # No prediction available. Asking is the safe failure
            # direction -- it wastes budget, whereas not asking
            # silently drops a teaching opportunity.
            return True, 'ask_unpredicted'

        teacher_action = peek_action(self.teacher, state)
        if teacher_action is None:
            # Paid teacher: the comparison cannot happen yet, so
            # commit to the query and settle it in the deliver gate.
            return True, 'ask'

        if int(teacher_action) == int(predicted):
            return False, 'predicted_agreement'

        return True, 'ask'

    def _gate_deliver(
        self, state, student_action, advice, importance, context
    ):
        """
        For a paid teacher, apply the prediction check now that the
        teacher's action is known.
        """

        predicted = self._last_prediction
        if predicted is None or advice.action is None:
            return True, 'deliver'

        # A free-lookup teacher already settled this in the ask
        # gate; repeating it here is harmless and keeps the two
        # teacher kinds on one code path.
        if int(advice.action) == int(predicted):
            return False, 'predicted_agreement'

        return True, 'deliver'

    def stats(self):
        """
        Add the predictor's prequential accuracy to the summary.

        The paper reports prediction accuracy alongside every
        predictive-advising result because the strategy's value
        depends on it directly (~50% on Mountain Car, 80-90% on
        Pac-Man, with correspondingly different outcomes), so ours
        travels with the results too.
        """

        out = super().stats()
        out['predictor'] = self.predictor.stats()
        return out


class SurrogateAdvisor(BaseAdvisor):
    """
    Predictive advising with the roles reversed: predict the
    TEACHER, not the student.

    This strategy has no counterpart in Torrey & Taylor because it
    solves a problem they do not have. Their scarce resource is the
    student's attention, so their predictor targets the student.
    Ours is the teacher's price per query, so ours targets the
    teacher: a cheap model is trained on advice already bought, and
    a real query is sent only when that model cannot cover the
    state.

    The decision rule at each state is:

    - surrogate has no confident guess  -> pay for a real query
    - surrogate's guess == student's intent -> no mistake, skip
      (free)
    - surrogate's guess != student's intent -> a mistake worth
      correcting. Either deliver the surrogate's own advice at zero
      cost (`use_surrogate_advice=True`) or pay for the real
      teacher's opinion to be sure.

    The free-advice path is the interesting one and the risky one.
    It can drive the whole run at a small fraction of the API spend,
    but the surrogate is trained on the teacher's *past* advice, so
    it inherits and amplifies the teacher's mistakes and cannot
    correct itself. `trust` sets how confident the surrogate must be
    before its advice is used; the `num_free_advice` counter reports
    how much of the run it ended up driving.

    Note that a per-state advice cache on the teacher (the LLM
    teacher here has one) already makes repeated states free. The
    surrogate's contribution must therefore be measured against
    cache-enabled spend, not against raw call count, or its saving
    will look far larger than it is.
    """

    def __init__(
        self,
        predictor: BasePredictor | None = None,
        num_actions: int = 3,
        predictor_kind: str = 'count',
        key_fn=None,
        trust: float = 0.8,
        use_surrogate_advice: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        predictor: BasePredictor | None
            A pre-built predictor of the TEACHER's action. Built
            from `predictor_kind` when None.
        num_actions: int
            Action-space size, used when building the predictor.
        predictor_kind: str
            'count' or 'linear'; see advising/predictor.py.
        key_fn: callable | None
            State-key function for the count predictor.
        trust: float
            Minimum surrogate confidence in [0, 1] before its guess
            is used at all. Below it, the state counts as uncovered
            and a real query is sent.
        use_surrogate_advice: bool
            When True, a confident surrogate that disagrees with the
            student delivers its own advice for free. When False,
            such states trigger a real query instead, so the
            surrogate only ever *saves* queries at states where the
            student already agrees with it.
        **kwargs
            Forwarded to BaseAdvisor.
        """

        kwargs.setdefault('name', 'surrogate')
        super().__init__(**kwargs)

        if not 0.0 <= trust <= 1.0:
            raise ValueError(
                f'trust must be in [0, 1]; got {trust}.'
            )

        self.predictor = predictor or make_predictor(
            predictor_kind, num_actions, key_fn=key_fn
        )
        self.trust = trust
        self.use_surrogate_advice = use_surrogate_advice

        # Monotonic counter behind the call ids of surrogate-authored
        # advice. Not a delivery count -- see `_free_advice`.
        self._free_call_counter = 0
        self._last_prediction: int | None = None

    def _surrogate_guess(self, state) -> int | None:
        """
        Return the surrogate's action for `state`, or None when it
        has no prediction or is not confident enough to be trusted.
        """

        predicted = self.predictor.predict(state)
        if predicted is None:
            return None

        # Predictors expose a confidence; treat one without as
        # fully confident so a custom predictor stays usable.
        confidence_fn = getattr(self.predictor, 'confidence', None)
        if confidence_fn is not None:
            if confidence_fn(state) < self.trust:
                return None

        return int(predicted)

    def observe(self, state, student_action=None, advice=None):
        """
        Train on the teacher's advice, not the student's actions.

        The controller passes both; this strategy uses only the
        advice, since the teacher is what it is trying to imitate.
        """

        if advice is not None and advice.action is not None:
            self.predictor.observe(state, int(advice.action))

    def note_episode_end(self, episode):
        """
        Refit the surrogate between episodes.
        """

        super().note_episode_end(episode)
        self.predictor.fit()

    def _gate_ask(self, state, student_action, importance, context):
        """
        Send a real query only where the surrogate cannot cover the
        state, or where it predicts a mistake it is not allowed to
        correct itself.
        """

        # Cleared before any early return. Without this, a state
        # rejected on importance -- or rejected by the base class's
        # budget checks, which run before this gate at all -- would
        # leave the PREVIOUS state's guess in place, and
        # `_free_advice` would then hand the student advice computed
        # for somewhere else entirely.
        self._last_prediction = None

        if importance < self.threshold:
            return False, 'below_threshold'

        guess = self._surrogate_guess(state)
        self._last_prediction = guess

        if guess is None:
            # Uncovered state: this is what the budget is for.
            return True, 'ask_uncovered'

        if student_action is not None and not self.is_mistake(
            state, student_action, guess, context
        ):
            # The surrogate believes the student is already doing
            # what the teacher would say, so there is nothing to
            # correct and nothing to buy.
            return False, 'surrogate_agrees'

        if self.use_surrogate_advice:
            # Handled for free by `_free_advice` below.
            return False, 'surrogate_advises'

        return True, 'ask'

    def _free_advice(self, state, student_action, importance, context):
        """
        Supply the surrogate's own advice at zero cost, when it is
        confident and believes the student is about to err.

        Consumes the advice budget (the student's attention is spent
        either way) but not the query budget.
        """

        if not self.use_surrogate_advice:
            return None

        # Recomputed here rather than read from `_last_prediction`.
        # The ask gate does not always run before this: the base
        # class short-circuits on an exhausted query budget without
        # consulting the strategy at all, and that is a path worth
        # supporting -- out of money but not out of the student's
        # attention is exactly when free advice is most valuable.
        # Reading a remembered value would serve the PREVIOUS
        # state's guess as advice for this one.
        guess = self._surrogate_guess(state)
        self._last_prediction = guess

        if guess is None:
            return None
        if student_action is not None and not self.is_mistake(
            state, student_action, guess, context
        ):
            return None

        confidence_fn = getattr(self.predictor, 'confidence', None)
        confidence = (
            confidence_fn(state) if confidence_fn else 1.0
        )

        # Only a call-id counter, deliberately not a statistic. The
        # caller may still discard this advice -- the advice budget
        # can be exhausted -- so counting deliveries here would
        # overstate them. `BaseAdvisor.num_free` counts the ones that
        # actually reached the student, and is what `stats` reports.
        self._free_call_counter += 1

        # A zero Cost is the literal truth here: no tokens, no
        # dollars, no teacher call. The teacher_id records that this
        # did not come from the real teacher, so a trace can always
        # separate bought advice from imitated advice.
        return Advice(
            action=int(guess),
            confidence=float(confidence),
            explanation=(
                "Surrogate model trained on the teacher's past "
                'advice; no query was sent.'
            ),
            teacher_id=f'{self.name}-surrogate',
            call_id=f'{self.name}-free-{self._free_call_counter:06d}',
            cost=Cost(),
        )

    def _gate_deliver(
        self, state, student_action, advice, importance, context
    ):
        """
        Learn from every real answer, then withhold it if it turns
        out the student was already right.
        """

        # Every paid answer is training data for the surrogate, so
        # the next visit to this state may be free.
        if advice.action is not None:
            self.predictor.observe(state, int(advice.action))

        if student_action is None or advice.action is None:
            return True, 'deliver'

        if not self.is_mistake(
            state, student_action, advice.action, context
        ):
            return False, 'not_a_mistake'

        return True, 'deliver'

    def stats(self):
        """
        Add surrogate accuracy and free-advice volume to the
        summary. `num_free_advice` against `num_asked` is the
        headline saving.
        """

        out = super().stats()
        out['predictor'] = self.predictor.stats()
        # `num_free` comes from the base class and counts advice the
        # student actually received from the surrogate. Against
        # `num_asked` it is the headline saving.
        out['num_surrogate_authored'] = self._free_call_counter
        return out
