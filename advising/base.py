"""
Advice budgeting: deciding WHEN a teacher is consulted and when its
advice actually reaches the student.

This package is the companion to `channels/`. A channel decides what
to DO with a piece of advice once it exists -- override the student's
action, mask actions, shape the reward. An advisor decides whether
advice should be produced and delivered at all, under a budget. The
two compose: any advisor pairs with any channel.

The strategies implemented here follow Torrey & Taylor (AAMAS 2013),
"Teaching on a Budget", which introduced four ways for a teacher to
spend a fixed quantity of action advice: early advising, importance
advising, mistake correcting, and predictive advising. That paper
assumed the teacher was itself an RL agent holding a Q-function, so
*asking* the teacher was free and only the student's attention was
scarce.

An LLM or VLM teacher breaks that assumption: every query costs money
and wall-clock time whether or not the advice is ultimately used.
This module therefore tracks two separate budgets.

- `query_budget`: how many times the teacher may be CONSULTED. This
  is the money/latency budget, and it is the binding constraint for
  an API teacher.
- `advice_budget`: how many times advice may be DELIVERED to the
  student. This is Torrey & Taylor's `n`, the student-attention
  budget.

Setting `query_budget` to unlimited recovers the original paper
exactly. Setting the two equal models a teacher that acts on every
consultation. Reporting both is what lets us say which strategies
save attention, which save money, and which save both -- a
distinction the 2013 framework could not express.

Following the convention used elsewhere in this repo (see
`--teacher-until-episode`), a budget of 0 means "unlimited".

Layout
------
- AdviceRecord: one row of the per-decision trace, for post-hoc
  analysis of where the budget went.
- BaseAdvisor: the budget bookkeeping, the ask/deliver pipeline, and
  the reporting. Concrete strategies subclass it and implement only
  `_gate_ask` (and optionally `_gate_deliver`).
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from teachers.base import Advice, BaseTeacher, Cost

# Sentinel meaning "this budget does not constrain anything". Kept as
# a module constant so the 0-means-unlimited convention is written
# down in one place rather than re-derived at each comparison.
UNLIMITED = 0


def exact_mistake(
    student_action: int, teacher_action: int, context: dict
) -> bool:
    """
    Torrey & Taylor's mistake test: the student is making a mistake
    if the action it intends differs from the teacher's.

    Exactly right for their setting and for any epsilon-greedy
    student, because such a student has one definite intended
    action. It is the wrong test for a student that SAMPLES its
    action from a distribution -- see `probability_mistake` in the
    benchmark repo's `advising/policy.py` for why, and for the
    replacement.

    Widened to the full set of optimal actions when the teacher can
    supply one. A teacher returns a single action, but a gridworld
    usually offers several that are equally good, and counting a
    student wrong for choosing a different member of a tie spends
    budget correcting something that was not an error.
    """

    optimal = context.get('optimal_actions')
    if optimal:
        return int(student_action) not in {int(a) for a in optimal}

    return student_action != teacher_action


@dataclass
class AdviceRecord:
    """
    One decision made by an advisor, kept for post-hoc analysis.

    Recording every decision -- not just the ones that produced
    advice -- is what makes it possible to plot where the budget went
    and to answer questions like "how much of the budget did mistake
    correcting save relative to importance advising, and at what
    point in training".

    Attributes
    ----------
    episode, step: int
        When the decision was made. `step` is the step index within
        the episode.
    importance: float
        The value of I(s) at this state, as computed by the
        advisor's importance function. Always recorded, even when
        the strategy ignores importance, so importance
        distributions can be compared across strategies.
    asked: bool
        Whether the teacher was actually consulted (a query was
        paid for).
    delivered: bool
        Whether the resulting advice reached the student. False
        either because no query was made or because the advice was
        withheld after the fact.
    student_action: int | None
        The action the student intended to take at this state.
    teacher_action: int | None
        The action the teacher recommended, when a query was made.
    predicted_action: int | None
        The action the advisor's predictor guessed, for the
        strategies that use one. None otherwise.
    reason: str
        Short tag explaining the outcome ('ask', 'no_advice_budget',
        'below_threshold', 'not_a_mistake', ...). Grouping records
        by reason gives a breakdown of why budget was or was not
        spent.
    """

    episode: int = 0
    step: int = 0
    importance: float = 0.0
    asked: bool = False
    delivered: bool = False
    student_action: int | None = None
    teacher_action: int | None = None
    predicted_action: int | None = None
    reason: str = ''


class BaseAdvisor(ABC):
    """
    Abstract base class for advice-budgeting strategies.

    The public entry point is `advise`, which runs the full pipeline
    for one student step:

        1. Compute I(s) from signals that are free to obtain.
        2. Ask gate -- may we spend a query here? (budget checks in
           this base class, strategy logic in `_gate_ask`.)
        3. If yes, consult the teacher and pay its cost.
        4. Deliver gate -- should the advice reach the student?
           (`_gate_deliver`; default yes.)

    Splitting the decision into an ask gate and a deliver gate is the
    adaptation that makes the 2013 algorithms meaningful with an
    expensive teacher. Mistake correcting, for example, needs the
    teacher's action in order to know whether the student is about to
    make a mistake. With a Q-function teacher that knowledge is free,
    so the check belongs in the ask gate. With an LLM teacher it
    costs a query, so the check can only happen in the deliver gate:
    the money is spent either way, and only the student's attention
    is saved. The two counters make that difference visible instead
    of hiding it.

    Attributes
    ----------
    name: str
        Short identifier for the strategy, used in logs and in run
        directory names.
    advice_budget, query_budget: int
        Caps on deliveries and consultations respectively. 0 means
        unlimited.
    threshold: float
        Importance threshold `t`. States with I(s) < t are skipped
        by strategies that use importance.
    importance_fn: callable
        Maps (state, context) -> float. Must be cheap: it is
        evaluated at every step, before any query is paid for. See
        `advising/importance.py`.
    min_confidence: float
        Advice whose `confidence` is below this value is withheld.
        Has no analogue in the 2013 paper, where teachers were
        always certain; an LLM teacher self-reports confidence and
        this is the cheapest way to use it.
    trace: bool
        When True, keep an AdviceRecord for every decision. Costs
        memory proportional to the number of student steps.
    """

    def __init__(
        self,
        advice_budget: int = UNLIMITED,
        query_budget: int = UNLIMITED,
        threshold: float = 0.0,
        importance_fn: Callable[[Any, dict], float] | None = None,
        min_confidence: float = 0.0,
        mistake_fn: Callable[[int, int, dict], bool] | None = None,
        target_rate: float | None = None,
        horizon: int | None = None,
        pace_gain: float = 0.5,
        pace_interval: int = 25,
        trace: bool = False,
        name: str = 'base',
    ):
        """
        Initialize the advisor and its counters.

        Parameters
        ----------
        advice_budget: int
            Maximum number of pieces of advice delivered to the
            student over the whole run. 0 means unlimited.
        query_budget: int
            Maximum number of teacher consultations over the whole
            run. 0 means unlimited.
        threshold: float
            Importance threshold `t`.
        importance_fn: callable | None
            Importance function I(s). None installs a constant-zero
            function, which makes every threshold of 0 pass and
            reduces importance-based strategies to their
            importance-free counterparts.
        min_confidence: float
            Withhold advice below this self-reported confidence.
        mistake_fn: callable | None
            Decides whether the student is making a mistake, given
            (student_action, teacher_action, context). None installs
            `exact_mistake`, the paper's test. A student that samples
            its actions needs a different test -- see
            `probability_mistake` in the benchmark repo.
        target_rate: float | None
            Fraction of ENVIRONMENT steps that should receive
            advice. When set, `threshold` stops being a
            hyperparameter and becomes a controlled variable: it is
            nudged up when the advisor is overspending and down when
            it is underspending, so the spend rate stays roughly
            constant across training instead of being front-loaded.

            The denominator is environment steps, not the steps this
            advisor happens to see. A hand-off schedule decides which
            steps reach the advisor at all, and in prefix mode that
            population shrinks to nothing as the prefix anneals -- so
            a rate measured against it would mean something different
            at every point in training, and two runs with different
            schedules could report the same rate while spending
            wildly different amounts.
        horizon: int | None
            Expected total environment steps. Supplying it together
            with `advice_budget` and no `target_rate` switches on
            budget pacing: the target becomes remaining advice over
            remaining horizon, recomputed as the run proceeds. That
            is the setting to prefer when comparing strategies,
            because the budget is then the controlled variable and
            every arm delivers the same amount of advice, evenly
            spread, with no separately chosen rate to reconcile.
        pace_gain: float
            Proportional gain of that controller. Larger tracks the
            target faster and oscillates more.
        pace_interval: int
            Steps between threshold updates. Frequent updates chase
            noise; infrequent ones lag the distribution.
        trace: bool
            Keep a per-decision trace.
        name: str
            Strategy name for reporting.
        """

        if advice_budget < 0:
            raise ValueError(
                f'advice_budget must be >= 0; got {advice_budget}.'
            )
        if query_budget < 0:
            raise ValueError(
                f'query_budget must be >= 0; got {query_budget}.'
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f'min_confidence must be in [0, 1]; got '
                f'{min_confidence}.'
            )

        self.name = name
        self.advice_budget = advice_budget
        self.query_budget = query_budget
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.importance_fn = importance_fn or (
            lambda state, context: 0.0
        )
        self.mistake_fn = mistake_fn or exact_mistake
        self.trace = trace

        if target_rate is not None and not 0.0 < target_rate <= 1.0:
            raise ValueError(
                f'target_rate must be in (0, 1]; got {target_rate}.'
            )
        self.target_rate = target_rate
        self.horizon = horizon
        self.pace_gain = pace_gain
        self.pace_interval = pace_interval

        # Budget pacing: with a budget and a horizon but no rate of
        # its own, the advisor derives the rate that spends exactly
        # the budget over exactly the run, and keeps re-deriving it.
        # This is what resolves the conflict between a hard budget
        # and an independently chosen rate -- previously both could
        # be set, the budget would run out early, and the controller
        # would drive the threshold to the floor chasing a rate that
        # a gate it does not control had already made unreachable.
        self.pace_to_budget = (
            target_rate is None
            and horizon is not None
            and advice_budget != UNLIMITED
        )
        if target_rate is not None and advice_budget != UNLIMITED:
            # Both given. The budget still caps, and
            # `budget_bound_first` records if it ever did, but the
            # comparison between strategies is no longer at equal
            # spend, so say so rather than letting it pass silently.
            warnings.warn(
                'both advice_budget and target_rate are set. The '
                'budget is a hard cap and the rate is a pacing '
                'target; if they disagree the budget wins and the '
                'run stops being a fixed-rate experiment. Pass a '
                'horizon and omit target_rate to pace to the budget '
                'instead.',
                UserWarning,
                stacklevel=2,
            )

        # Environment steps, as reported by the training loop. The
        # advisor's own `num_steps` counts only the steps a hand-off
        # schedule let through to it, which is the wrong denominator
        # for anything describing cost.
        self.num_env_steps = 0
        self._env_steps_at_last_pace = 0
        # Set by the training loop via
        # `set_expected_remaining_visible` when it can integrate its
        # hand-off schedule forward. None means fall back to
        # extrapolating observed visibility.
        self._expected_remaining_visible: float | None = None
        # Sum of within-episode step indices seen, so the trace can
        # show WHERE in an episode the advisor got to act. Under a
        # prefix hand-off this is systematically the opening, which
        # limits what an importance claim can say.
        self._ep_step_sum = 0

        # With a quantile-normalized importance signal, the cut that
        # admits the top `target_rate` of states is exactly
        # 1 - target_rate, so the controller starts already close to
        # its setpoint instead of spending the early training
        # sweeping toward it. Under raw units this is only a
        # starting guess and the ceiling below is what keeps the
        # target reachable.
        if target_rate is not None:
            self.threshold = 1.0 - target_rate
        elif self.pace_to_budget:
            self.threshold = 1.0 - min(
                1.0, advice_budget / max(1, horizon)
            )

        # Largest threshold the controller may set. Normalized
        # importance is a quantile so 1.0 is the true ceiling, but a
        # raw signal can range higher -- entropy in nats reaches
        # ln(num_actions) -- and a controller capped below the
        # signal's range can never make the gate strict enough to
        # hit a low target rate. Widened from observed values.
        self.threshold_ceiling = 1.0

        # Set once the budget, rather than importance, becomes the
        # binding constraint. Reported so a run can never be read as
        # "importance advising with threshold t" when exhaustion was
        # actually making every decision.
        self.budget_bound_first = False

        # Set when the hand-off schedule, rather than the budgeting
        # strategy, is what limits spending: the advisor is simply
        # not shown enough steps to place its advice on, whatever
        # the threshold. A run with this set is not a comparison
        # between strategies.
        self.rate_unreachable = False

        # Threshold history, sampled at each controller update, so a
        # run can be checked afterwards for whether the controller
        # settled or oscillated.
        self.threshold_history: list[float] = []

        # Delivery count at the previous controller update, used to
        # measure the rate over just the last interval.
        self._delivered_at_last_pace = 0

        # Lifetime counters. `num_asked` is the money-side count and
        # `num_delivered` is the attention-side count; the gap
        # between them is exactly the advice that was paid for and
        # then thrown away.
        self.num_steps = 0
        self.num_asked = 0
        self.num_delivered = 0
        self.num_withheld = 0
        self.num_free = 0
        self.cost_total = Cost()

        # The teacher currently being advised against, set for the
        # duration of each `advise` call. Strategies read it to
        # probe for a free action lookup (see
        # `advising/strategies.peek_action`) without every subclass
        # having to thread the teacher through its own signatures.
        self.teacher: BaseTeacher | None = None

        # Breakdown of why budget was not spent, keyed by the reason
        # tag returned from the gates. Turned into a histogram at
        # reporting time.
        self.reasons: dict[str, int] = {}

        # Position in the run, maintained by the controller through
        # `note_step` / `note_episode_end` so records are labelled.
        self.episode = 0
        self.step = 0

        # Per-decision trace, populated only when `trace` is True.
        self.records: list[AdviceRecord] = []

        # Running sum of I(s) over all steps, used to report the
        # average importance a strategy saw versus the average
        # importance it actually spent budget on.
        self._importance_sum = 0.0
        self._importance_spent_sum = 0.0

    # -- Budget helpers ------------------------------------------

    @property
    def advice_remaining(self) -> float:
        """
        Number of deliveries still allowed, or infinity if the
        advice budget is unlimited.
        """

        if self.advice_budget == UNLIMITED:
            return float('inf')
        return self.advice_budget - self.num_delivered

    @property
    def query_remaining(self) -> float:
        """
        Number of consultations still allowed, or infinity if the
        query budget is unlimited.
        """

        if self.query_budget == UNLIMITED:
            return float('inf')
        return self.query_budget - self.num_asked

    @property
    def exhausted(self) -> bool:
        """
        True once neither a query nor a delivery is possible, so the
        controller can stop calling the advisor entirely.
        """

        return self.advice_remaining <= 0 or self.query_remaining <= 0

    # -- Main entry point ----------------------------------------

    def advise(
        self,
        teacher: BaseTeacher,
        state: Any,
        student_action: int | None = None,
        context: dict | None = None,
    ) -> tuple[Advice | None, Cost]:
        """
        Run one advising decision and return what the channel should
        see plus what it cost.

        Parameters
        ----------
        teacher: BaseTeacher
            The teacher to consult if the ask gate passes.
        state: Any
            The student's current state, in whatever representation
            the teacher expects.
        student_action: int | None
            The action the student intends to take. Required by
            mistake correcting and predictive advising; ignored by
            the others.
        context: dict | None
            Free-form context forwarded to both the importance
            function and the teacher. The importance functions in
            this package read 'q_values' from it.

        Returns
        -------
        advice: Advice | None
            The advice to hand to the channel, or None when no
            advice should be applied at this step. None covers both
            "we did not ask" and "we asked and withheld the answer".
        cost: Cost
            What this decision actually cost. Non-zero whenever the
            teacher was consulted, INCLUDING when the advice was
            then withheld -- an LLM call that gets discarded still
            appears on the bill, and reporting it any other way
            would understate the price of mistake correcting.
        """

        ctx = dict(context or {})
        self.num_steps += 1

        # Re-aim the threshold at the requested spend rate. No-op
        # unless `target_rate` was set.
        self._pace()

        # Expose the teacher to the gates for the duration of this
        # decision, so a strategy can check whether it offers a free
        # action lookup before committing budget.
        self.teacher = teacher

        # Importance is evaluated from free signals only, so it can
        # gate the decision to ask before any query is paid for.
        importance = float(self.importance_fn(state, ctx))
        self._importance_sum += importance

        # Track how high the signal actually goes, so the pacing
        # controller can raise its cut above a raw signal's typical
        # values instead of being pinned at 1.0.
        if importance > self.threshold_ceiling:
            self.threshold_ceiling = importance

        ask, reason = self._check_ask(
            state, student_action, importance, ctx
        )
        if not ask:
            # Declining to ask is not always the same as declining
            # to advise. A strategy holding a model of the teacher
            # can cover some states itself, at zero marginal cost.
            # Such advice spends the student-attention budget --
            # the student is interrupted either way -- but never
            # the query budget.
            free = self._free_advice(
                state, student_action, importance, ctx
            )
            if free is not None and self.advice_remaining > 0:
                self.num_delivered += 1
                self.num_free += 1
                self._importance_spent_sum += importance
                self._finish(
                    importance=importance,
                    asked=False,
                    delivered=True,
                    student_action=student_action,
                    teacher_action=free.action,
                    reason='free_advice',
                )
                return free, Cost()

            self._finish(
                importance=importance,
                asked=False,
                delivered=False,
                student_action=student_action,
                teacher_action=None,
                reason=reason,
            )
            return None, Cost()

        # The ask gate passed: consult the teacher and pay for it.
        advice = teacher.recommend(state, ctx or None)
        self.num_asked += 1
        cost = advice.cost
        self.cost_total = self.cost_total + cost

        deliver, reason = self._check_deliver(
            state, student_action, advice, importance, ctx
        )
        if not deliver:
            self.num_withheld += 1
            self._finish(
                importance=importance,
                asked=True,
                delivered=False,
                student_action=student_action,
                teacher_action=advice.action,
                reason=reason,
            )
            return None, cost

        self.num_delivered += 1
        self._importance_spent_sum += importance
        self._finish(
            importance=importance,
            asked=True,
            delivered=True,
            student_action=student_action,
            teacher_action=advice.action,
            reason=reason,
        )
        return advice, cost

    # -- Observation and bookkeeping hooks -----------------------

    def observe(
        self,
        state: Any,
        student_action: int | None = None,
        advice: Advice | None = None,
    ) -> None:
        """
        Watch a student step for free.

        Torrey & Taylor's teacher observes every state the student
        visits and every action it takes; only *advising* is
        budgeted, not watching. Strategies that train a predictor
        use this stream as training data. The base implementation
        does nothing.

        Controllers should call this once per step with the action
        the student *intended*, not the action that was executed. If
        a channel overrode the action, the executed action is the
        teacher's, and training a student-action predictor on it
        would teach the predictor to imitate the teacher instead.
        """

    def note_step(self, step: int) -> None:
        """
        Tell the advisor which step of the episode it is on, so
        trace records are labelled correctly.
        """

        self.step = step
        self._ep_step_sum += step

    def set_expected_remaining_visible(self, count: float) -> None:
        """
        Tell the advisor how many more steps it is expected to be
        consulted on before the run ends.

        Only the training loop can answer this, because only it
        knows the hand-off schedule. Supplying it is what makes
        budget pacing correct under a schedule that closes: without
        it the advisor assumes visibility continues at the rate it
        has seen, paces leisurely, and is still holding most of its
        budget when the teacher stops being consulted at all.
        """

        self._expected_remaining_visible = max(0.0, float(count))

    def note_env_steps(self, count: int = 1) -> None:
        """
        Report environment steps that elapsed, whether or not the
        advisor was consulted on them.

        The training loop must call this once per step for every
        environment. Without it, every rate the advisor reports is a
        fraction of the steps a hand-off schedule chose to show it,
        which is not a cost and is not comparable between runs whose
        schedules differ.
        """

        self.num_env_steps += count

    def note_episode_end(self, episode: int) -> None:
        """
        Signal the end of an episode.

        The base implementation only advances the episode counter.
        Predictor-based strategies override this to retrain, which
        matches the paper's protocol of fitting a fresh classifier
        between episodes rather than during them.
        """

        self.episode = episode + 1
        self.step = 0

    def _pace(self) -> None:
        """
        Nudge the importance threshold toward the spend rate the
        caller asked for.

        A fixed threshold interacts badly with a finite budget,
        because the distribution of I(s) moves as the student
        learns. Early on a policy is uncertain nearly everywhere, so
        nearly every state clears any reasonable cut and the whole
        budget is consumed in the first few thousand steps -- at
        which point the strategy has quietly become early advising
        and the threshold never filters anything again.

        This is a plain proportional controller against the observed
        delivery rate. Overspending raises the bar, underspending
        lowers it. It needs no knowledge of the run's length, which
        matters because the advisor does not see every environment
        step -- a hand-off schedule decides how many reach it.
        """

        if self.target_rate is None and not self.pace_to_budget:
            return
        if self.num_steps % self.pace_interval != 0:
            return

        # Stay quiet while the importance signal is still warming
        # up. A normalizer with no history yet reports every state as
        # maximally important, so the controller would read a 100%
        # delivery rate, conclude it was wildly overspending, and
        # drive the threshold to its ceiling -- then spend the rest
        # of the run climbing back down from an artefact of the
        # warmup rather than converging on the target.
        importance_warm = getattr(self.importance_fn, 'is_warm', True)
        if not importance_warm:
            # Reset the measurement window so the first real update
            # is not polluted by warmup deliveries either.
            self._delivered_at_last_pace = self.num_delivered
            self._env_steps_at_last_pace = self.num_env_steps
            return

        # Anti-windup. Once the advice budget is gone, deliveries
        # stop for a reason the threshold cannot influence, and a
        # controller that kept reacting would read the zero rate as
        # "far too strict", drive the threshold to the floor, and
        # still deliver nothing -- leaving a run that reports itself
        # as importance advising with a threshold of 0, which is
        # early advising. Freezing here keeps the recorded threshold
        # meaningful and leaves the reason histogram to say plainly
        # that exhaustion, not importance, was the binding gate.
        if self.advice_remaining <= 0 or self.query_remaining <= 0:
            self.budget_bound_first = True
            return

        # Measure the rate over the LAST interval, not over the run
        # so far. A cumulative average is a lagging indicator: once
        # a few thousand steps are banked, it barely moves, so a
        # controller reading it cannot notice that spending has
        # stopped entirely and never corrects. The windowed rate
        # responds immediately.
        recent = self.num_delivered - self._delivered_at_last_pace
        self._delivered_at_last_pace = self.num_delivered

        # Denominator is ENVIRONMENT steps elapsed, not advisor
        # calls. A hand-off schedule decides how many steps reach
        # the advisor, and in prefix mode that number shrinks toward
        # zero as the prefix anneals -- so pacing against advisor
        # calls would silently redefine the target as training went
        # on. Falls back to advisor calls only when the loop never
        # reports environment steps, which is the tabular case where
        # the two are the same thing anyway.
        env_recent = (
            self.num_env_steps - self._env_steps_at_last_pace
        )
        self._env_steps_at_last_pace = self.num_env_steps

        # Under budget pacing the target is re-derived every time
        # from what is left: spend the remaining advice evenly over
        # the remaining run. This self-corrects after any stretch of
        # over- or under-spending, and needs no rate from the caller
        # at all -- the budget is the only thing specified, so every
        # arm of a comparison delivers the same amount.
        if self.pace_to_budget:
            remaining_steps = max(
                1, self.horizon - self.num_env_steps
            )
            target_env = min(
                1.0, self.advice_remaining / remaining_steps
            )
        else:
            target_env = self.target_rate

        # Both the target and the achievement must be fractions of
        # the population the threshold actually gates: the steps the
        # advisor is SHOWN. Expressing the target per environment
        # step and the achievement per visible step mixes two scales
        # that can differ by thirty times, and the controller
        # becomes almost inert.
        #
        # How many more visible steps there will be is not knowable
        # from here -- a hand-off schedule decides it, and under a
        # prefix hand-off the answer is "fewer and fewer, then
        # none". Extrapolating the visibility observed so far is the
        # best estimate available without teaching the advisor about
        # schedules, and it self-corrects as the real rate changes.
        if self._expected_remaining_visible is not None:
            # The training loop integrated its hand-off schedule
            # forward and told us. This is the only reliable source:
            # a schedule that closes -- a prefix annealing to zero --
            # front-loads every offerable step, and no amount of
            # observing the past reveals that the opportunity is
            # about to vanish.
            expected_visible = self._expected_remaining_visible
        elif self.horizon is not None and self.num_env_steps > 0:
            # Fallback: assume visibility continues at the rate seen
            # so far. Correct for a stationary schedule, optimistic
            # for a closing one -- which is why the loop should
            # supply the real number when it can.
            remaining_env = max(
                0, self.horizon - self.num_env_steps
            )
            expected_visible = self.num_steps * (
                remaining_env / self.num_env_steps
            )
        else:
            expected_visible = float(self.num_steps)

        if self.pace_to_budget:
            target = self.advice_remaining / max(1.0, expected_visible)
        else:
            # A caller-supplied rate is per environment step, so
            # convert it into the visible units the gate works in.
            observed_visibility = (
                self.pace_interval / env_recent
                if env_recent > 0
                else 1.0
            )
            target = target_env / max(1e-6, observed_visibility)

        achieved = recent / self.pace_interval

        # A target above 1 means no threshold, however permissive,
        # can spend at this rate: the hand-off schedule simply will
        # not show the advisor enough steps. The run is then bounded
        # by the schedule rather than by the budgeting strategy, and
        # comparing strategies across such runs would be comparing
        # the schedule with itself.
        if target > 1.0:
            self.rate_unreachable = True
        target = min(1.0, target)

        error = achieved - target

        # Clamped to the range the importance signal actually spans.
        # Normalized importance is a quantile in [0, 1], but a run
        # using raw units is not: entropy in nats tops out at
        # ln(num_actions), and clamping that to 1.0 would put most
        # of the signal's range above every reachable threshold, so
        # the target rate could never be hit. `threshold_ceiling` is
        # learned from the values actually seen.
        self.threshold = min(
            self.threshold_ceiling,
            max(0.0, self.threshold + self.pace_gain * error),
        )
        self.threshold_history.append(self.threshold)

    def is_mistake(
        self,
        state: Any,
        student_action: int | None,
        teacher_action: int | None,
        context: dict,
    ) -> bool:
        """
        Is the student about to make a mistake at this state?

        Wraps the configured `mistake_fn` and handles the unknown
        cases uniformly: when either action is missing, report a
        mistake. That is the safe failure direction -- it wastes
        some budget, whereas reporting "no mistake" would silently
        drop a teaching opportunity and be invisible in the logs.
        """

        if student_action is None or teacher_action is None:
            return True

        # Offer the mistake test the full set of optimal actions
        # when the teacher can name one for free. Torrey & Taylor's
        # teacher returns a single action, and comparing against it
        # counts a student that picked a DIFFERENT but equally
        # optimal move as mistaken -- common in a gridworld, where
        # two routes around an obstacle are often the same length.
        # Tests that do not understand sets ignore this key.
        if 'optimal_actions' not in context:
            optimal_fn = getattr(self.teacher, 'optimal_actions', None)
            if optimal_fn is not None:
                context['optimal_actions'] = optimal_fn(state)

        return bool(
            self.mistake_fn(
                int(student_action), int(teacher_action), context
            )
        )

    # -- Gates ---------------------------------------------------

    def _check_ask(
        self,
        state: Any,
        student_action: int | None,
        importance: float,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Apply the budget checks shared by every strategy, then defer
        to the strategy's own gate.
        """

        # A strategy can never consult the teacher once either
        # budget is gone: no queries left means no money, and no
        # deliveries left means any answer would be discarded, so
        # paying for it would be pure waste.
        if self.query_remaining <= 0:
            return False, 'no_query_budget'
        if self.advice_remaining <= 0:
            return False, 'no_advice_budget'

        return self._gate_ask(
            state, student_action, importance, context
        )

    def _check_deliver(
        self,
        state: Any,
        student_action: int | None,
        advice: Advice,
        importance: float,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Apply the confidence check shared by every strategy, then
        defer to the strategy's own gate.
        """

        # An abstaining teacher has nothing to deliver. This is not
        # a wasted delivery, but it is a wasted query, and the
        # 'abstained' tag keeps those visible in the breakdown.
        if advice.action is None and not advice.forbidden_actions:
            return False, 'abstained'
        if advice.confidence < self.min_confidence:
            return False, 'low_confidence'

        return self._gate_deliver(
            state, student_action, advice, importance, context
        )

    @abstractmethod
    def _gate_ask(
        self,
        state: Any,
        student_action: int | None,
        importance: float,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Decide whether to spend a query at this state.

        Called only after the budget checks have passed. Must rely
        exclusively on information that is free to obtain: the
        budget, the schedule, the student's own internals, and any
        predictor the strategy maintains. Anything that requires the
        teacher's answer belongs in `_gate_deliver` instead.

        Returns
        -------
        (ask, reason): tuple[bool, str]
            Whether to consult the teacher, and a short tag naming
            the reason for the decision.
        """

    def _gate_deliver(
        self,
        state: Any,
        student_action: int | None,
        advice: Advice,
        importance: float,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Decide whether advice already paid for should reach the
        student. Default: always deliver.
        """

        return True, 'deliver'

    def _free_advice(
        self,
        state: Any,
        student_action: int | None,
        importance: float,
        context: dict,
    ) -> Advice | None:
        """
        Optionally produce advice without consulting the teacher.

        Called only when the ask gate has declined. Returning an
        Advice here delivers it to the student at zero cost;
        returning None (the default, and the behaviour of every
        strategy that does not model the teacher) means the step
        passes without advice.

        Any Advice returned must carry a zero `Cost` and a
        `teacher_id` that identifies it as not having come from the
        real teacher, so traces can always separate bought advice
        from imitated advice.
        """

        return None

    # -- Reporting -----------------------------------------------

    def _finish(
        self,
        importance: float,
        asked: bool,
        delivered: bool,
        student_action: int | None,
        teacher_action: int | None,
        reason: str,
        predicted_action: int | None = None,
    ) -> None:
        """
        Record the outcome of one decision in the reason histogram
        and, when tracing is on, in the per-decision trace.
        """

        self.reasons[reason] = self.reasons.get(reason, 0) + 1

        if not self.trace:
            return

        self.records.append(
            AdviceRecord(
                episode=self.episode,
                step=self.step,
                importance=importance,
                asked=asked,
                delivered=delivered,
                student_action=student_action,
                teacher_action=teacher_action,
                predicted_action=predicted_action,
                reason=reason,
            )
        )

    def stats(self) -> dict:
        """
        Summary of how the budget was spent, for end-of-run
        reporting and for the per-seed JSON the controllers write.

        The two rates are the headline numbers: `query_rate` is the
        fraction of student steps that cost money, and
        `delivery_rate` is the fraction that reached the student.
        `waste_rate` -- deliveries withheld after payment -- is the
        price of doing mistake correcting with a teacher that
        cannot be consulted for free.
        """

        steps = max(1, self.num_steps)
        asked = max(1, self.num_asked)
        delivered = max(1, self.num_delivered)

        # A normalized importance function tracks the raw range it
        # observed. Recording it means a run's threshold can be
        # translated back into the teacher's own units afterwards,
        # which is what makes thresholds comparable across runs.
        importance_stats = {}
        if hasattr(self.importance_fn, 'stats'):
            importance_stats = self.importance_fn.stats()

        return {
            'advisor': self.name,
            'advice_budget': self.advice_budget,
            'query_budget': self.query_budget,
            'threshold': self.threshold,
            'target_rate': self.target_rate,
            # True when the budget ran out and took over from the
            # threshold as the gate deciding everything. A run with
            # this set is early advising however it is labelled.
            'budget_bound_first': self.budget_bound_first,
            # True when the hand-off schedule showed the advisor too
            # few steps to spend at the requested rate. The run is
            # then bounded by the schedule and says nothing about
            # the budgeting strategy.
            'rate_unreachable': self.rate_unreachable,
            # Where the controller ended up and how far it moved.
            # A threshold pinned at 0 or 1 means the target was
            # unreachable -- the signal could not separate states
            # finely enough, or another gate was the real
            # constraint.
            'threshold_final': self.threshold,
            'threshold_min': (
                min(self.threshold_history)
                if self.threshold_history
                else None
            ),
            'threshold_max': (
                max(self.threshold_history)
                if self.threshold_history
                else None
            ),
            'min_confidence': self.min_confidence,
            'num_steps': self.num_steps,
            'num_asked': self.num_asked,
            'num_delivered': self.num_delivered,
            'num_withheld': self.num_withheld,
            'num_free': self.num_free,
            'num_env_steps': self.num_env_steps,
            # Rates per step the advisor was CONSULTED on. Useful for
            # reading the gates, but not a cost: the hand-off
            # schedule sets this denominator.
            'query_rate': self.num_asked / steps,
            'delivery_rate': self.num_delivered / steps,
            # Rates per ENVIRONMENT step, which is what actually
            # compares across runs. None when the training loop
            # never reported env steps.
            'query_rate_env': (
                self.num_asked / self.num_env_steps
                if self.num_env_steps
                else None
            ),
            'delivery_rate_env': (
                self.num_delivered / self.num_env_steps
                if self.num_env_steps
                else None
            ),
            # What fraction of training the advisor was even allowed
            # to see. A low value means the hand-off schedule, not
            # the budgeting strategy, is deciding most of the run.
            'visible_fraction': (
                self.num_steps / self.num_env_steps
                if self.num_env_steps
                else None
            ),
            # Mean within-episode step index the advisor acted at.
            # Under a prefix hand-off this sits near the start,
            # which bounds what an importance claim can say: the
            # strategy chose among opening states, never the
            # endgame.
            'mean_ep_step_seen': self._ep_step_sum / steps,
            'pacing': (
                'budget'
                if self.pace_to_budget
                else ('rate' if self.target_rate is not None
                      else 'fixed')
            ),
            'waste_rate': self.num_withheld / asked,
            'mean_importance_seen': self._importance_sum / steps,
            'mean_importance_spent': (
                self._importance_spent_sum / delivered
            ),
            **importance_stats,
            'reasons': dict(self.reasons),
            'cost': {
                'wall_time_s': self.cost_total.wall_time_s,
                'dollars': self.cost_total.dollars,
                'tokens_in': self.cost_total.tokens_in,
                'tokens_out': self.cost_total.tokens_out,
                'compute_units': self.cost_total.compute_units,
                'model': self.cost_total.model,
            },
        }


class UnlimitedAdvisor(BaseAdvisor):
    """
    Degenerate advisor that consults the teacher at every step.

    This is the behaviour the controllers had before this package
    existed, kept as an explicit strategy so that "no budgeting" is
    a point in the same design space as the budgeted strategies
    rather than a separate code path. Pair it with a finite
    `advice_budget` and it becomes early advising; the dedicated
    `EarlyAdvisor` subclass exists mainly for naming.
    """

    def __init__(self, **kwargs):
        """
        Initialize with the strategy name fixed to 'unlimited'.
        """

        kwargs.setdefault('name', 'unlimited')
        super().__init__(**kwargs)

    def _gate_ask(
        self,
        state: Any,
        student_action: int | None,
        importance: float,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Always ask; the budget checks in the base class are the only
        thing that can stop this strategy.
        """

        return True, 'ask'
