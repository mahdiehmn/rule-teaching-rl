"""
Tests for the advice-budgeting package.

Runs under pytest, and also standalone (`py -m tests.test_advising`)
for environments where pytest is not installed.

The tests are built around two fake teachers that differ in exactly
one respect: whether their action can be looked up for free. That is
the single axis along which our setting departs from Torrey & Taylor
(2013), so it is the axis the tests exercise hardest. Several
assertions below encode claims made in the paper -- notably the
Section 3.2 equivalence between importance advising at t=0 and early
advising -- so a regression in the budgeting logic shows up as a
failed replication rather than as a silently different experiment.
"""

import sys
from pathlib import Path

import numpy as np

# Allow running this file directly from a checkout that has not been
# pip-installed, by putting the repo root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advising import make_advisor  # noqa: E402
from advising.base import UnlimitedAdvisor  # noqa: E402
from advising.importance import (  # noqa: E402
    Normalizer,
    constant_importance,
    student_q_importance,
)
from advising.predictor import (  # noqa: E402
    CountPredictor,
    LinearPredictor,
)
from advising.strategies import (  # noqa: E402
    EarlyAdvisor,
    ImportanceAdvisor,
    MistakeCorrectingAdvisor,
    PredictiveAdvisor,
    SurrogateAdvisor,
)
from teachers.base import Advice, BaseTeacher, Cost  # noqa: E402

# A fixed policy over five integer states, used by both fake
# teachers so their advice is identical and only their cost model
# differs.
POLICY = {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}


class PaidTeacher(BaseTeacher):
    """
    Stand-in for an LLM teacher: answers correctly, charges for every
    answer, and offers no free lookup.

    Deliberately does NOT define `peek_action`, which is what forces
    the strategies onto their deliver-gate code paths.
    """

    # Its answer is POLICY[state] and nothing else, so recall is
    # sound.
    is_stateless = True

    def __init__(self, teacher_id='paid', seed=0):
        """
        Initialize the fake paid teacher.
        """

        super().__init__(teacher_id=teacher_id, seed=seed)
        self.num_calls = 0

    def recommend(self, state, context=None):
        """
        Return the policy action for `state`, billed as an API call.
        """

        self.num_calls += 1
        return Advice(
            action=POLICY[int(state)],
            confidence=1.0,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=0.1,
                dollars=0.01,
                tokens_in=100,
                tokens_out=10,
                model='fake-model',
            ),
        )


class FreeTeacher(BaseTeacher):
    """
    Stand-in for the value-iteration teacher: same answers, no
    monetary cost, and a free `peek_action` lookup.
    """

    # Same as PaidTeacher: a pure function of the state.
    is_stateless = True

    def __init__(self, teacher_id='free', seed=0):
        """
        Initialize the fake free teacher.
        """

        super().__init__(teacher_id=teacher_id, seed=seed)
        self.num_calls = 0
        self.num_peeks = 0

    def recommend(self, state, context=None):
        """
        Return the policy action for `state` at zero dollar cost.
        """

        self.num_calls += 1
        return Advice(
            action=POLICY[int(state)],
            confidence=1.0,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(wall_time_s=0.0001, compute_units=1),
        )

    def peek_action(self, state):
        """
        Free table lookup of the same policy.
        """

        self.num_peeks += 1
        return POLICY[int(state)]


def run_episode(advisor, teacher, student_actions):
    """
    Drive an advisor through one synthetic episode.

    Walks states 0..4 in order, handing the advisor the student's
    intended action from `student_actions`, and returns the list of
    (advice, cost) pairs. Mirrors what the real controllers do per
    step, minus the environment.
    """

    results = []
    for step, (state, student_action) in enumerate(
        zip(range(5), student_actions)
    ):
        advisor.note_step(step)
        # Student-side importance needs a Q-row; give every state a
        # spread of 1.0 so importance is constant and only the
        # threshold logic is under test.
        context = {'q_values': np.array([0.0, 1.0, 0.5])}
        advice, cost = advisor.advise(
            teacher, state, student_action, context
        )
        advisor.observe(state, student_action, advice)
        results.append((advice, cost))
    return results


# -- Budget accounting ------------------------------------------


def test_unlimited_advisor_asks_every_step():
    """
    The default strategy must reproduce the pre-existing behaviour:
    consult the teacher at every step.
    """

    teacher = PaidTeacher()
    advisor = UnlimitedAdvisor()
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    assert advisor.num_asked == 5
    assert advisor.num_delivered == 5
    assert teacher.num_calls == 5


def test_advice_budget_caps_deliveries():
    """
    Early advising stops after `n` pieces of advice, and stops
    querying too -- paying for advice that can never be delivered
    would be pure waste.
    """

    teacher = PaidTeacher()
    advisor = EarlyAdvisor(advice_budget=3)
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    assert advisor.num_delivered == 3
    assert advisor.num_asked == 3
    assert teacher.num_calls == 3
    assert advisor.reasons['no_advice_budget'] == 2


def test_query_budget_caps_asks_independently():
    """
    The query budget binds separately from the advice budget. This
    is the distinction that does not exist in the 2013 framework.
    """

    teacher = PaidTeacher()
    advisor = EarlyAdvisor(advice_budget=0, query_budget=2)
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    assert advisor.num_asked == 2
    assert teacher.num_calls == 2
    assert advisor.reasons['no_query_budget'] == 3


def test_zero_budget_means_unlimited():
    """
    A budget of 0 means unlimited, matching the convention the
    controllers already use for --teacher-until-episode.
    """

    teacher = PaidTeacher()
    advisor = EarlyAdvisor(advice_budget=0, query_budget=0)
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    assert advisor.num_asked == 5
    assert advisor.advice_remaining == float('inf')
    assert advisor.query_remaining == float('inf')


# -- Importance -------------------------------------------------


def test_importance_at_threshold_zero_equals_early():
    """
    Torrey & Taylor, Section 3.2: "When t is 0, this becomes
    equivalent to early advising, assuming importance values are
    non-negative." Both arms must make identical decisions.
    """

    early = EarlyAdvisor(advice_budget=3)
    importance = ImportanceAdvisor(
        advice_budget=3,
        threshold=0.0,
        importance_fn=student_q_importance,
    )

    run_episode(early, PaidTeacher(), [0, 0, 0, 0, 0])
    run_episode(importance, PaidTeacher(), [0, 0, 0, 0, 0])

    assert early.num_asked == importance.num_asked
    assert early.num_delivered == importance.num_delivered


def test_importance_threshold_filters_states():
    """
    With a threshold above every state's importance, nothing is
    bought at all.
    """

    teacher = PaidTeacher()
    advisor = ImportanceAdvisor(
        threshold=99.0, importance_fn=student_q_importance
    )
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    assert advisor.num_asked == 0
    assert teacher.num_calls == 0
    assert advisor.reasons['below_threshold'] == 5


def test_student_q_importance_is_the_q_spread():
    """
    Student-side importance is max_a Q - min_a Q, and degrades to
    zero rather than raising when no Q-values are supplied.
    """

    q = {'q_values': np.array([-3.0, 1.0, 0.0])}
    assert student_q_importance(None, q) == 4.0
    assert student_q_importance(None, {}) == 0.0
    assert constant_importance(None, q) == 0.0


def test_normalizer_rescales_after_warmup():
    """
    The normalizer passes everything during warmup, then maps raw
    values into [0, 1] so a threshold means the same thing across
    teachers with different I(s) ranges.
    """

    raw_values = [10.0, 20.0, 30.0]
    idx = {'i': 0}

    def fn(state, context):
        """
        Cycle through a fixed list of raw importance values.
        """

        value = raw_values[idx['i'] % len(raw_values)]
        idx['i'] += 1
        return value

    normalizer = Normalizer(fn, warmup=2)

    # During warmup every state reports as maximally important.
    assert normalizer(None, {}) == 1.0
    assert normalizer(None, {}) == 1.0

    # Third call: raw 30 is the top of the observed range 10..30.
    assert normalizer(None, {}) == 1.0
    # Fourth call: raw 10 is the bottom of that range.
    assert normalizer(None, {}) == 0.0

    stats = normalizer.stats()
    assert stats['importance_raw_min'] == 10.0
    assert stats['importance_raw_max'] == 30.0


# -- Mistake correcting -----------------------------------------


def test_mistake_correcting_free_teacher_spends_only_on_mistakes():
    """
    With a free-lookup teacher the mistake check runs in the ask
    gate, so no query is spent where the student already agrees.
    This is the paper's behaviour exactly.
    """

    teacher = FreeTeacher()
    advisor = MistakeCorrectingAdvisor()

    # The student agrees with the teacher at states 0 and 1 and errs
    # at 2, 3, 4.
    student = [POLICY[0], POLICY[1], 0, 0, 2]
    run_episode(advisor, teacher, student)

    assert advisor.num_asked == 3
    assert advisor.num_delivered == 3
    assert advisor.num_withheld == 0
    assert teacher.num_calls == 3
    assert advisor.stats()['waste_rate'] == 0.0


def test_mistake_correcting_paid_teacher_pays_for_every_check():
    """
    With a paid teacher the same strategy cannot avoid the query --
    it only avoids interrupting the student. Money is spent at every
    state; attention only at the mistakes.

    This asymmetry is the central adaptation in this package, so it
    is asserted rather than assumed.
    """

    teacher = PaidTeacher()
    advisor = MistakeCorrectingAdvisor()

    student = [POLICY[0], POLICY[1], 0, 0, 2]
    run_episode(advisor, teacher, student)

    assert advisor.num_asked == 5
    assert advisor.num_delivered == 3
    assert advisor.num_withheld == 2
    assert teacher.num_calls == 5
    assert advisor.stats()['waste_rate'] == 2 / 5


def test_withheld_advice_still_reports_its_cost():
    """
    A query that gets discarded still appears on the bill. Reporting
    it any other way would understate the price of mistake
    correcting with a paid teacher.
    """

    teacher = PaidTeacher()
    advisor = MistakeCorrectingAdvisor()

    # The student agrees everywhere, so every piece of advice is
    # withheld -- and every query is still charged.
    student = [POLICY[s] for s in range(5)]
    results = run_episode(advisor, teacher, student)

    assert advisor.num_delivered == 0
    assert advisor.num_asked == 5
    assert all(advice is None for advice, _ in results)
    assert all(cost.dollars == 0.01 for _, cost in results)
    assert advisor.cost_total.dollars == 0.05


# -- Predictors -------------------------------------------------


def test_count_predictor_recovers_a_fixed_policy():
    """
    The count predictor should learn a deterministic per-state
    policy, and abstain on states it has never seen.
    """

    predictor = CountPredictor(num_actions=3)
    for _ in range(5):
        for state, action in POLICY.items():
            predictor.observe(state, action)

    for state, action in POLICY.items():
        assert predictor.predict(state) == action
        assert predictor.confidence(state) > 0.9

    # Abstention on an unvisited state is what lets the surrogate
    # strategy know when it must pay for a real query.
    assert predictor.predict(99) is None
    assert predictor.confidence(99) == 0.0


def test_count_predictor_tracks_a_changing_policy():
    """
    Recency weighting must let the predictor follow a student that
    changes its mind -- the non-stationarity the paper flags as the
    core difficulty of action prediction.
    """

    predictor = CountPredictor(num_actions=3, decay=0.5)
    for _ in range(10):
        predictor.observe(0, 1)
    assert predictor.predict(0) == 1

    for _ in range(6):
        predictor.observe(0, 2)
    assert predictor.predict(0) == 2


def test_linear_predictor_recovers_a_separable_policy():
    """
    The linear predictor should learn a policy that is linearly
    separable in the state features.
    """

    predictor = LinearPredictor(num_actions=2, lr=1.0, epochs=200)
    rng = np.random.default_rng(0)

    # Action 1 when the first feature is positive, else action 0.
    for _ in range(200):
        x = rng.normal(size=2)
        predictor.observe(x, 1 if x[0] > 0 else 0)
    predictor.fit()

    assert predictor.predict(np.array([2.0, 0.0])) == 1
    assert predictor.predict(np.array([-2.0, 0.0])) == 0

    # Before any fit, the predictor abstains rather than guessing.
    assert LinearPredictor(num_actions=2).predict([0.0, 0.0]) is None


# -- Predictive and surrogate -----------------------------------


def test_predictive_advising_skips_predicted_agreement():
    """
    Once the predictor has learned the student's policy, predictive
    advising with a free-lookup teacher stops querying at states
    where the student is predicted to agree -- without ever reading
    the student's actual intent.
    """

    teacher = FreeTeacher()
    advisor = PredictiveAdvisor(num_actions=3)

    # Train the predictor on a student that matches the teacher
    # everywhere, then check that a whole episode costs nothing.
    student = [POLICY[s] for s in range(5)]
    for episode in range(4):
        run_episode(advisor, teacher, student)
        advisor.note_episode_end(episode)

    calls_before = teacher.num_calls
    run_episode(advisor, teacher, student)

    assert teacher.num_calls == calls_before
    assert advisor.predictor.accuracy > 0.9


def test_surrogate_stops_paying_once_it_has_learned():
    """
    The surrogate pays for each state once, then covers it from its
    own model. This is the mechanism that reduces API spend rather
    than merely student attention.
    """

    teacher = PaidTeacher()
    advisor = SurrogateAdvisor(num_actions=3, trust=0.8)

    # A student that is wrong everywhere, so advice is always
    # warranted and only the source of it changes.
    student = [(POLICY[s] + 1) % 3 for s in range(5)]

    run_episode(advisor, teacher, student)
    assert advisor.num_asked == 5
    assert advisor.num_free == 0

    advisor.note_episode_end(0)
    run_episode(advisor, teacher, student)

    # Second pass: every state is covered by the surrogate, so no
    # further queries are sent, yet the student keeps receiving
    # advice.
    assert advisor.num_asked == 5
    assert advisor.num_free == 5
    assert advisor.num_delivered == 10
    assert teacher.num_calls == 5
    assert advisor.cost_total.dollars == 0.05


def test_surrogate_without_free_advice_still_saves_on_agreement():
    """
    With `use_surrogate_advice=False` the surrogate never speaks for
    the teacher; it only skips states where it predicts the student
    already agrees. Safer, and still cheaper than asking every step.
    """

    teacher = PaidTeacher()
    advisor = SurrogateAdvisor(
        num_actions=3, trust=0.8, use_surrogate_advice=False
    )

    agreeing = [POLICY[s] for s in range(5)]
    run_episode(advisor, teacher, agreeing)
    advisor.note_episode_end(0)
    asked_after_first = advisor.num_asked

    run_episode(advisor, teacher, agreeing)

    assert advisor.num_asked == asked_after_first
    assert advisor.num_free == 0
    assert advisor.reasons['surrogate_agrees'] == 5


# -- Factory ----------------------------------------------------


def test_surrogate_never_serves_a_stale_prediction():
    """
    Regression: `_free_advice` used to read a remembered prediction
    set by the ask gate, but the ask gate does not always run --
    the base class short-circuits on an exhausted query budget
    before consulting the strategy at all. The student then received
    the PREVIOUS state's guess as advice for this one.

    Here the surrogate learns state 0 only, then the query budget is
    spent. Visiting an unknown state must produce no advice rather
    than state 0's answer.
    """

    teacher = PaidTeacher()
    advisor = SurrogateAdvisor(
        num_actions=3, trust=0.8, query_budget=1
    )

    # One real query at state 0 teaches the surrogate that answer.
    advice, _ = advisor.advise(teacher, 0, (POLICY[0] + 1) % 3, {})
    assert advice is not None
    assert advisor.query_remaining <= 0

    # State 4 is unknown and the query budget is gone. The surrogate
    # has nothing to say, so nothing must be delivered.
    advice, cost = advisor.advise(teacher, 4, 0, {})
    assert advice is None
    assert cost.dollars == 0.0

    # Revisiting the known state IS legitimately free advice, which
    # is the behaviour the fix has to preserve.
    advice, cost = advisor.advise(
        teacher, 0, (POLICY[0] + 1) % 3, {}
    )
    assert advice is not None
    assert advice.action == POLICY[0]
    assert cost.dollars == 0.0


def test_pacing_stops_once_the_budget_binds():
    """
    Regression: with the budget exhausted, deliveries stop for a
    reason the threshold cannot influence. A controller that kept
    reacting read the zero rate as "far too strict" and drove the
    threshold to the floor -- leaving a run that reports itself as
    importance advising at threshold 0, which is early advising.
    """

    import random

    rng = random.Random(2)
    advisor = ImportanceAdvisor(
        importance_fn=lambda s, c: rng.random(),
        target_rate=0.2,
        pace_interval=25,
        advice_budget=20,
    )

    teacher = PaidTeacher()
    for step in range(2000):
        advisor.note_step(step)
        advisor.advise(teacher, step % 5, 0, {})

    stats = advisor.stats()
    assert advisor.num_delivered == 20
    # The run must say plainly that exhaustion, not importance, was
    # deciding -- and the threshold must not have been wound down to
    # the floor chasing a rate it could no longer affect.
    assert stats['budget_bound_first'] is True
    assert stats['threshold_final'] > 0.0


def test_quantile_normalizer_reports_true_percentiles():
    """
    Under quantile normalization a threshold means what it says:
    t=0.8 admits the top 20% of recent states. Min-max
    normalization does not have that property once the raw values
    are unevenly spread.
    """

    from advising.importance import QuantileNormalizer

    values = list(range(100))
    idx = {'i': 0}

    def fn(state, context):
        """
        Walk a uniform 0..99 sequence, then repeat it.
        """

        value = float(values[idx['i'] % len(values)])
        idx['i'] += 1
        return value

    normalizer = QuantileNormalizer(fn, window=100, warmup=100)

    # Burn through the warmup, filling the window.
    for _ in range(100):
        normalizer(None, {})

    # The window now holds 0..99. The next values repeat that
    # sequence, so each one's quantile is its own rank. Mid-rank
    # splits the single tied value, so the smallest lands just
    # above 0 rather than exactly at it.
    assert normalizer(None, {}) < 0.02  # raw 0 beats nothing
    quantiles = [normalizer(None, {}) for _ in range(49)]
    # Raw value 50 should sit near the middle of the window.
    assert 0.45 < quantiles[-1] < 0.55


def test_quantile_normalizer_survives_a_constant_signal():
    """
    Regression, and a nasty one: counting only values strictly below
    the current one reports EVERY value of a constant signal at
    quantile 0.0, so no threshold above zero ever passes anything.

    This is not a corner case. Early in training a policy is close
    to uniform, so normalized entropy is 1.0 at nearly every state
    -- the most important states in the run scoring as the least
    important. It showed up as an advisor delivering exactly its
    warmup count and then nothing for the rest of a run.
    """

    from advising.importance import QuantileNormalizer

    normalizer = QuantileNormalizer(
        lambda s, c: 1.0, window=50, warmup=10
    )
    for _ in range(10):
        normalizer(None, {})

    # A constant signal cannot be ranked, so the honest answer is
    # the middle -- which leaves a threshold on either side of 0.5
    # behaving sensibly instead of rejecting everything.
    assert normalizer(None, {}) == 0.5
    assert normalizer(None, {}) == 0.5


def test_minmax_normalizer_is_skewed_by_an_outlier():
    """
    Why quantile is the better default.

    One freak value far above the rest compresses everything else
    toward zero under min-max, so a modest threshold silently
    rejects nearly all states. The quantile normalizer is unmoved.
    """

    from advising.importance import Normalizer, QuantileNormalizer

    # Ninety-nine ordinary values then one enormous one, repeated.
    seq = [1.0] * 99 + [1000.0]
    idx = {'i': 0}

    def fn(state, context):
        """
        Emit mostly small values with a single huge outlier.
        """

        value = seq[idx['i'] % len(seq)]
        idx['i'] += 1
        return value

    minmax = Normalizer(fn, warmup=0)
    idx['i'] = 0
    for _ in range(100):
        minmax(None, {})
    # A typical value now scores essentially zero, so a threshold of
    # 0.3 would reject it despite it being perfectly ordinary.
    assert minmax(None, {}) < 0.01

    idx['i'] = 0
    quantile = QuantileNormalizer(fn, window=100, warmup=0)
    for _ in range(100):
        quantile(None, {})
    # The same ordinary value is not pushed to the floor.
    assert quantile(None, {}) >= 0.0


def test_target_rate_controls_spending():
    """
    The headline behaviour of budget-derived thresholds: ask for a
    spend rate and get roughly that rate, without choosing a
    threshold at all.
    """

    import random

    rng = random.Random(0)

    def noisy_importance(state, context):
        """
        A stationary but noisy importance signal.
        """

        return rng.random()

    advisor = ImportanceAdvisor(
        importance_fn=noisy_importance,
        target_rate=0.2,
        pace_interval=50,
    )

    teacher = PaidTeacher()
    for step in range(4000):
        advisor.note_step(step)
        advisor.advise(teacher, step % 5, 0, {})

    achieved = advisor.num_delivered / advisor.num_steps
    assert 0.12 < achieved < 0.28, achieved

    # The threshold was found by the controller, not supplied.
    stats = advisor.stats()
    assert stats['target_rate'] == 0.2
    assert stats['threshold_final'] != 0.0


def test_target_rate_spreads_spending_across_the_run():
    """
    A fixed threshold with a finite budget front-loads: everything
    is spent early and the strategy silently becomes early
    advising. A paced threshold should still have budget left in
    the second half of the run.
    """

    import random

    rng = random.Random(1)

    def drifting_importance(state, context):
        """
        Importance that starts high everywhere and falls, imitating
        a policy that begins uncertain and grows confident.
        """

        return rng.random() * drifting_importance.scale

    drifting_importance.scale = 1.0

    paced = ImportanceAdvisor(
        importance_fn=drifting_importance,
        target_rate=0.1,
        pace_interval=50,
        advice_budget=200,
    )

    teacher = PaidTeacher()
    first_half = 0
    for step in range(2000):
        paced.note_step(step)
        # Halfway through, the signal collapses toward zero.
        if step == 1000:
            drifting_importance.scale = 0.1
            first_half = paced.num_delivered
        paced.advise(teacher, step % 5, 0, {})

    # Some budget survived the first half rather than all of it
    # being consumed before the distribution shifted.
    assert first_half < 200
    assert paced.num_delivered > first_half


def test_peekable_halves_queries_without_changing_advice():
    """
    The claim the PeekableTeacher exists to make.

    Mistake correcting against a teacher with no free lookup must
    pay for a query at every state, because the only way to learn
    what the teacher would say is to ask. Wrapping that same teacher
    so previously bought answers can be replayed must cut the
    queries WITHOUT changing a single piece of advice the student
    receives -- the set of states where the student errs does not
    depend on when the check happens.

    Asserting both halves together is the point: a saving that
    changed what the student saw would not be a saving at all.
    """

    from advising.peekable import PeekableTeacher

    # A student that is wrong at states 0 and 1 and right elsewhere.
    student = [1, 0, POLICY[2], POLICY[3], POLICY[4]]

    bare_teacher = PaidTeacher()
    bare = MistakeCorrectingAdvisor()
    for _ in range(3):
        run_episode(bare, bare_teacher, student)

    wrapped_teacher = PeekableTeacher(PaidTeacher(), num_actions=3)
    wrapped = MistakeCorrectingAdvisor()
    for _ in range(3):
        run_episode(wrapped, wrapped_teacher, student)

    # Identical teaching: same advice, same states.
    assert bare.num_delivered == wrapped.num_delivered == 6

    # The unwrapped teacher is asked at all 15 states across the
    # three passes; the wrapped one only until it has seen each
    # state once.
    assert bare.num_asked == 15
    assert wrapped.num_asked < bare.num_asked
    assert wrapped.cost_total.dollars < bare.cost_total.dollars


def test_peekable_recall_is_exact_not_approximate():
    """
    A peek answered from recall must return what the teacher
    actually said, and a peek at an unvisited state must return None
    rather than a guess -- admitting ignorance is what makes the
    strategies fall back to paying.
    """

    from advising.peekable import PeekableTeacher

    teacher = PeekableTeacher(
        PaidTeacher(), num_actions=3, exact_only=True
    )

    # Nothing known yet.
    assert teacher.peek_action(0) is None

    # After a real query, recall is exact and free.
    advice = teacher.recommend(0)
    assert teacher.peek_action(0) == advice.action == POLICY[0]
    assert teacher.peek_action(1) is None

    stats = teacher.stats()
    assert stats['num_peek_exact'] == 1
    assert stats['num_states_known'] == 1


def test_peekable_delegates_to_a_free_teacher():
    """
    Wrapping a teacher that can already peek must not throw that
    away -- delegation wins, and is counted separately so a run can
    attribute where its free answers came from.
    """

    from advising.peekable import PeekableTeacher

    teacher = PeekableTeacher(FreeTeacher(), num_actions=3)

    # Answered immediately, with no prior query, via delegation.
    assert teacher.peek_action(3) == POLICY[3]
    assert teacher.stats()['num_peek_delegated'] == 1
    assert teacher.stats()['num_peek_exact'] == 0


def test_peekable_refuses_a_stateful_teacher():
    """
    Recall replays the answer recorded at a state, which is only
    sound if the same state always earns the same answer. A teacher
    holding a plan or a sub-goal across steps breaks that, and the
    failure would be silent: a large query saving that also quietly
    changes the advice. The wrapper must refuse to be built.

    This is the same refusal pattern the predictor strategies use
    for the stateful BabyAI bot.
    """

    class StatefulTeacher(PaidTeacher):
        """
        A teacher whose answer depends on a carried sub-goal, like
        llm_subgoal -- the same state legitimately earns different
        answers in different episodes.
        """

        is_stateless = False

    from advising.peekable import PeekableTeacher

    try:
        PeekableTeacher(StatefulTeacher(), num_actions=3)
    except ValueError as exc:
        assert 'is_stateless' in str(exc)
    else:
        raise AssertionError(
            'expected PeekableTeacher to refuse a stateful teacher'
        )

    # The blind ablation carries no recall, so it stays available
    # even for a stateful teacher.
    PeekableTeacher(
        StatefulTeacher(),
        num_actions=3,
        exact_only=True,
        recall=False,
        use_inner_peek=False,
    )


def test_peekable_blind_mode_answers_nothing():
    """
    The ablation setting: recall off and prediction off leaves a
    wrapper that can never answer, turning a free teacher into a
    stand-in for a paid one so the arms can be compared without
    spending on real API calls.
    """

    from advising.peekable import PeekableTeacher

    teacher = PeekableTeacher(
        FreeTeacher(),
        num_actions=3,
        exact_only=True,
        recall=False,
        use_inner_peek=False,
    )

    teacher.recommend(0)
    assert teacher.peek_action(0) is None
    assert teacher.stats()['num_peek_missed'] == 1


def test_factory_builds_every_strategy():
    """
    Every advertised strategy name must build and run, so a CLI
    choice can never reference a strategy that does not work.
    """

    from advising import ADVISOR_NAMES

    for name in ADVISOR_NAMES:
        advisor = make_advisor(
            name,
            num_actions=3,
            advice_budget=2,
            importance_source='student_q',
        )
        run_episode(advisor, PaidTeacher(), [0, 0, 0, 0, 0])
        stats = advisor.stats()
        assert stats['advisor'] == name
        assert stats['num_delivered'] <= 2


def test_factory_rejects_teacher_q_without_a_value_function():
    """
    Asking for teacher-side importance from a teacher that has no
    Q-function must fail loudly at setup, not silently report zero
    importance for an entire run.
    """

    try:
        make_advisor(
            'importance',
            teacher=PaidTeacher(),
            importance_source='teacher_q',
        )
    except TypeError as exc:
        assert 'q_row' in str(exc)
    else:
        raise AssertionError('expected a TypeError')


def test_teacher_q_importance_uses_the_teachers_spread():
    """
    A teacher that does expose a Q-function drives the paper's exact
    importance measure.
    """

    class QTeacher(FreeTeacher):
        """
        Free teacher that also publishes a Q-row per state.
        """

        def q_row(self, state):
            """
            Wide spread at state 0, flat everywhere else.
            """

            if int(state) == 0:
                return np.array([0.0, 10.0, 5.0])
            return np.array([1.0, 1.0, 1.0])

    advisor = make_advisor(
        'importance',
        teacher=QTeacher(),
        importance_source='teacher_q',
        normalize_importance=False,
        threshold=5.0,
    )

    teacher = QTeacher()
    run_episode(advisor, teacher, [0, 0, 0, 0, 0])

    # Only state 0 clears a threshold of 5.
    assert advisor.num_asked == 1
    assert advisor.reasons['below_threshold'] == 4


def _main():
    """
    Run every test function in this module and report the results,
    so the suite works without pytest installed.
    """

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]

    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f'FAIL {name}: {exc}')
        else:
            print(f'ok   {name}')

    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
