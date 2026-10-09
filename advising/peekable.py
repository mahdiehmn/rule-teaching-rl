"""
Making an expensive teacher answerable for free.

Torrey & Taylor's two strongest algorithms -- mistake correcting and
predictive advising -- both need to know what the teacher would say
BEFORE deciding whether to spend budget. Their teacher is a Q-table,
so that lookup is free and the requirement is invisible. An LLM
teacher cannot answer without being paid, and the consequence is
severe: both algorithms collapse into plain importance advising,
querying at every state that clears the importance threshold and
using the teacher's answer only to decide whether to bother the
student with it. The money is spent either way.

`PeekableTeacher` fixes that at the source rather than in each
strategy. It wraps any teacher and adds the free `peek_action(state)`
lookup the strategies probe for, answering from two sources:

1. **Exact recall.** Every answer that passes through `recommend` is
   recorded against its state. A second visit to the same state is
   answered from that record at zero cost. This is not an
   approximation -- it is the teacher's own answer, replayed -- and
   in a gridworld, where the agent revisits the same handful of
   states thousands of times, it is where most of the saving comes
   from.

2. **Prediction.** A small model trained on the answers bought so
   far (see `advising/predictor.py`) covers states never visited
   before, when it is confident enough. This one IS an
   approximation, and a wrong guess costs either a missed teaching
   opportunity or a wasted query. `trust` sets how sure the model
   must be; `exact_only=True` turns it off entirely and keeps the
   guaranteed-correct half.

Why a wrapper and not a strategy
--------------------------------
The obvious alternative is a strategy that predicts the teacher and
skips queries itself -- which is what `SurrogateAdvisor` does. That
placement is a mistake: it makes prediction compete with importance
advising and mistake correcting for the single `--advisor` slot, when
what you actually want is to combine them. As a wrapper, the saving
composes with every strategy, and the strategies stay literal
transcriptions of the paper. Prefer this over `SurrogateAdvisor`
unless you specifically want the surrogate to *deliver* advice in the
teacher's place.

The honesty of the accounting is preserved either way: a peeked
answer accumulates no Cost, because none was incurred, and
`stats()` reports how many peeks were exact, how many were predicted,
and how many failed -- so the saving can always be attributed.
"""

from typing import Any

from advising.predictor import BasePredictor, make_predictor
from teachers.base import Advice, BaseTeacher


class PeekableTeacher(BaseTeacher):
    """
    Wrap a teacher so its action can be looked up for free.

    Behaves exactly like the teacher it wraps -- `recommend` is
    delegated unchanged and its Advice, cost and all, is returned
    untouched -- but additionally offers `peek_action`, which the
    advice-budgeting strategies use to decide whether a query is
    worth paying for.

    Attributes
    ----------
    teacher: BaseTeacher
        The wrapped teacher. All real queries go to it.
    key_fn: callable
        Maps a state to a hashable key for the exact-recall table.
        Pass the student's own discretizer when there is one.
    predictor: BasePredictor | None
        Model of the teacher, trained on answers already bought.
        None when `exact_only` is set.
    trust: float
        Confidence the predictor needs before its guess is used.
    num_exact, num_predicted, num_misses: int
        How each peek was answered. Their ratio is the attribution
        of any saving this wrapper produced.
    """

    def __init__(
        self,
        teacher: BaseTeacher,
        num_actions: int = 3,
        key_fn=None,
        predictor: BasePredictor | None = None,
        predictor_kind: str = 'count',
        trust: float = 0.9,
        exact_only: bool = False,
        recall: bool = True,
        use_inner_peek: bool = True,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        teacher: BaseTeacher
            The teacher to wrap.
        num_actions: int
            Action-space size, used to build the predictor.
        key_fn: callable | None
            State-to-key function for exact recall. Defaults to the
            predictor module's generic key builder.
        predictor: BasePredictor | None
            Pre-built model of the teacher; built from
            `predictor_kind` when None.
        predictor_kind: str
            'count' or 'linear'.
        trust: float
            Minimum predictor confidence before a guess is used.
            Higher than the surrogate strategy's default because a
            wrong peek here silently changes what the paper's
            algorithms do, rather than being visible as advice.
        exact_only: bool
            Answer peeks only from states actually queried before.
            Slower to pay off, but never wrong.
        recall: bool
            Whether to answer peeks from previously bought answers
            at all. Setting this False together with `exact_only`
            produces a wrapper that can never answer a peek -- which
            is useless in production and exactly what an ablation
            wants, because it turns a free teacher into a stand-in
            for a paid one. That makes it possible to measure what
            recall and prediction are worth without spending a cent
            on real API calls.
        use_inner_peek: bool
            Delegate to the wrapped teacher's own `peek_action` when
            it has one. True is correct for library use -- wrapping
            a free teacher should not throw away its free lookup.
            Ablations set it False so every arm faces the same
            teacher.
        seed: int
            Unused; accepted for signature consistency.
        """

        super().__init__(
            teacher_id=f'peekable:{teacher.teacher_id}', seed=seed
        )

        # Replaying a recorded answer is only sound when the same
        # state always earns the same answer. A teacher holding a
        # plan or a sub-goal across steps breaks that, and the
        # failure is silent: the wrapper reports a large query saving
        # while quietly serving advice the live teacher would not
        # have given. Refuse loudly instead, the same way the
        # predictor strategies refuse the stateful bot.
        if (recall or not exact_only) and not getattr(
            teacher, 'is_stateless', False
        ):
            raise ValueError(
                f'{type(teacher).__name__} does not declare '
                f'is_stateless=True, so its answers may depend on '
                f'more than the state (a held plan or sub-goal). '
                f'Replaying them would silently change the advice. '
                f'Set is_stateless=True on the teacher if its '
                f'answer really is a function of the state alone, '
                f'or use recall=False with exact_only=True.'
            )

        self.teacher = teacher
        self.exact_only = exact_only
        self.recall = recall
        self.trust = trust

        # The wrapped teacher's own free lookup, if it advertises
        # one. Resolved once here so `peek_action` stays branch-light
        # on the hot path.
        self._inner_peek = (
            getattr(teacher, 'peek_action', None)
            if use_inner_peek
            else None
        )
        self.num_delegated = 0

        # Import lazily through the predictor module's helper so the
        # default key function stays defined in one place.
        from advising.predictor import default_key

        self.key_fn = key_fn or default_key

        # Exact recall: state key -> the action the teacher actually
        # gave there. Only ever written from a real answer.
        self._answers: dict[Any, int] = {}

        if exact_only:
            self.predictor = None
        else:
            self.predictor = predictor or make_predictor(
                predictor_kind, num_actions, key_fn=self.key_fn
            )

        self.num_exact = 0
        self.num_predicted = 0
        self.num_misses = 0

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Delegate to the wrapped teacher and record its answer.

        The Advice is returned exactly as the wrapped teacher built
        it, cost included -- this wrapper never hides what a query
        cost. The only side effect is that the answer becomes
        available to future peeks.
        """

        advice = self.teacher.recommend(state, context)

        if advice.action is not None:
            key = self.key_fn(state)
            self._answers[key] = int(advice.action)
            if self.predictor is not None:
                self.predictor.observe(state, int(advice.action))

        return advice

    def peek_action(self, state) -> int | None:
        """
        Return what the teacher would say, without paying for it.

        Returns None when neither source can answer, which the
        strategies read as "this teacher cannot be consulted for
        free here" and fall back to paying.

        Returns
        -------
        int | None
            The teacher's action, or None if unknown.
        """

        # If the wrapped teacher can answer for free itself, that is
        # both cheaper and more accurate than anything this wrapper
        # can reconstruct, so it wins outright.
        if self._inner_peek is not None:
            answer = self._inner_peek(state)
            if answer is not None:
                self.num_delegated += 1
                return int(answer)

        # Exact recall next: if we have actually asked here before,
        # that answer is ground truth and costs nothing to reuse.
        key = self.key_fn(state)
        recalled = self._answers.get(key) if self.recall else None
        if recalled is not None:
            self.num_exact += 1
            return recalled

        if self.predictor is None:
            self.num_misses += 1
            return None

        # Otherwise fall back to the learned model, but only where
        # it is confident. An unconfident guess is worse than no
        # guess: it silently redirects budget instead of admitting
        # ignorance.
        guess = self.predictor.predict(state)
        if guess is None:
            self.num_misses += 1
            return None

        confidence_fn = getattr(self.predictor, 'confidence', None)
        if confidence_fn is not None and confidence_fn(state) < (
            self.trust
        ):
            self.num_misses += 1
            return None

        self.num_predicted += 1
        return int(guess)

    def optimal_actions(self, state):
        """
        Forward the wrapped teacher's optimal-action set, if it has
        one.

        Without this the wrapper silently hides the set, and the
        tie-aware mistake test falls back to comparing against a
        single arbitrarily-chosen action -- so turning on `--peek`
        would quietly re-introduce the bug where a student choosing
        an equally optimal move is scored as mistaken.
        """

        fn = getattr(self.teacher, 'optimal_actions', None)
        return None if fn is None else fn(state)

    def q_row(self, state):
        """
        Forward the wrapped teacher's Q-values, if it has any.

        Needed so Torrey & Taylor's teacher-side importance survives
        wrapping; otherwise `--peek` and `--importance-source
        teacher_q` would be mutually exclusive for no reason.
        """

        fn = getattr(self.teacher, 'q_row', None)
        return None if fn is None else fn(state)

    def fit(self) -> None:
        """
        Refit the predictor, called between episodes by the training
        loop via the advisor's `note_episode_end`.
        """

        if self.predictor is not None:
            self.predictor.fit()

    def stats(self) -> dict:
        """
        Attribution of the peeks: how many were answered from real
        recorded answers, how many were guessed, and how many could
        not be answered at all.

        `exact_rate` is the number worth reporting -- it is the part
        of the saving that carries no risk of being wrong.
        """

        answered = (
            self.num_exact + self.num_predicted + self.num_delegated
        )
        total = max(1, answered + self.num_misses)
        out = {
            'peekable_wraps': self.teacher.teacher_id,
            'exact_only': self.exact_only,
            'recall': self.recall,
            'trust': self.trust,
            'num_states_known': len(self._answers),
            'num_peek_exact': self.num_exact,
            'num_peek_predicted': self.num_predicted,
            'num_peek_delegated': self.num_delegated,
            'num_peek_missed': self.num_misses,
            'exact_rate': self.num_exact / total,
            'answered_rate': answered / total,
        }
        if self.predictor is not None:
            out['predictor'] = self.predictor.stats()
        return out
