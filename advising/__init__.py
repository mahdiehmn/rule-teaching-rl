"""
Advice budgeting for teacher-student reinforcement learning.

Public surface of the package. `make_advisor` is the factory the
controllers call; it mirrors the `_build_teacher` / `_build_channel`
helpers those controllers already use, so adding a budgeting strategy
to a run is one more flag rather than a new code path.

The three axes of an experiment in this framework are now:

    teacher  x  advisor  x  channel
    (who)       (when)      (how)

`teachers/` answers who is being consulted and what it costs.
`advising/` answers when it is worth consulting them. `channels/`
answers what happens to the advice once it arrives. All three vary
independently, which is what makes the grid of runs meaningful.

See `advising/base.py` for the two-budget model and why an LLM
teacher needs one, and `advising/strategies.py` for the individual
algorithms and their relationship to Torrey & Taylor (2013).
"""

from advising.base import (
    UNLIMITED,
    AdviceRecord,
    BaseAdvisor,
    UnlimitedAdvisor,
    exact_mistake,
)
from advising.importance import (
    IMPORTANCE_SOURCES,
    Normalizer,
    QuantileNormalizer,
    make_importance,
)
from advising.peekable import PeekableTeacher
from advising.predictor import (
    PREDICTOR_KINDS,
    BasePredictor,
    CountPredictor,
    LinearPredictor,
    make_predictor,
)
from advising.strategies import (
    ADVISOR_NAMES,
    EarlyAdvisor,
    ImportanceAdvisor,
    MistakeCorrectingAdvisor,
    PredictiveAdvisor,
    SurrogateAdvisor,
    peek_action,
)

__all__ = [
    'ADVISOR_NAMES',
    'IMPORTANCE_SOURCES',
    'PREDICTOR_KINDS',
    'UNLIMITED',
    'AdviceRecord',
    'BaseAdvisor',
    'BasePredictor',
    'CountPredictor',
    'EarlyAdvisor',
    'ImportanceAdvisor',
    'LinearPredictor',
    'MistakeCorrectingAdvisor',
    'Normalizer',
    'PeekableTeacher',
    'PredictiveAdvisor',
    'QuantileNormalizer',
    'SurrogateAdvisor',
    'UnlimitedAdvisor',
    'exact_mistake',
    'make_advisor',
    'make_importance',
    'make_predictor',
    'peek_action',
]


# Maps the CLI-facing strategy name to its class. Kept beside the
# factory so adding a strategy means touching one dict and one
# tuple, not hunting through if-chains.
_ADVISOR_CLASSES = {
    'unlimited': UnlimitedAdvisor,
    'early': EarlyAdvisor,
    'importance': ImportanceAdvisor,
    'mistake': MistakeCorrectingAdvisor,
    'predictive': PredictiveAdvisor,
    'surrogate': SurrogateAdvisor,
}


def make_advisor(
    name: str,
    teacher=None,
    num_actions: int = 3,
    advice_budget: int = UNLIMITED,
    query_budget: int = UNLIMITED,
    threshold: float = 0.0,
    importance_source: str = 'none',
    importance_fn=None,
    normalize_importance: bool = True,
    importance_normalizer: str = 'quantile',
    importance_window: int = 2000,
    importance_warmup: int = 100,
    target_rate: float | None = None,
    horizon: int | None = None,
    pace_gain: float = 0.5,
    pace_interval: int = 25,
    min_confidence: float = 0.0,
    mistake_fn=None,
    predictor_kind: str = 'count',
    key_fn=None,
    trust: float = 0.8,
    use_surrogate_advice: bool = True,
    trace: bool = False,
) -> BaseAdvisor:
    """
    Build an advisor by name.

    Parameters
    ----------
    name: str
        One of ADVISOR_NAMES. 'unlimited' reproduces the behaviour
        the controllers had before this package existed and is the
        right default for backwards compatibility.
    teacher: BaseTeacher | None
        Needed only to build teacher-side importance; the advisor
        receives the teacher again on every `advise` call, so this
        is not where the teacher is bound for normal operation.
    num_actions: int
        Action-space size, needed by the predictor-based strategies.
    advice_budget: int
        Cap on advice DELIVERED to the student (Torrey & Taylor's
        `n`). 0 means unlimited.
    query_budget: int
        Cap on teacher CONSULTATIONS -- the money budget. 0 means
        unlimited, which recovers the paper's assumptions exactly.
    threshold: float
        Importance threshold `t`. With normalized importance this
        lives in [0, 1] regardless of teacher.
    importance_source: str
        One of IMPORTANCE_SOURCES; see advising/importance.py for
        which sources work with which teachers.
    importance_fn: callable | None
        A ready-made importance function, overriding
        `importance_source` entirely. This is the extension point
        for settings whose importance signal is not one of the
        built-ins -- a PPO student, for instance, has no Q-table but
        does have policy logits, and its importance function has to
        close over the live network. Supplying one here keeps that
        setting-specific code out of this package.
    normalize_importance: bool
        Rescale importance to [0, 1], so `t` means the same thing
        across teachers and signals.
    importance_normalizer: str
        'quantile' (default) makes `t` a true percentile -- 0.8
        means "the most important 20% of recent states" -- and is
        robust both to outliers and to the way importance drifts as
        the student learns. 'minmax' is the older affine rescale.
    importance_window: int
        Sliding-window size for the quantile normalizer.
    importance_warmup: int
        Steps before the normalizer trusts its estimate.
    target_rate: float | None
        Fraction of ENVIRONMENT steps to advise on. Setting this
        makes `threshold` self-adjusting rather than a
        hyperparameter to guess -- see BaseAdvisor for why a fixed
        threshold plus a finite budget quietly degenerates into
        early advising.
    horizon: int | None
        Expected total environment steps. With an `advice_budget`
        and no `target_rate`, this switches on budget pacing: the
        target is remaining advice over remaining horizon,
        re-derived as the run proceeds. Prefer it when comparing
        strategies -- the budget becomes the controlled variable and
        there is no separately chosen rate to conflict with it.
    pace_gain, pace_interval: float, int
        Controller gain and update period for that adjustment.
    min_confidence: float
        Withhold advice below this self-reported confidence.
    mistake_fn: callable | None
        Overrides how "the student is making a mistake" is decided.
        None uses the paper's exact-action comparison, which is
        right for an epsilon-greedy student. A student that samples
        its actions wants `probability_mistake` instead -- see the
        benchmark repo's `advising/policy.py`.
    predictor_kind: str
        'count' or 'linear', for the predictive and surrogate
        strategies.
    key_fn: callable | None
        State-key function for the count predictor. Pass the
        student's own discretizer so predictor bins match the bins
        the student learns over.
    trust: float
        Surrogate-only: confidence required before the surrogate's
        guess is used.
    use_surrogate_advice: bool
        Surrogate-only: allow the surrogate to advise for free.
    trace: bool
        Keep a per-decision trace for later analysis.

    Returns
    -------
    BaseAdvisor
    """

    if name not in _ADVISOR_CLASSES:
        raise ValueError(
            f'Unknown advisor {name!r}; expected one of '
            f'{ADVISOR_NAMES}.'
        )

    # A caller-supplied importance function wins outright, but is
    # still wrapped in the normalizer so that a threshold means the
    # same thing whether the signal came from this package or from
    # the caller.
    if importance_fn is None:
        importance_fn = make_importance(
            importance_source,
            teacher=teacher,
            normalize=normalize_importance,
            warmup=importance_warmup,
            normalizer=importance_normalizer,
            window=importance_window,
        )
    elif normalize_importance:
        wrapper = (
            QuantileNormalizer
            if importance_normalizer == 'quantile'
            else Normalizer
        )
        if wrapper is QuantileNormalizer:
            importance_fn = wrapper(
                importance_fn,
                window=importance_window,
                warmup=importance_warmup,
            )
        else:
            importance_fn = wrapper(
                importance_fn, warmup=importance_warmup
            )

    # Settings every strategy accepts.
    common = {
        'advice_budget': advice_budget,
        'query_budget': query_budget,
        'threshold': threshold,
        'importance_fn': importance_fn,
        'min_confidence': min_confidence,
        'mistake_fn': mistake_fn,
        'target_rate': target_rate,
        'horizon': horizon,
        'pace_gain': pace_gain,
        'pace_interval': pace_interval,
        'trace': trace,
    }

    cls = _ADVISOR_CLASSES[name]

    # The two predictor-based strategies need to know the action
    # space and how to bin states; the rest do not accept those.
    if name == 'predictive':
        return cls(
            num_actions=num_actions,
            predictor_kind=predictor_kind,
            key_fn=key_fn,
            **common,
        )
    if name == 'surrogate':
        return cls(
            num_actions=num_actions,
            predictor_kind=predictor_kind,
            key_fn=key_fn,
            trust=trust,
            use_surrogate_advice=use_surrogate_advice,
            **common,
        )

    return cls(**common)
