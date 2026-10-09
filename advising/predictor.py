"""
Cheap action predictors used by the predictive advising strategies.

Torrey & Taylor's *predictive advising* has the teacher train a
classifier on the student's observed behaviour, so it can guess what
the student is about to do and spend advice only where it expects a
mistake -- without requiring the student to announce its intentions.
They use an SVM (SVM-Light) fitted between episodes.

We need the same capability in two directions.

1. **Predict the student's action** (the paper's direction). Lets a
   teacher detect likely mistakes without extra communication.
2. **Predict the teacher's action** (the direction an expensive
   teacher forces on us). If a cheap surrogate can guess what the
   LLM would say, most queries never need to be sent. The paper had
   no reason to consider this because its teacher answered for free;
   with a paid teacher it is the only mechanism here that reduces
   *money* rather than merely student attention.

Both directions are the same supervised problem -- map a state to an
action -- so both use the same predictor classes.

Why not scikit-learn
--------------------
The paper's choice of SVM is not load-bearing; what matters is that
the predictor is cheap and refits quickly. scikit-learn is not a
dependency of this project (it is present only in the sibling
benchmark repo's virtualenv), so adding it for two small models would
make the tabular experiments harder to run, not easier. The two
predictors here are a few dozen lines of numpy each.

The paper reports its SVM reaching only ~50% accuracy on Mountain Car
and blames the feature-to-example ratio (1024-2048 tile features
against 150-500 examples). Our tabular students discretize the state
anyway, so `CountPredictor` sidesteps that problem entirely by
predicting per state bin. Expect noticeably better prediction
accuracy than the paper reports, and interpret the predictive-advising
results accordingly -- a difference in the instrument, not in the
algorithm.
"""

from collections import deque
from typing import Any, Callable

import numpy as np

# Names accepted by `make_predictor`, exposed for CLI choices.
PREDICTOR_KINDS = ('count', 'linear')


def default_key(state: Any) -> Any:
    """
    Turn a state into something hashable, for predictors that key on
    exact states.

    Handles the shapes the tabular controllers actually produce:
    plain ints (MiniGrid's discretized state), tuples, and small
    float arrays (Mountain Car's position/velocity pair, which gets
    rounded so that nearby continuous states share a key).
    """

    if isinstance(state, (int, np.integer)):
        return int(state)
    if isinstance(state, tuple):
        return state

    arr = np.asarray(state).ravel()
    if arr.dtype.kind in 'iu':
        return tuple(int(v) for v in arr)

    # Rounding is what makes a continuous state usable as a table
    # key at all. Three decimals is fine for Mountain Car, whose
    # position spans about 1.8 and velocity about 0.14; a caller
    # with a real discretizer should pass its own key function
    # instead so the predictor shares the student's bins.
    return tuple(round(float(v), 3) for v in arr)


class BasePredictor:
    """
    Shared bookkeeping for the action predictors.

    Every predictor is scored *prequentially*: each observation is
    first predicted, then compared against the truth, and only then
    used for training. That gives an honest running accuracy on data
    the model has not seen, without holding out a test set -- which
    matters here because the target is non-stationary (a learning
    student changes its policy constantly) so a static holdout would
    measure the wrong thing.

    Attributes
    ----------
    num_actions: int
        Size of the action space.
    num_seen: int
        Observations recorded.
    num_scored, num_correct: int
        Prequential accuracy counters. `num_scored` counts only the
        observations where the model was willing to make a
        prediction, so abstentions do not inflate accuracy.
    """

    def __init__(self, num_actions: int):
        """
        Parameters
        ----------
        num_actions: int
            Number of discrete actions.
        """

        if num_actions <= 0:
            raise ValueError(
                f'num_actions must be positive; got {num_actions}.'
            )

        self.num_actions = num_actions
        self.num_seen = 0
        self.num_scored = 0
        self.num_correct = 0

    @property
    def accuracy(self) -> float:
        """
        Prequential accuracy over the observations the model chose
        to predict on. 0.0 before anything has been scored.
        """

        if self.num_scored == 0:
            return 0.0
        return self.num_correct / self.num_scored

    @property
    def coverage(self) -> float:
        """
        Fraction of observations the model was willing to predict
        on. Low coverage with high accuracy means a predictor that
        knows what it does not know -- exactly what a surrogate
        needs to avoid sending the student wrong advice.
        """

        if self.num_seen == 0:
            return 0.0
        return self.num_scored / self.num_seen

    def observe(self, state: Any, action: int) -> None:
        """
        Record one (state, action) example, scoring it first.
        """

        # Score before learning, so accuracy always reflects
        # performance on unseen data.
        guess = self.predict(state)
        self.num_seen += 1
        if guess is not None:
            self.num_scored += 1
            if int(guess) == int(action):
                self.num_correct += 1

        self._learn(state, int(action))

    def _learn(self, state: Any, action: int) -> None:
        """
        Absorb one example. Subclass hook.
        """

        raise NotImplementedError

    def predict(self, state: Any) -> int | None:
        """
        Guess the action for `state`, or None when the predictor has
        no basis for a guess.

        Returning None rather than a default action is important:
        the strategies treat "I do not know" as a reason to consult
        the real teacher, which is the safe direction to fail in.
        """

        raise NotImplementedError

    def fit(self) -> None:
        """
        Refit on accumulated data. Called between episodes, matching
        the paper's protocol of retraining between episodes rather
        than during them (an in-episode refit would be a visible
        pause for a human student, and here it would just be a
        needless per-step cost). Online predictors may no-op.
        """

    def stats(self) -> dict:
        """
        Summary counters for end-of-run reporting.
        """

        return {
            'predictor': self.__class__.__name__,
            'num_seen': self.num_seen,
            'num_scored': self.num_scored,
            'accuracy': self.accuracy,
            'coverage': self.coverage,
        }


class CountPredictor(BasePredictor):
    """
    Per-state-bin majority vote with exponential recency weighting.

    Keeps one count vector per state key and predicts its argmax.
    Counts decay by `decay` on every update to that key, so recent
    behaviour outweighs old behaviour.

    That decay is the direct answer to the tension the paper
    identifies in Section 3.4: a predictor of a *learning* student
    needs lots of data but also needs recent data, and more data
    stops helping once it is stale. An exponential window keeps both
    without a hard cutoff -- `decay=0.9` means an observation is
    worth about a third of its original weight ten updates later.

    Attributes
    ----------
    key_fn: callable
        Maps a state to a hashable key. Pass the student's own
        discretizer so the predictor bins states exactly as the
        student does.
    decay: float
        Multiplicative decay applied to a key's counts before each
        new observation.
    min_count: float
        Total weight a key needs before the predictor will commit to
        a prediction. Below it, `predict` returns None.
    """

    def __init__(
        self,
        num_actions: int,
        key_fn: Callable[[Any], Any] | None = None,
        decay: float = 0.95,
        min_count: float = 1.0,
    ):
        """
        Parameters
        ----------
        num_actions: int
            Number of discrete actions.
        key_fn: callable | None
            State-to-key function; `default_key` when None.
        decay: float
            Recency decay in (0, 1]. 1.0 disables decay and gives a
            plain lifetime majority vote.
        min_count: float
            Minimum accumulated weight before predicting.
        """

        super().__init__(num_actions)

        if not 0.0 < decay <= 1.0:
            raise ValueError(
                f'decay must be in (0, 1]; got {decay}.'
            )

        self.key_fn = key_fn or default_key
        self.decay = decay
        self.min_count = min_count

        # One count vector per state key. A dict rather than an
        # array because the key space is defined by the caller's
        # key function and may not be densely indexed.
        self.counts: dict[Any, np.ndarray] = {}

    def _learn(self, state: Any, action: int) -> None:
        """
        Decay this key's counts and add the observed action.
        """

        key = self.key_fn(state)
        row = self.counts.get(key)
        if row is None:
            row = np.zeros(self.num_actions, dtype=np.float64)
            self.counts[key] = row

        row *= self.decay
        row[action] += 1.0

    def predict(self, state: Any) -> int | None:
        """
        Return the most frequent recent action at this state's key,
        or None if the key is unseen or too sparsely observed.
        """

        row = self.counts.get(self.key_fn(state))
        if row is None or row.sum() < self.min_count:
            return None
        return int(np.argmax(row))

    def confidence(self, state: Any) -> float:
        """
        Share of this key's weight held by the predicted action.

        Used by the surrogate strategy to decide whether its guess
        is trustworthy enough to substitute for a real query. 0.0
        for unseen keys.
        """

        row = self.counts.get(self.key_fn(state))
        if row is None:
            return 0.0
        total = row.sum()
        if total <= 0.0:
            return 0.0
        return float(row.max() / total)

    def stats(self) -> dict:
        """
        Add the number of distinct state keys to the base summary,
        which is the predictor's memory footprint and a rough proxy
        for how much of the state space has been visited.
        """

        out = super().stats()
        out['num_keys'] = len(self.counts)
        return out


class LinearPredictor(BasePredictor):
    """
    Multinomial logistic regression over raw state features, refit
    between episodes.

    This is the closest stand-in for the paper's SVM: a linear model
    over state features, trained on recent experience, generalizing
    across states rather than memorizing them. It is the right
    choice when the state is continuous and the caller has no
    discretizer to share, or when the point is specifically to
    reproduce the paper's "linear model on state features" setup.

    Attributes
    ----------
    feature_fn: callable
        Maps a state to a 1-D float feature vector. Defaults to
        flattening the state itself.
    window: int
        Number of most recent examples retained for refitting. The
        paper trains on exactly the previous episode; a fixed-size
        window is the same idea, made robust to episodes that are
        very short or very long.
    epochs, lr, l2: int, float, float
        Full-batch gradient-descent settings for the refit.
    """

    def __init__(
        self,
        num_actions: int,
        feature_fn: Callable[[Any], np.ndarray] | None = None,
        window: int = 2000,
        epochs: int = 40,
        lr: float = 0.5,
        l2: float = 1e-4,
    ):
        """
        Parameters
        ----------
        num_actions: int
            Number of discrete actions.
        feature_fn: callable | None
            State-to-feature-vector function. None flattens the
            state to a float array.
        window: int
            Sliding-window size of retained examples.
        epochs, lr, l2: int, float, float
            Optimizer settings for `fit`.
        """

        super().__init__(num_actions)

        self.feature_fn = feature_fn or (
            lambda s: np.asarray(s, dtype=np.float64).ravel()
        )
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2

        # Sliding window of (features, action) examples awaiting the
        # next refit.
        self.buffer: deque = deque(maxlen=window)

        # Weights, allocated lazily on the first fit because the
        # feature width is not known until a state is seen.
        self.weights: np.ndarray | None = None

        # Running feature mean and variance, used to standardize
        # inputs. Gradient descent on raw Mountain Car features
        # would be badly conditioned: position spans about 1.8 while
        # velocity spans about 0.14, so without scaling the velocity
        # dimension is effectively ignored.
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._n_feat_seen = 0

    def _features(self, state: Any) -> np.ndarray:
        """
        Build the standardized feature vector for `state`, with a
        trailing constant for the bias term.
        """

        raw = np.asarray(
            self.feature_fn(state), dtype=np.float64
        ).ravel()

        if self._mean is None:
            return np.concatenate([raw, [1.0]])

        # Standardize with the running statistics, guarding against
        # zero-variance features (a constant input would otherwise
        # divide by zero).
        std = np.sqrt(np.maximum(self._var, 1e-8))
        return np.concatenate([(raw - self._mean) / std, [1.0]])

    def _update_feature_stats(self, raw: np.ndarray) -> None:
        """
        Fold one raw feature vector into the running mean and
        variance with Welford's algorithm.
        """

        if self._mean is None:
            self._mean = np.zeros_like(raw)
            self._var = np.ones_like(raw)

        self._n_feat_seen += 1
        delta = raw - self._mean
        self._mean = self._mean + delta / self._n_feat_seen
        # Welford's incremental variance, kept as a plain running
        # average of squared deviations so it stays well-defined
        # from the very first sample.
        self._var = self._var + (
            (delta * (raw - self._mean)) - self._var
        ) / self._n_feat_seen

    def _learn(self, state: Any, action: int) -> None:
        """
        Buffer one example and update the feature statistics.

        No gradient step happens here; the model is refit in bulk by
        `fit` between episodes.
        """

        raw = np.asarray(
            self.feature_fn(state), dtype=np.float64
        ).ravel()
        self._update_feature_stats(raw)
        self.buffer.append((raw, action))

    def fit(self) -> None:
        """
        Refit the weights on the buffered window by full-batch
        gradient descent on softmax cross-entropy.

        Does nothing until the buffer holds at least two distinct
        actions -- a single-class dataset has no decision boundary
        to learn and would just drive the weights to infinity.
        """

        if len(self.buffer) < 2:
            return

        actions = np.array([a for _, a in self.buffer], dtype=int)
        if np.unique(actions).size < 2:
            return

        # Standardize the buffered rows with the current statistics
        # and append the bias column.
        std = np.sqrt(np.maximum(self._var, 1e-8))
        raw = np.stack([r for r, _ in self.buffer])
        x = np.concatenate(
            [
                (raw - self._mean) / std,
                np.ones((raw.shape[0], 1)),
            ],
            axis=1,
        )

        # One-hot targets.
        y = np.zeros((actions.size, self.num_actions))
        y[np.arange(actions.size), actions] = 1.0

        if (
            self.weights is None
            or self.weights.shape[0] != x.shape[1]
        ):
            self.weights = np.zeros(
                (x.shape[1], self.num_actions), dtype=np.float64
            )

        n = x.shape[0]
        for _ in range(self.epochs):
            # Softmax probabilities, shifted by the row max for
            # numerical stability.
            logits = x @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)

            # Cross-entropy gradient plus L2 shrinkage.
            grad = x.T @ (probs - y) / n
            grad += self.l2 * self.weights
            self.weights -= self.lr * grad

    def predict(self, state: Any) -> int | None:
        """
        Return the highest-scoring action, or None before the first
        successful fit.
        """

        if self.weights is None:
            return None

        x = self._features(state)
        if x.shape[0] != self.weights.shape[0]:
            return None

        return int(np.argmax(x @ self.weights))

    def confidence(self, state: Any) -> float:
        """
        Softmax probability of the predicted action, for the
        surrogate strategy's trust check. 0.0 before the first fit.
        """

        if self.weights is None:
            return 0.0

        x = self._features(state)
        if x.shape[0] != self.weights.shape[0]:
            return 0.0

        logits = x @ self.weights
        logits -= logits.max()
        exp = np.exp(logits)
        return float(exp.max() / exp.sum())

    def stats(self) -> dict:
        """
        Add buffer occupancy to the base summary.
        """

        out = super().stats()
        out['buffer_size'] = len(self.buffer)
        out['fitted'] = self.weights is not None
        return out


def make_predictor(
    kind: str,
    num_actions: int,
    key_fn: Callable[[Any], Any] | None = None,
    **kwargs,
) -> BasePredictor:
    """
    Build a predictor by name.

    Parameters
    ----------
    kind: str
        'count' or 'linear'.
    num_actions: int
        Number of discrete actions.
    key_fn: callable | None
        Passed to CountPredictor as its state-key function; ignored
        by LinearPredictor, which works on raw features.
    **kwargs
        Forwarded to the predictor constructor.

    Returns
    -------
    BasePredictor
    """

    if kind == 'count':
        return CountPredictor(
            num_actions, key_fn=key_fn, **kwargs
        )
    if kind == 'linear':
        return LinearPredictor(num_actions, **kwargs)

    raise ValueError(
        f'Unknown predictor kind {kind!r}; expected one of '
        f'{PREDICTOR_KINDS}.'
    )
