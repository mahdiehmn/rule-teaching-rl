"""
State-importance signals for a policy-gradient student.

The rest of this package is a byte-for-byte copy of the one in the
sibling `vlm4rl-main` repo -- it depends only on the `Advice` /
`Cost` / `BaseTeacher` contract, which both repos define identically.
This module is the one bench-only addition, and it exists because
Torrey & Taylor's importance measure does not survive the move to
deep RL unchanged.

Their measure is

    I(s) = max_a Q(s, a) - min_a Q(s, a)

which needs a Q-function per action. A PPO agent has no Q-function:
it has a policy distribution over actions and a single state value
V(s). So the measure has to be rebuilt from what a policy actually
exposes. Two candidates, both free -- they read tensors the rollout
loop has already computed, so they add no forward passes:

- `logit_gap_importance` is the closest structural analogue. The
  actor's logits play the role Q-values played: a wide spread means
  the policy strongly prefers one action, a flat spread means it is
  indifferent. Note the interpretation flips relative to the paper.
  Torrey & Taylor compute the spread on a *converged teacher*, where
  a wide spread genuinely means "this state matters". Here the
  spread is the *student's*, so a wide spread means the student is
  already confident -- which is where advice is least likely to be
  needed, not most.

- `policy_entropy_importance` reads the same signal the honest way
  round. High entropy means the student has no idea what to do,
  which is exactly Clouse's (1996) original use of the measure as a
  learner-confidence signal and the right place to spend a query.
  This is the recommended default for a PPO student.

Both return raw numbers whose scale depends on the action-space size
and the stage of training, so wrap them in the package's `Normalizer`
(which `make_advisor` does by default) before comparing a threshold
against them.

Usage
-----
These build importance functions that read a per-step context the
training loop fills in, rather than closing over the network. That
keeps them cheap -- the loop already has the logits in hand -- and
keeps this module free of any dependency on a particular agent
class:

    advisor = make_advisor(
        'importance',
        importance_fn=policy_entropy_importance,
        threshold=0.5,
    )
    ...
    advice, cost = advisor.advise(
        teacher, state, student_action, {'logits': logits_i}
    )
"""

import numpy as np


def _as_logits(context: dict):
    """
    Pull a 1-D logit vector out of the per-step context.

    Accepts either a torch tensor or anything numpy can consume, so
    the training loop can hand over its tensors directly without a
    conversion step of its own. Returns None when the context has no
    logits, which makes the importance functions degrade to zero
    rather than raise if the loop has not been wired up yet.
    """

    logits = context.get('logits')
    if logits is None:
        return None

    # Detach a torch tensor before it reaches numpy; the importance
    # signal must never participate in the policy gradient.
    detach = getattr(logits, 'detach', None)
    if detach is not None:
        logits = detach().cpu().numpy()

    row = np.asarray(logits, dtype=np.float64).ravel()
    if row.size == 0:
        return None
    return row


def logit_gap_importance(state, context: dict) -> float:
    """
    Spread of the actor's logits at this state.

    The direct structural translation of the paper's formula, with
    logits standing in for Q-values. See the module docstring for
    why its interpretation is inverted relative to the paper: this
    is the *student's* spread, so large means confident.

    Useful mainly as a contrast arm against
    `policy_entropy_importance` -- if the two produce similar
    learning curves, importance is not the mechanism doing the work.
    """

    row = _as_logits(context)
    if row is None:
        return 0.0
    return float(row.max() - row.min())


def policy_entropy_importance(state, context: dict) -> float:
    """
    Entropy of the student's action distribution at this state.

    High entropy means the student is undecided, which is where a
    teacher's query buys the most. This is Clouse's (1996) reading
    of the measure -- learner confidence -- which is the reading
    available to us, since our teacher (an LLM) has no value
    function of its own to compute the paper's version from.

    Returns entropy in nats. The scale depends on the action-space
    size, so normalize before thresholding.

    Reads `context['entropy']` when the training loop already has
    it -- PPO's `get_action_and_value` returns per-env entropy as a
    matter of course, so the common case costs literally nothing --
    and falls back to computing it from `context['logits']`.
    """

    entropy = context.get('entropy')
    if entropy is not None:
        return float(entropy)

    row = _as_logits(context)
    if row is None:
        return 0.0

    # Softmax, shifted by the max for numerical stability.
    shifted = row - row.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    # Clip before the log so a saturated policy (a probability of
    # exactly zero after underflow) does not produce a NaN.
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _as_probs(context: dict):
    """
    Pull a 1-D action-probability vector out of the per-step
    context, computing it from logits if that is all there is.

    Returns None when neither is available, so the importance
    functions degrade to zero rather than raising.
    """

    probs = context.get('action_probs')
    if probs is not None:
        row = np.asarray(probs, dtype=np.float64).ravel()
        return row if row.size else None

    logits = _as_logits(context)
    if logits is None:
        return None

    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def normalized_entropy_importance(state, context: dict) -> float:
    """
    Policy entropy divided by its maximum, so the value lives in
    [0, 1] in every environment.

    `policy_entropy_importance` returns raw nats, whose ceiling is
    `ln(num_actions)`: 1.946 for MiniGrid's 7 actions, 1.099 for a
    3-action task. A threshold tuned on one is meaningless on the
    other -- 1.4 nats selects the top half of MiniGrid states and
    selects nothing at all in a 3-action task, because no state can
    score that high. Dividing by the ceiling removes the dependence:
    1.0 always means "completely undecided" and 0.0 always means
    "certain", whatever the action space.

    This is the importance signal to reach for by default with a
    policy-gradient student. Reads the whole distribution, so mass
    spread thinly over many bad actions counts the same as a
    genuine two-way tie -- see `top2_gap_importance` when that
    distinction matters.
    """

    row = _as_probs(context)
    if row is None or row.size < 2:
        return 0.0

    entropy = -np.sum(row * np.log(np.clip(row, 1e-12, 1.0)))

    # ln(num_actions) is the entropy of the uniform distribution and
    # therefore the largest value achievable over this many actions.
    return float(entropy / np.log(row.size))


def max_prob_importance(state, context: dict) -> float:
    """
    One minus the probability of the student's favourite action.

    The simplest confidence reading: a policy with 0.9 on one action
    scores 0.1 (leave it alone), a uniform policy over 7 actions
    scores 0.86 (worth advising). Already in [0, 1] and already
    environment-independent, though its floor rises with the action
    count (a uniform 7-action policy cannot score below 0.86).

    Ignores how the remaining mass is arranged, which makes it
    cheaper to reason about than entropy and blunter.
    """

    row = _as_probs(context)
    if row is None:
        return 0.0
    return float(1.0 - row.max())


def top2_gap_importance(state, context: dict) -> float:
    """
    One minus the gap between the best and second-best action
    probabilities: high when the top two are neck-and-neck.

    The closest thing here to Torrey & Taylor's `max_a Q - min_a Q`
    in spirit -- both ask whether there is a meaningful difference
    between the options at this state -- and the sharpest signal for
    the question that actually matters when spending advice: would
    advice CHANGE what the agent does? If the top two are tied, a
    nudge flips the behaviour. If the favourite dominates, advice
    barely moves the policy and the budget is better spent
    elsewhere.

    It sees a distinction entropy misses. The policies [0.4, 0.35,
    0.25] and [0.4, 0.4, 0.2] have almost identical entropy, but
    only the second is a genuine coin-flip; this function scores it
    higher (1.0 versus 0.95).

    Returns a value in [0, 1] regardless of action count, since it
    reads only two probabilities.
    """

    row = _as_probs(context)
    if row is None or row.size < 2:
        return 0.0

    # Partial sort is enough -- only the top two matter.
    top2 = np.partition(row, -2)[-2:]
    return float(1.0 - abs(top2[1] - top2[0]))


def probability_mistake(threshold: float = 0.5):
    """
    Build a mistake test suited to a student that SAMPLES its
    actions, replacing Torrey & Taylor's exact-action comparison.

    Their test is `a_student != a_teacher`, which is exactly right
    for an epsilon-greedy student: such a student has one intended
    action, so the comparison reads its intent directly. A PPO
    student has no single intended action -- it draws one from
    `pi(. | s)` -- and that makes the same comparison a poor
    instrument in both directions:

    - A student holding 90% of its probability mass on the correct
      action still samples something else one time in ten. The
      comparison calls that a mistake and spends budget correcting
      a student that already knows the answer.
    - A student split 40/35/25 across three actions may sample the
      correct one by luck. The comparison calls that no mistake and
      skips a state where the student is genuinely lost.

    The replacement asks what the student BELIEVES rather than what
    a single draw produced: a mistake is

        pi(a_teacher | s) < threshold

    -- the student assigns low probability to the action the
    teacher would take. Equivalently, in surprisal terms, the
    mistake condition is `-log pi(a* | s) > -log threshold`, and
    that quantity is also proportional to how much a behaviour-
    cloning gradient at this state would move the policy. So the
    same number answers "is the student wrong here?" and "would
    advice here actually change anything?", which the exact
    comparison cannot do at all: it is a bit, and every mistake
    looks equally bad.

    Parameters
    ----------
    threshold: float
        Probability below which the teacher's action counts as one
        the student would not have taken. 1.0 makes every state a
        mistake; 0.0 makes none. Around 0.5 means "the student is
        not already favouring the right action"; lower values
        (0.1-0.2) reserve budget for states where the student is
        badly wrong rather than merely unsure.

    Returns
    -------
    callable
        A `(student_action, teacher_action, context) -> bool`
        function suitable for `make_advisor(mistake_fn=...)`.
        Falls back to the exact comparison whenever the context
        carries no action probabilities, so a partially wired
        training loop degrades to the paper's behaviour instead of
        silently treating everything as a mistake.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f'threshold must be in [0, 1]; got {threshold}.'
        )

    def mistake_fn(student_action, teacher_action, context) -> bool:
        """
        Report a mistake when the student's policy puts less than
        `threshold` probability on the actions the teacher endorses.
        """

        # When the teacher can name every optimal action, score
        # against the whole set. A gridworld usually has several
        # equally good moves -- two shortest routes around an
        # obstacle -- and the teacher returns one arbitrarily, so
        # scoring against that one alone counts a student that chose
        # a different member of the tie as mistaken.
        optimal = context.get('optimal_actions')
        endorsed = (
            {int(a) for a in optimal}
            if optimal
            else {int(teacher_action)}
        )

        probs = context.get('action_probs')
        if probs is None:
            # No distribution available: fall back to the paper's
            # test, still widened to the endorsed set.
            return int(student_action) not in endorsed

        row = np.asarray(probs, dtype=np.float64).ravel()
        mass = sum(
            float(row[a]) for a in endorsed if a < row.size
        )
        return bool(mass < threshold)

    return mistake_fn


def value_gap_importance(state, context: dict) -> float:
    """
    Absolute advantage of the sampled action, |Q(s,a) - V(s)|,
    approximated from whatever the loop can supply cheaply.

    Reads `context['advantage']`. Unlike the two functions above
    this is not available at action-selection time -- the advantage
    is only known after the rollout has been credited -- so it
    cannot gate a live query. It is here for offline analysis: given
    a recorded trace, it answers "which states would have been worth
    querying, in hindsight?", which is the ceiling any online
    importance signal is trying to approach.
    """

    advantage = context.get('advantage')
    if advantage is None:
        return 0.0
    return float(abs(advantage))
