"""
Keeping bought advice, so one query pays off more than once.

This is the part of the framework with no counterpart in Torrey &
Taylor (2013), and the reason is structural rather than incidental.
Their teacher's advice is an *intervention*: it moves the student on
one step, that step happens, and the moment is gone. That is exactly
why a budget of `n` is scarce -- `n` interventions buy `n` moments.

A distillation channel changes what advice is. The teacher's action
becomes a *labelled example*, and a labelled example can be replayed
into the behaviour-cloning term for the rest of training. One query
then contributes to thousands of gradient steps rather than one, and
the unit of budget stops being "interventions" and becomes "labels".

The consequence worth stating plainly: this reframes the whole
problem as **active learning**. The question is no longer "when
should the teacher speak" but "which states are worth paying to have
labelled", and Torrey & Taylor's four algorithms become
recognisable special cases -- importance advising is uncertainty
sampling, mistake correcting is disagreement sampling. Their ordering
should not be expected to carry over, because it was derived for a
mechanism where advice compounds through a value function rather than
through a replay buffer. Early advising in particular should fare
worse here: "label the first n states you meet" is a poor sampling
strategy with no diversity, though it was competitive in the paper.

`ppo_distill.py` already builds the soft target this stores; see
`soften_advice` there. What this class adds is memory.
"""

from collections import deque

import numpy as np


class AdviceReplay:
    """
    A bounded store of (observation, teacher target) pairs.

    Attributes
    ----------
    capacity: int
        Maximum labels retained. Observations dominate the memory
        cost -- a 56x56x3 uint8 frame is about 9 KB, so 10000 labels
        is roughly 90 MB -- which is why this is bounded and why the
        default is modest.
    obs: collections.deque
        Stored observations, oldest evicted first.
    targets: collections.deque
        Stored target distributions over actions.
    num_added: int
        Labels ever added, including those since evicted. Against
        `len(self)` it shows how much of the bought advice the
        buffer is still able to reuse.
    """

    def __init__(self, capacity: int = 10000, seed: int = 0):
        """
        Parameters
        ----------
        capacity: int
            Maximum number of labels to retain.
        seed: int
            RNG seed for sampling.
        """

        if capacity <= 0:
            raise ValueError(
                f'capacity must be positive; got {capacity}.'
            )

        self.capacity = capacity
        self.obs: deque = deque(maxlen=capacity)
        self.targets: deque = deque(maxlen=capacity)
        self.num_added = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """
        Number of labels currently retained.
        """

        return len(self.obs)

    def add(self, observation, target) -> None:
        """
        Store one labelled state.

        Parameters
        ----------
        observation: array-like
            The student's observation at the labelled state, copied
            so a later in-place write to the rollout buffer cannot
            corrupt the stored copy.
        target: array-like
            Target distribution over actions, as produced by
            `ppo_distill.soften_advice`.
        """

        self.obs.append(np.array(observation, copy=True))
        self.targets.append(
            np.asarray(target, dtype=np.float32).copy()
        )
        self.num_added += 1

    def sample(self, batch_size: int):
        """
        Draw a batch of labels uniformly at random, with
        replacement only if the buffer is smaller than the batch.

        Returns
        -------
        (obs, targets): tuple[numpy.ndarray, numpy.ndarray] | None
            Batched observations and targets, or None when the
            buffer is empty.
        """

        if not self.obs:
            return None

        n = len(self.obs)
        size = min(batch_size, n)
        idx = self.rng.choice(n, size=size, replace=False)

        obs = np.stack([self.obs[i] for i in idx])
        targets = np.stack([self.targets[i] for i in idx])
        return obs, targets

    def stats(self) -> dict:
        """
        Reuse accounting.

        `reuse_factor` is the headline number: how many gradient
        steps' worth of behaviour-cloning signal the buffer has
        supplied per label bought. In the intervention framing this
        is always 1 by construction, so anything above 1 is what the
        distillation channel adds to the value of a budget.
        """

        return {
            'replay_capacity': self.capacity,
            'replay_size': len(self.obs),
            'replay_added': self.num_added,
            'replay_evicted': max(
                0, self.num_added - len(self.obs)
            ),
        }
