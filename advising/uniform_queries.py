"""State-independent query slots for the MultiRoom advisor comparison.

No global RNG draws or adaptive resampling.
"""

import hashlib
import numpy as np


class UniformQueries:
    """Choose clock slots without replacement inside active rollouts.

    Episode-reset placeholders may consume a slot without a consultation.
    Such slots are not replaced, preserving the state-independent plan.
    """

    def __init__(self, active_rollouts, num_steps, num_envs, budget, seed):
        self.rollouts = tuple(active_rollouts)
        if (not self.rollouts or tuple(sorted(set(self.rollouts)))
                != self.rollouts or num_steps < 1 or num_envs < 1):
            raise ValueError('Need unique increasing active rollout indices')
        size = num_steps * num_envs
        if not 0 < budget <= len(self.rollouts) * size:
            raise ValueError('Query budget must fit the active clock slots')
        self.seed = int(seed) + 1_904_117
        rng = np.random.default_rng(self.seed)
        relative = rng.choice(len(self.rollouts) * size, budget,
                              replace=False)
        self.slots = tuple(sorted(
            self.rollouts[int(v) // size] * size + int(v) % size
            for v in relative))
        self.selected = frozenset(self.slots)
        self.observed = []
        self.num_steps, self.num_envs = num_steps, num_envs

    def allows(self, rollout, step, env):
        slot = ((rollout * self.num_steps + step) * self.num_envs + env)
        return slot in self.selected

    def record(self, rollout, step, env):
        slot = ((rollout * self.num_steps + step) * self.num_envs + env)
        if slot not in self.selected or slot in self.observed:
            raise ValueError('Duplicate or off-plan uniform consultation')
        self.observed.append(slot)

    def manifest(self):
        encoded = ','.join(map(str, self.slots)).encode('ascii')
        return dict(kind='uniform_without_replacement', rng_seed=self.seed,
                    slots=list(self.slots), slots_sha256=hashlib.sha256(
                        encoded).hexdigest(), observed=list(self.observed),
                    skipped_slots_not_replaced=True)
