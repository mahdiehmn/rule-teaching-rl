"""Common-clock query scheduling: shared times, selector-specific states.

`UniformQueries` samples (time, environment) slots,
so its random arm also fixes WHICH environment is asked. Here every
selector shares one state-independent schedule of query TIMES, and
only the choice among the parallel environments differs:

- `random`: uniform among eligible environments;
- `entropy`: the highest normalised student entropy among the same
  eligible pool, with ties broken uniformly.

Declared semantics:

- Times come from the (active rollout, step) grid over the
  teacher-active rollouts, with exactly ONE slot per time. The number of
  slots equals the budget, so consultations never exceed it.
  - spacing='random' (default, unchanged since a8ed13c): drawn without
    replacement from a private RNG.
  - spacing='regular' (added 2026-09-28): the k-th of B times sits at
    grid index floor((k + 1/2) N / B) of the N active times. It is
    deterministic and evenly spread over the same window, and uses no
    RNG. Only the spacing differs from the random clock.
- Eligible means not an autoreset placeholder. Those ticks show a fresh
  episode's first observation while the submitted action is discarded.
  A slot with no eligible environment is skipped and NOT replaced; so
  is a slot the trainer does not reach.
- Selection draws only from the supplied advisor RNG (isolated from PPO
  when `advisor_rng_isolation` is set). Nothing global is consumed.

This ranks the current parallel-environment pool at a fixed time. It is
not a global top-entropy selector, nor the unrestricted importance
advisor.
"""

import hashlib

import numpy as np

SELECTORS = ('random', 'entropy')
SPACINGS = ('random', 'regular')


class ClockQueries:
    def __init__(self, selector, active_rollouts, num_steps, budget, seed,
                 spacing='random'):
        if selector not in SELECTORS:
            raise ValueError(f'Clock selector must be one of {SELECTORS}')
        if spacing not in SPACINGS:
            raise ValueError(f'Clock spacing must be one of {SPACINGS}')
        self.rollouts = tuple(active_rollouts)
        if (not self.rollouts or tuple(sorted(set(self.rollouts)))
                != self.rollouts or num_steps < 1):
            raise ValueError('Need unique increasing active rollout indices')
        if not 0 < budget <= len(self.rollouts) * num_steps:
            raise ValueError('Query budget must fit the active clock times')
        self.selector, self.num_steps = selector, num_steps
        self.spacing = spacing
        # Independent of the selector, so every arm of a seed shares it.
        self.seed = int(seed) + 2_417_003
        total = len(self.rollouts) * num_steps
        if spacing == 'random':
            rng = np.random.default_rng(self.seed)
            relative = rng.choice(total, budget, replace=False)
        else:
            relative = [int((k + .5) * total / budget) for k in range(budget)]
        self.times = tuple(sorted(
            (self.rollouts[int(v) // num_steps], int(v) % num_steps)
            for v in relative))
        self.pending = set(self.times)
        self.events = []

    def scheduled(self, rollout, step):
        return (rollout, step) in self.pending

    def choose(self, rollout, step, eligible, entropy, num_actions, rng):
        """The environment to consult at a scheduled time, or None."""
        if (rollout, step) not in self.pending:
            raise ValueError('Not a scheduled clock time')
        self.pending.discard((rollout, step))
        eligible = [int(i) for i in eligible]
        event = dict(rollout=rollout, step=step, eligible=len(eligible),
                     chosen=None)
        if not eligible:
            event['skipped'] = 'no_eligible_env'
            self.events.append(event)
            return None
        if self.selector == 'random':
            chosen = eligible[int(rng.integers(len(eligible)))]
        else:
            scores = np.array([float(entropy[i]) for i in eligible])
            scores /= np.log(num_actions)
            best = [i for i, v in zip(eligible, scores)
                    if v >= scores.max() - 1e-12]
            chosen = best[int(rng.integers(len(best)))]
            event.update(chosen_entropy=float(scores.max()),
                         pool_mean_entropy=float(scores.mean()),
                         ties=len(best))
        event['chosen'] = chosen
        self.events.append(event)
        return chosen

    def outcome(self, consulted, delivered, global_step):
        """Attach what happened at the most recent chosen time."""
        self.events[-1].update(consulted=bool(consulted),
                               delivered=bool(delivered),
                               global_step=int(global_step))

    def manifest(self):
        encoded = ','.join(f'{r}:{s}' for r, s in self.times).encode()
        chosen = [e for e in self.events if e['chosen'] is not None]
        delivered = [e['global_step'] for e in chosen if e.get('delivered')]
        return dict(
            kind='common_clock_one_slot_per_time', selector=self.selector,
            spacing=self.spacing,
            rng_seed=self.seed, planned_times=len(self.times),
            times_sha256=hashlib.sha256(encoded).hexdigest(),
            reached=len(self.events),
            skipped_no_eligible=sum('skipped' in e for e in self.events),
            not_reached=len(self.pending), consultations=sum(
                e.get('consulted', False) for e in chosen),
            delivered_labels=len(delivered),
            first_delivered_step=min(delivered) if delivered else None,
            last_delivered_step=max(delivered) if delivered else None,
            mean_delivered_step=(float(np.mean(delivered))
                                 if delivered else None),
            events=self.events,
            skipped_slots_not_replaced=True)
