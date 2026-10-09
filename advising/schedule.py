"""
Fixed, rollout-aligned query windows for the explanation comparison.

The pacing controller in `advising/base.py` targets *advice* -- it moves
an importance threshold to hit a delivery rate -- so it cannot state
where consultations fall, and `early` spends the whole query budget in
the opening steps. Neither answers the question the paid comparison
needs answered: at which transitions did we pay the teacher?

This module answers it in advance and independently of the run. The
window list is a pure function of the training shape, so it is
identical for R1, R2, R3 and R4, and knowing it requires no reference to
the student's policy, the phase, the returned text, whether the call
succeeded, or any advisor score. Two properties follow, and both are
load-bearing:

- **The arms are matched on when they paid.** Their trajectories still
  diverge, so the *states* differ; the schedule does not.
- **The positions can be checked.** A recorded query outside the list is
  a defect, not a judgement call about a controller's behaviour.

Rollout alignment is the second deliberate choice. Donors for the R3
control are drawn from the rollout being resolved, so scattering single
consultations across training would leave every sample the only member
of its phase in its own pool and the control would mask itself out.
Clustering a window's slots inside one rollout puts all of them in one
donor pool. That removes the *automatic* singleton problem; it does not
promise multiple eligible targets per phase, which is a property of the
collected text and has to be measured there.
"""


class QueryWindows:
    """
    The resolved consultation slots for one run.

    Zero-based throughout: rollout 0 is the first rollout, step 0 the
    first vector step within it. The trainer counts iterations from 1,
    so it passes `iteration - 1`.
    """

    def __init__(self, total_timesteps, batch_size, num_steps, num_envs,
                 num_windows=25, window_steps=10, guidance_fraction=0.75):
        """
        Resolve the window list, or refuse a configuration that cannot
        carry it.

        Refusing is the point of the checks below. A silently truncated
        or duplicated window list would still run, still cost money,
        and produce a schedule nobody specified.
        """

        if num_windows < 2:
            raise ValueError(
                f'num_windows must be at least 2 to span the guidance '
                f'window; got {num_windows}'
            )
        if window_steps < 1:
            raise ValueError(f'window_steps must be >= 1; got {window_steps}')
        if window_steps > num_steps:
            raise ValueError(
                f'window of {window_steps} steps is wider than the '
                f'{num_steps}-step rollout it must fit inside'
            )
        if not 0.0 < guidance_fraction <= 1.0:
            raise ValueError(
                f'guidance_fraction must be in (0, 1]; got '
                f'{guidance_fraction}'
            )

        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_windows = num_windows
        self.window_steps = window_steps
        self.guidance_fraction = guidance_fraction

        # Complete rollouts, and how many of them fall inside the
        # guidance window. PPO discards the remainder, so both floor.
        self.num_rollouts = int(total_timesteps // batch_size)
        self.guided_rollouts = int(
            (guidance_fraction * total_timesteps) // batch_size
        )
        if self.guided_rollouts < num_windows:
            raise ValueError(
                f'{num_windows} windows do not fit in the '
                f'{self.guided_rollouts} rollouts of the guidance '
                f'window; lengthen training, widen the fraction, or '
                f'ask for fewer windows'
            )

        # Endpoints included: window 0 is the first guided rollout and
        # the last is the final one, so the schedule spans the window
        # rather than clustering inside it.
        span = self.guided_rollouts - 1
        rollouts = [
            (j * span) // (num_windows - 1) for j in range(num_windows)
        ]
        if len(set(rollouts)) != num_windows:
            # Reachable only if the floor above is loosened; kept
            # because a duplicated index would quietly halve the spend
            # and leave the manifest claiming the full schedule.
            raise ValueError(
                f'window indices collided: {rollouts}; too few distinct '
                f'rollouts for {num_windows} windows'
            )
        self.rollouts = tuple(rollouts)
        self._allowed = frozenset(rollouts)

    def allows(self, rollout, step):
        """
        May the teacher be consulted at this (zero-based) position?
        """

        return rollout in self._allowed and step < self.window_steps

    @property
    def scheduled_slots(self):
        """
        Consultation slots the schedule offers, across the whole run.

        An upper bound, not a prediction. A slot is skipped when its
        environment is at a post-terminal placeholder step, and the
        hard query cap can bite first.
        """

        return self.num_windows * self.window_steps * self.num_envs

    def manifest(self):
        """
        The resolved schedule, for the run summary.

        The full index list is included deliberately: it is what makes
        an audit of recorded query positions possible after the fact.
        """

        return {
            'num_windows': self.num_windows,
            'window_steps': self.window_steps,
            'guidance_fraction': self.guidance_fraction,
            'num_rollouts': self.num_rollouts,
            'guided_rollouts': self.guided_rollouts,
            'scheduled_slots': self.scheduled_slots,
            'rollout_indices': list(self.rollouts),
        }
