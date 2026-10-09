"""
Action override from teacher advice.

Substitutes the teacher's recommended action for the agent's sampled
action with some probability, annealed from high (lean on the teacher
early, when the agent is clueless) to low or zero (act on the learned
policy late). This is the most direct form of guidance: the teacher
literally drives some of the steps, so the agent collects successful
trajectories it could not have produced on its own, and PPO then
raises the probability of the actions on those trajectories.

This module owns only the *policy* of how much to override and the
mechanics of applying it. Deciding which envs to override and
querying the teacher live in the training loop, because they need
access to the environments. Keeping the schedule here lets a future
masking strategy sit beside it as a sibling without touching the
trainer.
"""

import numpy as np


class ActionOverrider:
    """
    Annealed action-override schedule.

    Holds the override probability schedule and applies a teacher's
    advice to a batch of sampled actions. The probability decays
    linearly from `start_prob` to `end_prob` over the first
    `fraction` of training, then stays at `end_prob`.
    """

    def __init__(
        self,
        start_prob,
        end_prob,
        fraction,
        total_timesteps,
    ):
        """
        Parameters
        ----------
        start_prob: float
            Override probability at the start of training (per env,
            per step).
        end_prob: float
            Override probability once annealing has finished.
        fraction: float
            Fraction of total training over which to anneal from
            start to end (e.g. 0.5 = anneal over the first half).
        total_timesteps: int
            Total environment frames; sets the annealing horizon.
        """

        self.start_prob = start_prob
        self.end_prob = end_prob
        self.fraction = fraction
        self.total_timesteps = total_timesteps

    def current_prob(self, global_step):
        """
        Return the override probability at the given training step.
        """

        # Linear interpolation from start to end over the annealing
        # window, clamped to end_prob afterward.
        horizon = max(1.0, self.fraction * self.total_timesteps)
        frac = min(1.0, global_step / horizon)
        return self.start_prob + frac * (
            self.end_prob - self.start_prob
        )

    def expected_remaining_visible(
        self, global_step, num_envs, mean_ep_len=None
    ):
        """
        Estimate how many more steps the teacher will be consulted
        on before training ends.

        An advice budget can only be spent on steps this schedule
        actually offers, and that number is not something the
        budgeting layer can work out for itself: extrapolating the
        visibility observed so far says "plenty" right up until the
        schedule closes. Integrating the schedule forward gives the
        real answer.

        Parameters
        ----------
        global_step: int
            Steps elapsed.
        num_envs: int
            Parallel environments; each contributes one candidate
            step per environment step.
        mean_ep_len: float | None
            Unused here, accepted so both overriders share a
            signature.

        Returns
        -------
        float
            Expected remaining teacher-visible steps.
        """

        # The probability schedule is a straight line from
        # start_prob to end_prob over `fraction` of training, then
        # flat. Summing it over the remaining steps is exact enough
        # sampled once per 1% of the run.
        total = 0.0
        remaining = max(0, self.total_timesteps - global_step)
        if remaining <= 0:
            return 0.0

        chunk = max(1, remaining // 100)
        step = global_step
        while step < self.total_timesteps:
            span = min(chunk, self.total_timesteps - step)
            total += self.current_prob(step) * span
            step += span

        return total

    def select(self, num_envs, global_step, rng):
        """
        Decide, per environment, whether to override this step.

        Parameters
        ----------
        num_envs: int
            Number of parallel environments.
        global_step: int
            Current training step, used to read the schedule.
        rng: numpy.random.Generator
            RNG for the per-env Bernoulli draws.

        Returns
        -------
        numpy.ndarray of bool, shape (num_envs,)
            True where this env's action should be overridden.
        """

        prob = self.current_prob(global_step)
        return rng.random(num_envs) < prob

    def apply(self, sampled_actions, advices, flags):
        """
        Replace sampled actions with teacher actions where flagged.

        Parameters
        ----------
        sampled_actions: numpy.ndarray, shape (num_envs,)
            The actions the policy sampled.
        advices: list
            Per-env Advice objects (or None where no advice was
            requested). Only entries whose env is flagged are read.
        flags: numpy.ndarray of bool, shape (num_envs,)
            Which envs to override (from `select`).

        Returns
        -------
        executed: numpy.ndarray, shape (num_envs,)
            The actions to actually execute.
        n_overridden: int
            How many actions were replaced (teacher gave a concrete
            action). Abstentions leave the sampled action in place.
        """

        # The replacement logic is identical for both hand-off
        # strategies, so it lives in a shared module-level helper.
        return apply_override(sampled_actions, advices, flags)


def apply_override(sampled_actions, advices, flags):
    """
    Replace sampled actions with teacher actions where flagged.

    Shared by ActionOverrider (random-step) and PrefixOverrider
    (Jump-Start prefix): given the policy's sampled actions, the
    per-env teacher advice, and a boolean override mask, return the
    actions to actually execute plus how many were replaced.

    Parameters
    ----------
    sampled_actions: numpy.ndarray, shape (num_envs,)
        The actions the policy sampled.
    advices: list
        Per-env Advice objects (or None where no advice was
        requested). Only entries whose env is flagged are read.
    flags: numpy.ndarray of bool, shape (num_envs,)
        Which envs to override.

    Returns
    -------
    executed: numpy.ndarray, shape (num_envs,)
        The actions to actually execute.
    n_overridden: int
        How many actions were replaced (teacher gave a concrete
        action). Abstentions leave the sampled action in place.
    """

    executed = np.array(sampled_actions, copy=True)
    n_overridden = 0
    for i, flag in enumerate(flags):
        if not flag:
            continue
        advice = advices[i]
        # A teacher may abstain (action is None); only override
        # when it actually recommends something.
        if advice is not None and advice.action is not None:
            executed[i] = advice.action
            n_overridden += 1
    return executed, n_overridden


class PrefixOverrider:
    """
    Jump-Start-style prefix hand-off (Uchendu et al., 2023).

    Unlike ActionOverrider, which overrides RANDOM steps scattered
    through an episode, the teacher here drives a contiguous PREFIX of
    every episode -- the first H steps -- and the agent acts on its
    own for the entire rest. H is annealed from `start_len` down to
    `end_len` over the first `fraction` of training.

    The effect is a reverse curriculum: early on (H large) the teacher
    solves almost the whole episode and the agent only has to finish
    the last few steps near the goal -- but it finishes them with its
    OWN policy and collects the real reward. As H shrinks, the agent
    owns progressively earlier segments, always learning to reach the
    start of the segment it has already mastered. Because the agent
    always controls -- and is honestly evaluated on -- the suffix it
    reaches, it never just rides the teacher, so removing guidance
    does not collapse it. (Contrast ActionOverrider, where random
    patching masks the agent's mistakes anywhere in the episode, so
    the agent's own policy is never trained on the hard middle and
    falls over once the override stops.)
    """

    def __init__(self, start_len, end_len, fraction, total_timesteps):
        """
        Parameters
        ----------
        start_len: float
            Guide prefix length H at the start of training. Should be
            at least the task's optimal path length so the teacher can
            solve a whole episode early on.
        end_len: float
            Prefix length once annealing has finished (0 = full
            hand-off, the agent eventually drives from step one).
        fraction: float
            Fraction of total training over which to anneal H from
            start to end.
        total_timesteps: int
            Total environment frames; sets the annealing horizon.
        """

        self.start_len = start_len
        self.end_len = end_len
        self.fraction = fraction
        self.total_timesteps = total_timesteps

    def current_len(self, global_step):
        """
        Return the guide prefix length H at the given training step.
        """

        # Linear interpolation from start to end over the annealing
        # window, clamped to end_len afterward.
        horizon = max(1.0, self.fraction * self.total_timesteps)
        frac = min(1.0, global_step / horizon)
        return self.start_len + frac * (self.end_len - self.start_len)

    def expected_remaining_visible(
        self, global_step, num_envs, mean_ep_len=None
    ):
        """
        Estimate how many more steps the teacher will be consulted
        on before training ends.

        The prefix hand-off makes this question urgent rather than
        academic. H anneals to zero, so teacher-visible steps are
        front-loaded and then stop entirely -- a budget paced as if
        the opportunity were spread evenly across training will
        still be mostly unspent when the opportunity disappears.

        Within an episode of length L the teacher drives the first
        H steps, so a fraction min(H, L) / L of steps are visible.
        Integrating that over the remaining schedule gives the
        estimate.

        Parameters
        ----------
        global_step: int
            Steps elapsed.
        num_envs: int
            Parallel environments.
        mean_ep_len: float | None
            Observed mean episode length. Without it the estimate
            cannot be made, and 0 is returned so the caller falls
            back rather than trusting a guess.

        Returns
        -------
        float
            Expected remaining teacher-visible steps.
        """

        if not mean_ep_len or mean_ep_len <= 0:
            return 0.0

        remaining = max(0, self.total_timesteps - global_step)
        if remaining <= 0:
            return 0.0

        total = 0.0
        chunk = max(1, remaining // 100)
        step = global_step
        while step < self.total_timesteps:
            span = min(chunk, self.total_timesteps - step)
            visible_fraction = min(
                1.0, self.current_len(step) / mean_ep_len
            )
            total += visible_fraction * span
            step += span

        return total

    def select(self, ep_step, global_step):
        """
        Override env i iff it is still inside the guide prefix.

        Parameters
        ----------
        ep_step: numpy.ndarray of int, shape (num_envs,)
            Steps already taken in each env's current episode.
        global_step: int
            Current training step, used to read the H schedule.

        Returns
        -------
        numpy.ndarray of bool, shape (num_envs,)
            True where this env is still within the first H steps.
        """

        return ep_step < self.current_len(global_step)

    def apply(self, sampled_actions, advices, flags):
        """
        Replace sampled actions with teacher actions where flagged.
        """

        return apply_override(sampled_actions, advices, flags)
