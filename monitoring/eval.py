"""
Teacher-off greedy evaluation, shared by every algorithm.

The project's success criterion is performance with the teacher
OFF: a student that only performs while assisted has demonstrated
assisted execution, not knowledge transfer. This module is the one
implementation of that measurement, used identically by the plain
PPO baseline and by every teacher-guided variant, so the
`eval_success_rate` curves in different runs are directly
comparable (same episode seeds, same greedy action rule, same
episode budget).

Design choices:

- Greedy (argmax) actions rather than sampling: removes sampling
  noise from the measurement, so the curve reflects the policy,
  not the temperature.
- A FIXED set of eval episode seeds per run (seed_base + episode
  index): every evaluation point replays the same episodes, so
  consecutive eval points are paired measurements and the curve is
  smooth enough to read with few episodes.
- The eval env is built fresh by the caller-supplied factory and
  closed here, so evaluation can never leak state into (or from)
  the training envs.

A recurrent policy (algos/ppo_babyai.py and friends) needs to reset
its hidden state at the start of each episode, since episodes here
run strictly one at a time and nothing else in this function's
signature tells the closure where an episode boundary is. Rather
than change `select_action`'s calling convention (which would break
every existing memoryless caller), an optional `select_action.reset`
attribute is called, if present, right after each `env.reset()` --
plain functions are objects, so a caller can attach one without
`select_action` needing any special type. Memoryless callers simply
never define it, so `getattr(..., None)` finds nothing and this is a
no-op for them.
"""


def greedy_eval(select_action, make_env, num_episodes, seed_base):
    """
    Run `num_episodes` teacher-off greedy episodes; return stats.

    Parameters
    ----------
    select_action: callable obs -> int
        Greedy policy: maps a single raw observation (numpy array
        or dict of arrays, as returned by the env) to a discrete
        action. The caller wraps its network, including any argmax
        and no_grad logic. May optionally carry a zero-argument
        `.reset()` attribute (a stateful/recurrent policy's hidden-
        state reset); see the module docstring.
    make_env: callable () -> gymnasium.Env
        Factory for the evaluation environment, configured with
        the same task and observation pipeline as training.
    num_episodes: int
        Episodes to run. Kept modest (default 10 in the algos)
        because early policies time out every episode, making eval
        cost proportional to max_steps * num_episodes.
    seed_base: int
        Episode k is reset with seed `seed_base + k`, giving every
        evaluation call the same fixed episode set.

    Returns
    -------
    dict
        'success_rate', 'mean_return', 'mean_length' over the run
        episodes ('success' = positive episodic return, matching
        the convention used everywhere else in this repo).
    """

    env = make_env()
    # Read once: a plain function has no `.reset` attribute, so this
    # is None (and stays a no-op) for every non-recurrent caller.
    reset_policy = getattr(select_action, 'reset', None)
    successes, returns, lengths = [], [], []
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + episode)
            if reset_policy is not None:
                reset_policy()
            total_reward, steps = 0.0, 0
            while True:
                action = select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(
                    action
                )
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            successes.append(1.0 if total_reward > 0 else 0.0)
            returns.append(total_reward)
            lengths.append(steps)
    finally:
        env.close()

    n = max(1, len(successes))
    return {
        'success_rate': sum(successes) / n,
        'mean_return': sum(returns) / n,
        'mean_length': sum(lengths) / n,
    }
