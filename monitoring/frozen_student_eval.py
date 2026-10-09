"""Batched teacher-free evaluation of frozen recurrent students."""

import numpy as np
import torch


def evaluate_frozen_student(agent, make_env, episode_seeds, *,
                            batch_size=10, device='cpu', on_episode=None):
    """
    Evaluate greedy episodes with native limits and fresh recurrent state.

    Every environment has exactly one episode. Completed slots remain
    inactive until the batch ends, avoiding vector autoreset ambiguity.
    No teacher, optimizer, reward modification or parameter update exists
    in this evaluator. The caller verifies checkpoint hashes separately.
    """

    seeds = list(episode_seeds)
    if batch_size < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Need a positive batch and unique episode seeds')
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(seeds), batch_size):
            chunk = seeds[offset:offset + batch_size]
            envs = []
            try:
                observations, limits = [], []
                for seed in chunk:
                    env = make_env()
                    envs.append(env)
                    obs, _ = env.reset(seed=int(seed))
                    observations.append(obs)
                    limits.append(int(env.unwrapped.max_steps))
                obs = np.stack(observations)
                core = agent.initial_core_state(len(chunk), device)
                starts = torch.zeros(len(chunk), device=device)
                active = np.ones(len(chunk), dtype=bool)
                returns = np.zeros(len(chunk), dtype=float)
                lengths = np.zeros(len(chunk), dtype=int)
                outcomes = [None] * len(chunk)
                for _ in range(max(limits)):
                    hidden, core = agent.get_states(
                        torch.as_tensor(obs, dtype=torch.float32,
                                        device=device), core, starts)
                    actions = agent.actor(hidden).argmax(-1).cpu().numpy()
                    for i in np.flatnonzero(active):
                        next_obs, reward, done, trunc, _ = envs[i].step(
                            int(actions[i]))
                        obs[i] = next_obs
                        returns[i] += float(reward)
                        lengths[i] += 1
                        if done or trunc:
                            active[i] = False
                            outcomes[i] = dict(
                                seed=int(chunk[i]),
                                success=int(returns[i] > 0),
                                reward=float(returns[i]),
                                length=int(lengths[i]),
                                terminated=bool(done),
                                truncated=bool(trunc),
                                native_max_steps=limits[i])
                            # Persist each finished episode immediately
                            # so a later failure cannot erase it.
                            if on_episode is not None:
                                on_episode(dict(outcomes[i]))
                    if not active.any():
                        break
                if active.any():
                    raise RuntimeError('Environment exceeded its native limit')
                rows.extend(outcomes)
            finally:
                for env in envs:
                    env.close()
    return rows
