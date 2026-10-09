"""
Count exploration and return targets for the matched distillation grid.

The count partition, reward normalization, and continuing intrinsic
return follow ppo_intrinsic. No-bonus arms can measure the same count
coverage while their PPO objective uses only the environment reward.
"""

import numpy as np
import torch

from algos.intrinsic import CountBonus, RunningMeanStd


def local_count_observations(sync_envs, device):
    """
    Read the native local grid without changing the policy's wrapper.

    MiniGrid gen_obs is a deterministic read of current state. Call it
    after vector stepping, including NEXT_STEP resets, just as the
    original counter reads the returned next observation. Float32
    preserves the byte representation used by local-policy counts.
    """

    images = [env.unwrapped.gen_obs()['image'] for env in sync_envs.envs]
    return torch.as_tensor(np.stack(images), dtype=torch.float32,
                           device=device)


def intrinsic_gae(rewards, values, bootstrap, gamma, gae_lambda, keep=None):
    """
    Compute the continuing intrinsic stream used by ppo_intrinsic.

    `keep[t]` (1 where the student chose the action at step t) stops the
    trace before a step another policy chose (the RLingua baseline); None
    leaves the computation unchanged.
    """

    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        if keep is not None and t < len(rewards) - 1:
            carry = carry * keep[t + 1]
        carry = delta + gamma * gae_lambda * carry
        advantages[t] = carry
    return advantages, advantages + values


class CountExploration:
    """
    Keep episodic observation counts and an optional reward stream.

    Coverage here means distinct observation hashes, not distinct world
    positions or teacher phases. Under partial symbolic views different
    positions can share a hash. Reset transitions use the same convention
    as the existing ppo_intrinsic benchmark.
    """

    def __init__(self, obs_shape, num_envs, device, args):
        self.counter = CountBonus(obs_shape, num_envs, device,
                                  scope='episodic')
        self.enabled = args.bonus == 'count'
        self.normalize = args.norm_int_reward
        self.gamma = args.int_gamma
        self.reward_rms = RunningMeanStd()
        self.discounted = None

    def rollout(self, obs, next_obs, actions, dones):
        """
        Return reward and comparable coverage diagnostics per rollout.
        """

        with torch.no_grad():
            raw = self.counter.rollout_bonus(
                obs, next_obs, actions.long(), dones)
            stats = {
                'first_visit_frac': self.counter.last_unique_frac,
                'count_reward_mean_raw': float(raw.mean()),
                'count_reward_max_raw': float(raw.max()),
            }
            if not self.enabled:
                stats['reward_mean_used'] = 0.0
                return torch.zeros_like(raw), stats
            reward = raw
            if self.normalize:
                # The discounted sum estimates scale only. Keeping it
                # across rollouts matches the baseline normalization.
                discounted = []
                for row in raw.cpu().numpy():
                    self.discounted = (
                        row.copy() if self.discounted is None
                        else self.gamma * self.discounted + row)
                    discounted.append(self.discounted.copy())
                self.reward_rms.update(np.asarray(discounted).reshape(-1))
                reward = raw / (np.sqrt(self.reward_rms.var) + 1e-8)
            stats['reward_mean_used'] = float(reward.mean())
            stats['reward_running_std'] = float(
                np.sqrt(self.reward_rms.var))
            return reward, stats
