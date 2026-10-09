"""
Tabular Q-learning baseline -- the "can't even represent it" floor.

This baseline exists to *demonstrate*, rather than hide, the
limitation that motivates the whole benchmark: a lookup table cannot
index a raw image. It runs honest tabular Q-learning, but the only
way to turn a 56x56x3 image into a table key is to treat each
distinct image as its own key (here, the raw pixel bytes). The
consequence is visible in two logged signals:

- charts/qtable_size: the number of distinct states in the table. On
  a procedurally generated task it grows by roughly one per step and
  never saturates -- the table is just memorizing pixels.
- charts/new_state_rate: the fraction of recent steps whose state had
  never been seen before. When this stays near 1.0, the agent almost
  never revisits a state, so there is nothing to bootstrap from and
  Q-learning cannot actually learn.

On a tiny fixed-layout task (empty5x5) the set of possible views is
small and recurs, so the table stays bounded and the agent can even
solve it -- which makes the contrast honest: tabular works only when
the state space is tiny and repeats. The moment the task is
procedurally generated (multiroom, keycorridor, any BabyAI level),
the table explodes and learning fails. That failure is the point of
this baseline.

Run it like the others:

    python -m algos.qlearn --task empty5x5
    python -m algos.qlearn --task multiroom_n6   # watch it explode
"""

import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter

from envs.registry import build_env
from monitoring.metrics import RunTracker, runtime_metadata


@dataclass
class Args:
    """
    Command-line arguments for a tabular Q-learning run.

    tyro exposes each field as a `--field value` flag.
    """

    # Friendly task id from envs.registry.TASKS.
    task: str = 'empty5x5'
    # Base RNG seed for the run.
    seed: int = 0
    # 'partial' = egocentric agent view; 'full' / 'fully_obs' = full
    # RGB map. In both cases the table still hashes raw pixels.
    obs_mode: str = 'partial'
    # MiniGrid partial-view size. Must be odd and >= 3.
    agent_view_size: int = 7
    # RGB pixels per grid cell. 0 means auto-scale near
    # obs_target_size so images stay around 56x56.
    obs_tile_size: int = 0
    obs_target_size: int = 56
    # Total environment steps to train for.
    total_timesteps: int = 1_000_000
    # Q-learning step size (alpha).
    learning_rate: float = 0.1
    # Discount factor.
    gamma: float = 0.99
    # Epsilon-greedy exploration, annealed linearly from start to end
    # over the first `exploration_fraction` of training.
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.5
    # Console print frequency, in environment steps.
    log_interval: int = 2000


def obs_key(obs):
    """
    Turn an image observation into a hashable table key.

    The raw pixel bytes are used directly: two identical views map to
    the same key, but any single-pixel difference produces a brand-new
    key. That brittleness is exactly why a table cannot represent
    image observations -- it is the failure this baseline exhibits.
    """

    return obs.tobytes()


def train(args):
    """
    Train a tabular Q-learning agent on the configured task.

    Standard epsilon-greedy Q-learning over a dictionary Q-table keyed
    by hashed image bytes, with extra logging (table size, new-state
    rate) that exposes why the table approach breaks on image inputs.
    """

    run_name = f'{args.task}__qlearn__{args.seed}__{int(time.time())}'

    # Anchor logs to the repo root regardless of launch directory.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    run_dir = os.path.join(repo_root, 'results', 'runs', run_name)
    writer = SummaryWriter(run_dir)
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n'
        + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()),
    )

    # Seed the RNGs for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # A single environment is enough for tabular learning. Wrap it so
    # episode return and length are recorded automatically on the
    # step that ends an episode.
    env = build_env(
        args.task,
        seed=args.seed,
        obs_mode=args.obs_mode,
        agent_view_size=args.agent_view_size,
        obs_tile_size=args.obs_tile_size,
        obs_target_size=args.obs_target_size,
    )
    env = RecordEpisodeStatistics(env)
    n_actions = env.action_space.n
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo='qlearn',
        args=args,
        obs_shape=env.observation_space.shape,
        action_space=env.action_space,
        unwrapped_env=env.unwrapped,
        extra=runtime_metadata(),
    )

    # The Q-table: a dict mapping a hashed observation to one Q-value
    # per action. defaultdict lazily creates an all-zero row the first
    # time a state is seen -- so len(q) is exactly the number of
    # distinct states encountered so far.
    q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    # Rolling window tracking how often the current state is brand new.
    recent_new = deque(maxlen=1000)
    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    start_time = tracker.start_time
    obs, _ = env.reset(seed=args.seed)
    key = obs_key(obs)

    for step in range(1, args.total_timesteps + 1):

        # Linearly anneal epsilon over the first part of training.
        frac = min(
            1.0, step / (args.exploration_fraction * args.total_timesteps)
        )
        epsilon = args.start_epsilon + frac * (
            args.end_epsilon - args.start_epsilon
        )

        # Record whether this state was already in the table before we
        # touch it, so new_state_rate reflects genuine first visits.
        recent_new.append(0.0 if key in q else 1.0)

        # Epsilon-greedy action selection. Accessing q[key] for the
        # argmax creates the row if it did not exist yet.
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q[key]))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_key = obs_key(next_obs)

        # Q-learning update. The bootstrap value is zero at a terminal
        # state, and also zero for an as-yet-unseen next state (its
        # default row is all zeros). Crucially we check membership
        # with `not in` instead of indexing q[next_key], because
        # indexing would create the row and make the new_state_rate
        # metric below always read zero.
        if done or next_key not in q:
            best_next = 0.0
        else:
            best_next = float(np.max(q[next_key]))
        td_target = reward + args.gamma * best_next
        q[key][action] += args.learning_rate * (
            td_target - q[key][action]
        )

        # Log finished episodes. A single env does not auto-reset, so
        # we reset here and start the next episode from a fresh state.
        if done:
            ep_return = float(info['episode']['r'])
            writer.add_scalar(
                'charts/episodic_return', ep_return, step
            )
            writer.add_scalar(
                'charts/episodic_length',
                float(info['episode']['l']),
                step,
            )
            writer.add_scalar(
                'charts/episodic_success',
                1.0 if ep_return > 0 else 0.0,
                step,
            )
            recent_returns.append(ep_return)
            recent_successes.append(1.0 if ep_return > 0 else 0.0)
            tracker.record_episode(
                step, 0, ep_return, float(info['episode']['l'])
            )

            next_obs, _ = env.reset()
            next_key = obs_key(next_obs)

        key = next_key

        # Periodic diagnostics. The two signals that expose the
        # failure are qtable_size and new_state_rate.
        if step % args.log_interval == 0:
            new_rate = (
                sum(recent_new) / len(recent_new)
                if recent_new
                else 1.0
            )
            sps = int(step / (time.time() - start_time))
            writer.add_scalar('charts/qtable_size', len(q), step)
            writer.add_scalar('charts/new_state_rate', new_rate, step)
            writer.add_scalar('charts/epsilon', epsilon, step)
            writer.add_scalar('charts/SPS', sps, step)
            writer.add_scalar(
                'charts/wall_time_sec', tracker.elapsed(), step
            )
            writer.add_scalar(
                'charts/total_episodes', tracker.episode_count, step
            )
            if recent_returns:
                avg_succ = sum(recent_successes) / len(recent_successes)
                avg_ret = sum(recent_returns) / len(recent_returns)
                writer.add_scalar(
                    'charts/success_rate_recent_100', avg_succ, step
                )
                progress = (
                    f'success {avg_succ:4.2f} return {avg_ret:6.3f}'
                )
            else:
                progress = 'success  --  return    --'
            print(
                f'step {step}/{args.total_timesteps} '
                f'SPS {sps} | table {len(q):>7} '
                f'new {new_rate:4.2f} | {progress}'
            )
            tracker.write_summary(
                status='running',
                global_step=step,
                extra={
                    'qtable_size': len(q),
                    'new_state_rate': new_rate,
                    'epsilon': epsilon,
                },
            )

    tracker.close(
        status='completed',
        global_step=args.total_timesteps,
        extra={'qtable_size': len(q)},
    )
    env.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
