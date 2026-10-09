"""
Vanilla PPO baseline (cleanrl-style, single file).

This is the unguided, expensive bound: PPO on the partial-observation
image with no intrinsic reward and no teacher. On the hard tasks it is
expected to need very many frames or to plateau near zero -- that is
the point of including it. On the trivial `empty5x5` task it should
solve quickly, which is how we validate the whole training loop
(README milestone 4) before writing any other algorithm.

The RND, tabular, and teacher variants are all diffs against this
file, so it is kept deliberately plain and self-contained.

Run it from the repo root with the venv active:

    python -m algos.ppo --task empty5x5

Metrics (episodic return, episodic success, losses) are written to
TensorBoard under results/runs/<run-name>. View them with:

    tensorboard --logdir results/runs
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.registry import build_env, make_thunk
from monitoring.eval import greedy_eval
from monitoring.metrics import RunTracker, runtime_metadata


@dataclass
class Args:
    """
    Command-line arguments for a PPO run.

    Defaults are tuned as sensible starting points for MiniGrid image
    observations on CPU; tyro turns this dataclass into a typed CLI,
    so every field below is settable as `--field value`.
    """

    # --- Experiment identity and logging ---
    # Friendly task id from envs.registry.TASKS.
    task: str = 'empty5x5'
    # Base RNG seed for the run.
    seed: int = 0
    # When True, make cuDNN deterministic for reproducibility.
    torch_deterministic: bool = True
    # Use the GPU if one is available; harmless to leave True on CPU.
    cuda: bool = True
    # When True, mirror metrics to Weights & Biases in addition to
    # TensorBoard. Off by default so the loop runs without a login.
    track: bool = False
    # W&B project name, only used when track is True.
    wandb_project: str = 'vlm-rl-bench'

    # --- Observation settings ---
    # 'historical' = the accumulated fog-of-war map (what the agent
    # has seen so far); 'partial' = egocentric agent view only;
    # 'full' / 'fully_obs' = the whole map. In every case the agent
    # still receives only pixels. 'historical' is the project-wide
    # default every reported result uses -- keep it in sync with the
    # other algos and with scripts/submit_cc.sh.
    obs_mode: str = 'historical'
    # MiniGrid partial-view size. Must be odd and >= 3.
    agent_view_size: int = 7
    # RGB pixels per grid cell. 0 means auto-scale near
    # obs_target_size so the CNN input stays around 56x56.
    obs_tile_size: int = 0
    obs_target_size: int = 56

    # --- Core PPO hyperparameters ---
    # Total environment frames to train for, summed across all envs.
    total_timesteps: int = 1_000_000
    # Optimizer step size.
    learning_rate: float = 2.5e-4
    # Number of parallel environments collected synchronously.
    num_envs: int = 8
    # Steps collected per env before each policy update; the rollout
    # length. batch_size = num_envs * num_steps.
    num_steps: int = 128
    # Linearly decay the learning rate to zero over training.
    anneal_lr: bool = True
    # Discount factor.
    gamma: float = 0.99
    # GAE lambda, trading bias against variance in the advantage.
    gae_lambda: float = 0.95
    # Number of minibatches each rollout is split into per update.
    num_minibatches: int = 4
    # Optimization epochs over each rollout.
    update_epochs: int = 4
    # Normalize advantages per minibatch.
    norm_adv: bool = True
    # PPO clipping coefficient on the policy ratio.
    clip_coef: float = 0.2
    # Also clip the value loss, as in the original PPO implementation.
    clip_vloss: bool = True
    # Entropy bonus weight, encouraging exploration.
    ent_coef: float = 0.01
    # Value loss weight.
    vf_coef: float = 0.5
    # Gradient norm clip threshold.
    max_grad_norm: float = 0.5
    # Early-stop an update if the approximate KL exceeds this. None
    # disables the check.
    target_kl: float | None = None

    # --- Greedy evaluation (monitoring/eval.py) ---
    # Every this many iterations, run a fixed set of greedy-policy
    # episodes and log charts/eval_*. This baseline has no teacher,
    # so its training curve is already "teacher-off"; the greedy
    # eval exists so every method in the benchmark (this floor
    # included) reports the exact same evaluation curve, per the
    # consistency protocol in docs/stage1_distill.md. 0 disables.
    eval_interval: int = 50
    # Modest count because a weak early policy times out every
    # episode, making eval cost ~ max_steps * episodes.
    eval_episodes: int = 10
    # Also evaluate the SAMPLED (non-argmax) policy on the same
    # episodes, logged as charts/eval_sampled_success_rate. See the
    # eval block in train() for why a greedy-only measurement can
    # read 0.00 for a policy that has clearly learned something.
    eval_sampled: bool = True

    # --- Derived at runtime in train(); not set on the CLI ---
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill its
    bias.

    Orthogonal initialization with a per-layer gain is the standard
    PPO initialization and noticeably stabilizes early training. The
    actor head uses a tiny std so the initial policy is close to
    uniform; the critic head uses std=1.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Shared-CNN actor-critic for MiniGrid image observations.

    A small convolutional encoder maps the partial-observation RGB
    image to a feature vector, which feeds two linear heads: the actor
    (a categorical policy over the discrete actions) and the critic (a
    scalar state value). The encoder is shared, which is the usual
    choice for image-based PPO.
    """

    def __init__(self, envs):
        """
        Build the encoder and heads sized to the env's spaces.

        The flattened convolution output size is computed from a dummy
        forward pass so the network adapts to whatever image shape the
        wrappers produce, rather than hard-coding 56x56.
        """

        super().__init__()

        # Observations arrive as (H, W, C) uint8; record the channel
        # count and spatial size to build and size the encoder.
        h, w, c = envs.single_observation_space.shape

        # An Atari-style three-layer CNN, which is ample for the small
        # MiniGrid view. Inputs are permuted to (N, C, H, W) and
        # normalized to [0, 1] in the forward helpers below.
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Run a zero image through the conv stack once to discover the
        # flattened feature dimension without hard-coding it.
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.network(dummy).shape[1]

        # Project the conv features to a 512-d hidden representation
        # shared by both heads.
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )

        # Actor head: logits over the discrete action set. Small init
        # std keeps the starting policy near-uniform.
        self.actor = layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01
        )
        # Critic head: a single state-value scalar.
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def _encode(self, x):
        """
        Encode a batch of (N, H, W, C) uint8 images into features.
        """

        # Permute to channels-first and scale pixel values to [0, 1].
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.fc(self.network(x))

    def get_value(self, x):
        """
        Return the critic's value estimate for a batch of states.
        """

        return self.critic(self._encode(x))

    def get_action_and_value(self, x, action=None):
        """
        Sample (or evaluate) an action and return it alongside its log
        probability, the policy entropy, and the state value.

        When `action` is None a fresh action is sampled from the
        policy; otherwise the provided action is scored, which is what
        the PPO update needs when recomputing log-probs under the
        current policy.
        """

        hidden = self._encode(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
        )


def train(args):
    """
    Train a PPO agent on the configured task and log to TensorBoard.

    This is the canonical cleanrl PPO loop: collect a fixed-length
    rollout from the vectorized envs, compute GAE advantages, then run
    several clipped-surrogate optimization epochs over minibatches of
    that rollout. Episodic return and success are logged so the
    success-rate-vs-frames curve can be read directly.
    """

    # Fill in the sizes derived from the core hyperparameters.
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # A unique, sortable run name keyed by task, algo, seed, and time.
    run_name = f'{args.task}__ppo__{args.seed}__{int(time.time())}'

    # Optionally mirror everything to Weights & Biases.
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

    # TensorBoard is the primary logger; runs live under results/.
    # Anchor the path to the repo root (the parent of this algos/
    # package) so logs always land in the repo's results/ folder no
    # matter which directory the script was launched from.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    run_dir = os.path.join(repo_root, 'results', 'runs', run_name)
    writer = SummaryWriter(run_dir)
    # Record the full hyperparameter set as a markdown table.
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n'
        + '\n'.join(
            f'|{k}|{v}|' for k, v in vars(args).items()
        ),
    )

    # Seed every RNG so a run is reproducible end to end.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # PyTorch defaults to spawning as many CPU threads as the
    # machine has cores for intra-op parallelism. On a shared HPC
    # node running many array jobs at once, each job is only
    # allocated --cpus-per-task cores via SLURM, but torch does not
    # know that on its own -- left unset, every job tries to use
    # every core on the node, and the resulting thread contention
    # (not the model itself) can make training an order of
    # magnitude slower than running alone. Capping to the actual
    # SLURM allocation avoids that; SLURM_CPUS_PER_TASK is unset
    # outside a SLURM job, so this is a no-op on a local machine.
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        torch.set_num_threads(int(slurm_cpus))

    # Select the device. torch is CPU-only here, but honoring the
    # flag keeps the file portable to a CUDA machine unchanged.
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # Build the synchronous vector of environments from the registry
    # thunks, then wrap it so episodic return and length are recorded
    # automatically in the step info.
    sync_envs = gym.vector.SyncVectorEnv(
        [
            make_thunk(
                args.task,
                args.seed,
                i,
                obs_mode=args.obs_mode,
                agent_view_size=args.agent_view_size,
                obs_tile_size=args.obs_tile_size,
                obs_target_size=args.obs_target_size,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = RecordEpisodeStatistics(sync_envs)

    # PPO assumes a discrete action space here; fail early otherwise.
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), 'this PPO baseline only supports discrete actions'

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    # Rollout storage. One slot per (step, env) for each quantity GAE
    # and the PPO update consume.
    obs_shape = envs.single_observation_space.shape
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo='ppo',
        args=args,
        obs_shape=obs_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=runtime_metadata(),
    )
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the clock and reset the envs to get the first observation.
    global_step = 0
    start_time = tracker.start_time
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Keep a rolling window of the most recent episode outcomes so the
    # console can show how learning is progressing, not just the
    # speed. 100 episodes is enough to smooth out the per-episode
    # noise without lagging far behind the current policy.
    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    # Most recent greedy-eval stats, carried between iterations so
    # the summary always reports the latest measurement.
    last_eval = None

    # The outer loop: one iteration collects a rollout and updates.
    for iteration in range(1, args.num_iterations + 1):

        # Linearly anneal the learning rate toward zero across
        # training, a standard PPO schedule that aids late-stage
        # stability.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # --- Rollout collection ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Act under the current policy without tracking gradients;
            # the update recomputes log-probs later.
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step every env. gymnasium 1.x auto-resets a finished env
            # on the following step, so terminal observations are
            # handled for us; we only need the termination flags.
            next_obs, reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = (
                torch.tensor(reward).to(device).view(-1)
            )
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Log any episodes that finished this step. The vector
            # RecordEpisodeStatistics wrapper exposes per-env episode
            # stats under 'episode' with a boolean mask '_episode'.
            # A positive MiniGrid return means the goal was reached,
            # so return > 0 doubles as the success signal.
            if 'episode' in infos:
                mask = infos['_episode']
                for i in range(args.num_envs):
                    if not mask[i]:
                        continue
                    ep_return = float(infos['episode']['r'][i])
                    ep_length = float(infos['episode']['l'][i])
                    writer.add_scalar(
                        'charts/episodic_return',
                        ep_return,
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_length',
                        ep_length,
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_success',
                        1.0 if ep_return > 0 else 0.0,
                        global_step,
                    )
                    # Feed the rolling console window too.
                    recent_returns.append(ep_return)
                    recent_successes.append(
                        1.0 if ep_return > 0 else 0.0
                    )
                    tracker.record_episode(
                        global_step, i, ep_return, ep_length
                    )

        # --- Advantage estimation (GAE) ---
        # Bootstrap from the value of the state we are about to step
        # into, then sweep backward accumulating the GAE advantage.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # Flatten the rollout from (steps, envs, ...) to one big batch.
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # --- PPO update ---
        # Optimize the clipped surrogate objective for several epochs
        # over shuffled minibatches of the collected rollout.
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, args.batch_size, args.minibatch_size
            ):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Recompute log-probs, entropy, and values for the
                # minibatch actions under the current policy.
                _, newlogprob, entropy, newvalue = (
                    agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                # Clamp the log-ratio before exponentiating to guard
                # against an extreme importance ratio blowing the
                # update up into NaNs on long runs.
                logratio = torch.clamp(logratio, -10.0, 10.0)
                ratio = logratio.exp()

                # Track the approximate KL and clip fraction as
                # diagnostics of how far the update is moving.
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages within the minibatch to keep the
                # gradient scale stable.
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                # Clipped policy-gradient (surrogate) loss.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss, optionally clipped like the policy loss.
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_inds]
                    ) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (
                        v_clipped - b_returns[mb_inds]
                    ) ** 2
                    v_loss_max = torch.max(
                        v_loss_unclipped, v_loss_clipped
                    )
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = (
                        0.5
                        * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    )

                # Combine policy, value, and entropy terms.
                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                # Clip the global gradient norm before stepping.
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            # Stop early this iteration if the policy has moved too
            # far, when a target KL is configured.
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # --- Greedy evaluation ---
        # The same fixed-episode greedy eval every algorithm in the
        # benchmark runs (see monitoring/eval.py), so this floor's
        # eval curve is directly comparable to the teacher-guided
        # variants' teacher-off curves.
        run_eval = args.eval_interval > 0 and (
            iteration % args.eval_interval == 0
            or iteration == args.num_iterations
        )
        if run_eval:

            def select_action(o):
                with torch.no_grad():
                    t = torch.Tensor(o).unsqueeze(0).to(device)
                    logits_1 = agent.actor(agent._encode(t))
                    return int(logits_1.argmax(dim=-1).item())

            def sample_action(o):
                """
                Sample from the policy instead of taking its argmax.
                """

                with torch.no_grad():
                    t = torch.Tensor(o).unsqueeze(0).to(device)
                    logits_1 = agent.actor(agent._encode(t))
                    return int(
                        Categorical(logits=logits_1).sample().item()
                    )

            def make_eval_env():
                """
                Build a fresh eval env matching training's pipeline.
                """

                return build_env(
                    args.task,
                    seed=args.seed + 50_000,
                    obs_mode=args.obs_mode,
                    agent_view_size=args.agent_view_size,
                    obs_tile_size=args.obs_tile_size,
                    obs_target_size=args.obs_target_size,
                )

            last_eval = greedy_eval(
                select_action,
                make_env=make_eval_env,
                num_episodes=args.eval_episodes,
                seed_base=args.seed + 50_000,
            )

            # Re-run the SAME teacher-off episodes with the sampled
            # policy. An argmax policy is deterministic and can cycle
            # (turn left, turn right, forever) where the stochastic
            # policy it came from escapes, which reports as a flat
            # 0.00 however much was learned. The completed
            # DoorKey-8x8 runs in results/ look like exactly that:
            # eval_success_rate 0.00 on all five seeds while
            # success_rate_recent_100 sat at 0.10-0.70. Recording
            # both separates "learned nothing" from "the argmax of
            # this policy loops", which changes what a 0.00 in the
            # comparison tables is allowed to mean.
            if args.eval_sampled:
                sampled_eval = greedy_eval(
                    sample_action,
                    make_env=make_eval_env,
                    num_episodes=args.eval_episodes,
                    seed_base=args.seed + 50_000,
                )
                last_eval['sampled_success_rate'] = sampled_eval[
                    'success_rate'
                ]
                writer.add_scalar(
                    'charts/eval_sampled_success_rate',
                    sampled_eval['success_rate'],
                    global_step,
                )

            writer.add_scalar(
                'charts/eval_success_rate',
                last_eval['success_rate'],
                global_step,
            )
            writer.add_scalar(
                'charts/eval_return',
                last_eval['mean_return'],
                global_step,
            )
            writer.add_scalar(
                'charts/eval_length',
                last_eval['mean_length'],
                global_step,
            )

        # --- Per-iteration diagnostics ---
        # Explained variance reports how well the value function
        # predicts the returns; near 1 is good, near 0 is no better
        # than predicting the mean.
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan
            if var_y == 0
            else 1 - np.var(y_true - y_pred) / var_y
        )

        writer.add_scalar(
            'charts/learning_rate',
            optimizer.param_groups[0]['lr'],
            global_step,
        )
        writer.add_scalar(
            'losses/value_loss', v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/policy_loss', pg_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/entropy', entropy_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/approx_kl', approx_kl.item(), global_step
        )
        writer.add_scalar(
            'losses/clipfrac', np.mean(clipfracs), global_step
        )
        writer.add_scalar(
            'losses/explained_variance', explained_var, global_step
        )

        # Throughput in environment frames per second, a quick health
        # check on the loop and a comparable wall-clock measure.
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar('charts/SPS', sps, global_step)
        writer.add_scalar(
            'charts/wall_time_sec', tracker.elapsed(), global_step
        )
        writer.add_scalar(
            'charts/total_episodes', tracker.episode_count, global_step
        )
        if recent_successes:
            writer.add_scalar(
                'charts/success_rate_recent_100',
                sum(recent_successes) / len(recent_successes),
                global_step,
            )
        extra = {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'value_loss': v_loss.item(),
            'policy_loss': pg_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': float(np.mean(clipfracs)),
            'explained_variance': float(explained_var),
        }
        if last_eval is not None:
            extra['eval_success_rate'] = last_eval['success_rate']
            extra['eval_mean_return'] = last_eval['mean_return']
            if 'sampled_success_rate' in last_eval:
                extra['eval_sampled_success_rate'] = last_eval[
                    'sampled_success_rate'
                ]
        tracker.write_summary(
            status='running', global_step=global_step, extra=extra
        )
        # Summarize the last up-to-100 episodes for the console so the
        # learning trend is visible live. Until the first episodes
        # finish, show dashes instead of dividing by zero.
        if recent_returns:
            avg_succ = sum(recent_successes) / len(recent_successes)
            avg_ret = sum(recent_returns) / len(recent_returns)
            progress = (
                f'success {avg_succ:4.2f} return {avg_ret:6.3f}'
            )
        else:
            progress = 'success  --  return    --'
        print(
            f'iter {iteration}/{args.num_iterations} '
            f'step {global_step} SPS {sps} | {progress}'
        )

    tracker.close(status='completed', global_step=global_step)
    # Save the trained policy weights next to the run summary so the
    # sub-goal probe (scripts/probe_subgoal.py) can use this plain-PPO
    # student as the negative control: if the teacher's sub-goal is
    # decodable from the distillation student but not from this one,
    # that is evidence the teacher's reasoning was internalized.
    torch.save(agent.state_dict(), os.path.join(run_dir, 'agent.pt'))
    envs.close()
    writer.close()


if __name__ == '__main__':
    # tyro builds a typed CLI from the Args dataclass, so any field is
    # overridable as `--field value`.
    train(tyro.cli(Args))
