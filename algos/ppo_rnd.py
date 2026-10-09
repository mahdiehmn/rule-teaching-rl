"""
PPO + Random Network Distillation (RND).

This is the curiosity baseline the VLM/LLM teacher must beat. It is a
diff against algos/ppo.py: the same PPO loop, plus an unsupervised
*intrinsic reward* that pushes the agent toward novel observations.

The idea in one paragraph
-------------------------
RND keeps two networks that both map an observation to a feature
vector: a fixed, randomly initialized *target* and a trained
*predictor*. The predictor is trained to copy the target's output on
states the agent visits. On a familiar state the predictor matches
the target well (low error); on a novel state it has not learned the
mapping yet (high error). That prediction error is handed to the
agent as a bonus reward, so "go somewhere I cannot yet predict" =
"go somewhere new". No labels, no human, no language -- which is
exactly why beating it is the bar that proves *semantic* guidance
adds something curiosity alone does not.

What changes vs. vanilla PPO (the five differences)
---------------------------------------------------
1. Two value heads. The agent now predicts two values per state: an
   *extrinsic* value (task reward, episodic) and an *intrinsic*
   value (curiosity reward, treated as non-episodic). They are
   advantaged separately and combined.
2. An RND module (target + predictor) that produces the intrinsic
   reward as the predictor's squared error.
3. Observation normalization. RND only works if the inputs to its
   networks are standardized, so we track a running mean/std over
   observations and normalize before feeding the RND nets. The stats
   are warmed up with a short random rollout before training.
4. Intrinsic-reward normalization. The raw prediction error has an
   arbitrary scale, so we divide it by the running std of the
   (discounted) intrinsic returns to keep it comparable to the task
   reward.
5. An extra loss term. The predictor is trained alongside the policy
   by adding its regression loss to the total loss.

Run it the same way as the PPO baseline:

    python -m algos.ppo_rnd --task empty5x5
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
import torch.nn.functional as F
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
    Command-line arguments for a PPO+RND run.

    Shares the core PPO knobs with algos/ppo.py and adds the RND-
    specific ones at the bottom. tyro exposes every field as a
    `--field value` flag.
    """

    # --- Experiment identity and logging ---
    task: str = 'empty5x5'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
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

    # --- Core PPO hyperparameters (same meaning as in ppo.py) ---
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    # Extrinsic discount. RND papers use a longer horizon (0.999) for
    # the task reward, since curiosity handles short-term novelty.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- RND-specific hyperparameters ---
    # Discount for the (non-episodic) intrinsic value stream.
    int_gamma: float = 0.99
    # Weights blending the two advantage streams into one. Intrinsic
    # is weighted lower so task reward dominates once it appears.
    int_coef: float = 1.0
    ext_coef: float = 2.0
    # Fraction of each minibatch used to train the predictor, which
    # subsamples the regression target to slow predictor learning.
    update_proportion: float = 0.25
    # Number of rollouts of random actions used to warm up the
    # observation normalization stats before training begins.
    num_iterations_obs_norm_init: int = 5

    # --- Greedy evaluation (monitoring/eval.py) ---
    # The teacher-off greedy eval every other algorithm in the
    # benchmark reports. This file previously logged only
    # `success_rate_recent_100`, which is measured under the
    # STOCHASTIC training policy while it is chasing the intrinsic
    # reward -- not the same quantity as the teacher-guided runs'
    # `eval_success_rate`, so the two were never comparable and any
    # table putting them side by side understated this baseline.
    # 0 disables.
    eval_interval: int = 50
    eval_episodes: int = 10
    # Also evaluate the SAMPLED (non-argmax) policy on the same
    # episodes; see algos/ppo.py's eval block for why.
    eval_sampled: bool = True

    # --- Derived at runtime in train(); not set on the CLI ---
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill its
    bias, the standard PPO initialization.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """
    Online mean and variance via Welford's parallel algorithm.

    Used to standardize observations (and intrinsic returns) without
    storing all the data: it updates running statistics from each new
    batch. This is the normalization RND depends on to behave.
    """

    def __init__(self, epsilon=1e-4, shape=()):
        """
        Start with a zero mean, unit variance, and a tiny pseudo-count
        so the very first update is well defined.
        """

        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        """
        Fold a new batch (first axis is the batch axis) into the
        running statistics.
        """

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Combine the existing statistics with a batch's moments. This
        is the Welford merge step that keeps mean/var exact.
        """

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = (
            m_a
            + m_b
            + delta**2 * self.count * batch_count / tot_count
        )
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class RewardForwardFilter:
    """
    Compute a running, forward-discounted sum of intrinsic rewards.

    Used only to estimate the scale (std) of the intrinsic return so
    the intrinsic reward can be normalized. It is not the reward the
    agent learns from directly.
    """

    def __init__(self, gamma):
        """
        Store the discount and start with no accumulated return.
        """

        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        """
        Discount the running total and add this step's rewards.
        """

        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class Agent(nn.Module):
    """
    Shared-CNN actor-critic with two value heads for RND.

    Identical to the ppo.py agent except the single critic is split
    into an extrinsic head (task value) and an intrinsic head
    (curiosity value), since RND learns two value functions.
    """

    def __init__(self, envs):
        """
        Build the shared encoder, the actor head, and the two critic
        heads, sized to the env's observation and action spaces.
        """

        super().__init__()
        h, w, c = envs.single_observation_space.shape

        # Shared convolutional encoder (same as the PPO baseline).
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.network(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )

        # Actor head over the discrete actions.
        self.actor = layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01
        )
        # Two critic heads: extrinsic (task) and intrinsic (novelty).
        self.critic_ext = layer_init(nn.Linear(512, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(512, 1), std=0.01)

    def _encode(self, x):
        """
        Encode a batch of (N, H, W, C) uint8 images into features.
        """

        x = x.permute(0, 3, 1, 2) / 255.0
        return self.fc(self.network(x))

    def get_value(self, x):
        """
        Return both value estimates (extrinsic, intrinsic) for states.
        """

        hidden = self._encode(x)
        return self.critic_ext(hidden), self.critic_int(hidden)

    def get_action_and_value(self, x, action=None):
        """
        Sample (or score) an action and return it with its log
        probability, entropy, and both value estimates.
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
            self.critic_ext(hidden),
            self.critic_int(hidden),
        )


class RNDModel(nn.Module):
    """
    The Random Network Distillation target and predictor networks.

    Both map a normalized observation to a 512-d feature vector. The
    target is randomly initialized and frozen; the predictor is
    trained to match it. Their squared difference on a state is that
    state's novelty -- large when the predictor has not seen states
    like it before.
    """

    def __init__(self, c, h, w):
        """
        Build the frozen target and the trainable predictor, sizing
        the linear layers from a dummy convolution pass.
        """

        super().__init__()

        # Convolutional trunk shared in shape by both networks.
        def conv_stack():
            return nn.Sequential(
                layer_init(nn.Conv2d(c, 32, 8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            n_flatten = conv_stack()(
                torch.zeros(1, c, h, w)
            ).shape[1]

        # Target: conv trunk to a single linear projection. Frozen.
        self.target = nn.Sequential(
            conv_stack(),
            layer_init(nn.Linear(n_flatten, 512)),
        )
        # Predictor: same trunk plus extra layers, so it has the
        # capacity to fit the target on seen states.
        self.predictor = nn.Sequential(
            conv_stack(),
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Freeze the target: it is a fixed random function, never
        # trained. Only the predictor chases it.
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        """
        Return (predictor_features, target_features) for the given
        already-normalized, channels-first observation batch.
        """

        return self.predictor(obs), self.target(obs)


def train(args):
    """
    Train a PPO+RND agent and log to TensorBoard.

    Same skeleton as algos/ppo.train, with the five RND additions
    noted in the module docstring folded into the rollout, advantage,
    and update phases.
    """

    # Derive batch sizes from the core knobs.
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f'{args.task}__ppo_rnd__{args.seed}__{int(time.time())}'

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

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

    # Seed everything for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Cap CPU threads to the SLURM allocation so many array jobs
    # sharing one node do not oversubscribe it -- see algos/ppo.py's
    # seeding block for the full reasoning. No-op outside SLURM.
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        torch.set_num_threads(int(slurm_cpus))

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # Build the vectorized envs and the episode-stats wrapper.
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
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), 'this PPO+RND baseline only supports discrete actions'

    obs_shape = envs.single_observation_space.shape
    h, w, c = obs_shape
    run_extra = runtime_metadata()
    run_extra['obs_norm_init_frames'] = (
        args.num_steps
        * args.num_iterations_obs_norm_init
        * args.num_envs
    )
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo='ppo_rnd',
        args=args,
        obs_shape=obs_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=run_extra,
    )

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(c, h, w).to(device)

    # One optimizer trains both the agent and the RND predictor; the
    # frozen target is excluded because its params need no gradient.
    optimizer = optim.Adam(
        list(agent.parameters())
        + list(rnd_model.predictor.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # Running stats for observation and intrinsic-reward
    # normalization, plus the forward filter that estimates intrinsic
    # return scale. obs stats are kept channels-first to match the
    # RND network inputs.
    obs_rms = RunningMeanStd(shape=(c, h, w))
    reward_rms = RunningMeanStd()
    reward_filter = RewardForwardFilter(args.int_gamma)

    def normalize_obs(obs_tensor):
        """
        Standardize a (N, H, W, C) image batch for the RND networks.

        Converts to channels-first, subtracts the running mean,
        divides by the running std, and clips to [-5, 5] so rare
        extreme pixels cannot blow up the prediction error.
        """

        obs_cf = obs_tensor.permute(0, 3, 1, 2)
        mean = torch.tensor(
            obs_rms.mean, device=device, dtype=torch.float32
        )
        std = torch.tensor(
            np.sqrt(obs_rms.var), device=device, dtype=torch.float32
        )
        return torch.clamp((obs_cf - mean) / std, -5.0, 5.0)

    # Rollout storage, now with two value streams and a curiosity
    # reward buffer alongside the task reward buffer.
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = tracker.start_time
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    # Most recent greedy-eval stats, carried between iterations so
    # the run summary always reports the latest measurement.
    last_eval = None

    # --- Warm up observation normalization ---
    # Step random actions for a few rollouts and fit obs_rms on what
    # we see, so the RND networks get sensibly scaled inputs from the
    # first real update. Without this RND's error signal is noise.
    print('initializing observation normalization stats...')
    init_obs = []
    for _ in range(
        args.num_steps * args.num_iterations_obs_norm_init
    ):
        acts = np.array(
            [
                envs.single_action_space.sample()
                for _ in range(args.num_envs)
            ]
        )
        s, _, _, _, _ = envs.step(acts)
        # Store channels-first so it matches obs_rms's shape.
        init_obs.append(s.transpose(0, 3, 1, 2))
        if len(init_obs) >= 128:
            obs_rms.update(np.concatenate(init_obs, axis=0))
            init_obs = []
    if init_obs:
        obs_rms.update(np.concatenate(init_obs, axis=0))

    # The outer loop: collect a rollout, then update.
    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # --- Rollout collection ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Act, and read both value heads.
            with torch.no_grad():
                action, logprob, _, ext_v, int_v = (
                    agent.get_action_and_value(next_obs)
                )
                ext_values[step] = ext_v.flatten()
                int_values[step] = int_v.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step the envs to get the next observation and task
            # reward.
            next_obs, reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Curiosity reward: the predictor's error on the new
            # observation. High error = novel = bonus. Detached
            # because this is a reward, not part of the policy graph.
            with torch.no_grad():
                rnd_obs = normalize_obs(next_obs)
                predict_f, target_f = rnd_model(rnd_obs)
                curiosity_rewards[step] = (
                    ((target_f - predict_f).pow(2).sum(dim=1)) / 2.0
                )

            # Log finished episodes (task reward only -- the agent's
            # actual objective).
            if 'episode' in infos:
                mask = infos['_episode']
                for i in range(args.num_envs):
                    if not mask[i]:
                        continue
                    ep_return = float(infos['episode']['r'][i])
                    writer.add_scalar(
                        'charts/episodic_return',
                        ep_return,
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_length',
                        float(infos['episode']['l'][i]),
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_success',
                        1.0 if ep_return > 0 else 0.0,
                        global_step,
                    )
                    recent_returns.append(ep_return)
                    recent_successes.append(
                        1.0 if ep_return > 0 else 0.0
                    )
                    tracker.record_episode(
                        global_step,
                        i,
                        ep_return,
                        float(infos['episode']['l'][i]),
                    )

        # --- Normalize the intrinsic reward ---
        # Estimate the scale of the discounted intrinsic return and
        # divide the curiosity rewards by its std, so novelty stays
        # comparable to task reward instead of swamping it.
        curiosity_np = curiosity_rewards.cpu().numpy()
        per_step_returns = np.array(
            [reward_filter.update(r) for r in curiosity_np]
        )
        reward_rms.update(per_step_returns.reshape(-1))
        curiosity_rewards /= np.sqrt(reward_rms.var) + 1e-8

        # --- Two-stream advantage estimation (GAE) ---
        # The extrinsic stream respects episode boundaries; the
        # intrinsic stream is treated as never-ending (novelty does
        # not "reset"), so its non-terminal factor is always 1.
        with torch.no_grad():
            next_ext_v, next_int_v = agent.get_value(next_obs)
            next_ext_v = next_ext_v.reshape(1, -1)
            next_int_v = next_int_v.reshape(1, -1)
            ext_adv = torch.zeros_like(rewards).to(device)
            int_adv = torch.zeros_like(curiosity_rewards).to(device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nonterm = 1.0 - next_done
                    ext_next = next_ext_v
                    int_next = next_int_v
                else:
                    ext_nonterm = 1.0 - dones[t + 1]
                    ext_next = ext_values[t + 1]
                    int_next = int_values[t + 1]
                # Intrinsic transitions are always "non-terminal".
                int_nonterm = 1.0
                ext_delta = (
                    rewards[t]
                    + args.gamma * ext_next * ext_nonterm
                    - ext_values[t]
                )
                int_delta = (
                    curiosity_rewards[t]
                    + args.int_gamma * int_next * int_nonterm
                    - int_values[t]
                )
                ext_adv[t] = ext_lastgaelam = (
                    ext_delta
                    + args.gamma
                    * args.gae_lambda
                    * ext_nonterm
                    * ext_lastgaelam
                )
                int_adv[t] = int_lastgaelam = (
                    int_delta
                    + args.int_gamma
                    * args.gae_lambda
                    * int_nonterm
                    * int_lastgaelam
                )
            ext_returns = ext_adv + ext_values
            int_returns = int_adv + int_values

        # Flatten the rollout into one big batch.
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_ext_adv = ext_adv.reshape(-1)
        b_int_adv = int_adv.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)

        # Blend the two advantage streams into the single signal the
        # policy is optimized against.
        b_advantages = (
            b_ext_adv * args.ext_coef + b_int_adv * args.int_coef
        )

        # Pre-compute the normalized observations the predictor will
        # be trained on (channels-first, standardized).
        b_obs_norm = normalize_obs(b_obs)

        # --- PPO + predictor update ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, args.batch_size, args.minibatch_size
            ):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Predictor regression loss on this minibatch's
                # observations. A random mask trains the predictor on
                # only a fraction of samples, slowing it so novelty
                # signal persists.
                predict_f, target_f = rnd_model(b_obs_norm[mb_inds])
                forward_loss = F.mse_loss(
                    predict_f, target_f.detach(), reduction='none'
                ).mean(dim=-1)
                mask = (
                    torch.rand(len(forward_loss), device=device)
                    < args.update_proportion
                ).float()
                forward_loss = (forward_loss * mask).sum() / torch.clamp(
                    mask.sum(), min=1.0
                )

                # Recompute policy quantities under the current net.
                _, newlogprob, entropy, new_ext_v, new_int_v = (
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

                with torch.no_grad():
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (
                        mb_adv.std() + 1e-8
                    )

                # Clipped policy loss against the blended advantage.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Separate value loss for each head, then summed.
                new_ext_v = new_ext_v.view(-1)
                new_int_v = new_int_v.view(-1)
                ext_v_loss = (
                    0.5
                    * ((new_ext_v - b_ext_returns[mb_inds]) ** 2).mean()
                )
                int_v_loss = (
                    0.5
                    * ((new_int_v - b_int_returns[mb_inds]) ** 2).mean()
                )
                v_loss = ext_v_loss + int_v_loss

                entropy_loss = entropy.mean()
                # Total loss: policy + value + entropy (as in PPO),
                # plus the predictor's regression loss.
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + forward_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.parameters())
                    + list(rnd_model.predictor.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

        # Refresh the observation normalization stats with the
        # observations just collected, so it tracks the changing
        # state distribution as the policy improves.
        obs_rms.update(
            b_obs.permute(0, 3, 1, 2).cpu().numpy()
        )

        # --- Greedy teacher-off evaluation ---
        # The identical fixed-episode greedy eval algos/ppo.py and
        # every teacher-guided variant run (monitoring/eval.py), so
        # this baseline's curve can finally be compared with theirs
        # on the same quantity rather than against its training-time
        # success rate.
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

            # The same episodes under the SAMPLED policy, for the
            # reason spelled out in algos/ppo.py's eval block: a
            # deterministic argmax policy can cycle and report 0.00
            # for a policy that has clearly learned something.
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
        writer.add_scalar(
            'charts/learning_rate',
            optimizer.param_groups[0]['lr'],
            global_step,
        )
        writer.add_scalar(
            'losses/ext_value_loss', ext_v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/int_value_loss', int_v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/policy_loss', pg_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/rnd_predictor_loss',
            forward_loss.item(),
            global_step,
        )
        writer.add_scalar(
            'losses/entropy', entropy_loss.item(), global_step
        )
        writer.add_scalar(
            'charts/mean_curiosity_reward',
            curiosity_rewards.mean().item(),
            global_step,
        )

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
            'ext_value_loss': ext_v_loss.item(),
            'int_value_loss': int_v_loss.item(),
            'policy_loss': pg_loss.item(),
            'rnd_predictor_loss': forward_loss.item(),
            'entropy': entropy_loss.item(),
            'mean_curiosity_reward': curiosity_rewards.mean().item(),
            'clipfrac': float(np.mean(clipfracs)),
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
        if recent_returns:
            avg_succ = sum(recent_successes) / len(recent_successes)
            avg_ret = sum(recent_returns) / len(recent_returns)
            progress = f'success {avg_succ:4.2f} return {avg_ret:6.3f}'
        else:
            progress = 'success  --  return    --'
        print(
            f'iter {iteration}/{args.num_iterations} '
            f'step {global_step} SPS {sps} | {progress}'
        )

    tracker.close(status='completed', global_step=global_step)
    # Save the trained policy next to the run summary, matching
    # algos/ppo.py, so this curiosity baseline can also serve as a
    # control for scripts/probe_subgoal.py without being retrained.
    torch.save(agent.state_dict(), os.path.join(run_dir, 'agent.pt'))
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
