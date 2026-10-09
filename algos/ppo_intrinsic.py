"""
PPO with a pluggable exploration bonus, optional memory, and either
pixel or symbolic observations.

This file exists to make the project's "no teacher-free method can do
this" claim actually testable. The previous teacher-free baselines
were plain PPO and one RND implementation, both feed-forward, both on
rendered pixels, both at 2M frames. That configuration differs from
the published MiniGrid exploration results on four axes at once, so a
0.00 from it cannot distinguish "the task needs a teacher" from "this
particular baseline was under-powered". Each axis is a flag here:

    --bonus      none | count | rnd | noveld | re3 | e3b
    --recurrent  add a GRU core, for tasks whose difficulty is memory
    --obs_mode   symbolic (the literature's representation) vs the
                 repo default historical RGB
    --total-timesteps  the budget itself

Everything else -- the PPO loop, the greedy teacher-off evaluation,
the run naming, the metrics written -- is deliberately identical to
algos/ppo.py, so a run from this file drops straight into the
existing comparison tables and plots.

`--bonus none` is the matched control: same architecture, same
optimizer, same evaluation, no intrinsic reward. Prefer it over
algos/ppo.py when attributing a difference to the bonus, since it
differs from the bonus runs in exactly one thing.

Examples
--------
    # The strong pure-RL baseline: NovelD on symbolic observations.
    python -m algos.ppo_intrinsic --task keycorridor_s3r3 \
        --bonus noveld --obs-mode symbolic --total-timesteps 10000000

    # The cheapest useful check: RE3, nothing trained, 1M frames.
    python -m algos.ppo_intrinsic --task doorkey_8x8 --bonus re3

    # Memory diagnostic: same bonus, with and without the GRU.
    python -m algos.ppo_intrinsic --task keycorridor_s3r3 \
        --bonus noveld --recurrent

Metrics land in results/runs/<task>__ppo_<bonus>[_gru]__<seed>__<t>/
and are read by scripts/plot_runs.py unchanged.
"""

import hashlib
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from algos.intrinsic import (
    BONUSES,
    SMALL_INPUT_POOL,
    SMALL_INPUT_THRESHOLD,
    RunningMeanStd,
    build_bonus,
    layer_init,
)
from algos.nets import MissionEncoder
from envs.mission_vocab import MissionVocab
from envs.registry import BABYAI_LANGUAGE_TASKS, build_env, make_thunk
from monitoring.eval import greedy_eval
from monitoring.metrics import RunTracker, runtime_metadata

# Width of the shared trunk and of the recurrent core. Kept at 512 to
# match algos/ppo.py's hidden size so architecture is not a
# confounder when comparing against the existing runs.
HIDDEN = 512
# Embedding width per symbolic channel. MiniGrid's grid encoding is
# three small category indices per cell (object, color, state); each
# is embedded and the three vectors concatenated before the convs.
SYMBOL_EMBED_DIM = 8
# Upper bound on any MiniGrid category index (max object index is 10),
# with headroom so the table never needs resizing per task.
SYMBOL_VOCAB = 16


@dataclass
class Args:
    """
    Command-line arguments, shared with algos/ppo.py where the field
    means the same thing, plus the bonus/recurrence/observation knobs
    this file adds. tyro exposes every field as `--field value`.
    """

    # --- Experiment identity and logging ---
    task: str = 'empty5x5'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = 'vlm-rl-bench'
    experiment_id: str = ''
    record_initial_policy: bool = False

    # --- What this file adds ---
    # Which exploration bonus to add to the task reward. 'none' is
    # the architecture-matched plain-PPO control.
    bonus: str = 'none'
    # Add a GRU core so the policy can remember earlier observations.
    # This is the diagnostic for tasks whose difficulty is memory
    # (KeyCorridor hides the key) rather than exploration.
    recurrent: bool = False

    # --- Observation settings ---
    # 'symbolic' is MiniGrid's native integer grid encoding, which is
    # what the exploration literature trains on and what BabyAI 1.1
    # measured as up to 3x more sample-efficient than pixels.
    # 'historical' (the repo default everywhere else) is the
    # fog-of-war RGB render, kept as the default here so a run from
    # this file is comparable to the existing results by default.
    obs_mode: str = 'historical'
    agent_view_size: int = 7
    obs_tile_size: int = 0
    obs_target_size: int = 56
    # Feed the tokenized instruction to the policy AND to the bonus.
    # None auto-detects: on the BabyAI language tasks the mission is
    # the only thing that says which object to go to, so without it
    # the task is not merely hard but ill-posed -- the same
    # observation implies different correct actions depending on an
    # instruction the agent cannot see. Set True/False to override.
    obs_mission: Optional[bool] = None

    # --- Core PPO hyperparameters (same meaning as algos/ppo.py) ---
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    # Extrinsic discount. The RND line of work uses a long horizon
    # (0.999) for the task reward on sparse-reward mazes.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # --- Intrinsic-reward blending ---
    # Discount for the intrinsic value stream, which is treated as
    # non-episodic (novelty does not reset at an episode boundary).
    int_gamma: float = 0.99
    # Weights on the two advantage streams. The defaults follow the
    # RND paper; they are the knob to sweep if a bonus drowns out
    # the task reward or vanishes beneath it.
    int_coef: float = 1.0
    ext_coef: float = 2.0
    # Divide the raw bonus by the running std of intrinsic returns,
    # so different bonuses (whose raw scales differ by orders of
    # magnitude) are comparable under one int_coef.
    norm_int_reward: bool = True

    # --- Per-bonus knobs ---
    # RND / NovelD: fraction of each minibatch used to train the
    # predictor, which slows it so novelty persists past one visit.
    update_proportion: float = 0.25
    # NovelD: weight on the previous state's novelty in the
    # difference, and whether to apply the first-visit gate.
    noveld_alpha: float = 0.5
    noveld_episodic_gate: bool = True
    # RE3: which nearest neighbour defines the entropy estimate.
    re3_k: int = 3
    # E3B: the ridge lambda in C_0 = lambda * I.
    e3b_ridge: float = 0.1
    # Count: 'episodic' resets per episode, 'global' never resets.
    count_scope: str = 'episodic'

    # --- Greedy evaluation (monitoring/eval.py) ---
    # The teacher-off measurement every method in the benchmark
    # reports. It is the ONLY number comparable across methods --
    # the training-time success rate is measured under a stochastic
    # policy and, for teacher-guided runs, under teacher assistance.
    eval_interval: int = 50
    eval_episodes: int = 10
    # Also evaluate the SAMPLED (non-argmax) policy on the same
    # episodes. Cheap insurance against reading a cycling argmax
    # policy as a policy that learned nothing.
    eval_sampled: bool = True
    eval_frame_milestones: str = ''

    # --- Derived at runtime in train(); not set on the CLI ---
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class Agent(nn.Module):
    """
    Actor-critic with two value heads, an observation encoder that
    handles both pixels and symbolic grids, and an optional GRU core.

    There are always two critics (extrinsic and intrinsic) even when
    no bonus is configured; the intrinsic head is then simply trained
    on zeros and weighted out. Keeping one architecture means the
    `--bonus none` control differs from a bonus run in exactly the
    reward, not in parameter count.
    """

    def __init__(self, envs, recurrent, symbolic, obs_shape,
                 mission_vocab=0):
        """
        Build the encoder, the optional recurrent core, and the
        actor/critic heads, sized to the env's spaces.

        Parameters
        ----------
        recurrent: bool
            Insert a single-layer GRU between the trunk and the
            heads.
        symbolic: bool
            The observation is MiniGrid's integer grid encoding
            rather than a render, so each of the three channels is
            embedded instead of the array being scaled by 1/255.
        obs_shape: tuple
            The (H, W, C) image shape. Passed explicitly because on
            the language tasks the observation space is a Dict and
            `.shape` is not defined on it.
        mission_vocab: int
            Vocabulary size on the BabyAI language tasks, else 0. When
            non-zero, a GRU-summarized instruction vector is
            concatenated to the image features before the trunk.
        """

        super().__init__()

        h, w, c = obs_shape
        self.recurrent = recurrent
        self.symbolic = symbolic

        if symbolic:
            # Embed each cell's (object, color, state) indices and
            # concatenate, giving a dense c * SYMBOL_EMBED_DIM
            # channel image. Dividing small category indices by 255
            # instead (as the pixel path does) would leave the input
            # in a 0-0.04 band, which trains poorly and is the usual
            # reason a naive symbolic port underperforms its pixel
            # equivalent.
            self.embedding = nn.Embedding(
                SYMBOL_VOCAB, SYMBOL_EMBED_DIM
            )
            conv_in = c * SYMBOL_EMBED_DIM
        else:
            self.embedding = None
            conv_in = c

        # Symbolic grids are at most 16 cells across, so the
        # Atari-style large-stride stack does not fit them (its
        # second layer's 4x4 kernel is bigger than what the first
        # layer leaves of a 16x16 input); use 3x3 stride-1
        # convolutions there instead. The threshold is imported from
        # algos/intrinsic.py so the policy and the bonus networks
        # always make the same choice for a given observation.
        if min(h, w) < SMALL_INPUT_THRESHOLD:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(conv_in, 32, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, padding=1)),
                nn.ReLU(),
                # Cap the spatial size before flattening so a
                # whole-map symbolic encoding (16x16) does not need
                # an 8.5M-parameter projection; see SMALL_INPUT_POOL.
                nn.AdaptiveMaxPool2d(
                    min(SMALL_INPUT_POOL, min(h, w))
                ),
                nn.Flatten(),
            )
        else:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(conv_in, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Size the projection from a dummy pass so the same code
        # works for every obs_mode and task size.
        with torch.no_grad():
            n_flatten = self.network(
                torch.zeros(1, conv_in, h, w)
            ).shape[1]

        # On the language tasks the instruction joins the image
        # features before the trunk, so the policy conditions on what
        # it was actually asked to do.
        if mission_vocab:
            self.mission_encoder = MissionEncoder(mission_vocab)
            n_flatten += self.mission_encoder.output_dim
        else:
            self.mission_encoder = None

        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, HIDDEN)), nn.ReLU()
        )

        if recurrent:
            # batch_first=False (the nn.GRU default): inputs are
            # (seq_len, batch, features), which is the layout the
            # rollout is already stored in.
            self.core = nn.GRU(HIDDEN, HIDDEN)
            for name, param in self.core.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.orthogonal_(param, 1.0)
        else:
            self.core = None

        self.actor = layer_init(
            nn.Linear(HIDDEN, envs.single_action_space.n), std=0.01
        )
        # Separate value heads for the task reward and the bonus, as
        # in RND: they have different discounts and different
        # episodicity, so one head cannot serve both.
        self.critic_ext = layer_init(nn.Linear(HIDDEN, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(HIDDEN, 1), std=1.0)

    def initial_core_state(self, num_envs, device):
        """
        Return a zeroed recurrent state, or None when feed-forward.
        """

        if self.core is None:
            return None
        return torch.zeros(1, num_envs, HIDDEN, device=device)

    def _encode(self, x, mission_ids=None, mission_len=None):
        """
        Encode a batch of (N, H, W, C) observations into trunk
        features, taking the embedding path for symbolic grids and
        the scaled-pixel path for renders.
        """

        if self.symbolic:
            # (N, H, W, C) indices -> (N, H, W, C, E) embeddings ->
            # (N, C*E, H, W) channels-first for the convolutions.
            emb = self.embedding(x.long())
            n, h, w = emb.shape[0], emb.shape[1], emb.shape[2]
            emb = emb.reshape(n, h, w, -1)
            features = emb.permute(0, 3, 1, 2)
        else:
            features = x.permute(0, 3, 1, 2) / 255.0
        flat = self.network(features)
        if self.mission_encoder is not None:
            flat = torch.cat(
                [flat, self.mission_encoder(mission_ids, mission_len)],
                dim=1,
            )
        return self.fc(flat)

    def get_states(self, x, core_state, episode_start,
                   mission_ids=None, mission_len=None):
        """
        Encode observations and, when recurrent, carry them through
        the GRU while zeroing the hidden state at genuine episode
        starts.

        Parameters
        ----------
        x: FloatTensor, shape (B,) + obs_shape
            B is either E (one rollout step across all envs) or T*E
            in T-major order (a stored sequence replayed during the
            update).
        core_state: FloatTensor (1, E, HIDDEN) or None
            E is read from this tensor's own shape, which is what
            lets one function serve both the rollout and the update.
        episode_start: FloatTensor, shape (B,)
            1.0 exactly where an observation is the first of a fresh
            episode. This is `dones` shifted by one step, NOT
            `dones` itself: dones[t] says the episode ended AFTER
            obs[t], whereas the core must be zeroed BEFORE the first
            observation of the next one.

        Returns
        -------
        (hidden, core_state)
        """

        hidden = self._encode(x, mission_ids, mission_len)
        if self.core is None:
            return hidden, None

        num_envs_here = core_state.shape[1]
        # (B, HIDDEN) -> (T, E, HIDDEN); T == 1 in the rollout case.
        seq = hidden.reshape(-1, num_envs_here, hidden.shape[-1])
        ep_start = episode_start.reshape(-1, num_envs_here)
        seq_len = seq.shape[0]

        # The GRU can consume a whole boundary-free run in ONE call;
        # only a step where some env resets forces a new call, since
        # that is the only place the hidden state must be forcibly
        # zeroed rather than carried. Splitting on those boundaries
        # instead of stepping one timestep at a time is
        # mathematically identical and avoids paying per-call
        # dispatch overhead seq_len times -- the same fix
        # algos/ppo_babyai.py documents after it caused a >30x
        # cluster slowdown there.
        resets_any_env = (ep_start != 0).any(dim=1)
        boundaries = [0] + (
            torch.nonzero(resets_any_env[1:], as_tuple=True)[0]
            .add(1)
            .tolist()
        )

        outputs = []
        for i, start in enumerate(boundaries):
            end = (
                boundaries[i + 1]
                if i + 1 < len(boundaries)
                else seq_len
            )
            # Zero exactly the envs resetting at this chunk's first
            # step; every other env's state carries in unchanged.
            gate = (1.0 - ep_start[start]).view(1, -1, 1)
            chunk_out, core_state = self.core(
                seq[start:end], gate * core_state
            )
            outputs.append(chunk_out)

        hidden = torch.cat(outputs, dim=0).reshape(-1, HIDDEN)
        return hidden, core_state

    def get_value(self, x, core_state=None, episode_start=None,
                  mission_ids=None, mission_len=None):
        """
        Return both value estimates, discarding the updated core
        state (a read-only bootstrap query must not perturb the
        state the next rollout step will use).
        """

        hidden, _ = self.get_states(
            x, core_state, episode_start, mission_ids, mission_len
        )
        return self.critic_ext(hidden), self.critic_int(hidden)

    def get_action_and_value(
        self, x, core_state=None, episode_start=None, action=None,
        mission_ids=None, mission_len=None
    ):
        """
        Sample (or score) an action and return it with its log
        probability, entropy, both values, and the updated core
        state.
        """

        hidden, core_state = self.get_states(
            x, core_state, episode_start, mission_ids, mission_len
        )
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
            core_state,
        )


class RewardForwardFilter:
    """
    Running, forward-discounted sum of intrinsic rewards.

    Used only to estimate the SCALE of the intrinsic return so the
    bonus can be divided by its standard deviation; it is never the
    reward the agent learns from.
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


def build_select_action(agent, device, greedy=True):
    """
    Build an action closure for monitoring.eval.greedy_eval.

    Parameters
    ----------
    greedy: bool
        True takes the argmax action (the benchmark's headline
        teacher-off measurement). False samples from the policy
        instead.

    Why both are worth measuring
    ----------------------------
    An argmax policy is deterministic, so in a gridworld it can fall
    into a cycle -- turn left, turn right, forever -- that the
    stochastic policy it was derived from escapes routinely. That
    failure reports as a flat 0.00 success rate no matter how much
    the policy actually learned. The existing DoorKey-8x8 plain-PPO
    runs look exactly like this: 0.00 greedy on all five seeds while
    the same policies were finishing 52-70% of their training
    episodes. Logging the sampled rate alongside the greedy one
    costs one extra evaluation and distinguishes "the policy learned
    nothing" from "the argmax of this policy loops".

    For a recurrent policy the closure carries a hidden state and
    exposes a `.reset` attribute, which greedy_eval calls after each
    env.reset(). Because greedy_eval runs episodes strictly one at a
    time, `episode_start` can always be zero there -- reset() has
    already zeroed the state at exactly the right moment.
    """

    state = {'core': agent.initial_core_state(1, device)}

    def reset():
        state['core'] = agent.initial_core_state(1, device)

    def select_action(obs):
        with torch.no_grad():
            # On the language tasks the env hands back a dict; split
            # it the same way the rollout does so evaluation reads
            # the instruction too. Evaluating a mission-conditioned
            # policy without the mission would measure a different
            # agent from the one being trained.
            if isinstance(obs, dict):
                image = obs['image']
                mission_ids = torch.as_tensor(
                    obs['mission_ids'], dtype=torch.long
                ).unsqueeze(0).to(device)
                mission_len = torch.as_tensor(
                    obs['mission_len'], dtype=torch.long
                ).reshape(1).to(device)
            else:
                image = obs
                mission_ids = None
                mission_len = None
            x = torch.as_tensor(image).float().unsqueeze(0).to(device)
            zero_start = torch.zeros(1, device=device)
            hidden, state['core'] = agent.get_states(
                x, state['core'], zero_start,
                mission_ids=mission_ids, mission_len=mission_len,
            )
            logits = agent.actor(hidden)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            return int(Categorical(logits=logits).sample().item())

    if agent.recurrent:
        select_action.reset = reset
    return select_action


def train(args):
    """
    Train PPO with the configured exploration bonus and log to
    TensorBoard.

    The loop is algos/ppo.py's, with three additions: the bonus is
    scored once per rollout and folded in as a second, non-episodic
    reward stream with its own value head and discount; the update
    optionally replays contiguous per-env sequences so a GRU core
    can be trained; and the bonus's own loss (if it trains anything)
    is added to the PPO objective.
    """

    if args.bonus not in BONUSES:
        valid = ', '.join(BONUSES)
        raise ValueError(
            f'unknown --bonus {args.bonus!r}; valid: {valid}'
        )

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    eval_milestones = {int(n) // args.batch_size
                       for n in args.eval_frame_milestones.split(',')
                       if n.strip()}

    # Name the run after what actually varies, so the results
    # directory stays self-describing: ppo_noveld_gru, ppo_re3, ...
    algo_name = f'ppo_{args.bonus}'
    if args.recurrent:
        algo_name += '_gru'
    # The episodicity ablations must NOT collide with their own
    # baselines. Without this, a global-scope count run and an
    # episodic one both land under 'ppo_count_gru', and
    # scripts/summarize_runs.py -- which groups by this name --
    # would average the ablation into the control rather than
    # compare it against one. Suffix whatever was flipped.
    if args.count_scope == 'global' and args.bonus == 'count':
        algo_name += '_global'
    if not args.noveld_episodic_gate and args.bonus == 'noveld':
        algo_name += '_nogate'
    if args.experiment_id:
        if not all(c.isalnum() or c in '_-' for c in args.experiment_id):
            raise ValueError('experiment_id must contain letters/digits/_/-')
        algo_name += f'_{args.experiment_id}'
    run_name = (
        f'{args.task}__{algo_name}__{args.seed}__{int(time.time())}'
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Cap CPU threads to the SLURM allocation so concurrent array
    # jobs on one node do not oversubscribe it; see algos/ppo.py for
    # the full reasoning. No-op outside SLURM.
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        torch.set_num_threads(int(slurm_cpus))

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # Whether the observation is MiniGrid's integer grid encoding
    # rather than a render. This changes how the policy and the
    # bonus networks read it, and whether the episodic bonuses can
    # detect a revisit at all.
    symbolic = args.obs_mode.lower().startswith(
        ('symbolic', 'grid')
    )

    # The episodic bonuses recognize a repeat visit by hashing the
    # observation, so what counts as "the same state" depends on the
    # observation mode -- see algos/intrinsic.py's module docstring
    # for how the three modes partition states differently. This is
    # measured, not assumed: the first_visit_frac metric logged
    # below reports the fraction of each rollout that was a first
    # visit, and a value pinned near 1.0 means the episodic
    # component is inert and the run is only a global-novelty
    # baseline.
    episodic_bonus = args.bonus in ('count', 'noveld', 'e3b')
    if episodic_bonus:
        print(
            f'--bonus {args.bonus} has an episodic component; '
            'watch charts/first_visit_frac to confirm revisits are '
            'being detected (near 1.0 means they are not).'
        )

    # Read the instruction on tasks whose mission varies per episode.
    # Without it those tasks are ill-posed rather than hard: the same
    # observation implies different correct actions depending on an
    # instruction the agent cannot see, so no policy and no bonus can
    # do better than guessing which object was meant.
    use_mission = (
        args.task in BABYAI_LANGUAGE_TASKS
        if args.obs_mission is None
        else args.obs_mission
    )
    # One vocabulary instance shared by every env, rather than each
    # re-reading the frozen vocab file.
    vocab = MissionVocab.load() if use_mission else None
    mission_vocab_size = vocab.size if use_mission else 0
    # build_bonus reads this off args, so the bonus is mission-aware
    # exactly when the policy is.
    args.mission_vocab_size = mission_vocab_size
    if use_mission:
        print(
            f'--task {args.task} carries a per-episode instruction; '
            f'policy and bonus are mission-conditioned '
            f'(vocab {mission_vocab_size}).'
        )

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
                obs_mission=use_mission,
                mission_vocab=vocab,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = RecordEpisodeStatistics(sync_envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), 'this PPO variant only supports discrete actions'

    # A recurrent update replays whole envs' contiguous sequences,
    # so minibatches are groups of ENVS rather than arbitrary
    # timesteps; that only divides evenly when num_envs does.
    if args.recurrent:
        assert args.num_envs % args.num_minibatches == 0, (
            f'recurrent minibatching needs num_envs '
            f'({args.num_envs}) divisible by num_minibatches '
            f'({args.num_minibatches})'
        )
        envs_per_batch = args.num_envs // args.num_minibatches

    # With a mission the observation space is a Dict, so the image
    # shape has to be read from its 'image' entry.
    obs_shape = (
        envs.single_observation_space['image'].shape
        if use_mission
        else envs.single_observation_space.shape
    )
    mission_len_max = (
        envs.single_observation_space['mission_ids'].shape[0]
        if use_mission
        else 0
    )
    agent = Agent(
        envs, args.recurrent, symbolic, obs_shape,
        mission_vocab=mission_vocab_size,
    ).to(device)
    if args.record_initial_policy:
        # Hash before constructing a trainable exploration module.
        initial = hashlib.sha256()
        for name, tensor in sorted(agent.state_dict().items()):
            initial.update(name.encode() + b'\0')
            initial.update(tensor.detach().cpu().numpy().tobytes())
        with open(os.path.join(run_dir, 'initial_policy.sha256'), 'w',
                  encoding='utf-8') as handle:
            handle.write(initial.hexdigest() + '\n')
    bonus = build_bonus(
        args.bonus,
        obs_shape,
        args.num_envs,
        device,
        int(envs.single_action_space.n),
        args,
    )

    # One optimizer trains the policy and whatever the bonus learns
    # (RND's predictor, E3B's inverse-dynamics encoder). Stateless
    # bonuses contribute an empty list.
    bonus_params = (
        bonus.trainable_parameters() if bonus is not None else []
    )
    optimizer = optim.Adam(
        list(agent.parameters()) + bonus_params,
        lr=args.learning_rate,
        eps=1e-5,
    )

    run_extra = runtime_metadata()
    run_extra['bonus'] = args.bonus
    run_extra['recurrent'] = args.recurrent
    run_extra['symbolic_obs'] = symbolic
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo=algo_name,
        args=args,
        obs_shape=obs_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=run_extra,
    )

    # Rollout storage. next_obs is stored explicitly (unlike
    # algos/ppo.py) because every bonus scores the state a
    # transition LANDS in, and under auto-reset the next iteration's
    # obs[0] is not that state whenever an episode just ended.
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape
    ).to(device)
    next_obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    int_values = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    # episode_starts is `dones` shifted one step, which is what the
    # recurrent core consumes; see Agent.get_states for why they are
    # not the same array.
    episode_starts = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    # Instruction storage, allocated only on the language tasks. The
    # mission is constant within an episode but a rollout spans
    # episode boundaries, so it genuinely varies down the T axis.
    if use_mission:
        mission_ids_buf = torch.zeros(
            (args.num_steps, args.num_envs, mission_len_max),
            dtype=torch.long,
        ).to(device)
        mission_len_buf = torch.zeros(
            (args.num_steps, args.num_envs), dtype=torch.long
        ).to(device)
    else:
        mission_ids_buf = None
        mission_len_buf = None


    def split_obs(raw):
        """
        Split an env observation into (image, mission ids, lengths).

        Returns tensors on the training device, with both mission
        entries None on the goal-fixed tasks so every downstream call
        can pass them through unconditionally.
        """

        if not use_mission:
            return torch.Tensor(raw).to(device), None, None
        return (
            torch.Tensor(raw['image']).to(device),
            torch.as_tensor(
                raw['mission_ids'], dtype=torch.long
            ).to(device),
            torch.as_tensor(
                raw['mission_len'], dtype=torch.long
            ).reshape(-1).to(device),
        )

    # Running scale of the intrinsic return, used to normalize the
    # bonus so one int_coef is meaningful across bonuses.
    reward_rms = RunningMeanStd()
    reward_filter = RewardForwardFilter(args.int_gamma)

    global_step = 0
    start_time = tracker.start_time
    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs, next_mission_ids, next_mission_len = split_obs(raw_obs)
    next_done = torch.zeros(args.num_envs).to(device)
    # The first observation of training genuinely follows a reset.
    episode_start = torch.ones(args.num_envs).to(device)
    core_state = agent.initial_core_state(args.num_envs, device)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    last_eval = None

    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # The update replays each minibatch's stored sequence from
        # exactly the state the rollout started from, so snapshot it.
        initial_core_state = (
            core_state.detach().clone()
            if core_state is not None
            else None
        )

        # --- Rollout collection ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            episode_starts[step] = episode_start
            if use_mission:
                mission_ids_buf[step] = next_mission_ids
                mission_len_buf[step] = next_mission_len

            with torch.no_grad():
                (
                    action,
                    logprob,
                    _,
                    ext_v,
                    int_v,
                    core_state,
                ) = agent.get_action_and_value(
                    next_obs, core_state, episode_start,
                    mission_ids=next_mission_ids,
                    mission_len=next_mission_len,
                )
                ext_values[step] = ext_v.flatten()
                int_values[step] = int_v.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # dones[step] (just stored) says whether the NEXT
            # observation begins a fresh episode.
            episode_start = dones[step].clone()

            raw_obs, reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_mission_ids, next_mission_len = split_obs(
                raw_obs
            )
            next_done = torch.Tensor(next_done).to(device)
            # Record where this transition landed, for the bonus.
            next_obs_buf[step] = next_obs

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
                    # A positive MiniGrid return means the goal was
                    # reached, so return > 0 is the success signal.
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
                        global_step, i, ep_return, ep_length
                    )

        # --- Intrinsic reward ---
        # Score the whole rollout at once. `dones` marks where
        # obs[t] begins a fresh episode, which is the only signal
        # the episodic bonuses need to reset their state.
        if bonus is not None:
            with torch.no_grad():
                curiosity = bonus.rollout_bonus(
                    obs, next_obs_buf, actions.long(), dones,
                    mission_ids=mission_ids_buf,
                    mission_len=mission_len_buf,
                )
            if args.norm_int_reward:
                # Divide by the running std of the discounted
                # intrinsic return, so the bonus stays on a scale
                # comparable to the task reward instead of either
                # swamping it or vanishing.
                curiosity_np = curiosity.cpu().numpy()
                per_step_returns = np.array(
                    [reward_filter.update(r) for r in curiosity_np]
                )
                reward_rms.update(per_step_returns.reshape(-1))
                curiosity = curiosity / (
                    np.sqrt(reward_rms.var) + 1e-8
                )
        else:
            curiosity = torch.zeros_like(rewards)

        # --- Two-stream advantage estimation (GAE) ---
        # The extrinsic stream respects episode boundaries; the
        # intrinsic stream is treated as never-ending, since novelty
        # does not reset when an episode does.
        with torch.no_grad():
            next_ext_v, next_int_v = agent.get_value(
                next_obs, core_state, episode_start,
                mission_ids=next_mission_ids,
                mission_len=next_mission_len,
            )
            next_ext_v = next_ext_v.reshape(1, -1)
            next_int_v = next_int_v.reshape(1, -1)
            ext_adv = torch.zeros_like(rewards).to(device)
            int_adv = torch.zeros_like(curiosity).to(device)
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
                ext_delta = (
                    rewards[t]
                    + args.gamma * ext_next * ext_nonterm
                    - ext_values[t]
                )
                int_delta = (
                    curiosity[t]
                    + args.int_gamma * int_next
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
                    * int_lastgaelam
                )
            ext_returns = ext_adv + ext_values
            int_returns = int_adv + int_values

        # Flatten the rollout into one batch.
        b_obs = obs.reshape((-1,) + obs_shape)
        b_next_obs = next_obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_ext_adv = ext_adv.reshape(-1)
        b_int_adv = int_adv.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_episode_starts = episode_starts.reshape(-1)
        b_mission_ids = (
            mission_ids_buf.reshape(-1, mission_len_max)
            if use_mission
            else None
        )
        b_mission_len = (
            mission_len_buf.reshape(-1) if use_mission else None
        )

        # The single signal the policy is optimized against.
        b_advantages = (
            b_ext_adv * args.ext_coef + b_int_adv * args.int_coef
        )

        # flat_grid[t, e] locates obs[t, e] in the flat arrays, which
        # is how a recurrent minibatch selects every timestep of a
        # chosen group of envs in the right (T-major) order.
        flat_grid = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )

        # --- PPO update ---
        clipfracs = []
        flat_inds = np.arange(args.batch_size)
        env_inds = np.arange(args.num_envs)
        approx_kl = torch.zeros((), device=device)

        for epoch in range(args.update_epochs):
            # A recurrent core needs temporally contiguous data, so
            # minibatches are whole ENVS there; the feed-forward
            # path keeps algos/ppo.py's flat timestep shuffle so it
            # remains directly comparable to the existing runs.
            if args.recurrent:
                np.random.shuffle(env_inds)
                batches = [
                    (
                        flat_grid[
                            :, env_inds[s:s + envs_per_batch]
                        ].ravel(),
                        env_inds[s:s + envs_per_batch],
                    )
                    for s in range(
                        0, args.num_envs, envs_per_batch
                    )
                ]
            else:
                np.random.shuffle(flat_inds)
                batches = [
                    (flat_inds[s:s + args.minibatch_size], None)
                    for s in range(
                        0, args.batch_size, args.minibatch_size
                    )
                ]

            for mb_inds, mb_envs in batches:
                # Replay starts from the state this iteration's
                # rollout started from, for these envs specifically.
                mb_core = (
                    initial_core_state[:, mb_envs]
                    if args.recurrent
                    else None
                )
                mb_starts = (
                    b_episode_starts[mb_inds]
                    if args.recurrent
                    else None
                )

                (
                    _,
                    newlogprob,
                    entropy,
                    new_ext_v,
                    new_int_v,
                    _,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_core,
                    mb_starts,
                    mission_ids=(
                        None if b_mission_ids is None
                        else b_mission_ids[mb_inds]
                    ),
                    mission_len=(
                        None if b_mission_len is None
                        else b_mission_len[mb_inds]
                    ),
                    action=b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                # Clamp before exponentiating so an extreme
                # importance ratio cannot blow the update into NaNs.
                logratio = torch.clamp(logratio, -10.0, 10.0)
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
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

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # One value loss per stream. Only the extrinsic head
                # is clipped, matching ppo.py; the intrinsic
                # target's scale drifts as the bonus decays, which
                # makes clipping against a stale value unhelpful.
                new_ext_v = new_ext_v.view(-1)
                new_int_v = new_int_v.view(-1)
                if args.clip_vloss:
                    v_unclipped = (
                        new_ext_v - b_ext_returns[mb_inds]
                    ) ** 2
                    v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_v - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_clipped_loss = (
                        v_clipped - b_ext_returns[mb_inds]
                    ) ** 2
                    ext_v_loss = 0.5 * torch.max(
                        v_unclipped, v_clipped_loss
                    ).mean()
                else:
                    ext_v_loss = (
                        0.5
                        * (
                            (new_ext_v - b_ext_returns[mb_inds]) ** 2
                        ).mean()
                    )
                int_v_loss = (
                    0.5
                    * ((new_int_v - b_int_returns[mb_inds]) ** 2).mean()
                )
                v_loss = ext_v_loss + int_v_loss

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                # Whatever the bonus learns is trained on the same
                # minibatch, by the same optimizer step.
                bonus_loss = (
                    bonus.aux_loss(
                        b_obs[mb_inds],
                        b_next_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        mb_mission_ids=(
                            None if b_mission_ids is None
                            else b_mission_ids[mb_inds]
                        ),
                        mb_mission_len=(
                            None if b_mission_len is None
                            else b_mission_len[mb_inds]
                        ),
                    )
                    if bonus is not None
                    else None
                )
                if bonus_loss is not None:
                    loss = loss + bonus_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.parameters()) + bonus_params,
                    args.max_grad_norm,
                )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # --- Greedy teacher-off evaluation ---
        # The same fixed-episode greedy eval every method in the
        # benchmark runs, so this curve is directly comparable to
        # the teacher-guided variants' teacher-off curves. Without
        # it a bonus run could only be compared on its training-time
        # success rate, which is measured under a different
        # (stochastic, bonus-seeking) policy.
        run_eval = args.eval_interval > 0 and (
            iteration % args.eval_interval == 0
            or iteration in eval_milestones
            or iteration == args.num_iterations
        )
        if run_eval:

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
                    obs_mission=use_mission,
                    mission_vocab=vocab,
                )

            last_eval = greedy_eval(
                build_select_action(agent, device, greedy=True),
                make_env=make_eval_env,
                num_episodes=args.eval_episodes,
                seed_base=args.seed + 50_000,
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

            # The same teacher-off episodes under the SAMPLED policy.
            # A large gap below the greedy number means the argmax
            # policy is cycling, not that nothing was learned -- see
            # build_select_action for why this matters here.
            if args.eval_sampled:
                sampled_eval = greedy_eval(
                    build_select_action(agent, device, greedy=False),
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

            # Save each actual evaluation for portable curve analysis.
            with open(os.path.join(run_dir, 'evaluations.jsonl'), 'a',
                      encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'global_step': global_step,
                    'iteration': iteration,
                    'teacher_on': False,
                    'wall_time_sec': time.time() - start_time,
                    'training_episodes': tracker.episode_count,
                    **last_eval,
                }) + '\n')

        # --- Per-iteration diagnostics ---
        y_pred = b_ext_values.cpu().numpy()
        y_true = b_ext_returns.cpu().numpy()
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
            'losses/ext_value_loss', ext_v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/int_value_loss', int_v_loss.item(), global_step
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

        extra = {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'ext_value_loss': ext_v_loss.item(),
            'int_value_loss': int_v_loss.item(),
            'policy_loss': pg_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': float(np.mean(clipfracs)),
            'explained_variance': float(explained_var),
        }
        if bonus is not None:
            writer.add_scalar(
                'charts/mean_intrinsic_reward',
                curiosity.mean().item(),
                global_step,
            )
            extra['mean_intrinsic_reward'] = curiosity.mean().item()
            if bonus_loss is not None:
                writer.add_scalar(
                    'losses/bonus_loss',
                    bonus_loss.item(),
                    global_step,
                )
                extra['bonus_loss'] = bonus_loss.item()
            # Per-bonus health metrics, notably first_visit_frac:
            # if it sits at 1.0 the episodic gate is inert and the
            # observation mode is wrong for this bonus.
            for key, value in bonus.diagnostics().items():
                writer.add_scalar(
                    f'charts/{key}', value, global_step
                )
                extra[key] = value

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
    # Save the policy alongside the summary, matching algos/ppo.py,
    # so scripts/probe_subgoal.py can use any of these runs as a
    # control without retraining.
    torch.save(agent.state_dict(), os.path.join(run_dir, 'agent.pt'))
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
