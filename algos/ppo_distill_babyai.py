"""
Kickstarting distillation on top of the mission-conditioned BabyAI
agent (algos/ppo_babyai.py).

Exactly what algos/ppo_distill.py is to algos/ppo.py: the student
always samples and executes its own action (the rollout is
byte-for-byte the ppo_babyai.py rollout), and the teacher's advice
enters only as an annealed cross-entropy term in the loss,
evaluated on the states the student itself visited. See
algos/ppo_distill.py's module docstring for the full mechanism and
its verification against LLM4Teach's released code -- none of that
changes here. `soften_advice` and `distill_coef` are imported from
there rather than duplicated, since they are already
architecture-independent pure functions with their own tests
(tests/test_distill.py); everything that IS architecture-specific
(the recurrent, mission-conditioned agent, and the env-level
minibatching it requires) is duplicated from algos/ppo_babyai.py,
matching this project's "each algorithm is a self-contained diff"
convention.

Why this file exists rather than just adding a teacher to
ppo_babyai.py: algos/ppo_distill.py's own Agent differs from
algos/ppo.py's Agent only in returning `logits` from
get_action_and_value (needed for the CE term without a second
forward pass); the same one-method difference applies here.

Teacher: BabyAIBotTeacher ('bot', the default -- free, offline,
built for exactly this advising role, see
teachers/minigrid/bot_teacher.py) or the topology-agnostic vision
teacher ('vlm_general' -- paid, needs internet, local-only). The
BFS oracle ('oracle') is DoorKey-only and therefore never valid on
a BabyAI task; selecting it raises at startup, same as
algos/ppo_distill.py's own validation.

Two "done-like" signals coexist in the rollout loop and must NOT be
conflated (see algos/ppo_babyai.py's module docstring for the first
one): `episode_start` (shifted one step, feeds the recurrent core's
hidden-state reset) and the bot's own `fresh_episode`/`last_actions`
bookkeeping (same-index as `envs.step()`'s own done flag, feeds
`BabyAIBotTeacher.recommend`'s statefulness contract). They answer
different questions -- "should the STUDENT's memory reset before
this observation" vs. "did the episode the TEACHER is tracking just
end" -- and are kept as distinctly named variables throughout.

Run it locally:

    python -m algos.ppo_distill_babyai --task gotoseq --teacher bot
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

from algos.nets import ObsEncoder, is_symbolic_mode
from algos.ppo_distill import distill_coef, soften_advice
from envs.mission_vocab import MAX_MISSION_LEN, MissionVocab
from envs.registry import TASKS, build_env, make_thunk
from monitoring.eval import greedy_eval
from monitoring.metrics import RunTracker, runtime_metadata
from teachers.base import Cost
from teachers.factory import make_teacher

# The language-conditioned tasks this file supports (see
# algos/ppo_babyai.py's module docstring for why the fix is scoped
# to exactly these, and why the BabyAI-registered keycorridor_*
# tasks are deliberately absent).
BABYAI_TASKS = (
    'gotolocal',
    'gotoseq',
    'gotoseq_s5r2',
    'putnextlocal',
    'bosslevel',
)

# Every BabyAI task exposes .instrs, so 'bot' always applies; the
# BFS oracle is DoorKey-only and 'llm'/'vlm' are DoorKey-shaped and
# frozen (see teachers/factory.py). 'vlm_general' (vision API) and
# 'llm_general' (text-only API) both work on any task; comparing
# them against each other and against the free bot -- same task,
# same channel -- is the teacher-identity study.
SUPPORTED_TEACHERS = ('bot', 'vlm_general', 'llm_general')

MISSION_EMBED_DIM = 32
MISSION_HIDDEN = 64
CNN_PROJ_DIM = 256
CORE_HIDDEN = 512


@dataclass
class Args:
    """
    Command-line arguments for a BabyAI distillation run.

    PPO and recurrence knobs match algos/ppo_babyai.py exactly; the
    teacher and distillation-schedule knobs match algos/ppo_distill.py
    exactly (same defaults, same meanings), so a difference between
    the two distillation runs (this file vs. algos/ppo_distill.py on
    the same task) is attributable to the mission input and
    recurrence, not to schedule tuning drift.
    """

    # --- Experiment identity and logging ---
    task: str = 'gotoseq'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = 'vlm-rl-bench'

    # --- Observation settings ---
    obs_mode: str = 'historical'
    agent_view_size: int = 7
    obs_tile_size: int = 0
    obs_target_size: int = 56

    # --- Core PPO hyperparameters (identical to ppo_babyai.py) ---
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
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

    # --- Teacher ---
    teacher: str = 'bot'
    # Model for the vlm_general/llm_general teachers. Ignored by
    # bot.
    teacher_model: str = 'gpt-4.1-mini'
    # Query the teacher only every k-th rollout step (1 = every
    # step). Unqueried steps carry no distillation target (the
    # mask handles that), so this thins the supervision rather
    # than changing its meaning; it caps API cost for the paid
    # teachers (cost ~1/k). Keep 1 for the free bot.
    query_interval: int = 1
    # HARD ceiling on this run's teacher spend, in dollars. 0
    # disables the cap. Once total_cost.dollars reaches this, the
    # teacher stops being queried for the REST of the run and
    # training continues as pure PPO for free -- a real, code-
    # enforced ceiling, unlike query_interval/total_timesteps which
    # only estimate cost in advance. See algos/ppo_distill.py's
    # Args for the full reasoning.
    max_cost_dollars: float = 0.0

    # --- Distillation schedule (identical to ppo_distill.py) ---
    distill_coef_start: float = 10.0
    distill_coef_min: float = 0.1
    distill_fraction: float = 0.5
    distill_cutoff: float = 0.75

    # --- Teacher-off greedy evaluation ---
    eval_interval: int = 50
    eval_episodes: int = 10
    # Evaluate the SAMPLED policy alongside the greedy one, so a
    # teacher arm scoring 0.00 can be distinguished from one whose
    # argmax merely cycles.
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


class MissionEncoder(nn.Module):
    """
    Embed a tokenized mission and summarize it with a GRU. Identical
    to algos/ppo_babyai.py's MissionEncoder -- see that file for the
    reasoning behind the final-hidden-state choice and the
    pack_padded_sequence usage.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size):
        """
        Build the embedding table and the summarizing GRU.
        """

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.output_dim = hidden_size

    def forward(self, mission_ids, mission_len):
        """
        Encode a batch of tokenized missions into fixed-size
        vectors.
        """

        embedded = self.embedding(mission_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            mission_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden.squeeze(0)


class RecurrentAgent(nn.Module):
    """
    CNN + mission encoder + recurrent core, actor-critic heads.

    Identical to algos/ppo_babyai.py's RecurrentAgent except
    get_action_and_value also returns the raw policy logits: the
    distillation term needs the full action distribution (not just
    the log-prob of one action) without a second forward pass, the
    same reason algos/ppo_distill.py's Agent differs from
    algos/ppo.py's Agent by exactly one return value.
    """

    def __init__(self, envs, vocab_size, symbolic=False):
        """
        Build the encoder, mission branch, recurrent core, and
        heads, sized to the env's spaces and the module-level
        MISSION_*/CNN_PROJ_DIM/CORE_HIDDEN constants.

        Parameters
        ----------
        symbolic: bool
            True when the observation is MiniGrid's integer grid
            rather than a render. See ppo_babyai.py's RecurrentAgent
            for why the two need different encoders: category codes
            must be embedded rather than scaled, and the small grids
            do not fit the Atari stack's 8x8 stride-4 first layer.
            Derive it with algos.nets.is_symbolic_mode(args.obs_mode).
        """

        super().__init__()

        obs_shape = envs.single_observation_space['image'].shape

        # Shared with ppo_babyai.py, ppo_distill.py and
        # ppo_teacher.py, so a teacher arm and a teacher-free arm at
        # the same obs_mode differ in the teacher and nothing else.
        self.encoder = ObsEncoder(
            obs_shape, symbolic, hidden=CNN_PROJ_DIM
        )

        self.mission_encoder = MissionEncoder(
            vocab_size, MISSION_EMBED_DIM, MISSION_HIDDEN
        )

        self.fusion = nn.Sequential(
            layer_init(
                nn.Linear(CNN_PROJ_DIM + MISSION_HIDDEN, CORE_HIDDEN)
            ),
            nn.ReLU(),
        )

        self.core = nn.GRU(CORE_HIDDEN, CORE_HIDDEN)
        for name, param in self.core.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(
            nn.Linear(CORE_HIDDEN, envs.single_action_space.n),
            std=0.01,
        )
        self.critic = layer_init(nn.Linear(CORE_HIDDEN, 1), std=1.0)

    def initial_core_state(self, num_envs, device):
        """
        A zeroed recurrent state for `num_envs` fresh episodes.
        """

        return torch.zeros(1, num_envs, CORE_HIDDEN, device=device)

    def get_states(self, x, core_state, episode_start):
        """
        Encode a batch of observations and step them through the
        recurrent core, resetting it wherever an episode genuinely
        starts. See algos/ppo_babyai.py's RecurrentAgent.get_states
        for the full parameter/shape contract -- unchanged here.
        """

        cnn_feat = self.encoder(x['image'])
        mission_feat = self.mission_encoder(
            x['mission_ids'], x['mission_len']
        )
        fused = self.fusion(
            torch.cat([cnn_feat, mission_feat], dim=-1)
        )

        num_envs_here = core_state.shape[1]
        fused = fused.reshape(-1, num_envs_here, fused.shape[-1])
        ep_start = episode_start.reshape(-1, num_envs_here)
        seq_len = fused.shape[0]

        # Batch reset-free runs into single GRU calls instead of one
        # call per timestep -- see algos/ppo_babyai.py's
        # RecurrentAgent.get_states for the full reasoning (this was
        # the dominant cost behind a >30x cluster-vs-local slowdown
        # observed in practice; the result is mathematically
        # identical, only the number of Python-level calls changes).
        resets_any_env = (ep_start != 0).any(dim=1)
        boundaries = [0] + (
            torch.nonzero(resets_any_env[1:], as_tuple=True)[0]
            .add(1)
            .tolist()
        )

        outputs = []
        for i, start in enumerate(boundaries):
            end = (
                boundaries[i + 1] if i + 1 < len(boundaries)
                else seq_len
            )
            gate = (1.0 - ep_start[start]).view(1, -1, 1)
            chunk_out, core_state = self.core(
                fused[start:end], gate * core_state
            )
            outputs.append(chunk_out)

        hidden = torch.cat(outputs, dim=0).reshape(-1, CORE_HIDDEN)
        return hidden, core_state

    def get_value(self, x, core_state, episode_start):
        """
        Return the critic's value estimate, discarding the updated
        core state.
        """

        hidden, _ = self.get_states(x, core_state, episode_start)
        return self.critic(hidden)

    def get_action_and_value(
        self, x, core_state, episode_start, action=None
    ):
        """
        Sample (or evaluate) an action; return it with log-prob,
        entropy, value, the updated core state, and the raw logits
        (for the distillation loss).
        """

        hidden, core_state = self.get_states(
            x, core_state, episode_start
        )
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            core_state,
            logits,
        )


def build_select_action(agent, device, greedy=True):
    """
    Build a stateful action-selection closure for
    monitoring.eval.greedy_eval. Identical to
    algos/ppo_babyai.py's build_select_action -- see there for why
    episode_start can simply always be 0 here.
    """

    state = {'core': agent.initial_core_state(1, device)}

    def reset():
        state['core'] = agent.initial_core_state(1, device)

    def select_action(obs):
        with torch.no_grad():
            x = {
                'image': torch.as_tensor(obs['image'])
                .float()
                .unsqueeze(0)
                .to(device),
                'mission_ids': torch.as_tensor(obs['mission_ids'])
                .long()
                .unsqueeze(0)
                .to(device),
                'mission_len': torch.as_tensor(
                    np.asarray(obs['mission_len'])
                )
                .long()
                .view(1)
                .to(device),
            }
            zero_start = torch.zeros(1, device=device)
            hidden, state['core'] = agent.get_states(
                x, state['core'], zero_start
            )
            logits = agent.actor(hidden)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            return int(Categorical(logits=logits).sample().item())

    select_action.reset = reset
    return select_action


def train(args):
    """
    Train a recurrent, mission-conditioned, distillation-guided PPO
    agent on a BabyAI task and log to TensorBoard.

    The skeleton is algos/ppo_babyai.train; the additions are (1)
    teacher queries + soft-target storage during rollout, (2) the
    annealed CE term in the update, both composed with the recurrent
    env-level minibatching exactly as algos/ppo_distill.py composes
    the same additions with algos/ppo.py's flat minibatching.
    """

    if args.task not in BABYAI_TASKS:
        raise ValueError(
            f'algos.ppo_distill_babyai only supports the language-'
            f'conditioned BabyAI tasks {BABYAI_TASKS}; got '
            f'{args.task!r}.'
        )
    if args.teacher not in SUPPORTED_TEACHERS:
        raise ValueError(
            f'algos.ppo_distill_babyai supports teachers '
            f'{SUPPORTED_TEACHERS} on BabyAI tasks (the BFS oracle '
            f'is DoorKey-only, and llm/vlm are frozen and '
            f'DoorKey-shaped -- see teachers/factory.py); got '
            f'{args.teacher!r}.'
        )
    if args.teacher == 'bot' and args.query_interval != 1:
        raise ValueError(
            'query_interval > 1 is only valid for STATELESS '
            'teachers (vlm_general, llm_general): the bot tracks '
            'the student one executed action at a time via '
            'replan(last_action), so skipping steps silently '
            'desynchronizes its plan. The bot is free anyway -- '
            'there is no cost to cap.'
        )

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.num_envs % args.num_minibatches != 0:
        raise ValueError(
            f'recurrent minibatching needs num_envs '
            f'({args.num_envs}) divisible by num_minibatches '
            f'({args.num_minibatches}).'
        )
    envs_per_batch = args.num_envs // args.num_minibatches

    algo_tag = f'ppo_distill_babyai_{args.teacher}'
    run_name = f'{args.task}__{algo_tag}__{args.seed}__{int(time.time())}'

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
    # Cap CPU threads to the SLURM allocation so many array jobs
    # sharing one node do not oversubscribe it -- see algos/ppo.py's
    # seeding block for the full reasoning. No-op outside SLURM.
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        torch.set_num_threads(int(slurm_cpus))

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    vocab = MissionVocab.load()

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
                obs_mission=True,
                mission_vocab=vocab,
                mission_max_len=MAX_MISSION_LEN,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = RecordEpisodeStatistics(sync_envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), 'this algorithm only supports discrete actions'
    n_actions = int(envs.single_action_space.n)

    # --- Teachers: one instance per parallel env ---
    # The bot carries per-episode plan state, so per-env instances
    # are required; vlm_general is stateless per call but using one
    # instance per env keeps the code uniform (and gives each env
    # its own cache/counters).
    env_id = TASKS[args.task]
    teachers = [
        make_teacher(
            args.teacher, env_id, args.seed + i, model=args.teacher_model
        )
        for i in range(args.num_envs)
    ]
    # Per-env bookkeeping the bot teacher needs: the action the
    # student last executed (fed to replan) and whether this env
    # just started a fresh episode (rebuild the bot's plan). This is
    # a DIFFERENT signal from the recurrent core's `episode_start`
    # below -- see the module docstring.
    last_actions = [None] * args.num_envs
    fresh_episode = [True] * args.num_envs

    symbolic = is_symbolic_mode(args.obs_mode)
    agent = RecurrentAgent(
        envs, vocab_size=vocab.size, symbolic=symbolic
    ).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    image_shape = envs.single_observation_space['image'].shape
    mission_shape = envs.single_observation_space['mission_ids'].shape
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo=algo_tag,
        args=args,
        obs_shape=image_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=runtime_metadata(),
    )

    obs_image = torch.zeros(
        (args.num_steps, args.num_envs) + image_shape
    ).to(device)
    obs_mission_ids = torch.zeros(
        (args.num_steps, args.num_envs) + mission_shape,
        dtype=torch.long,
    ).to(device)
    obs_mission_len = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.long
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    episode_starts = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    # Teacher targets q(.|s) per (step, env), plus a mask that is 0
    # where the teacher abstained, the step is a post-terminal
    # placeholder, or queries are off (post-cutoff) -- identical
    # meaning to algos/ppo_distill.py's buffers of the same name.
    teacher_targets = torch.zeros(
        (args.num_steps, args.num_envs, n_actions)
    ).to(device)
    teacher_mask = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)

    global_step = 0
    start_time = tracker.start_time

    def _to_device_obs(obs_np):
        """
        Convert one vectorized Dict observation (numpy) to tensors
        on `device`, with the dtypes each field actually needs.
        """

        return {
            'image': torch.as_tensor(obs_np['image']).float().to(
                device
            ),
            'mission_ids': torch.as_tensor(obs_np['mission_ids'])
            .long()
            .to(device),
            'mission_len': torch.as_tensor(obs_np['mission_len'])
            .long()
            .to(device),
        }

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = _to_device_obs(next_obs_np)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_start = torch.ones(args.num_envs).to(device)
    core_state = agent.initial_core_state(args.num_envs, device)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    total_queries = 0
    total_abstains = 0
    total_cost = Cost()
    last_eval = None
    # Printed once, the first time max_cost_dollars stops queries.
    budget_notice_printed = False

    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        coef = distill_coef(iteration, args.num_iterations, args)

        initial_core_state = core_state.detach().clone()

        # --- Rollout collection ---
        # Identical to ppo_babyai.py except the teacher is also
        # queried on each visited state. The executed action is
        # ALWAYS the student's own sample.
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_image[step] = next_obs['image']
            obs_mission_ids[step] = next_obs['mission_ids']
            obs_mission_len[step] = next_obs['mission_len']
            dones[step] = next_done
            episode_starts[step] = episode_start

            with torch.no_grad():
                action, logprob, _, value, core_state, _ = (
                    agent.get_action_and_value(
                        next_obs, core_state, episode_start
                    )
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            episode_start = dones[step].clone()

            # Query the teacher for every env, unless the schedule
            # has cut distillation off, the step falls between
            # query-interval ticks (paid-teacher cost control), or
            # this step is the post-terminal placeholder gymnasium
            # inserts before auto-reset (same skip condition as
            # algos/ppo_distill.py, and the SAME-index `next_done`
            # signal it uses -- deliberately not `episode_start`,
            # since the teacher needs "did the call that produced
            # this observation itself terminate", not the shifted
            # recurrent-reset signal).
            under_budget = (
                args.max_cost_dollars <= 0
                or total_cost.dollars < args.max_cost_dollars
            )
            if (
                coef > 0.0
                and step % args.query_interval == 0
                and under_budget
            ):
                for i in range(args.num_envs):
                    if next_done[i] > 0:
                        continue
                    u_i = sync_envs.envs[i].unwrapped
                    if args.teacher == 'bot':
                        advice = teachers[i].recommend(
                            u_i,
                            {
                                'new_episode': fresh_episode[i],
                                'last_action': last_actions[i],
                            },
                        )
                    elif args.teacher == 'vlm_general':
                        advice = teachers[i].recommend(
                            u_i,
                            {
                                'image': u_i.get_frame(
                                    highlight=False, tile_size=32
                                ),
                                'mission': getattr(
                                    u_i, 'mission', ''
                                ),
                            },
                        )
                    else:  # llm_general (text-only: renders its
                        # own ASCII map from the unwrapped env).
                        advice = teachers[i].recommend(
                            u_i,
                            {
                                'mission': getattr(
                                    u_i, 'mission', ''
                                ),
                            },
                        )
                    fresh_episode[i] = False
                    total_queries += 1
                    total_cost = total_cost + advice.cost
                    target = soften_advice(advice, n_actions)
                    if target is None:
                        total_abstains += 1
                    else:
                        teacher_targets[step, i] = torch.from_numpy(
                            target
                        )
                        teacher_mask[step, i] = 1.0
            elif (
                coef > 0.0
                and not under_budget
                and not budget_notice_printed
            ):
                print(
                    f'[budget] teacher spend reached '
                    f'${total_cost.dollars:.4f} >= '
                    f'max_cost_dollars={args.max_cost_dollars}; '
                    f'no further teacher queries this run -- '
                    f'training continues as pure PPO.'
                )
                budget_notice_printed = True

            executed = action.cpu().numpy()
            next_obs_np, reward, terminations, truncations, infos = (
                envs.step(executed)
            )
            done_mask = np.logical_or(terminations, truncations)
            next_done = torch.Tensor(done_mask).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = _to_device_obs(next_obs_np)

            # Keep the stateful bot synchronized: remember what the
            # student actually did, and flag envs whose episode just
            # ended so the next query rebuilds the plan for the new
            # mission.
            for i in range(args.num_envs):
                last_actions[i] = int(executed[i])
                if done_mask[i]:
                    fresh_episode[i] = True
                    last_actions[i] = None

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
                    recent_returns.append(ep_return)
                    recent_successes.append(
                        1.0 if ep_return > 0 else 0.0
                    )
                    tracker.record_episode(
                        global_step, i, ep_return, ep_length
                    )

        # --- Advantage estimation (GAE), identical to ppo_babyai.py ---
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, core_state, episode_start
            ).reshape(1, -1)
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

        b_obs_image = obs_image.reshape((-1,) + image_shape)
        b_obs_mission_ids = obs_mission_ids.reshape(
            (-1,) + mission_shape
        )
        b_obs_mission_len = obs_mission_len.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_episode_starts = episode_starts.reshape(-1)
        b_teacher_targets = teacher_targets.reshape(-1, n_actions)
        b_teacher_mask = teacher_mask.reshape(-1)

        flat_grid = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )

        # --- PPO update + the distillation term: env-level minibatches ---
        env_inds = np.arange(args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, envs_per_batch):
                mb_envs = env_inds[start:start + envs_per_batch]
                mb_flat = flat_grid[:, mb_envs].ravel()

                mb_x = {
                    'image': b_obs_image[mb_flat],
                    'mission_ids': b_obs_mission_ids[mb_flat],
                    'mission_len': b_obs_mission_len[mb_flat],
                }
                mb_episode_start = b_episode_starts[mb_flat]
                mb_init_core = initial_core_state[:, mb_envs]

                _, newlogprob, entropy, newvalue, _, logits = (
                    agent.get_action_and_value(
                        mb_x,
                        mb_init_core,
                        mb_episode_start,
                        action=b_actions.long()[mb_flat],
                    )
                )
                logratio = newlogprob - b_logprobs[mb_flat]
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

                mb_advantages = b_advantages[mb_flat]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_flat]
                    ) ** 2
                    v_clipped = b_values[mb_flat] + torch.clamp(
                        newvalue - b_values[mb_flat],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (
                        v_clipped - b_returns[mb_flat]
                    ) ** 2
                    v_loss_max = torch.max(
                        v_loss_unclipped, v_loss_clipped
                    )
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = (
                        0.5
                        * ((newvalue - b_returns[mb_flat]) ** 2).mean()
                    )

                # The kickstarting term: identical formula to
                # algos/ppo_distill.py, indexed by mb_flat.
                mb_mask = b_teacher_mask[mb_flat]
                if coef > 0.0 and mb_mask.sum() > 0:
                    log_pi = F.log_softmax(logits, dim=-1)
                    ce = -(
                        b_teacher_targets[mb_flat] * log_pi
                    ).sum(dim=-1)
                    distill_loss = (
                        (ce * mb_mask).sum() / mb_mask.sum()
                    )
                else:
                    distill_loss = torch.zeros((), device=device)

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + coef * distill_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Reset the target buffers for the next rollout so a step
        # that goes unqueried next iteration cannot inherit this
        # iteration's stale target.
        teacher_targets.zero_()
        teacher_mask.zero_()

        # --- Teacher-off greedy evaluation ---
        run_eval = args.eval_interval > 0 and (
            iteration % args.eval_interval == 0
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
                    obs_mission=True,
                    mission_vocab=vocab,
                    mission_max_len=MAX_MISSION_LEN,
                )

            last_eval = greedy_eval(
                build_select_action(agent, device, greedy=True),
                make_env=make_eval_env,
                num_episodes=args.eval_episodes,
                seed_base=args.seed + 50_000,
            )
            # A teacher arm reading 0.00 is only evidence of failure
            # if the sampled policy also fails; the argmax of a
            # learned policy can cycle in a gridworld.
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
        writer.add_scalar(
            'losses/distill_loss', distill_loss.item(), global_step
        )
        writer.add_scalar('teacher/distill_coef', coef, global_step)
        writer.add_scalar(
            'teacher/total_queries', total_queries, global_step
        )
        writer.add_scalar(
            'teacher/total_abstains', total_abstains, global_step
        )
        writer.add_scalar(
            'teacher/cost_dollars', total_cost.dollars, global_step
        )
        writer.add_scalar(
            'teacher/budget_exhausted',
            1.0 if budget_notice_printed else 0.0,
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
            'value_loss': v_loss.item(),
            'policy_loss': pg_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': float(np.mean(clipfracs)),
            'explained_variance': float(explained_var),
            'distill_loss': distill_loss.item(),
            'distill_coef': coef,
            'teacher_total_queries': int(total_queries),
            'teacher_total_abstains': int(total_abstains),
            'teacher_cost_dollars': total_cost.dollars,
            'teacher_wall_time_s': total_cost.wall_time_s,
            'teacher_compute_units': int(total_cost.compute_units),
        }
        if last_eval is not None:
            extra['eval_success_rate'] = last_eval['success_rate']
            extra['eval_mean_return'] = last_eval['mean_return']
            # A teacher arm's 0.00 is only evidence of failed
            # internalization if the sampled policy fails too.
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
        eval_str = (
            f' | eval {last_eval["success_rate"]:4.2f}'
            if last_eval is not None
            else ''
        )
        print(
            f'iter {iteration}/{args.num_iterations} '
            f'step {global_step} SPS {sps} | {progress} '
            f'| ks {coef:5.2f}{eval_str}'
        )

    # Keep the trained policy, so a finished run can be reloaded for
    # a transfer or hand-off experiment instead of repeated.
    torch.save(
        agent.state_dict(), os.path.join(run_dir, 'agent.pt')
    )

    tracker.close(
        status='completed', global_step=global_step, extra=extra
    )
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
