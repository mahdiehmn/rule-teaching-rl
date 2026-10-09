"""
Recurrent, mission-conditioned PPO for BabyAI (language) tasks.

On gotoseq/gotolocal/putnextlocal/bosslevel the per-episode mission
string is often the ONLY thing that disambiguates the goal (e.g.
"go to a grey box" when several colored boxes are visible) -- the
rendered image alone is ambiguous. algos/ppo.py's Agent never sees
the mission at all (envs.wrappers.KeepMissionWrapper only copies it
into info, for the teacher's benefit), so on these four tasks it is
being asked to solve a genuinely ill-posed problem: no amount of
training, with or without a teacher, can fix a missing input. This
file is the fix, scoped to exactly the tasks that need it -- the
Axis-A tasks (doorkey_*, keycorridor_s6r3, multiroom_n6,
obstructedmaze_full) have goals that are visually unambiguous
without reading the mission (verified: DoorKey's mission is a
single invariant string across every seed; KeyCorridor's target is
the only pickup-able ball in the level), so ppo.py is left exactly
as it is for those.

Two additions on top of ppo.py's skeleton:

1. The student's observation includes the tokenized mission
   (envs.wrappers.MissionTokenWrapper, via build_env(...,
   obs_mission=True)), and the network has a small GRU mission
   encoder whose output is concatenated with the CNN's image
   features before the shared trunk.
2. The trunk is followed by a recurrent core (another GRU), because
   the environment gives no observable signal for "have I already
   satisfied the first clause of this sequential mission" -- e.g.
   GoToSeq's "go to X then go to Y". The policy needs memory of its
   OWN progress through the instruction, which is a different thing
   from envs/historical_obs.py's fog-of-war memory of the map (that
   wrapper is still used underneath this one; see obs_mode below).

The trickiest part of the recurrent core is exactly WHEN to reset
its hidden state to zero. This repo's SyncVectorEnv runs
gymnasium's AutoresetMode.NEXT_STEP: when an episode ends on step
t, `dones[t]` is 1 and the observation stored at that step IS the
real terminal frame (not a placeholder) -- the fresh reset only
happens on step t+1, whose own `dones[t+1]` is 0. A naive port of
the textbook "reset hidden state where done" trick, applied at the
SAME index as the observation being processed, would therefore
reset the core one step too late: it would zero the state right
before re-processing the (already-known) terminal frame, but leave
the state UNRESET going into the very first observation of the new
episode -- exactly the moment a fresh mission needs a clean slate.
This file uses a separate `episode_start` signal, shifted one step
relative to `dones`, to avoid that: `episode_start` fed into step t
equals `dones[t-1]`, not `dones[t]`. See the rollout loop below.

Run it locally:

    python -m algos.ppo_babyai --task gotolocal
"""

import json
import os
import random
import tempfile
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

from algos.nets import ObsEncoder, is_symbolic_mode
from envs.mission_vocab import MAX_MISSION_LEN, MissionVocab
from envs.registry import build_env, make_thunk
from monitoring.eval import greedy_eval
from monitoring.metrics import RunTracker, runtime_metadata

# The language-conditioned tasks this file supports (gotoseq_s5r2 is
# the small, fast member of the gotoseq family -- same compositional
# missions, ~23x shorter episodes). Any other task is rejected at
# startup (see train()) rather than silently training a mission
# encoder nobody needs. Note the keycorridor_* tasks are NOT here
# even though some are BabyAI-registered: their mission ('pick up
# the ball', exactly one ball) disambiguates nothing, so they stay
# Axis-A tasks trained by the pixels-only algos/ppo.py agent.
BABYAI_TASKS = (
    'gotolocal',
    'gotoseq',
    'gotoseq_s5r2',
    'gotoseq_s5r2_sequence',
    'putnextlocal',
    'bosslevel',
)

# Network sizes. Not exposed on the CLI, matching ppo.py's own
# choice to hard-code its conv stack rather than parameterize it.
MISSION_EMBED_DIM = 32
MISSION_HIDDEN = 64
CNN_PROJ_DIM = 256
CORE_HIDDEN = 512


@dataclass
class Args:
    """
    Command-line arguments for a BabyAI PPO run.

    The core PPO knobs match algos/ppo.py exactly (same defaults,
    same meanings), so any behavior difference on a BabyAI task is
    attributable to the mission input and recurrence, not to tuning
    drift. New knobs are the recurrent minibatching constraint
    (num_minibatches must divide num_envs, checked in train()) and
    the greedy-eval block shared with every other algorithm.
    """

    # --- Experiment identity and logging ---
    # Must be one of BABYAI_TASKS; gotolocal is the simplest one,
    # good for a first smoke run.
    task: str = 'gotolocal'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = 'vlm-rl-bench'

    # --- Observation settings ---
    # 'historical' (fog of war) is the project's standard for every
    # Stage 1+ comparison -- see docs/stage0_ppo.md. This wrapper
    # governs what the agent remembers about the MAP; it is
    # unrelated to the mission-progress memory this file adds.
    obs_mode: str = 'historical'
    agent_view_size: int = 7
    obs_tile_size: int = 0
    obs_target_size: int = 56

    # --- Core PPO hyperparameters (identical to ppo.py) ---
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

    # --- Greedy evaluation (monitoring/eval.py) ---
    eval_interval: int = 50
    eval_episodes: int = 10
    # Also evaluate the SAMPLED policy alongside the greedy one. A
    # deterministic argmax policy can cycle in a gridworld and report
    # 0.00 however much it learned, so the two numbers together
    # separate 'learned nothing' from 'the argmax loops'.
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
    Embed a tokenized mission and summarize it with a GRU.

    The final hidden state (not a pooled average) is used as the
    mission vector: a GRU's last state is already a function of the
    whole sequence, and BabyAI missions are short (observed 2-33
    tokens across gotolocal/gotoseq/putnextlocal/bosslevel, a
    provable ceiling of 72 -- see envs/mission_vocab.py), so there
    is little to gain from pooling. pack_padded_sequence with the
    TRUE lengths means padding tokens are never fed to the GRU at
    all, regardless of how large mission_max_len is.
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

        Parameters
        ----------
        mission_ids: LongTensor, shape (B, L)
        mission_len: LongTensor, shape (B,)
            True (unpadded) token counts. Clamped to at least 1 so
            pack_padded_sequence never sees a zero-length sequence,
            which it rejects.

        Returns
        -------
        FloatTensor, shape (B, hidden_size)
        """

        embedded = self.embedding(mission_ids)
        # Lengths must live on the CPU regardless of where the data
        # tensor lives; this is a hard requirement of
        # pack_padded_sequence, not a style choice.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            mission_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        # hidden: (num_layers=1, B, hidden_size) -> (B, hidden_size).
        return hidden.squeeze(0)


class RecurrentAgent(nn.Module):
    """
    CNN + mission encoder + recurrent core, actor-critic heads.

    Structurally: the image CNN is the exact conv stack
    algos/ppo.py's Agent uses (so the visual features it can learn
    are not the reason for any behavior difference), projected and
    concatenated with the mission vector, fused through one linear
    layer, then carried through a single-layer GRU across time
    before the actor/critic heads. Concatenation was chosen over
    FiLM conditioning (mission-vector-generated per-channel scale/
    shift inside the conv stack) for this first version: simpler to
    get right, still a legitimate architecture, with FiLM flagged
    as a possible upgrade if this underperforms.
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
            rather than a render, which differs from pixels in two
            ways. The values are category codes (4 means 'door', not
            'brightness 4'), so they are embedded rather than scaled
            by 1/255, which would assert a magnitude ordering that
            does not exist. And the grids are small -- 7x7 up to
            22x22 -- so the Atari stack's 8x8 stride-4 first layer is
            larger than the whole input and raises. Derive this with
            algos.nets.is_symbolic_mode(args.obs_mode).
        """

        super().__init__()

        obs_shape = envs.single_observation_space['image'].shape

        # The shared encoder handles both observation families and is
        # the same one ppo_distill.py and ppo_teacher.py use, so a
        # BabyAI run is comparable to the goal-fixed grid at the same
        # obs_mode. Its pixel branch is the identical Atari stack this
        # file used before, so existing RGB results are unaffected.
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

        # The recurrent core tracks the agent's own progress through
        # the mission across steps -- e.g. whether GoToSeq's first
        # clause has already been satisfied -- which nothing in a
        # single frame (however much of the map it shows) can
        # reveal on its own. batch_first=False (the nn.GRU default):
        # inputs are (seq_len, batch, input_size).
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
        recurrent core, resetting the core wherever an episode
        genuinely starts.

        Parameters
        ----------
        x: dict of tensors
            x['image']: (B, H, W, C) ; x['mission_ids']: (B, L) ;
            x['mission_len']: (B,). B is either E (one rollout
            step, all envs) or T*E (a full stored sequence replayed
            for T*num_envs_per_minibatch, T-major -- see
            algos/ppo_babyai.py's train() for how callers build
            this).
        core_state: FloatTensor, shape (1, E, CORE_HIDDEN)
            E (envs in THIS call) is read from this tensor's own
            shape, which is how a single function handles both the
            one-step rollout call and the full-sequence replay call
            uniformly.
        episode_start: FloatTensor, shape (B,)
            1.0 at exactly the observations that are a genuine
            fresh episode start (see the module docstring for why
            this is NOT the same as the `dones` array): the core
            state is zeroed before processing those, and only
            those.

        Returns
        -------
        hidden: FloatTensor, shape (B, CORE_HIDDEN)
        core_state: FloatTensor, shape (1, E, CORE_HIDDEN)
            The state after processing the last timestep in `x`.
        """

        cnn_feat = self.encoder(x['image'])
        mission_feat = self.mission_encoder(
            x['mission_ids'], x['mission_len']
        )
        fused = self.fusion(
            torch.cat([cnn_feat, mission_feat], dim=-1)
        )

        num_envs_here = core_state.shape[1]
        # (B, CORE_HIDDEN) -> (T, E, CORE_HIDDEN). B == E means
        # T == 1 (the rollout case); B == T*E replays a full stored
        # sequence (the update case, T = num_steps -- 128 by
        # default). Both are handled by the same chunking below.
        fused = fused.reshape(-1, num_envs_here, fused.shape[-1])
        ep_start = episode_start.reshape(-1, num_envs_here)
        seq_len = fused.shape[0]

        # nn.GRU can process an entire multi-step sequence in ONE
        # call; only a genuine episode boundary forces starting a
        # new call, because that is the only place the hidden state
        # needs to be forcibly zeroed for some envs rather than
        # carried forward. Finding those boundaries once, up front,
        # replaces what used to be `seq_len` separate single-step
        # GRU calls (up to 128 of them, called seq_len times per
        # minibatch during the PPO update) with typically just one
        # or a handful of calls -- the interior of a boundary-free
        # run is mathematically IDENTICAL whether processed as one
        # batched call or one step at a time (that is what a GRU's
        # own internal recurrence already does), so this changes
        # nothing about the result, only how many Python-level calls
        # it costs to compute it. This was the dominant cost behind
        # a >30x cluster-vs-local slowdown observed in practice:
        # per-call dispatch overhead is cheap on an idle local CPU
        # but gets punished hard on a contended/shared cluster node,
        # and the naive version paid that cost seq_len times over.
        resets_any_env = (ep_start != 0).any(dim=1)
        # Index 0 always starts the first chunk; any later index
        # where some env resets also starts a new chunk.
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
            # Zero exactly the envs resetting at this chunk's first
            # step; every other env's state carries in unchanged.
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
        core state (a read-only bootstrap query must not perturb
        the state the next rollout step will actually use).
        """

        hidden, _ = self.get_states(x, core_state, episode_start)
        return self.critic(hidden)

    def get_action_and_value(
        self, x, core_state, episode_start, action=None
    ):
        """
        Sample (or evaluate) an action and return it alongside its
        log probability, the policy entropy, the state value, and
        the updated recurrent core state.
        """

        hidden, core_state = self.get_states(
            x, core_state, episode_start
        )
        logits = self.actor(hidden)
        if getattr(self, 'capture_auxiliary_features', False):
            self.auxiliary_features = hidden
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            core_state,
        )


def build_select_action(agent, device, greedy=True):
    """
    Build a stateful action-selection closure for
    monitoring.eval.greedy_eval.

    greedy_eval runs one episode at a time and calls the optional
    `.reset()` attribute (if present) right after each env.reset(),
    which is the only signal this closure needs: since episodes
    never interleave here (unlike training's num_envs parallel
    envs), `episode_start` can simply always be 0 -- reset() already
    zeroes the hidden state at exactly the right moment.

    Parameters
    ----------
    greedy: bool
        True takes the argmax action, the headline teacher-off
        measurement. False samples from the policy instead.

    Both are worth logging because an argmax policy is deterministic
    and can cycle in a gridworld -- turn left, turn right, forever --
    which reports as a flat 0.00 however much the policy learned.
    That was reproduced elsewhere in this repo at 0.00 greedy against
    0.67 sampled on identical episodes, so a teacher arm scoring 0.00
    here cannot be called a failure without the sampled number beside
    it.
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


def create_run_directory(args, repo_root):
    """
    Allocate a unique output directory even for simultaneous launches.

    Views share task and seed in a factorial. A timestamp alone lets
    concurrent jobs overwrite one another, including their checkpoints.
    Atomic directory creation also separates deliberate same-view reruns.
    """

    runs_root = os.path.join(repo_root, 'results', 'runs')
    os.makedirs(runs_root, exist_ok=True)
    identity = getattr(args, 'experiment_id', '')
    identity = f'_{identity}' if identity else ''
    prefix = (f'{args.task}__ppo_babyai_{args.obs_mode}{identity}__'
              f'{args.seed}__{int(time.time())}__')
    return tempfile.mkdtemp(prefix=prefix, dir=runs_root)


def train(args, auxiliary=None):
    """
    Train a recurrent, mission-conditioned PPO agent on a BabyAI
    task and log to TensorBoard.

    The skeleton matches algos/ppo.train; the differences are (1)
    the Dict observation (image + tokenized mission) built via
    build_env(..., obs_mission=True), (2) the recurrent core and
    its episode_start-gated reset, and (3) env-level minibatching in
    place of ppo.py's flat (step, env) shuffle, required because a
    recurrent core needs a temporally contiguous sequence to replay.
    """

    if args.task not in BABYAI_TASKS:
        raise ValueError(
            f'algos.ppo_babyai only supports the language-'
            f'conditioned BabyAI tasks {BABYAI_TASKS}; got '
            f'{args.task!r}. Non-language tasks (doorkey_*, '
            f'keycorridor_s6r3, ...) do not need a mission input '
            f'-- use algos.ppo instead.'
        )

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # A recurrent minibatch must contain a WHOLE env's contiguous
    # sequence (see the module docstring), so minibatches are built
    # by splitting envs into equal-sized groups -- num_envs must
    # divide evenly, a stricter constraint than plain PPO's
    # batch_size % num_minibatches (which always holds trivially).
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError(
            f'recurrent minibatching needs num_envs '
            f'({args.num_envs}) divisible by num_minibatches '
            f'({args.num_minibatches}): a minibatch is a group of '
            f'whole envs replayed over their full stored sequence, '
            f'not an arbitrary sample of timesteps.'
        )
    envs_per_batch = args.num_envs // args.num_minibatches

    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    run_dir = create_run_directory(args, repo_root)
    run_name = os.path.basename(run_dir)
    print(f'Run directory: {run_dir}', flush=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

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

    # Load the frozen mission vocabulary once and share the same
    # instance across every parallel env (and the eval env below),
    # rather than each sub-env re-reading the vocab file.
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

    symbolic = is_symbolic_mode(args.obs_mode)
    agent = RecurrentAgent(
        envs, vocab_size=vocab.size, symbolic=symbolic
    ).to(device)
    if auxiliary is not None:
        auxiliary.initialize(agent, sync_envs, run_dir, device, args)
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    image_shape = envs.single_observation_space['image'].shape
    mission_shape = envs.single_observation_space['mission_ids'].shape
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo='ppo_babyai',
        args=args,
        obs_shape=image_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=runtime_metadata(),
    )

    # Rollout storage. Observations are stored per-key (a Dict
    # space, unlike ppo.py's single image tensor); mission_ids and
    # mission_len are integer buffers since they are used as
    # nn.Embedding indices and pack_padded_sequence lengths, never
    # as network input pixels.
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
    # episode_starts is the shifted reset signal the recurrent core
    # consumes -- see the module docstring for why it is NOT the
    # same array as `dones`.
    episode_starts = torch.zeros(
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
    # The very first observation of training genuinely follows a
    # reset, so this starts at 1; numerically it makes no difference
    # here since core_state is already all zeros, but it is the
    # semantically honest value.
    episode_start = torch.ones(args.num_envs).to(device)
    core_state = agent.initial_core_state(args.num_envs, device)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    last_eval = None

    for iteration in range(1, args.num_iterations + 1):

        if auxiliary is not None:
            auxiliary.begin_rollout(iteration)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # Snapshot the recurrent state at the start of this
        # iteration's rollout: the update phase below replays each
        # minibatch's stored sequence starting from exactly this
        # state, so the recomputed quantities match what the
        # rollout actually saw.
        initial_core_state = core_state.detach().clone()

        # --- Rollout collection ---
        for step in range(args.num_steps):
            if auxiliary is not None:
                auxiliary.observe(step, global_step, sync_envs, next_done)
            global_step += args.num_envs
            obs_image[step] = next_obs['image']
            obs_mission_ids[step] = next_obs['mission_ids']
            obs_mission_len[step] = next_obs['mission_len']
            dones[step] = next_done
            # episode_start here is the value carried in from the
            # PREVIOUS step (or the previous iteration's last step,
            # or the pre-loop initialization) -- i.e. dones[step-1],
            # the one-step shift the module docstring explains.
            episode_starts[step] = episode_start

            with torch.no_grad():
                action, logprob, _, value, core_state = (
                    agent.get_action_and_value(
                        next_obs, core_state, episode_start
                    )
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # dones[step] (just stored) tells us whether the NEXT
            # observation will be a fresh reset -- feed it forward
            # as next step's episode_start.
            episode_start = dones[step].clone()

            next_obs_np, reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            next_done = torch.Tensor(
                np.logical_or(terminations, truncations)
            ).to(device)
            rewards[step] = (
                torch.tensor(reward).to(device).view(-1)
            )
            next_obs = _to_device_obs(next_obs_np)

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

        # --- Advantage estimation (GAE) ---
        # Unchanged from ppo.py: GAE only ever consults `dones`,
        # never `episode_starts` -- bootstrapping across a genuine
        # episode boundary is exactly what GAE's own nonterminal
        # mask already handles correctly.
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

        # Flatten (num_steps, num_envs, ...) to (num_steps*num_envs,
        # ...), step-major (env varies fastest) -- this is the same
        # C-order reshape gymnasium/torch use by default, and it is
        # what flat_grid below assumes when selecting a minibatch's
        # timesteps.
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

        # flat_grid[t, e] is obs_image[t, e]'s index in the flat
        # arrays above -- used below to select, for a chosen group
        # of envs, every timestep of their stored sequence in the
        # right (T-major) order for get_states to reshape correctly.
        flat_grid = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )

        # --- PPO update: env-level minibatches ---
        # A recurrent core needs a temporally contiguous run of an
        # env's full stored sequence to replay meaningfully, so
        # (unlike ppo.py's flat (step, env) shuffle) minibatches
        # here are whole ENVS, not arbitrary timesteps.
        env_inds = np.arange(args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, envs_per_batch):
                mb_envs = env_inds[start:start + envs_per_batch]
                # T-major flat indices for exactly these envs' full
                # num_steps sequence.
                mb_flat = flat_grid[:, mb_envs].ravel()

                mb_x = {
                    'image': b_obs_image[mb_flat],
                    'mission_ids': b_obs_mission_ids[mb_flat],
                    'mission_len': b_obs_mission_len[mb_flat],
                }
                mb_episode_start = b_episode_starts[mb_flat]
                # The recurrent replay starts from exactly the state
                # this iteration's rollout started from, for these
                # envs specifically.
                mb_init_core = initial_core_state[:, mb_envs]

                _, newlogprob, entropy, newvalue, _ = (
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

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                if auxiliary is not None:
                    loss = loss + auxiliary.loss(
                        agent.auxiliary_features, mb_flat, ppo_loss=loss
                    )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()
                if auxiliary is not None:
                    auxiliary.optimizer_step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # --- Teacher-off greedy evaluation ---
        # There is no teacher in this file, but the same shared
        # evaluator every other algorithm uses keeps the reported
        # curve directly comparable (see monitoring/eval.py).
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
            # The sampled pass distinguishes 'the policy learned
            # nothing' from 'the argmax of this policy loops', which
            # a single greedy number cannot.
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

            with open(os.path.join(run_dir, 'evaluations.jsonl'), 'a',
                      encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'global_step': global_step,
                    'iteration': iteration,
                    'teacher_on': False,
                    'episodes': args.eval_episodes,
                    'seed_base': args.seed + 50_000,
                    'training_episodes': tracker.episode_count,
                    'wall_time_sec': time.time() - start_time,
                    **last_eval,
                }, allow_nan=False) + '\n')

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
            # Without the sampled rate a 0.00 is unreadable: it could
            # mean the policy learned nothing, or that its argmax
            # cycles in the gridworld.
            if 'sampled_success_rate' in last_eval:
                extra['eval_sampled_success_rate'] = last_eval[
                    'sampled_success_rate'
                ]
        tracker.write_summary(
            status='running', global_step=global_step, extra=extra
        )
        if auxiliary is not None:
            auxiliary.record_iteration(global_step, last_eval)
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

    # Keep the trained policy. Without it a finished run cannot be
    # reloaded for a transfer or hand-off experiment, and the whole
    # run would have to be repeated to get one.
    torch.save(
        agent.state_dict(), os.path.join(run_dir, 'agent.pt')
    )

    tracker.close(status='completed', global_step=global_step)
    if auxiliary is not None:
        auxiliary.finish(agent, global_step)
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
