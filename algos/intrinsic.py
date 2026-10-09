"""
Pluggable intrinsic-reward (exploration bonus) modules.

Why this file exists
--------------------
The repo's only teacher-free exploration baseline was RND
(algos/ppo_rnd.py), and it scored 0.00 on KeyCorridorS3R3 -- a level
the published RND results DO solve. A baseline that fails where the
literature says it succeeds cannot support the claim "no teacher-free
method reaches this task"; it only says our RND was too weak. RND is
also six years old and is the *first* method every later MiniGrid
exploration paper beats.

This module implements the bonuses that actually define the MiniGrid
frontier, behind one interface, so they can be swapped on the command
line and compared under an identical PPO loop, budget, and evaluation
protocol:

- `count`  -- episodic or global visitation counts, 1/sqrt(n). The
              cheapest possible bonus and a sanity check: if this
              solves a task, no learned novelty model is needed.
- `rnd`    -- Random Network Distillation (Burda et al. 2019). Global
              novelty from a frozen random target's prediction error.
- `noveld`  -- NovelD (Zhang et al. 2021). RND novelty *differences*
              across a transition, gated to the first visit of a
              state within the episode. Reported to solve every
              static procedurally-generated MiniGrid task at 120M
              frames -- including KeyCorridorS6R3 and
              ObstructedMaze-Full -- where RND/RIDE/AMIGo/ICM plateau.
- `re3`    -- Random Encoders for Efficient Exploration (Seo et al.
              2021). k-nearest-neighbour state entropy in a FIXED
              random embedding: no learned model at all, so it is
              nearly free, yet reported at 0.95 return on
              DoorKey-8x8 in 1M steps.
- `e3b`    -- Elliptical Episodic Bonus (Henaff et al. 2022). An
              episodic bonus in a learned inverse-dynamics embedding,
              which extends count-based episodic novelty to
              continuous features.

The common interface
--------------------
Every bonus is computed ONCE PER ROLLOUT rather than once per step:

    bonus.rollout_bonus(obs, next_obs, actions, dones) -> (T, E)

`dones[t, e] == 1` marks that `obs[t, e]` is the first observation of
a fresh episode (this is exactly the convention algos/ppo.py already
stores), which is the only signal the episodic bonuses need to know
where to reset their per-episode state. Bonuses that keep per-episode
state therefore walk the rollout in time order internally; the purely
batched ones (RE3) ignore `dones` entirely.

Bonuses with trainable parts expose them through
`trainable_parameters()` (added to the PPO optimizer) and
`aux_loss(...)` (added to the PPO loss). Stateless bonuses return an
empty list and None, so the training loop needs no special cases.

A note on observation modes
---------------------------
The episodic bonuses (`count`, `noveld`'s first-visit gate, `e3b`)
have to recognize a repeat visit, and they do so by hashing the
observation. That makes the definition of "the same state" depend on
the observation mode, in ways worth stating because they are not
obvious:

- `historical` (the repo default) is a full-map render with a
  persistent visibility mask, so the hash covers the agent's pose AND
  everything revealed so far. Two visits to one cell count as a
  revisit only if the revealed set also matches -- a FINER partition
  than the position-counting used in the literature.
- `partial` / `symbolic` hash only the current egocentric view, so
  two different locations that happen to look identical are counted
  as the same state -- a COARSER partition.

An earlier version of this file asserted that `historical` made every
observation unique and therefore broke the episodic bonuses. Measured
on 4k-frame random-ish rollouts, that is simply false; the fraction of
steps that were first visits came out at 0.24 (KeyCorridorS3R3) and
0.11 (DoorKey-8x8) under `historical`, versus 0.13/0.13 under
`symbolic` and 0.15/0.23 under `partial`. Revisits are detected in
every mode, and `historical` is not the outlier.

The `first_visit_frac` diagnostic is logged every iteration precisely
so this is measured rather than assumed: a value pinned near 1.0 would
mean the gate is inert and the episodic component is doing nothing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.nets import MissionEncoder


# Observations with fewer than this many cells on a side are treated
# as symbolic grids and get small convolution kernels. Shared with
# algos/ppo_intrinsic.py's Agent so the policy and the bonus networks
# always make the same choice for a given observation.
SMALL_INPUT_THRESHOLD = 32

# Spatial size the small-input (symbolic) conv stack is pooled down
# to before flattening. The stride-1 padded convolutions preserve the
# grid, so a whole-map 16x16 encoding would flatten to 64*16*16 =
# 16384 features and an 8.5M-parameter projection that measured at
# 31 SPS -- an order of magnitude slower than every other mode. This
# caps the flatten at 64*7*7 = 3136 regardless of map size, and is a
# no-op on the 7x7 egocentric modes.
SMALL_INPUT_POOL = 7


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill its
    bias, matching the initialization used across algos/.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_conv_encoder(obs_shape, out_dim, activation=nn.ReLU):
    """
    Build a convolutional encoder sized to the observation shape.

    Two regimes matter here, and getting this wrong is silent: the
    RGB modes give ~56x56 images, for which the standard Atari conv
    stack (8x8 stride 4, then 4x4 stride 2) is right, but the
    symbolic modes give a 7x7 (or map-sized) integer grid, on which
    an 8x8 stride-4 kernel is larger than the entire input and would
    either error or collapse everything to a single cell. Small
    inputs therefore get 3x3 stride-1 convolutions, which is the
    encoder shape the MiniGrid exploration literature uses.

    Parameters
    ----------
    obs_shape: tuple
        The (H, W, C) observation shape as reported by the env.
    out_dim: int
        Width of the final linear projection.
    activation: callable
        Activation module constructor. RND conventionally uses
        LeakyReLU for its two networks; ReLU elsewhere.

    Returns
    -------
    nn.Module
        A module mapping (N, C, H, W) float input to (N, out_dim).
    """

    h, w, c = obs_shape

    # The Atari stack shrinks its input hard: 8x8/stride-4, then
    # 4x4/stride-2, then 3x3. A 16x16 input is already down to 3x3
    # after the first layer, so the second layer's 4x4 kernel is
    # larger than what is left and torch raises. Anything under 32 on
    # a side is therefore a symbolic grid (the largest map here is
    # 16x16) and gets 3x3 stride-1 kernels that preserve the grid.
    if min(h, w) < SMALL_INPUT_THRESHOLD:
        conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 3, stride=1, padding=1)),
            activation(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            activation(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            activation(),
            # Cap the spatial size so a whole-map symbolic encoding
            # does not blow up the projection; see SMALL_INPUT_POOL.
            nn.AdaptiveMaxPool2d(min(SMALL_INPUT_POOL, min(h, w))),
            nn.Flatten(),
        )
    else:
        conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            activation(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            activation(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            activation(),
            nn.Flatten(),
        )

    # Discover the flattened width with a dummy pass rather than
    # hard-coding it, so the encoder adapts to any obs_mode.
    with torch.no_grad():
        n_flatten = conv(torch.zeros(1, c, h, w)).shape[1]

    return nn.Sequential(conv, layer_init(nn.Linear(n_flatten, out_dim)))


class StateEncoder(nn.Module):
    """
    Encode an observation, and the instruction when there is one.

    RND, RE3 and E3B all need one vector per state. On the goal-fixed
    tasks that is a convolution over the observation and nothing
    else, which is what `build_conv_encoder` gives. On the BabyAI
    language tasks the instruction is part of the state: E3B's
    encoder is trained to predict the action taken between two
    observations, and on GoToSeq that action depends on which object
    the instruction names. A mission-blind encoder cannot fit that
    mapping, so its features -- and the elliptical bonus computed
    from them -- describe the wrong space.

    The mission vector is concatenated to the convolutional features
    before the output projection, so `out_dim` is unchanged and every
    consumer downstream is unaffected.
    """

    def __init__(self, obs_shape, out_dim, activation=nn.ReLU,
                 mission_vocab=0):
        """
        Build the convolutional path and, when the task carries an
        instruction, the mission path feeding a joint projection.
        """

        super().__init__()

        h, w, c = obs_shape
        encoder = build_conv_encoder(
            obs_shape, out_dim, activation=activation
        )
        if not mission_vocab:
            self.conv = encoder
            self.mission = None
            self.project = None
            return

        # Keep only the convolution half of the sized stack and
        # re-project, so the mission vector joins before the output.
        self.conv = encoder[0]
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.mission = MissionEncoder(mission_vocab)
        self.project = layer_init(
            nn.Linear(n_flatten + self.mission.output_dim, out_dim)
        )

    def forward(self, obs_cf, mission_ids=None, mission_len=None):
        """
        Map channels-first observations, and optional instructions,
        to a feature vector of width out_dim.
        """

        if self.mission is None:
            return self.conv(obs_cf)
        # An encoder built for a language task must be given the
        # language. Silently substituting zeros would train the
        # projection on an input it never sees at rollout time.
        if mission_ids is None:
            raise ValueError(
                'mission-aware StateEncoder called without a mission'
            )
        return self.project(
            torch.cat(
                [self.conv(obs_cf),
                 self.mission(mission_ids, mission_len)],
                dim=1,
            )
        )


class RNDPredictor(nn.Module):
    """
    RND's trainable predictor: a state encoder plus an MLP head.

    A module rather than two loose attributes, so that
    `bonus.predictor.parameters()` keeps working and the mission
    arguments reach the encoder without every call site unpacking it.
    """

    def __init__(self, encoder, feature_dim):
        """
        Wrap an encoder with the extra capacity the predictor needs to
        actually fit the frozen target on states it has visited.
        """

        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
        )

    def forward(self, obs_cf, mission_ids=None, mission_len=None):
        """
        Map normalized observations to the predictor's features.
        """

        return self.head(
            self.encoder(obs_cf, mission_ids, mission_len)
        )


class RunningMeanStd:
    """
    Online mean and variance via Welford's parallel algorithm.

    Used to standardize the observations fed to the bonus networks
    (RND is well known to produce noise without this) without having
    to store the data. Duplicated from algos/ppo_rnd.py so this
    module has no import dependency on a specific algorithm file.
    """

    def __init__(self, epsilon=1e-4, shape=()):
        """
        Start from zero mean, unit variance, and a tiny pseudo-count
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

        self._update_from_moments(
            np.mean(x, axis=0), np.var(x, axis=0), x.shape[0]
        )

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Combine the stored statistics with a batch's moments -- the
        Welford merge step that keeps mean and variance exact.
        """

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = (
            m_a + m_b
            + delta**2 * self.count * batch_count / tot_count
        )
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class IntrinsicBonus(nn.Module):
    """
    Base class defining the interface the PPO loop consumes.

    Subclasses override `rollout_bonus`, and optionally `aux_loss`
    (a term added to the PPO loss) and `reset_all` (clearing
    per-episode state). Defaults make a stateless, untrained bonus
    work with no special-casing in the training loop.
    """

    def __init__(self, obs_shape, num_envs, device, mission_vocab=0):
        """
        Record the shapes and device every subclass needs.

        Parameters
        ----------
        mission_vocab: int
            Vocabulary size for the BabyAI language tasks, or 0 for
            the goal-fixed tasks that carry no instruction. When
            non-zero the bonus is mission-aware: novelty is measured
            over (observation, instruction) rather than observation
            alone, which is required for correctness on tasks where
            the same observation implies different behaviour under
            different instructions.
        """

        super().__init__()
        self.obs_shape = obs_shape
        self.num_envs = num_envs
        self.device = device
        self.mission_vocab = mission_vocab

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Return the per-step intrinsic reward for a whole rollout.

        Parameters
        ----------
        obs: FloatTensor, shape (T, E) + obs_shape
            The observation the agent acted on at each step.
        next_obs: FloatTensor, shape (T, E) + obs_shape
            The observation that resulted from that action.
        actions: LongTensor, shape (T, E)
            The action taken, needed only by bonuses that train an
            inverse-dynamics model.
        dones: FloatTensor, shape (T, E)
            1.0 where obs[t, e] is the first observation of a fresh
            episode, which is where episodic state must be reset.
        mission_ids: LongTensor, shape (T, E, L) or None
            Tokenized instruction, on the language tasks only.
        mission_len: LongTensor, shape (T, E) or None
            True token counts for those instructions.

        Returns
        -------
        FloatTensor, shape (T, E)
            The raw (un-normalized) intrinsic reward.
        """

        raise NotImplementedError

    def aux_loss(self, mb_obs, mb_next_obs, mb_actions,
                 mb_mission_ids=None, mb_mission_len=None):
        """
        Return a scalar loss to add to the PPO objective, or None if
        this bonus trains nothing.
        """

        return None

    def trainable_parameters(self):
        """
        Return the parameters the PPO optimizer should also update.
        """

        return [p for p in self.parameters() if p.requires_grad]

    def reset_all(self):
        """
        Clear any per-episode state, for all envs at once.
        """

        return None

    def diagnostics(self):
        """
        Return a dict of scalars worth logging for this bonus.
        """

        return {}


class ObsNormalizer:
    """
    Running standardization of observations for the bonus networks.

    RND and its descendants regress one network onto another in
    observation space, which only behaves when the inputs are
    standardized; the raw scale differs wildly between the RGB modes
    (0-255 pixels) and the symbolic modes (small category indices),
    so a single hard-coded /255 would be wrong for half the configs.
    Tracking the actual statistics handles both without a flag.
    """

    def __init__(self, obs_shape, device, clip=5.0):
        """
        Track statistics in channels-first layout, matching the
        conv encoders these values are fed to.
        """

        h, w, c = obs_shape
        self.rms = RunningMeanStd(shape=(c, h, w))
        self.device = device
        self.clip = clip

    def update(self, obs_flat):
        """
        Fold a flat (N, H, W, C) batch of observations into the
        running statistics.
        """

        self.rms.update(
            obs_flat.permute(0, 3, 1, 2).cpu().numpy().astype('float64')
        )

    def __call__(self, obs_flat):
        """
        Standardize and clip a flat (N, H, W, C) batch, returning it
        channels-first and ready for a conv encoder.
        """

        obs_cf = obs_flat.permute(0, 3, 1, 2)
        mean = torch.as_tensor(
            self.rms.mean, device=self.device, dtype=torch.float32
        )
        std = torch.as_tensor(
            np.sqrt(self.rms.var),
            device=self.device,
            dtype=torch.float32,
        )
        return torch.clamp(
            (obs_cf - mean) / (std + 1e-8), -self.clip, self.clip
        )


def _flatten_mission(mission_ids, mission_len):
    """
    Collapse a (T, E, L) mission batch to (T*E, L) so it lines up
    with a flattened observation batch.

    None passes straight through, which is what the goal-fixed tasks
    supply -- they carry no instruction.
    """

    if mission_ids is None:
        return None, None
    return (
        mission_ids.reshape(-1, mission_ids.shape[-1]),
        mission_len.reshape(-1),
    )


def _hash_obs(obs_row, mission_row=None):
    """
    Turn one observation array into a hashable key.

    Counting visits needs an exact identity for a state. Hashing the
    observation's raw bytes is the standard choice for MiniGrid count
    baselines and, unlike reading the simulator's internal agent
    position, it stays honest to what the agent can actually perceive
    -- two states the agent cannot tell apart are counted as one.

    On the language tasks the instruction is part of what the agent
    perceives, so it must be part of the identity. Without it the
    same cell under 'go to the red ball' and 'go to the green key'
    hashes to one state, and the bonus reports 'already visited'
    somewhere the agent has never been under this instruction. Within
    a single episode the mission is constant, so this changes nothing
    for the episodic scopes -- it matters for global scope and for
    NovelD's lifetime novelty table, which span missions.
    """

    if mission_row is None:
        return hash(obs_row.tobytes())
    return hash((obs_row.tobytes(), mission_row.tobytes()))


class CountBonus(IntrinsicBonus):
    """
    Classic count-based novelty: reward 1 / sqrt(n(s)).

    Kept deliberately as the floor of the comparison. It has no
    learned parts and costs almost nothing, so if it moves a task
    that plain PPO cannot, the task's difficulty was bookkeeping,
    not representation learning -- a result worth knowing before
    attributing anything to a teacher.
    """

    def __init__(self, obs_shape, num_envs, device, scope='episodic',
                 mission_vocab=0):
        """
        Build the per-env visitation tables.

        Parameters
        ----------
        scope: str
            'episodic' counts within the current episode only (reset
            at every episode boundary), 'global' counts across the
            whole run. Episodic is the variant the modern methods
            build on.
        """

        super().__init__(obs_shape, num_envs, device,
                         mission_vocab=mission_vocab)
        if scope not in ('episodic', 'global'):
            raise ValueError(
                f'count scope must be episodic or global, got {scope!r}'
            )
        self.scope = scope
        # One dictionary per env for episodic counts; a single shared
        # dictionary when counting globally.
        self.counts = [{} for _ in range(num_envs)]
        self.global_counts = {}
        self.last_unique_frac = 0.0

    def reset_all(self):
        """
        Clear every env's episodic table.
        """

        self.counts = [{} for _ in range(self.num_envs)]

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Walk the rollout in time order, counting each visited next
        observation and emitting 1/sqrt(count).
        """

        num_steps, num_envs = dones.shape
        next_np = next_obs.cpu().numpy()
        dones_np = dones.cpu().numpy()
        mission_np = (
            None if mission_ids is None else mission_ids.cpu().numpy()
        )
        bonus = np.zeros((num_steps, num_envs), dtype=np.float32)

        first_visits = 0
        for t in range(num_steps):
            for e in range(num_envs):
                # A done flag on obs[t] means a new episode starts
                # here, so this env's episodic table is stale.
                if self.scope == 'episodic' and dones_np[t, e] > 0:
                    self.counts[e] = {}
                table = (
                    self.global_counts
                    if self.scope == 'global'
                    else self.counts[e]
                )
                key = _hash_obs(
                    next_np[t, e],
                    None if mission_np is None else mission_np[t, e],
                )
                n = table.get(key, 0) + 1
                table[key] = n
                if n == 1:
                    first_visits += 1
                bonus[t, e] = 1.0 / np.sqrt(n)

        # Report how much of the rollout was genuinely new. Near 1.0
        # under obs_mode='historical' is the degenerate case the
        # module docstring warns about, not successful exploration.
        self.last_unique_frac = first_visits / max(
            1, num_steps * num_envs
        )
        return torch.as_tensor(bonus, device=self.device)

    def diagnostics(self):
        """
        Report the fraction of the last rollout that was first-visit.
        """

        return {'first_visit_frac': self.last_unique_frac}


class RNDBonus(IntrinsicBonus):
    """
    Random Network Distillation: novelty as prediction error.

    A frozen, randomly initialized target network defines an
    arbitrary function of the observation; a predictor is trained to
    copy it on visited states. Where the predictor is still wrong,
    the agent has not been often -- so the squared error is a global
    (not per-episode) novelty signal.
    """

    def __init__(
        self,
        obs_shape,
        num_envs,
        device,
        feature_dim=512,
        update_proportion=0.25,
        mission_vocab=0,
    ):
        """
        Build the frozen target and the deeper trainable predictor.
        """

        super().__init__(obs_shape, num_envs, device,
                         mission_vocab=mission_vocab)

        self.normalizer = ObsNormalizer(obs_shape, device)
        self.update_proportion = update_proportion

        # The target is a fixed random function, never trained.
        self.target = StateEncoder(
            obs_shape, feature_dim, activation=nn.LeakyReLU,
            mission_vocab=mission_vocab,
        )
        for param in self.target.parameters():
            param.requires_grad = False

        # The predictor gets extra capacity so it can actually fit
        # the target on the states it has seen.
        self.predictor = RNDPredictor(
            StateEncoder(
                obs_shape, feature_dim, activation=nn.LeakyReLU,
                mission_vocab=mission_vocab,
            ),
            feature_dim,
        )

    def novelty(self, obs_flat, mission_ids=None, mission_len=None):
        """
        Return the per-state prediction error for a flat batch of
        observations, which is this bonus's notion of novelty.
        """

        normalized = self.normalizer(obs_flat)
        with torch.no_grad():
            target_f = self.target(normalized, mission_ids, mission_len)
            predict_f = self.predictor(
                normalized, mission_ids, mission_len
            )
        return (target_f - predict_f).pow(2).sum(dim=1) / 2.0

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Score every next observation's novelty, and refresh the
        observation statistics with what was just collected.
        """

        num_steps, num_envs = dones.shape
        flat = next_obs.reshape((-1,) + tuple(self.obs_shape))
        ids, lens = _flatten_mission(mission_ids, mission_len)
        bonus = self.novelty(flat, ids, lens).reshape(
            num_steps, num_envs
        )
        # Track the shifting state distribution as the policy
        # improves, so normalization does not go stale.
        self.normalizer.update(flat)
        return bonus

    def aux_loss(self, mb_obs, mb_next_obs, mb_actions,
                 mb_mission_ids=None, mb_mission_len=None):
        """
        Train the predictor toward the target on a random subset of
        the minibatch. Subsampling slows the predictor down so the
        novelty signal is not erased immediately after one visit.
        """

        normalized = self.normalizer(mb_next_obs)
        target_f = self.target(
            normalized, mb_mission_ids, mb_mission_len
        ).detach()
        predict_f = self.predictor(
            normalized, mb_mission_ids, mb_mission_len
        )
        per_sample = F.mse_loss(
            predict_f, target_f, reduction='none'
        ).mean(dim=-1)
        mask = (
            torch.rand(len(per_sample), device=self.device)
            < self.update_proportion
        ).float()
        return (per_sample * mask).sum() / torch.clamp(
            mask.sum(), min=1.0
        )


class NovelDBonus(RNDBonus):
    """
    NovelD: reward the novelty DIFFERENCE across a transition, only
    on the first visit within the episode.

        b_t = max(N(s_{t+1}) - alpha * N(s_t), 0) * 1[first visit]

    The problem NovelD fixes is RND's failure mode on procedurally
    generated mazes: RND rewards absolute novelty, so an agent that
    finds one novel corridor keeps collecting reward for loitering
    there instead of pushing to the *boundary* of what it has
    explored. Taking the difference makes only crossing that
    boundary pay, and the first-visit gate stops the agent farming
    the same boundary crossing repeatedly within an episode.

    This is the method reported to solve every static
    procedurally-generated MiniGrid task at 120M frames without a
    curriculum -- including the KeyCorridorS6R3 and
    ObstructedMaze-Full levels where RND, RIDE, ICM and AMIGo all
    plateau near zero.
    """

    def __init__(
        self,
        obs_shape,
        num_envs,
        device,
        feature_dim=512,
        update_proportion=0.25,
        alpha=0.5,
        episodic_gate=True,
        mission_vocab=0,
    ):
        """
        Build the RND machinery plus the per-episode visit tables
        that implement the first-visit gate.

        Parameters
        ----------
        alpha: float
            Weight on the previous state's novelty in the
            difference. 0.5 is the value used in the paper.
        episodic_gate: bool
            When False the first-visit indicator is dropped, leaving
            the bare novelty difference. Useful as an ablation, and
            the honest setting when the observation cannot support
            revisit detection (see the module docstring).
        """

        super().__init__(
            obs_shape,
            num_envs,
            device,
            feature_dim=feature_dim,
            update_proportion=update_proportion,
            mission_vocab=mission_vocab,
        )
        self.alpha = alpha
        self.episodic_gate = episodic_gate
        self.visited = [set() for _ in range(num_envs)]
        self.last_gate_frac = 1.0

    def reset_all(self):
        """
        Clear every env's per-episode visited set.
        """

        self.visited = [set() for _ in range(self.num_envs)]

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Score both endpoints of every transition, take the clipped
        novelty difference, and gate it on first visit.
        """

        num_steps, num_envs = dones.shape
        shape = tuple(self.obs_shape)

        # Score both endpoints in one batched pass each. The mission
        # is constant across a transition, so the same flattened
        # instruction serves both endpoints.
        flat_obs = obs.reshape((-1,) + shape)
        flat_next = next_obs.reshape((-1,) + shape)
        ids, lens = _flatten_mission(mission_ids, mission_len)
        n_cur = self.novelty(flat_obs, ids, lens).reshape(
            num_steps, num_envs
        )
        n_next = self.novelty(flat_next, ids, lens).reshape(
            num_steps, num_envs
        )

        # The core NovelD criterion, floored at zero so that moving
        # BACK toward familiar states is never rewarded (and never
        # punished either -- the bonus is one-sided).
        diff = torch.clamp(n_next - self.alpha * n_cur, min=0.0)

        if self.episodic_gate:
            next_np = next_obs.cpu().numpy()
            dones_np = dones.cpu().numpy()
            mission_np = (
                None if mission_ids is None
                else mission_ids.cpu().numpy()
            )
            gate = np.zeros((num_steps, num_envs), dtype=np.float32)
            for t in range(num_steps):
                for e in range(num_envs):
                    # A fresh episode starts at obs[t], so anything
                    # remembered from the previous one is stale.
                    if dones_np[t, e] > 0:
                        self.visited[e] = set()
                    key = _hash_obs(
                        next_np[t, e],
                        None if mission_np is None
                        else mission_np[t, e],
                    )
                    if key not in self.visited[e]:
                        self.visited[e].add(key)
                        gate[t, e] = 1.0
            self.last_gate_frac = float(gate.mean())
            diff = diff * torch.as_tensor(gate, device=self.device)

        self.normalizer.update(flat_next)
        return diff

    def diagnostics(self):
        """
        Report how often the first-visit gate let a bonus through.
        A value pinned at 1.0 means the gate is doing nothing.
        """

        return {'first_visit_frac': self.last_gate_frac}


class RE3Bonus(IntrinsicBonus):
    """
    RE3: k-nearest-neighbour state entropy in a FIXED random
    embedding.

    A randomly initialized, never-trained encoder maps observations
    to features; the bonus is the distance from each state to its
    k-th nearest neighbour among the states in the same rollout.
    Spreading out in embedding space maximizes a particle estimate
    of state entropy, so the agent is pushed to cover the space
    rather than to any particular novel point.

    Its appeal here is cost: nothing is trained, so it adds one
    forward pass per rollout and no optimizer parameters at all --
    yet published MiniGrid results put it at 0.95 return on
    DoorKey-8x8 within 1M steps, which is the budget regime this
    project actually runs in.
    """

    def __init__(
        self, obs_shape, num_envs, device, feature_dim=128, k=3,
        mission_vocab=0
    ):
        """
        Build (and immediately freeze) the random encoder.
        """

        super().__init__(obs_shape, num_envs, device,
                         mission_vocab=mission_vocab)
        self.k = k
        self.encoder = StateEncoder(
            obs_shape, feature_dim, mission_vocab=mission_vocab
        )
        # The encoder is a fixed random projection: freezing it is
        # the whole point, not an optimization.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Embed every state in the rollout and return the log distance
        to each one's k-th nearest neighbour.
        """

        num_steps, num_envs = dones.shape
        flat = next_obs.reshape((-1,) + tuple(self.obs_shape))
        ids, lens = _flatten_mission(mission_ids, mission_len)
        with torch.no_grad():
            # Standardize scale-free: the random encoder does not
            # care about offset, but the pairwise distances do, so
            # feed it channels-first floats directly. A rollout spans
            # episode boundaries, so on the language tasks it holds
            # several different instructions at once -- the mission
            # must enter the embedding or states from unrelated
            # instructions become each other's nearest neighbours.
            z = self.encoder(flat.permute(0, 3, 1, 2), ids, lens)
            # Full pairwise distances over the rollout. At the
            # default 128x8 = 1024 states this is a 1024x1024
            # matrix, which is trivially affordable once per update.
            dist = torch.cdist(z, z, p=2)
            # Column 0 of the sorted distances is each point's
            # distance to itself (zero), so the k-th neighbour is
            # at index k. Clamp to the batch size so a rollout with
            # fewer than k+1 states falls back to its furthest
            # available neighbour instead of indexing out of bounds.
            k = min(self.k, dist.shape[1] - 1)
            knn = dist.sort(dim=1).values[:, k]
        bonus = torch.log(knn + 1.0)
        return bonus.reshape(num_steps, num_envs)


class E3BBonus(IntrinsicBonus):
    """
    E3B: an elliptical episodic bonus in a learned inverse-dynamics
    embedding.

    Episodic count bonuses need to recognize a repeat visit, which
    only works when states are discrete. E3B generalizes that to
    continuous features: it keeps a per-episode covariance of the
    features seen so far and rewards a state by its squared
    Mahalanobis norm under the inverse of that covariance, which is
    large exactly for directions the episode has not covered yet.

    The embedding is trained with an inverse dynamics model (predict
    the action from consecutive states), which keeps only the
    *controllable* part of the observation. That matters on any
    observation carrying content the agent cannot influence -- the
    failure that made vanilla NovelD collapse on MiniHack, where a
    step counter in the observation made every state trivially
    novel. This repo's 'historical' fog-of-war observation has the
    same monotone-content property, so E3B is the bonus most likely
    to survive that mode.
    """

    def __init__(
        self,
        obs_shape,
        num_envs,
        device,
        num_actions,
        feature_dim=128,
        ridge=0.1,
        mission_vocab=0,
    ):
        """
        Build the encoder and inverse-dynamics head, and allocate one
        inverse covariance matrix per env.

        Parameters
        ----------
        ridge: float
            The lambda in C_0 = lambda * I. It both keeps the matrix
            invertible and sets the bonus's starting magnitude
            (1/lambda for the first state of an episode).
        """

        super().__init__(obs_shape, num_envs, device,
                         mission_vocab=mission_vocab)

        self.feature_dim = feature_dim
        self.ridge = ridge

        # On a language task the action between two states depends on
        # which object the instruction names, so a mission-blind
        # encoder cannot fit the inverse-dynamics objective at all --
        # and the elliptical bonus is computed from exactly these
        # features.
        self.encoder = StateEncoder(
            obs_shape, feature_dim, mission_vocab=mission_vocab
        )
        # Inverse dynamics head: from a pair of consecutive
        # embeddings, classify which action caused the transition.
        self.inverse_head = nn.Sequential(
            layer_init(nn.Linear(2 * feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_actions)),
        )

        # C_inv[e] is env e's running inverse covariance. Registered
        # as a buffer so .to(device) moves it with the module.
        self.register_buffer(
            'c_inv',
            torch.eye(feature_dim, device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1, 1)
            / ridge,
        )

    def reset_all(self):
        """
        Reset every env's episodic covariance to (1/ridge) * I.
        """

        eye = torch.eye(self.feature_dim, device=self.c_inv.device)
        self.c_inv = (
            eye.unsqueeze(0).repeat(self.num_envs, 1, 1) / self.ridge
        )

    def rollout_bonus(self, obs, next_obs, actions, dones,
                      mission_ids=None, mission_len=None):
        """
        Walk the rollout in time order, emitting the elliptical bonus
        and folding each state into its env's covariance.

        The per-step inverse update is done with the Sherman-Morrison
        identity, so no matrix is ever inverted: a rank-one update to
        C has a closed-form rank-one update to C^-1.
        """

        num_steps, num_envs = dones.shape
        shape = tuple(self.obs_shape)

        # Embed the whole rollout once; only the covariance update
        # has to be sequential.
        ids, lens = _flatten_mission(mission_ids, mission_len)
        with torch.no_grad():
            flat = next_obs.reshape((-1,) + shape)
            z_all = self.encoder(flat.permute(0, 3, 1, 2), ids, lens)
            z_all = z_all.reshape(num_steps, num_envs, -1)

        eye = torch.eye(self.feature_dim, device=self.c_inv.device)
        bonus = torch.zeros(
            (num_steps, num_envs), device=self.c_inv.device
        )

        for t in range(num_steps):
            # Reset the covariance of any env whose episode restarts
            # at this step, before that env's state is scored.
            resets = torch.nonzero(dones[t] > 0, as_tuple=True)[0]
            if len(resets) > 0:
                self.c_inv[resets] = eye / self.ridge

            # z: (E, d) -> (E, d, 1) so the batched matmuls below
            # read as ordinary column-vector algebra.
            z = z_all[t].unsqueeze(-1)
            # u = C^-1 z, reused by both the bonus and the update.
            u = torch.bmm(self.c_inv, z)
            # The elliptical bonus itself: z^T C^-1 z.
            quad = torch.bmm(z.transpose(1, 2), u).squeeze(-1).squeeze(-1)
            bonus[t] = quad
            # Sherman-Morrison: after C <- C + z z^T,
            # C^-1 <- C^-1 - (u u^T) / (1 + z^T C^-1 z).
            denom = (1.0 + quad).view(-1, 1, 1)
            self.c_inv = self.c_inv - torch.bmm(
                u, u.transpose(1, 2)
            ) / denom

        return bonus

    def aux_loss(self, mb_obs, mb_next_obs, mb_actions,
                 mb_mission_ids=None, mb_mission_len=None):
        """
        Train the embedding by predicting which action produced each
        transition, so the features keep what the agent controls and
        discard what it does not.
        """

        z_cur = self.encoder(
            mb_obs.permute(0, 3, 1, 2), mb_mission_ids, mb_mission_len
        )
        z_next = self.encoder(
            mb_next_obs.permute(0, 3, 1, 2),
            mb_mission_ids, mb_mission_len,
        )
        logits = self.inverse_head(
            torch.cat([z_cur, z_next], dim=-1)
        )
        return F.cross_entropy(logits, mb_actions.long())


# Command-line name -> constructor. Adding a bonus is a one-line
# edit here plus the class above; the training loop reads only this.
BONUSES = ('none', 'count', 'rnd', 'noveld', 're3', 'e3b')


def build_bonus(name, obs_shape, num_envs, device, num_actions, args):
    """
    Construct the intrinsic bonus named on the command line.

    Parameters
    ----------
    name: str
        One of BONUSES. 'none' returns None, which the training loop
        treats as "run plain PPO", so the exact same file provides
        the no-bonus control.
    obs_shape: tuple
        The (H, W, C) observation shape.
    num_envs: int
        Number of parallel envs, which sets how much per-episode
        state the episodic bonuses allocate.
    device: torch.device
        Device the bonus networks live on.
    num_actions: int
        Size of the discrete action set, needed by E3B's inverse
        dynamics head.
    args: object
        The parsed CLI args, read for the per-bonus knobs
        (`noveld_alpha`, `re3_k`, `e3b_ridge`, ...).

    Returns
    -------
    IntrinsicBonus or None
    """

    # Zero on the goal-fixed tasks, which carry no instruction. On the
    # BabyAI language tasks it is the mission vocabulary size, and
    # every bonus below becomes mission-aware.
    mission_vocab = getattr(args, 'mission_vocab_size', 0) or 0

    if name == 'none':
        return None
    if name == 'count':
        return CountBonus(
            obs_shape, num_envs, device, scope=args.count_scope,
            mission_vocab=mission_vocab,
        ).to(device)
    if name == 'rnd':
        return RNDBonus(
            obs_shape,
            num_envs,
            device,
            update_proportion=args.update_proportion,
            mission_vocab=mission_vocab,
        ).to(device)
    if name == 'noveld':
        return NovelDBonus(
            obs_shape,
            num_envs,
            device,
            update_proportion=args.update_proportion,
            alpha=args.noveld_alpha,
            episodic_gate=args.noveld_episodic_gate,
            mission_vocab=mission_vocab,
        ).to(device)
    if name == 're3':
        return RE3Bonus(
            obs_shape, num_envs, device, k=args.re3_k,
            mission_vocab=mission_vocab,
        ).to(device)
    if name == 'e3b':
        return E3BBonus(
            obs_shape,
            num_envs,
            device,
            num_actions=num_actions,
            ridge=args.e3b_ridge,
            mission_vocab=mission_vocab,
        ).to(device)

    valid = ', '.join(BONUSES)
    raise ValueError(f'unknown bonus {name!r}; valid: {valid}')
