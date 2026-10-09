"""
Shared observation encoder for the pixel and symbolic modes.

Why this exists
---------------
`algos/ppo.py`, `algos/ppo_distill.py` and `algos/ppo_teacher.py` all
carry a byte-identical copy of the same trunk: an Atari-style conv
stack over a `(N, H, W, C)` uint8 image divided by 255. That is
correct for the RGB observation modes and silently wrong for the
symbolic ones, in two separate ways:

1. **Scale.** MiniGrid's symbolic encoding is three small category
   indices per cell (object 0-10, colour 0-5, state 0-2). Dividing
   those by 255 leaves every input in a 0-0.04 band, which trains
   badly and is the usual reason a naive symbolic port underperforms
   its pixel equivalent. Category indices are not magnitudes anyway --
   "door" (4) is not twice "floor" (3) -- so they want an embedding,
   not a division.
2. **Kernel size.** The symbolic observation is 7x7 (egocentric) or
   up to 25x25 (whole map). An 8x8 stride-4 first layer is larger
   than a 7x7 input, and on a 16x16 input it leaves 3x3, which the
   next 4x4 kernel cannot consume -- torch raises rather than
   degrading quietly.

This module owns both decisions once, so a new observation mode is
supported everywhere at the same time instead of in whichever file
was remembered.

`algos/ppo_intrinsic.py` deliberately keeps its own copy: it has
completed multi-seed results, and re-pointing it at a shared module
would risk changing a validated pipeline for no experimental gain.
"""

import numpy as np
import torch
import torch.nn as nn

# Observations with fewer cells than this on a side are symbolic
# grids and get small kernels; anything larger is a render.
SMALL_INPUT_THRESHOLD = 32
# Spatial size the small-input stack is pooled to before flattening,
# so a whole-map 25x25 encoding cannot blow up the projection.
SMALL_INPUT_POOL = 7
# Embedding width per symbolic channel.
SYMBOL_EMBED_DIM = 8
# Upper bound on any MiniGrid category index (max object index is 10),
# with headroom so the table never needs resizing per task.
SYMBOL_VOCAB = 16
# Width of the feature vector handed to the actor/critic heads. 512
# matches every existing agent in algos/, so switching a file to this
# encoder does not change its capacity.
HIDDEN = 512


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill its
    bias, the standard PPO initialization used across algos/.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RecurrentCore(nn.Module):
    """
    A single-layer GRU over trunk features, with correct resets.

    Shared rather than copied because the reset logic is the easiest
    thing in this repo to get subtly wrong, in two independent ways.

    WHEN to reset. This project's SyncVectorEnv runs gymnasium's
    AutoresetMode.NEXT_STEP: when an episode ends at step t, dones[t]
    is 1 and the observation stored at t IS the real terminal frame;
    the fresh reset lands at t+1, whose own dones[t+1] is 0. Zeroing
    the hidden state wherever `dones` is 1 therefore resets one step
    too late -- right before re-processing a frame the agent has
    already seen, and NOT before the first frame of the new episode,
    which is the only place it matters. This module consumes a
    separate `episode_start` signal, which is `dones` shifted by one.

    HOW MANY GRU calls. Stepping one timestep at a time is correct
    but pays per-call dispatch overhead seq_len times, which was
    traced to a >30x slowdown on the cluster in algos/ppo_babyai.py.
    A reset-free run can be consumed in ONE call, so this splits the
    sequence only at steps where some env actually resets. That is
    mathematically identical, not an approximation.
    """

    def __init__(self, hidden=HIDDEN):
        """
        Build the GRU with the orthogonal init used across algos/.
        """

        super().__init__()

        # batch_first=False (the nn.GRU default): inputs are
        # (seq_len, batch, features), the layout rollouts are already
        # stored in.
        self.gru = nn.GRU(hidden, hidden)
        self.hidden = hidden
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.orthogonal_(param, 1.0)

    def initial_state(self, num_envs, device):
        """
        Return a zeroed hidden state for `num_envs` fresh episodes.
        """

        return torch.zeros(1, num_envs, self.hidden, device=device)

    def forward(self, features, core_state, episode_start):
        """
        Carry trunk features through the GRU across time.

        Parameters
        ----------
        features: FloatTensor, shape (B, hidden)
            B is either E (one rollout step across all envs) or T*E
            in T-major order (a stored sequence replayed during the
            update).
        core_state: FloatTensor, shape (1, E, hidden)
            E is read from this tensor's own shape, which is what
            lets one call serve both the rollout and the update.
        episode_start: FloatTensor, shape (B,)
            1.0 exactly where an observation is the first of a fresh
            episode. NOT `dones` -- see the class docstring.

        Returns
        -------
        (features, core_state)
        """

        num_envs_here = core_state.shape[1]
        seq = features.reshape(-1, num_envs_here, features.shape[-1])
        ep_start = episode_start.reshape(-1, num_envs_here)
        seq_len = seq.shape[0]

        # Split only where some env resets; everywhere else the state
        # simply carries forward and one call covers the whole run.
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
            chunk_out, core_state = self.gru(
                seq[start:end], gate * core_state
            )
            outputs.append(chunk_out)

        return (
            torch.cat(outputs, dim=0).reshape(-1, self.hidden),
            core_state,
        )


class MissionEncoder(nn.Module):
    """
    Embed a tokenized BabyAI mission and summarize it with a GRU.

    The final hidden state (not a pooled average) is the mission
    vector: a GRU's last state is already a function of the whole
    sequence, and BabyAI missions are short -- 2-33 tokens observed
    across gotolocal/gotoseq/putnextlocal/bosslevel against a ceiling
    of 80 (envs/mission_vocab.py) -- so pooling buys little.
    pack_padded_sequence with the TRUE lengths means padding tokens
    never reach the GRU, however large mission_max_len is.

    This lives here rather than in one algorithm because both the
    policy and the exploration bonuses need it. A bonus that cannot
    see the instruction measures novelty in a mission-blind space:
    on GoToSeq the same cell under 'go to the red ball' and 'go to
    the green key' would be one state, so a count bonus would report
    'already visited' somewhere the agent has never been under this
    instruction.
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
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
        Encode a batch of tokenized missions into fixed-size vectors.

        Parameters
        ----------
        mission_ids: LongTensor, shape (B, L)
        mission_len: LongTensor, shape (B,)
            True (unpadded) token counts. Clamped to at least 1 so
            pack_padded_sequence never sees a zero-length sequence,
            which it rejects outright.

        Returns
        -------
        FloatTensor, shape (B, hidden_size)
        """

        embedded = self.embedding(mission_ids.long())
        # Lengths must be on the CPU wherever the data lives. That is
        # a hard requirement of pack_padded_sequence, not a style
        # choice.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            mission_len.long().clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        # (num_layers=1, B, hidden) -> (B, hidden).
        return hidden.squeeze(0)


def is_symbolic_mode(obs_mode):
    """
    Report whether an obs_mode string yields MiniGrid's integer grid
    encoding rather than a rendered image.

    Accepts the same aliases envs/registry.py does, so callers can
    pass the raw command-line value straight through.
    """

    return str(obs_mode).lower().replace('-', '_').startswith(
        ('symbolic', 'grid')
    )


class ObsEncoder(nn.Module):
    """
    Encode a batch of MiniGrid observations into a feature vector.

    Handles both observation families behind one `forward`: rendered
    RGB (scaled to [0, 1]) and the symbolic grid (embedded per
    channel). Callers only need to say which one they built the env
    with.
    """

    def __init__(self, obs_shape, symbolic, hidden=HIDDEN):
        """
        Build the embedding (symbolic only) and the conv trunk sized
        to the observation.

        Parameters
        ----------
        obs_shape: tuple
            The (H, W, C) shape the env reports.
        symbolic: bool
            True when the observation is the integer grid encoding.
            Use is_symbolic_mode(args.obs_mode) to derive it.
        hidden: int
            Width of the output feature vector.
        """

        super().__init__()

        h, w, c = obs_shape
        self.symbolic = symbolic

        if symbolic:
            # Each cell's (object, colour, state) indices are embedded
            # and concatenated, giving a dense c * SYMBOL_EMBED_DIM
            # channel image for the convolutions.
            self.embedding = nn.Embedding(SYMBOL_VOCAB, SYMBOL_EMBED_DIM)
            conv_in = c * SYMBOL_EMBED_DIM
        else:
            self.embedding = None
            conv_in = c

        if min(h, w) < SMALL_INPUT_THRESHOLD:
            # Small symbolic grids: stride-1 padded 3x3 kernels keep
            # the grid intact, then pool so a large map does not
            # produce an enormous flatten.
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(conv_in, 32, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, padding=1)),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(min(SMALL_INPUT_POOL, min(h, w))),
                nn.Flatten(),
            )
        else:
            # The Atari stack every existing agent in algos/ uses, so
            # a pixel run through this encoder is unchanged.
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(conv_in, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Discover the flattened width with a dummy pass rather than
        # hard-coding it, so any obs_mode and map size works.
        with torch.no_grad():
            n_flatten = self.network(
                torch.zeros(1, conv_in, h, w)
            ).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, hidden)), nn.ReLU()
        )

    def forward(self, x):
        """
        Map an (N, H, W, C) observation batch to (N, hidden).
        """

        if self.symbolic:
            # (N, H, W, C) indices -> (N, H, W, C*E) -> channels-first.
            emb = self.embedding(x.long())
            n, h, w = emb.shape[0], emb.shape[1], emb.shape[2]
            features = emb.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        else:
            features = x.permute(0, 3, 1, 2) / 255.0
        return self.fc(self.network(features))
