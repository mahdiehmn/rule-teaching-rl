"""
Tests for the recurrent, mission-conditioned BabyAI agent
(algos/ppo_babyai.py) and the autoreset assumption its rollout loop
depends on.

The most important thing tested here is NOT "does the network run"
but "does the episode_start-gated hidden-state reset actually reset
at the right moments" -- see algos/ppo_babyai.py's module docstring
for why a naive same-index (1 - done) reset would be wrong under
this repo's autoreset semantics. That empirical claim about
gymnasium's SyncVectorEnv is pinned here as an assertion, not left
as an ad-hoc script, so a future gymnasium upgrade that changes
autoreset behavior fails this test loudly instead of silently
corrupting every BabyAI run's mission-progress memory.
"""

import gymnasium as gym
import torch

import pytest

pytest.importorskip('minigrid')

from algos.ppo_babyai import RecurrentAgent, build_select_action
from envs.mission_vocab import MissionVocab
from envs.registry import make_thunk


def _make_babyai_envs(num_envs=2, task='gotolocal'):
    """
    Build a small, un-recorded SyncVectorEnv with mission tokens
    turned on, sharing one in-memory vocab (never touches the
    committed vocab file, so these tests do not depend on it having
    been (re)built).
    """

    vocab = MissionVocab.build_from_missions(
        [
            'go to the red ball',
            'go to a grey box',
            'go to the blue key',
        ]
    )
    envs = gym.vector.SyncVectorEnv(
        [
            make_thunk(
                task, 0, i, obs_mission=True, mission_vocab=vocab
            )
            for i in range(num_envs)
        ]
    )
    return envs, vocab


def _obs_to_tensors(obs_np):
    """
    Convert one vectorized Dict observation to the tensor dtypes
    RecurrentAgent expects, mirroring algos/ppo_babyai.py's own
    _to_device_obs helper.
    """

    return {
        'image': torch.as_tensor(obs_np['image']).float(),
        'mission_ids': torch.as_tensor(obs_np['mission_ids']).long(),
        'mission_len': torch.as_tensor(obs_np['mission_len']).long(),
    }


def _reference_get_states(agent, x, core_state, episode_start):
    """
    Pre-optimization reference implementation: one GRU call per
    timestep, unconditionally -- exactly what
    RecurrentAgent.get_states did before it was rewritten to batch
    reset-free runs into single calls (a >30x cluster slowdown was
    traced to this loop's per-call dispatch overhead). Kept ONLY
    here, for the differential test below, never in production code.
    """

    # The observation encoder is shared (algos.nets.ObsEncoder) so it
    # handles pixels and symbolic grids alike; what this reference
    # reimplements is only the recurrent core's per-step stepping.
    cnn_feat = agent.encoder(x['image'])
    mission_feat = agent.mission_encoder(
        x['mission_ids'], x['mission_len']
    )
    fused = agent.fusion(torch.cat([cnn_feat, mission_feat], dim=-1))

    num_envs_here = core_state.shape[1]
    fused = fused.reshape(-1, num_envs_here, fused.shape[-1])
    ep_start = episode_start.reshape(-1, num_envs_here)

    outputs = []
    for t in range(fused.shape[0]):
        gate = (1.0 - ep_start[t]).view(1, -1, 1)
        step_out, core_state = agent.core(
            fused[t].unsqueeze(0), gate * core_state
        )
        outputs.append(step_out)
    hidden = torch.cat(outputs, dim=0).reshape(
        -1, agent.core.hidden_size
    )
    return hidden, core_state


def test_recurrent_agent_forward_pass_shapes():
    """
    A single rollout-style forward pass (one timestep, all envs)
    must return correctly shaped outputs and an unchanged core-state
    shape.
    """

    envs, vocab = _make_babyai_envs(num_envs=3)
    try:
        agent = RecurrentAgent(envs, vocab_size=vocab.size)
        obs_np, _ = envs.reset(seed=0)
        x = _obs_to_tensors(obs_np)
        core_state = agent.initial_core_state(3, device='cpu')
        episode_start = torch.ones(3)

        action, logprob, entropy, value, new_core = (
            agent.get_action_and_value(x, core_state, episode_start)
        )

        assert action.shape == (3,)
        assert logprob.shape == (3,)
        assert entropy.shape == (3,)
        assert value.shape == (3, 1)
        assert new_core.shape == (1, 3, agent.core.hidden_size)
    finally:
        envs.close()


def test_episode_start_zeroes_the_core_state():
    """
    episode_start=1 must produce EXACTLY the same result as if the
    core state had genuinely been zero -- this is the core
    correctness property the whole recurrent design depends on (see
    the module docstring). Compared via the agent's own
    deterministic forward pass, not an approximation.
    """

    envs, vocab = _make_babyai_envs(num_envs=2)
    try:
        agent = RecurrentAgent(envs, vocab_size=vocab.size)
        agent.eval()
        obs_np, _ = envs.reset(seed=0)
        x = _obs_to_tensors(obs_np)

        # An arbitrary, deliberately nonzero "stale" state, as if
        # carried over from a previous (unrelated) episode.
        torch.manual_seed(0)
        stale_state = torch.randn(1, 2, agent.core.hidden_size)

        with torch.no_grad():
            hidden_gated, core_gated = agent.get_states(
                x, stale_state.clone(), episode_start=torch.ones(2)
            )
            hidden_zero, core_zero = agent.get_states(
                x,
                torch.zeros(1, 2, agent.core.hidden_size),
                episode_start=torch.zeros(2),
            )

        assert torch.allclose(hidden_gated, hidden_zero)
        assert torch.allclose(core_gated, core_zero)
    finally:
        envs.close()


def test_episode_start_zero_preserves_the_core_state():
    """
    The flip side of the previous test: episode_start=0 must NOT
    reset the state -- feeding the same stale state through with and
    without the gate must give different results, or the gate is not
    actually doing anything.
    """

    envs, vocab = _make_babyai_envs(num_envs=2)
    try:
        agent = RecurrentAgent(envs, vocab_size=vocab.size)
        agent.eval()
        obs_np, _ = envs.reset(seed=0)
        x = _obs_to_tensors(obs_np)

        torch.manual_seed(0)
        stale_state = torch.randn(1, 2, agent.core.hidden_size)

        with torch.no_grad():
            hidden_carried, _ = agent.get_states(
                x, stale_state.clone(), episode_start=torch.zeros(2)
            )
            hidden_reset, _ = agent.get_states(
                x, stale_state.clone(), episode_start=torch.ones(2)
            )

        assert not torch.allclose(hidden_carried, hidden_reset)
    finally:
        envs.close()


def test_batched_recurrent_replay_matches_per_step_reference():
    """
    Differential test: the chunked-GRU implementation in
    RecurrentAgent.get_states must produce EXACTLY the result a
    naive one-call-per-timestep loop would, for a sequence with
    resets scattered at different points for different envs. This
    is the correctness guarantee behind the performance fix -- the
    optimization must never change what training computes, only how
    many Python-level calls it costs to compute it.
    """

    envs, vocab = _make_babyai_envs(num_envs=3)
    try:
        agent = RecurrentAgent(envs, vocab_size=vocab.size)
        agent.eval()
        obs_np, _ = envs.reset(seed=0)
        single_x = _obs_to_tensors(obs_np)  # one real step, 3 envs

        seq_len = 10
        # Repeat the same per-env observation across seq_len
        # timesteps, T-major (env varies fastest) -- matching the
        # layout get_states expects. The actual pixel/mission
        # content is irrelevant here; what is being tested is the
        # reset-gating arithmetic, not perception.
        x = {
            k: v.unsqueeze(0)
            .expand(seq_len, *v.shape)
            .reshape(seq_len * v.shape[0], *v.shape[1:])
            .contiguous()
            for k, v in single_x.items()
        }

        torch.manual_seed(0)
        init_state = torch.randn(1, 3, agent.core.hidden_size)

        # Scatter resets: env 0 resets mid-sequence (t=3), env 1
        # resets at the very start AND again mid-sequence (t=0,
        # t=7), env 2 never resets after the initial state -- this
        # exercises chunk boundaries that differ per env, multiple
        # boundaries in one sequence, and a boundary at t=0.
        ep_start = torch.zeros(seq_len, 3)
        ep_start[3, 0] = 1.0
        ep_start[0, 1] = 1.0
        ep_start[7, 1] = 1.0
        ep_start_flat = ep_start.reshape(-1)

        with torch.no_grad():
            hidden_batched, core_batched = agent.get_states(
                x, init_state.clone(), ep_start_flat
            )
            hidden_ref, core_ref = _reference_get_states(
                agent, x, init_state.clone(), ep_start_flat
            )

        assert torch.allclose(hidden_batched, hidden_ref, atol=1e-6)
        assert torch.allclose(core_batched, core_ref, atol=1e-6)
    finally:
        envs.close()


def test_distill_babyai_recurrent_agent_also_matches_reference():
    """
    algos/ppo_distill_babyai.py's RecurrentAgent duplicates
    get_states verbatim (this project's diff-file convention means
    it is a separate class, not a shared import) -- so the same
    batching fix and the same correctness risk both exist there
    independently, and need their own check rather than trusting
    the copy-paste. _reference_get_states is reused unchanged: it
    only relies on attribute names (encoder/mission_encoder
    /fusion/core) both classes share, not on which module they came
    from.
    """

    from algos.ppo_distill_babyai import (
        RecurrentAgent as DistillRecurrentAgent,
    )

    envs, vocab = _make_babyai_envs(num_envs=3)
    try:
        agent = DistillRecurrentAgent(envs, vocab_size=vocab.size)
        agent.eval()
        obs_np, _ = envs.reset(seed=0)
        single_x = _obs_to_tensors(obs_np)

        seq_len = 10
        x = {
            k: v.unsqueeze(0)
            .expand(seq_len, *v.shape)
            .reshape(seq_len * v.shape[0], *v.shape[1:])
            .contiguous()
            for k, v in single_x.items()
        }

        torch.manual_seed(0)
        init_state = torch.randn(1, 3, agent.core.hidden_size)

        ep_start = torch.zeros(seq_len, 3)
        ep_start[3, 0] = 1.0
        ep_start[0, 1] = 1.0
        ep_start[7, 1] = 1.0
        ep_start_flat = ep_start.reshape(-1)

        with torch.no_grad():
            hidden_batched, core_batched = agent.get_states(
                x, init_state.clone(), ep_start_flat
            )
            hidden_ref, core_ref = _reference_get_states(
                agent, x, init_state.clone(), ep_start_flat
            )

        assert torch.allclose(hidden_batched, hidden_ref, atol=1e-6)
        assert torch.allclose(core_batched, core_ref, atol=1e-6)
    finally:
        envs.close()


def test_build_select_action_reset_zeroes_hidden_state():
    """
    build_select_action's `.reset()` attribute -- the hook
    monitoring.eval.greedy_eval calls between episodes -- must
    actually restore the initial (zeroed) core state.

    Tested black-box rather than by reaching into closure internals:
    since the network is deterministic (no dropout/batchnorm), a
    closure that has been driven away from its initial state and
    then reset() must agree EXACTLY with a brand-new closure's very
    first call on the same observation, because both are now
    starting from the same zeroed core state.
    """

    envs, vocab = _make_babyai_envs(num_envs=1)
    try:
        agent = RecurrentAgent(envs, vocab_size=vocab.size)
        agent.eval()
        select_action = build_select_action(agent, device='cpu')

        obs_np, _ = envs.reset(seed=0)
        single_obs = {k: v[0] for k, v in obs_np.items()}

        # Drive a few steps so the internal core state moves away
        # from its initial value.
        for _ in range(3):
            select_action(single_obs)
        select_action.reset()
        action_after_reset = select_action(single_obs)

        fresh_select_action = build_select_action(agent, device='cpu')
        action_fresh = fresh_select_action(single_obs)

        assert action_after_reset == action_fresh
    finally:
        envs.close()


def test_sync_vector_env_autoreset_is_next_step():
    """
    Pins down the exact autoreset semantics
    algos/ppo_babyai.py's episode_start bookkeeping depends on: the
    call that terminates an episode returns the REAL terminal frame
    (not a placeholder), and the fresh reset only happens on the
    FOLLOWING call, which itself reports done=False. If a future
    gymnasium version changes this, this test fails loudly instead
    of the recurrent core silently resetting one step early or late
    on every BabyAI episode boundary.
    """

    envs = gym.vector.SyncVectorEnv([make_thunk('empty5x5', 0, 0)])
    try:
        assert (
            envs.autoreset_mode == gym.vector.AutoresetMode.NEXT_STEP
        )

        obs, _ = envs.reset(seed=0)
        u = envs.envs[0].unwrapped
        start_pos = tuple(int(v) for v in u.agent_pos)

        # Force a one-step win: stand the agent right next to the
        # goal, facing it.
        u.agent_pos = (3, 2)
        u.agent_dir = 1  # facing south, toward the goal at (3, 3)

        obs, reward, term, trunc, info = envs.step([2])  # forward
        # The call that ends the episode reports the win directly.
        assert bool(term[0]) is True
        assert float(reward[0]) > 0.0
        assert tuple(int(v) for v in u.agent_pos) == (3, 3)

        # The NEXT call is the fresh reset: no termination flag, and
        # the agent is back at its original start position -- the
        # action passed here is ignored by the env.
        obs, reward, term, trunc, info = envs.step([2])
        assert bool(term[0]) is False
        assert bool(trunc[0]) is False
        assert tuple(int(v) for v in u.agent_pos) == start_pos
    finally:
        envs.close()
