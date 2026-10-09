"""
Differential tests for the shared recurrent core.

algos/nets.RecurrentCore batches every reset-free run of timesteps
into a single GRU call instead of stepping one timestep at a time.
That is an optimization -- a >30x cluster slowdown was traced to
per-call dispatch overhead in the per-step version -- and
optimizations of this shape are exactly where a silent correctness
bug hides: the wrong version still trains, still produces plausible
curves, and simply carries the wrong hidden state across an episode
boundary.

So the batched implementation is checked against a deliberately naive
per-step reference, on inputs that contain resets in the awkward
places: at the very first step, mid-sequence, on some envs but not
others, and on consecutive steps.
"""

import pytest
import torch

from algos.nets import HIDDEN, RecurrentCore


def reference_forward(core, features, core_state, episode_start):
    """
    One GRU call per timestep, unconditionally.

    This is what the batched implementation must reproduce exactly.
    Kept only in tests, never in production code.
    """

    num_envs = core_state.shape[1]
    seq = features.reshape(-1, num_envs, features.shape[-1])
    ep_start = episode_start.reshape(-1, num_envs)

    outputs = []
    for t in range(seq.shape[0]):
        gate = (1.0 - ep_start[t]).view(1, -1, 1)
        out, core_state = core.gru(
            seq[t:t + 1], gate * core_state
        )
        outputs.append(out)
    return (
        torch.cat(outputs, dim=0).reshape(-1, core.hidden),
        core_state,
    )


def run_case(starts, seed=0):
    """
    Compare batched and per-step output for one reset pattern.
    """

    torch.manual_seed(seed)
    core = RecurrentCore(HIDDEN)
    steps, envs = starts.shape
    features = torch.randn(steps * envs, HIDDEN)
    state = torch.randn(1, envs, HIDDEN)

    batched_out, batched_state = core(
        features, state.clone(), starts.reshape(-1)
    )
    ref_out, ref_state = reference_forward(
        core, features, state.clone(), starts.reshape(-1)
    )
    return batched_out, batched_state, ref_out, ref_state


def test_no_resets_anywhere():
    """
    A reset-free run must collapse to a single GRU call and still
    match the per-step version.
    """

    starts = torch.zeros(6, 4)
    a, sa, b, sb = run_case(starts)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)


def test_reset_on_the_first_step():
    """
    A reset at index 0 must zero the incoming state, which is the
    case the boundary list has to handle without a leading split.
    """

    starts = torch.zeros(6, 4)
    starts[0] = 1.0
    a, sa, b, sb = run_case(starts)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)


def test_reset_mid_sequence_on_one_env_only():
    """
    Resetting a single env must leave the others' state carried
    through untouched -- the case a naive whole-batch zeroing breaks.
    """

    starts = torch.zeros(6, 4)
    starts[3, 2] = 1.0
    a, sa, b, sb = run_case(starts)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)


def test_resets_on_consecutive_steps():
    """
    Back-to-back boundaries produce zero-length runs between splits;
    the batched version must not emit an empty GRU call.
    """

    starts = torch.zeros(6, 4)
    starts[2, 1] = 1.0
    starts[3, 1] = 1.0
    starts[3, 0] = 1.0
    a, sa, b, sb = run_case(starts)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)


def test_every_env_resets_every_step():
    """
    The degenerate all-resets case: every step is its own chunk.
    """

    starts = torch.ones(5, 3)
    a, sa, b, sb = run_case(starts)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)


def test_single_timestep_rollout_shape():
    """
    The rollout calls this with one timestep across all envs, which
    must return one feature row per env.
    """

    torch.manual_seed(0)
    core = RecurrentCore(HIDDEN)
    features = torch.randn(4, HIDDEN)
    state = core.initial_state(4, torch.device('cpu'))
    out, new_state = core(features, state, torch.zeros(4))
    assert out.shape == (4, HIDDEN)
    assert new_state.shape == (1, 4, HIDDEN)


def test_reset_actually_changes_the_output():
    """
    Guard against a reset that is silently a no-op: the same inputs
    with and without a reset must differ.
    """

    torch.manual_seed(0)
    core = RecurrentCore(HIDDEN)
    features = torch.randn(3 * 2, HIDDEN)
    state = torch.randn(1, 2, HIDDEN)

    without = core(
        features, state.clone(), torch.zeros(6)
    )[0]
    starts = torch.zeros(3, 2)
    starts[0] = 1.0
    with_reset = core(
        features, state.clone(), starts.reshape(-1)
    )[0]
    assert not torch.allclose(without, with_reset, atol=1e-6)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_random_reset_patterns(seed):
    """
    Random boundaries, to catch patterns the hand-written cases miss.
    """

    torch.manual_seed(seed)
    starts = (torch.rand(8, 4) < 0.25).float()
    a, sa, b, sb = run_case(starts, seed=seed)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(sa, sb, atol=1e-6)