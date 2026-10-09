"""
Tests for the advice replay buffer (advising/replay.py).

The buffer exists to change what an advice budget BUYS. Under action
override a piece of advice is an intervention consumed on the step it
is given; under distillation it is a labelled example that can be
replayed for the rest of training. These tests pin the properties
that difference depends on: stored labels must be independent of the
rollout memory they came from, sampling must not repeat within a
batch, and eviction must be reported rather than silent.

Runs under pytest, and standalone via
`python -m tests.test_advice_replay`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advising.replay import AdviceReplay  # noqa: E402


def test_stored_labels_survive_rollout_reuse():
    """
    The rollout observation buffer is written in place every
    iteration. If the replay buffer kept a view rather than a copy,
    stored labels would silently become whatever the latest rollout
    put in that slot -- training on corrupted pairs with no error.
    """

    rollout_slot = np.ones((4, 4, 3), dtype=np.uint8)
    buffer = AdviceReplay(capacity=10)
    buffer.add(rollout_slot, np.array([0.9, 0.1]))

    # The training loop overwrites the slot for the next rollout.
    rollout_slot[:] = 7

    obs, targets = buffer.sample(1)
    assert obs[0].max() == 1, 'buffer aliased the rollout memory'
    assert np.isclose(targets[0][0], 0.9)


def test_sampling_does_not_repeat_within_a_batch():
    """
    A batch drawn from the buffer should be distinct labels, so one
    stored state cannot dominate a gradient step.
    """

    buffer = AdviceReplay(capacity=100, seed=0)
    for i in range(20):
        buffer.add(np.full((2, 2, 1), i, dtype=np.uint8),
                   np.array([1.0, 0.0]))

    obs, _ = buffer.sample(10)
    values = sorted(int(o[0, 0, 0]) for o in obs)
    assert len(set(values)) == 10


def test_sampling_is_capped_by_buffer_size():
    """
    Asking for more than the buffer holds returns what there is,
    rather than raising or padding with repeats.
    """

    buffer = AdviceReplay(capacity=100)
    for i in range(3):
        buffer.add(np.zeros((2, 2, 1), dtype=np.uint8),
                   np.array([1.0, 0.0]))

    obs, targets = buffer.sample(64)
    assert len(obs) == 3
    assert len(targets) == 3

    assert AdviceReplay(capacity=5).sample(4) is None


def test_eviction_is_reported():
    """
    A full buffer drops its oldest labels. That is bought advice
    being discarded, so it has to be visible in the accounting
    rather than inferred from a size that stopped growing.
    """

    buffer = AdviceReplay(capacity=5)
    for i in range(12):
        buffer.add(np.zeros((2, 2, 1), dtype=np.uint8),
                   np.array([1.0, 0.0]))

    stats = buffer.stats()
    assert stats['replay_size'] == 5
    assert stats['replay_added'] == 12
    assert stats['replay_evicted'] == 7


def test_capacity_must_be_positive():
    """
    A zero-capacity buffer would silently discard every label.
    """

    try:
        AdviceReplay(capacity=0)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def _main():
    """
    Run every test in this module without pytest.
    """

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]

    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f'FAIL {name}: {exc}')
        else:
            print(f'ok   {name}')

    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
