"""
The general LLM teacher must remember what the STUDENT executed.

The bug these tests pin: `llm_general` recorded the action it had
*advised*, not the action the student actually took. During teacher-alone
evaluation those coincide, so it went unnoticed; during distillation the
student picks its own action, so every later rationale was conditioned on
a trajectory that never happened. The history is also folded into the
cache key, so a wrong history corrupts cache identity too.

These tests exercise `_observe_outcome` directly. They need no API key,
no network and no environment.
"""

import pytest

from teachers.minigrid.llm_general import MiniGridGeneralLLMTeacher


@pytest.fixture
def teacher():
    """
    Build a teacher without touching the network.

    `__init__` builds an OpenAI client, so bypass it: these tests only
    exercise the pure history bookkeeping.
    """

    obj = object.__new__(MiniGridGeneralLLMTeacher)
    from collections import deque
    obj._history = deque(maxlen=6)
    obj._pending_action = None
    obj._prev_state_key = None
    return obj


def test_records_the_students_action_not_the_advised_one(teacher):
    """
    When the student disagrees with the advice, the history must show
    what the student did.
    """

    # The teacher advised 2 at the previous step.
    teacher._pending_action = 2
    teacher._prev_state_key = ('a',)
    # The student actually executed 0.
    teacher._observe_outcome(('b',), executed=[0])

    assert list(teacher._history) == [(0, False)], (
        'history must record the executed action 0, not the advised 2'
    )


def test_falls_back_to_advice_when_no_executed_action_is_given(teacher):
    """
    Teacher-alone evaluation executes the advice verbatim, so the old
    behaviour must survive when the caller passes nothing.
    """

    teacher._pending_action = 2
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('a',))

    # Same state key means the action provably did nothing.
    assert list(teacher._history) == [(2, True)]


def test_every_skipped_step_is_recorded_under_query_interval(teacher):
    """
    With query_interval > 1 several actions elapse between queries, and
    all of them belong in the history in order.
    """

    teacher._pending_action = 1
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=[0, 1, 2])

    assert [a for a, _ in teacher._history] == [0, 1, 2]


def test_blocked_is_only_claimed_across_a_single_action(teacher):
    """
    An unchanged state after several moves does not imply any one of
    them failed, so no intermediate action may be flagged blocked.
    """

    teacher._pending_action = 1
    teacher._prev_state_key = ('a',)
    # State is unchanged, but three actions elapsed: the agent could
    # have moved away and come back.
    teacher._observe_outcome(('a',), executed=[0, 1, 2])

    assert not any(blocked for _, blocked in teacher._history), (
        'blocked must not be inferred across a multi-step gap'
    )


def test_blocked_is_still_detected_for_a_single_executed_action(teacher):
    """
    The wall-bump signal must survive the fix for the one case where it
    is actually valid.
    """

    teacher._pending_action = 1
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('a',), executed=[3])

    assert list(teacher._history) == [(3, True)]


def test_pending_action_is_cleared_after_observing(teacher):
    """
    A stale pending action must not be double-counted at the next step.
    """

    teacher._pending_action = 2
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=[0])
    teacher._observe_outcome(('c',))

    assert list(teacher._history) == [(0, False)]
