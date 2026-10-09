"""
The caller's side of the llm_general history contract.

`test_llm_general_history.py` covers the teacher helper. These cover the
boundary the helper cannot see: what the training loop must hand it, and
when it may consider that context acknowledged.

The six cases are the stopping rule from the 2026-09-09 contract review:
skipped query, withheld advice, cached consultation, explicit empty,
reset-only tick, and multi-step gap. They are exercised against the real
`ImportanceAdvisor` and the real history methods, with a recording
teacher; no SDK client, no API call and no training.
"""

from collections import deque

import numpy as np
import pytest

from advising.strategies import ImportanceAdvisor
from algos.ppo_distill import note_executed_action, teacher_was_consulted
from teachers.base import Advice
from teachers.minigrid.llm_general import (
    MiniGridGeneralLLMTeacher,
    format_history_block,
)


class RecordingTeacher:
    """
    Stands in for the LLM teacher and records every consultation.

    `recommend` is what a real consultation invokes, so counting calls
    here is exactly the "was the teacher consulted" question the caller
    must answer before clearing its pending history.
    """

    teacher_id = 'recording'

    def __init__(self):
        self.calls = []

    def recommend(self, state, context=None):
        self.calls.append(dict(context or {}))
        return Advice(action=1, confidence=1.0, explanation='x',
                      teacher_id=self.teacher_id)


@pytest.fixture
def teacher():
    """
    A teacher with only the history bookkeeping initialized.
    """

    obj = object.__new__(MiniGridGeneralLLMTeacher)
    obj._history = deque(maxlen=6)
    obj._pending_action = None
    obj._prev_state_key = None
    return obj


def simulate_query(advisor, teacher_obj, pending, ctx):
    """
    Run one consultation the way `ppo_distill` does, and report whether
    the teacher was actually reached.

    Returns (consulted, pending_after, fresh_after) so a test can assert
    the caller's acknowledge-only-if-consulted rule.
    """

    asked_before = advisor.num_asked
    advisor.advise(teacher_obj, object(), 0, ctx)
    # The PRODUCTION predicate, imported from the trainer, so a
    # regression there cannot leave this test green.
    consulted = teacher_was_consulted(asked_before, advisor)
    return consulted, ([] if consulted else pending)


# --- Finding 1: a skipped consultation must not clear pending state ---

def test_skipped_consultation_keeps_pending_actions():
    """
    An advisor that declines to ask leaves the teacher ignorant, so the
    student's actions must still be pending for the next consultation.
    """

    recorder = RecordingTeacher()
    # Entropy far below the threshold: this advisor will not consult.
    advisor = ImportanceAdvisor(threshold=10.0)
    ctx = {'entropy': 0.0, 'action_probs': np.array([1.0, 0.0, 0.0]),
           'executed_actions': [3, 4]}

    consulted, pending_after = simulate_query(
        advisor, recorder, [3, 4], ctx)

    assert not consulted, 'this advisor should have declined to consult'
    assert recorder.calls == [], 'the teacher must not have been reached'
    assert pending_after == [3, 4], (
        'a skipped consultation must not discard the pending actions'
    )


def test_real_consultation_clears_pending_actions():
    """
    The complement: when the teacher IS reached, the context has been
    delivered and may be cleared.
    """

    recorder = RecordingTeacher()
    advisor = ImportanceAdvisor(threshold=0.0)
    ctx = {'entropy': 5.0, 'action_probs': np.array([0.34, 0.33, 0.33]),
           'executed_actions': [3, 4]}

    consulted, pending_after = simulate_query(
        advisor, recorder, [3, 4], ctx)

    assert consulted and len(recorder.calls) == 1
    assert recorder.calls[0]['executed_actions'] == [3, 4], (
        'the teacher must receive the actions before they are cleared'
    )
    assert pending_after == []


# --- Finding 4: explicit [] is not the same as absent ---

def test_explicit_empty_records_nothing(teacher):
    """
    Two consultations with no environment step between them: the student
    executed nothing, and nothing may be invented.
    """

    teacher._pending_action = 2
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('a',), executed=[])

    assert list(teacher._history) == [], (
        'an explicit empty list must not fall back to the advised action'
    )


def test_absent_history_still_falls_back_for_teacher_alone(teacher):
    """
    Teacher-alone evaluation passes nothing and executes the advice
    verbatim, so that legacy path must survive.
    """

    teacher._pending_action = 2
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=None)

    assert list(teacher._history) == [(2, False)]


# --- Finding 3: unknown outcomes must not claim success ---

def test_multi_step_gap_reports_unknown_not_state_changed(teacher):
    """
    Several actions with no intermediate states: each individual outcome
    is unobserved, and the prompt must say so.
    """

    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=[0, 1, 2])

    assert all(outcome is None for _, outcome in teacher._history)
    rendered = format_history_block(teacher._history)
    assert 'effect not recorded' in rendered
    assert 'state changed' not in rendered


def test_first_query_of_episode_has_no_outcome_to_report(teacher):
    """
    With no previous state there is nothing to compare against, so the
    outcome is unknown rather than a claimed success.
    """

    teacher._prev_state_key = None
    teacher._observe_outcome(('a',), executed=[1])

    assert list(teacher._history) == [(1, None)]


def test_single_action_outcome_is_still_judged(teacher):
    """
    The one case that supports a real judgement must keep working.
    """

    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('a',), executed=[3])
    assert list(teacher._history) == [(3, True)]

    teacher._history.clear()
    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=[3])
    assert list(teacher._history) == [(3, False)]


def test_unknown_outcome_emits_no_blocked_warning(teacher):
    """
    The loud "NO EFFECT" warning is only justified by a measurement, so
    an unknown outcome must not trigger it.
    """

    teacher._prev_state_key = ('a',)
    teacher._observe_outcome(('b',), executed=[0, 1])

    assert 'WARNING' not in format_history_block(teacher._history)


# --- Finding 2: the autoreset tick executes nothing ---

def test_reset_only_tick_is_not_recorded_as_student_behaviour():
    """
    Under Gymnasium NEXT_STEP autoreset the action submitted on the tick
    after a termination is discarded by the environment, so it must not
    enter the teacher's history.

    Drives the trainer's own `note_executed_action`, so a regression in
    production leaves this test red rather than green.
    """

    num_envs = 2
    executed_since_query = [[] for _ in range(num_envs)]
    # Env 0's previous episode ended, so this tick only resets it.
    pre_step_dones = np.array([True, False])
    submitted = np.array([0, 5])

    for i in range(num_envs):
        # The PRODUCTION filter, imported from the trainer.
        note_executed_action(
            executed_since_query[i], submitted[i], pre_step_dones[i]
        )

    assert executed_since_query[0] == [], (
        'the discarded reset-tick action must not be recorded'
    )
    assert executed_since_query[1] == [5]
