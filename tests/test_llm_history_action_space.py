"""
History formatting must survive every action the student can execute.

The bug: `_ACTION_NAMES` held six entries while the student's actor
samples from `Discrete(7)`. Action 6 (`done`) is ordinary for a policy to
emit; it was invisible in teacher-alone evaluation only because the
teacher's recommendation schema never proposes it. On a real
`keycorridor_s3r3` state, `done` leaves the symbolic state unchanged
without ending the episode, so the history correctly records `(6, True)`
-- and the blocked-action warning then indexed past the end of the tuple.
Prompt formatting runs *before* the API call's exception handler, so that
IndexError took down the whole next consultation.

These exercise the real production functions rather than copies of their
expressions, and they read the action count from a real environment so
the coverage cannot silently drift from the action space.
"""

import pytest

from envs.registry import build_env
from teachers.minigrid.llm_general import (
    _ACTION_NAMES,
    format_history_block,
)


@pytest.fixture(scope='module')
def action_count():
    """
    The number of actions the student can actually execute on the task
    the explanation experiment runs on.
    """

    env = build_env('keycorridor_s3r3', seed=0, obs_mode='symbolic',
                    obs_target_size=56, agent_view_size=7)
    n = int(env.action_space.n)
    env.close()
    return n


def test_vocabulary_covers_the_whole_student_action_space(action_count):
    """
    The names table must not be shorter than the action space; that
    mismatch is what produced the crash.
    """

    assert len(_ACTION_NAMES) >= action_count, (
        f'{len(_ACTION_NAMES)} names for {action_count} student actions'
    )


def test_every_action_formats_when_its_effect_was_observed(action_count):
    """
    Both observed outcomes, for every executable action, through the real
    formatter.
    """

    for action in range(action_count):
        for blocked in (True, False):
            rendered = format_history_block([(action, blocked)])
            assert rendered, f'no history rendered for action {action}'
            assert 'UNKNOWN ACTION' not in rendered, (
                f'action {action} is valid and must be named'
            )


def test_blocked_warning_survives_every_action(action_count):
    """
    The warning path is the one that crashed, because it indexed the
    names table directly. Action 6 with `blocked=True` is the exact
    reproduction from the review.
    """

    for action in range(action_count):
        rendered = format_history_block([(action, True)])
        assert 'WARNING' in rendered, (
            f'a blocked action {action} must still warn'
        )


def test_done_action_is_named_not_called_unknown(action_count):
    """
    Action 6 is `done`, a real MiniGrid action. Describing it as unknown
    would mislead the teacher about what the student did.
    """

    assert action_count == 7, (
        'keycorridor_s3r3 is expected to expose the full MiniGrid '
        'action space; if this changes, revisit the history vocabulary'
    )
    rendered = format_history_block([(6, True)])
    assert 'DONE' in rendered
    assert 'UNKNOWN ACTION' not in rendered
    assert '?' not in rendered


def test_unknown_ids_degrade_instead_of_crashing():
    """
    Formatting happens before the API call's exception handler, so an
    out-of-range id must never raise -- it must render something a
    reader can act on.
    """

    for bogus in (-1, 99):
        rendered = format_history_block([(bogus, True)])
        assert 'UNKNOWN ACTION' in rendered


def test_done_leaves_the_symbolic_state_unchanged(action_count):
    """
    Ground the premise in the real environment: `done` on this task does
    not end the episode and does not change the symbolic state, which is
    why it is recorded as blocked and reaches the warning path at all.
    """

    from envs.state import extract_generic_state

    env = build_env('keycorridor_s3r3', seed=0, obs_mode='symbolic',
                    obs_target_size=56, agent_view_size=7)
    env.reset(seed=0)
    before = extract_generic_state(env.unwrapped)
    _, _, terminated, truncated, _ = env.step(action_count - 1)
    after = extract_generic_state(env.unwrapped)
    env.close()

    assert not (terminated or truncated), (
        'done is expected not to end the episode on this task'
    )
    assert before == after, (
        'done is expected to leave the symbolic state unchanged, which '
        'is what makes the history record it as blocked'
    )
