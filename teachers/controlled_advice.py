"""Explicit teacher-stream and structured-rationale experimental controls."""

import hashlib
from copy import deepcopy
from dataclasses import replace

import numpy as np

from envs.state import extract_doorkey_state, extract_generic_state
from teachers.base import Cost


SUBGOALS = (
    'explore', 'get_key', 'unlock_door', 'open_door',
    'reach_object', 'pick_up_target', 'reach_goal', 'other',
)


class QueryOnlyTeacher:
    """Expose the selected answer only after a counted consultation."""

    def __init__(self, teacher):
        self._teacher = teacher
        self.teacher_id = teacher.teacher_id

    def recommend(self, state, context=None):
        return self._teacher.recommend(state, context)


def advisor_teacher(teacher, no_peek=False):
    """Keep legacy capabilities unless query-before-filter is requested."""
    return QueryOnlyTeacher(teacher) if no_peek else teacher


class CurrentAdvice:
    """Let an advisor select a precomputed label without advancing a bot."""

    def __init__(self, advice):
        self.advice = replace(advice, cost=Cost())
        self.teacher_id = advice.teacher_id

    def peek_action(self, state):
        return self.advice.action

    def recommend(self, state, context=None):
        return self.advice


def consult_reference(teacher, kind, env, new_episode, last_action):
    """Query the actual free teacher once at the current executed state."""

    state = extract_doorkey_state(env) if kind == 'oracle' else env
    return teacher.recommend(state, {
        'new_episode': new_episode, 'last_action': last_action,
    })


def reference_target(advice, num_actions):
    """Use one deterministic action target, independent of LLM confidence."""

    if advice is None or advice.action is None:
        return None
    if not 0 <= advice.action < num_actions:
        raise ValueError('Reference action is outside the action space')
    vector = np.zeros(num_actions, dtype=np.float32)
    vector[advice.action] = 1.0
    return vector


def rationale_schema(base, reference_action=None):
    """Request text and an LLM-selected subgoal in the same API response."""

    result = deepcopy(base)
    schema = result['format']['schema']
    schema['properties']['subgoal'] = {'type': 'string',
                                        'enum': list(SUBGOALS)}
    schema['required'].append('subgoal')
    if reference_action is not None:
        schema['properties']['action']['enum'] = [int(reference_action)]
    return result


def subgoal_vectors(rows):
    """Encode model-written subgoal fields, not symbolic grouping phases."""

    indices = [SUBGOALS.index(row['subgoal']) for row in rows]
    return np.eye(len(SUBGOALS), dtype=np.float32)[indices]


def aliased_advice(advice, unwrapped, num_actions, rate):
    """
    Redirect a fraction of labels as a fixed function of hidden state.

    This is aliasing, not noise, and the difference is the whole point.
    Noise drawn afresh each visit averages out: a state seen often still
    teaches its majority action. Aliasing is *consistent* -- the same
    unobservable configuration always yields the same wrong label -- so
    repetition never resolves it. That is what a privileged teacher
    inflicts on a student whose 7x7 egocentric view omits the key, the
    door and its own absolute position.

    The decision is keyed on the full symbolic state, none of which is
    recoverable from a local observation, so two states that look
    identical to the student can disagree about the label. `rate` is the
    share of state space redirected; measured contradiction rates were
    .033 (DoorKey), .097 (MultiRoom) and .130 (KeyCorridor S3R3) under
    each task's own teacher distribution.
    """

    if rate <= 0 or advice is None or advice.action is None:
        return advice
    if num_actions < 2:
        return advice
    state = extract_generic_state(unwrapped)
    digest = hashlib.blake2b(repr(state).encode('utf-8'),
                             digest_size=8).digest()
    value = int.from_bytes(digest, 'big')
    metadata = dict(advice.cost.metadata or {})
    if (value / 2 ** 64) >= rate:
        metadata['aliased'] = False
        return replace(advice, cost=replace(advice.cost, metadata=metadata))
    # Deterministic alternative action, never the teacher's own choice.
    offset = 1 + (value % (num_actions - 1))
    metadata['aliased'] = True
    metadata['teacher_action_before_alias'] = int(advice.action)
    return replace(advice, action=(int(advice.action) + offset) % num_actions,
                   cost=replace(advice.cost, metadata=metadata))
