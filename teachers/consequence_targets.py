"""Raw categorical LLM targets and matched rollout controls, without APIs.

Simulator truth never enters this module.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


EFFECTS = {
    'movement': ('stay', 'forward'),
    'inventory': ('same', 'picked_up', 'dropped'),
    'front_door': ('same', 'opened', 'closed'),
    'goal_reached': (False, True),
}
FIELDS = tuple(EFFECTS)


class ConsequenceSchemaError(ValueError):
    """An auxiliary reply does not satisfy the frozen categorical schema."""


def _action(value, maximum=6):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ConsequenceSchemaError('Invalid discrete action')
    return value


def component_mask(action):
    """Mask action-determined fields, rotations and DONE from the loss."""
    action = _action(action)
    return (action == 2, action in (3, 4), action == 5, action == 2)


def action_label(payload):
    """Read a valid action independently of auxiliary-schema validity."""
    if not isinstance(payload, Mapping):
        return None
    try:
        return _action(payload.get('action'), maximum=5)
    except ConsequenceSchemaError:
        return None


def decode_response(payload, student_action):
    """Strictly validate syntax, not physical correctness; return a copy."""
    _action(student_action)
    required = {
        'action',
        'recommended',
        'student',
        'why_recommended',
        'why_student',
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ConsequenceSchemaError('Response fields differ from schema')
    result = {'action': _action(payload['action'], maximum=5)}
    for branch in ('recommended', 'student'):
        values = payload[branch]
        if not isinstance(values, Mapping) or set(values) != set(FIELDS):
            raise ConsequenceSchemaError('Effect fields differ from schema')
        result[branch] = {}
        for field, choices in EFFECTS.items():
            value = values[field]
            if not any(
                type(value) is type(choice) and value == choice
                for choice in choices
            ):
                raise ConsequenceSchemaError(f'Invalid effect: {field}')
            result[branch][field] = value
    for key in ('why_recommended', 'why_student'):
        value = payload[key]
        if not isinstance(value, str) or not 1 <= len(value) <= 500:
            raise ConsequenceSchemaError('Invalid explanation string')
        result[key] = value
    return result


@dataclass(frozen=True)
class ConsequenceRecord:
    sample_id: str
    rollout: int
    phase: str
    student_action: int
    response: object
    # Hash the world/history input, excluding sample ID. Fixtures may
    # omit it; integration must provide it to detect repeated queries.
    query_identity: str | None = None


@dataclass(frozen=True)
class RolloutTargets:
    sample_ids: tuple[str, ...]
    action_labels: np.ndarray
    actions: np.ndarray
    correct: np.ndarray
    shuffled: np.ndarray
    mask: np.ndarray
    donor_ids: tuple[str | None, ...]
    status: tuple[str, ...]
    stats: dict


def resolve_rollout_targets(
    records: Sequence[ConsequenceRecord], *, seed: int
) -> RolloutTargets:
    """Resolve once per rollout and retain the result through PPO epochs.

    Permute complete consultation bundles within phase/action-set groups.
    A random cycle preserves the joint target multiset without self donors;
    it is not a uniform draw over all possible derangements. Correct,
    detached and shuffled consumers MUST all use the returned shared mask.
    Invalid or donor-ineligible auxiliaries never erase a valid action label.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    ids = tuple(record.sample_id for record in records)
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Nonempty sample IDs required')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate consultation IDs')
    rollouts = {record.rollout for record in records}
    if any(type(x) is not int or x < 0 for x in rollouts):
        raise ValueError('Nonnegative rollout indices required')
    if len(rollouts) > 1:
        raise ValueError('Donors must belong to a single rollout')
    n = len(records)
    actions = np.full((n, 2), 6, dtype=np.int64)
    correct = np.full((n, 2, len(FIELDS)), -1, dtype=np.int64)
    mask = np.zeros_like(correct, dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    status = ['invalid_auxiliary'] * n
    groups = defaultdict(list)
    for i, record in enumerate(records):
        _action(record.student_action)
        if not isinstance(record.phase, str) or not record.phase:
            raise ValueError('Nonempty symbolic phase required')
        if record.query_identity is not None and (
            not isinstance(record.query_identity, str)
            or not record.query_identity
        ):
            raise ValueError('query_identity must be a nonempty hash or None')
        label = action_label(record.response)
        if label is not None:
            labels[i] = label
        try:
            parsed = decode_response(record.response, record.student_action)
        except ConsequenceSchemaError:
            continue
        if label == record.student_action and (
            parsed['recommended'] != parsed['student']
        ):
            status[i] = 'conflicting_duplicate_action'
            continue
        branches = {
            record.student_action: parsed['student'],
            label: parsed['recommended'],
        }
        active = sorted(a for a in branches if any(component_mask(a)))
        if not active:
            status[i] = 'no_state_dependent_action'
            continue
        for b, action in enumerate(active):
            actions[i, b] = action
            mask[i, b] = component_mask(action)
            correct[i, b] = [
                EFFECTS[field].index(branches[action][field])
                for field in FIELDS
            ]
        # Bundles preserve loss weights, actions and joint labels.
        # Branch role is not a head feature or a grouping variable.
        key = (record.phase, tuple(active), tuple(mask[i].ravel()))
        groups[key].append(i)
        status[i] = 'awaiting_donor'

    shuffled = correct.copy()
    donors = [None] * n
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, next(iter(rollouts), 0)])
    )
    eligible_groups = 0
    # Stable IDs make the mapping independent of input row order.
    for key in sorted(groups):
        indices = groups[key]
        identities = Counter(
            records[i].query_identity
            for i in indices
            if records[i].query_identity is not None
        )
        unique = []
        for i in indices:
            identity = records[i].query_identity
            if identity is not None and identities[identity] > 1:
                status[i] = 'repeated_query_identity'
                mask[i] = False
            else:
                unique.append(i)
        if len(unique) < 2:
            for i in unique:
                status[i] = 'no_donor'
                mask[i] = False
            continue
        eligible_groups += 1
        ordered = sorted(unique, key=lambda i: ids[i])
        cycle = rng.permutation(ordered).tolist()
        for recipient, donor in zip(cycle, cycle[1:] + cycle[:1]):
            shuffled[recipient] = correct[donor]
            donors[recipient] = ids[donor]
            status[recipient] = 'eligible'

    eligible = mask.any(axis=(1, 2))
    unchanged = np.logical_or(correct == shuffled, ~mask).all(axis=(1, 2))
    count = int(eligible.sum())
    active_count = int(mask.sum())
    same_count = int(((correct == shuffled) & mask).sum())
    class_counts = {
        field: dict(Counter(correct[:, :, j][mask[:, :, j]].tolist()))
        for j, field in enumerate(FIELDS)
    }
    stats = {
        'consultations': n,
        'valid_action_labels': int((labels >= 0).sum()),
        'eligible_consultations': count,
        'eligible_groups': eligible_groups,
        'status_counts': dict(Counter(status)),
        'eligible_without_query_identity': sum(
            bool(eligible[i]) and record.query_identity is None
            for i, record in enumerate(records)
        ),
        'active_components': active_count,
        'unchanged_components': same_count,
        'unchanged_component_fraction': (
            same_count / active_count if active_count else None
        ),
        'unchanged_bundle_fraction': (
            int((eligible & unchanged).sum()) / count if count else None
        ),
        'class_counts': class_counts,
    }
    return RolloutTargets(
        ids,
        labels,
        actions,
        correct,
        shuffled,
        mask,
        tuple(donors),
        tuple(status),
        stats,
    )
