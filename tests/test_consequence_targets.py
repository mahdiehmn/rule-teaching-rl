"""Independent action validity and matched categorical donor controls."""

from copy import deepcopy

import numpy as np
import pytest

from teachers.consequence_targets import (
    ConsequenceRecord,
    ConsequenceSchemaError,
    action_label,
    decode_response,
    resolve_rollout_targets,
)


def reply(action=2, forward=False):
    effects = dict(
        movement='forward' if forward else 'stay',
        inventory='same',
        front_door='same',
        goal_reached=forward,
    )
    return dict(
        action=action,
        recommended=effects.copy(),
        student=effects.copy(),
        why_recommended='Immediate effect.',
        why_student='Immediate effect.',
    )


def record(i, *, phase='seek_key', student=2, payload=None, identity=None):
    return ConsequenceRecord(
        str(i),
        7,
        phase,
        student,
        reply() if payload is None else payload,
        identity,
    )


def resolve(records):
    return resolve_rollout_targets(records, seed=42)


@pytest.mark.parametrize('invalid', [1, 'false', None])
def test_boolean_effect_is_not_coerced(invalid):
    payload = reply()
    payload['student']['goal_reached'] = invalid
    with pytest.raises(ConsequenceSchemaError):
        decode_response(payload, 2)
    assert action_label(payload) == 2
    result = resolve([record(0, payload=payload)])
    assert result.action_labels.tolist() == [2]
    assert not result.mask.any()


def test_schema_is_exact_and_boolean_action_is_invalid():
    payload = reply()
    payload['action'] = True
    assert action_label(payload) is None
    with pytest.raises(ConsequenceSchemaError):
        decode_response(payload, 2)
    payload = reply()
    payload['recommended']['invented'] = 'value'
    with pytest.raises(ConsequenceSchemaError):
        decode_response(payload, 2)


def test_conflicting_duplicate_masks_auxiliary_only():
    payload = reply()
    payload['student']['movement'] = 'forward'
    result = resolve([record(0, payload=payload), record(1)])
    assert result.action_labels.tolist() == [2, 2]
    assert result.status == ('conflicting_duplicate_action', 'no_donor')
    assert not result.mask.any()


def test_agreement_deduplicates_and_masks_inactive_fields():
    result = resolve([record(0), record(1, payload=reply(forward=True))])
    assert result.actions.tolist() == [[2, 6], [2, 6]]
    assert result.mask.tolist() == [
        [[True, False, False, True], [False] * 4],
        [[True, False, False, True], [False] * 4],
    ]
    assert result.donor_ids == ('1', '0')
    assert result.stats['unchanged_component_fraction'] == 0
    assert result.stats['unchanged_bundle_fraction'] == 0


def test_raw_physically_wrong_targets_are_not_corrected_or_filtered():
    payload = reply(action=3)
    payload['recommended']['inventory'] = 'dropped'
    payload['student']['inventory'] = 'dropped'
    result = resolve(
        [
            record(0, student=3, payload=payload),
            record(1, student=3, payload=payload),
        ]
    )
    # A pickup cannot drop inventory. Test raw-label preservation;
    # it does not establish the physical truth of a prediction.
    assert result.stats['eligible_consultations'] == 2
    assert result.correct[:, 0, 1].tolist() == [2, 2]
    assert result.stats['unchanged_component_fraction'] == 1


def test_whole_bundles_preserve_actions_roles_and_weighted_targets():
    rows = []
    for i in range(4):
        payload = reply(action=2 if i % 2 == 0 else 5)
        student = 5 if i % 2 == 0 else 2
        for branch in ('recommended', 'student'):
            payload[branch]['movement'] = 'forward' if i > 1 else 'stay'
            payload[branch]['front_door'] = 'opened' if i > 1 else 'same'
        rows.append(record(i, student=student, payload=payload))
    # These single-action bundles MUST NOT mix with two-action bundles.
    rows.extend([record(4), record(5, payload=reply(forward=True))])
    original = deepcopy(rows)
    result = resolve(rows)
    assert rows == original
    assert result.stats['eligible_consultations'] == 6
    assert result.action_labels.tolist() == [2, 5, 2, 5, 2, 2]
    for i, donor in enumerate(result.donor_ids):
        d = int(donor)
        assert d != i
        assert (i < 4) == (d < 4)
        np.testing.assert_array_equal(result.actions[i], result.actions[d])
        np.testing.assert_array_equal(result.shuffled[i], result.correct[d])
    # Preserve the joint bundle multiset and each action's loss weight.
    for indices in (range(4), range(4, 6)):
        left = sorted(tuple(result.correct[i].ravel()) for i in indices)
        right = sorted(tuple(result.shuffled[i].ravel()) for i in indices)
        assert left == right


def test_phase_singletons_and_rotations_never_get_donors():
    result = resolve(
        [
            record(0),
            record(1, phase='unlock_door'),
            record(2, student=0, payload=reply(action=1)),
        ]
    )
    assert result.status == (
        'no_donor',
        'no_donor',
        'no_state_dependent_action',
    )
    assert result.action_labels.tolist() == [2, 2, 1]
    assert result.stats['unchanged_component_fraction'] is None
    assert not result.mask.any()


def test_repeated_query_content_is_excluded_not_silent_corruption():
    result = resolve(
        [
            record(0, identity='same'),
            record(1, identity='same'),
            record(2, identity='different'),
        ]
    )
    assert result.status == (
        'repeated_query_identity',
        'repeated_query_identity',
        'no_donor',
    )
    assert result.stats['valid_action_labels'] == 3
    assert not result.mask.any()


def test_deterministic_donors_ignore_row_order_and_global_rng():
    records = [record(i, payload=reply(forward=bool(i % 2))) for i in range(6)]
    np.random.seed(8)
    expected = np.random.random(4)
    np.random.seed(8)
    a = resolve(records)
    np.testing.assert_array_equal(np.random.random(4), expected)
    b = resolve(records[::-1])
    assert dict(zip(a.sample_ids, a.donor_ids)) == dict(
        zip(b.sample_ids, b.donor_ids)
    )


def test_input_identity_and_rollout_fail_closed():
    with pytest.raises(ValueError, match='Duplicate'):
        resolve([record(0), record(0)])
    with pytest.raises(ValueError, match='single rollout'):
        resolve([record(0), ConsequenceRecord('1', 8, 'seek_key', 2, reply())])
    result = resolve([])
    assert result.stats['consultations'] == 0
    assert result.mask.shape == (0, 2, 4)


def test_resolved_targets_feed_head_with_matched_action_updates():
    import torch
    from torch import nn

    from algos.consequence_head import ConsequenceHead, consequence_loss

    resolved = resolve([record(0), record(1, payload=reply(forward=True))])
    actions = torch.as_tensor(resolved.actions)
    mask = torch.as_tensor(resolved.mask)
    action_labels = torch.as_tensor(resolved.action_labels)
    torch.manual_seed(3)
    trunk = nn.Linear(3, 8)
    actor = nn.Linear(8, 7)
    head = ConsequenceHead(hidden_dim=8, seed=2)
    inputs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])

    def update(kind):
        shared, policy, auxiliary = deepcopy((trunk, actor, head))
        params = [*shared.parameters(), *policy.parameters()]
        optimizer = torch.optim.SGD(
            [*params, *auxiliary.parameters()],
            lr=0.1,
        )
        hidden = shared(inputs)
        loss = nn.functional.cross_entropy(policy(hidden), action_labels)
        if kind != 'action_only':
            labels = (
                resolved.shuffled if kind == 'shuffled' else resolved.correct
            )
            prediction = auxiliary(
                hidden,
                actions,
                detach_features=kind == 'detached',
            )
            loss = loss + consequence_loss(
                prediction,
                torch.as_tensor(labels),
                mask,
                actions,
            )
        loss.backward()
        optimizer.step()
        return torch.cat([p.detach().flatten() for p in params])

    baseline = update('action_only')
    assert torch.equal(baseline, update('detached'))
    aligned = update('aligned')
    assert not torch.equal(baseline, aligned)
    assert not torch.equal(aligned, update('shuffled'))
