"""
Categorical consequence gradients and masking.

Synthetic labels exercise a software primitive, not an LLM quality claim.
"""

import math

import pytest
import torch
from torch import nn

from algos.consequence_head import (
    ACTIONS,
    EFFECTS,
    FIELDS,
    WIDTHS,
    ConsequenceHead,
    action_component_masks,
    consequence_loss,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    """
    Keep small orthogonal initializations from oversubscribing the host.
    """
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def zero_predictions(actions):
    """
    Uniform categorical logits make expected CE analytically checkable.
    """
    return {
        name: torch.zeros(*actions.shape, width, requires_grad=True)
        for name, width in zip(FIELDS, WIDTHS)
    }


def labels(actions):
    """
    Start with class zero on every component before action eligibility.
    """
    shape = (*actions.shape, len(FIELDS))
    return torch.zeros(shape, dtype=torch.long), torch.ones(
        shape, dtype=torch.bool
    )


def test_vocabulary_action_conditioning_and_duplicate_role_invariance():
    head = ConsequenceHead(seed=5)
    assert FIELDS == ('movement', 'inventory', 'front_door', 'goal_reached')
    assert [list(EFFECTS[k]) for k in FIELDS] == [
        ['stay', 'forward'],
        ['same', 'picked_up', 'dropped'],
        ['same', 'opened', 'closed'],
        [False, True],
    ]
    assert WIDTHS == (2, 3, 3, 2) and len(ACTIONS) == 7
    assert head.net[0].in_features == 519
    assert head.net[0].out_features == 512
    assert head.net[2].out_features == 10
    hidden = torch.zeros(7, 512)
    actions = torch.arange(7)
    output = head(hidden, actions)
    assert {k: p.shape for k, p in output.items()} == {
        k: (7, width) for k, width in zip(FIELDS, WIDTHS)
    }
    assert not torch.equal(output['movement'][0], output['movement'][1])
    paired = head(hidden, torch.stack((actions, actions), dim=1))
    for name in FIELDS:
        assert torch.equal(paired[name][:, 0], paired[name][:, 1])
    assert head.metadata()['effects']['goal_reached'] == [False, True]


def test_seeded_initialization_preserves_cpu_stream_and_reproduces_weights():
    torch.manual_seed(991)
    before = torch.get_rng_state().clone()
    first = ConsequenceHead(seed=19)
    assert torch.equal(torch.get_rng_state(), before)
    second = ConsequenceHead(seed=19)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(
        torch.equal(a, b)
        for a, b in zip(first.parameters(), second.parameters())
    )
    third = ConsequenceHead(seed=20)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(first.net[0].weight, third.net[0].weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_initialization_preserves_existing_cuda_rng_stream():
    torch.cuda.manual_seed_all(337)
    before = [s.clone() for s in torch.cuda.get_rng_state_all()]
    ConsequenceHead(seed=2).to('cuda')
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, torch.cuda.get_rng_state_all())
    )


@pytest.mark.parametrize('detached', [False, True])
def test_auxiliary_updates_head_and_only_attached_trunk(detached):
    torch.manual_seed(44)
    trunk = nn.Linear(3, 512)
    head = ConsequenceHead(seed=6)
    x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    actions = torch.tensor([[2, 3], [5, 4]])
    targets, mask = labels(actions)
    targets = targets.float().requires_grad_()
    before_trunk = trunk.weight.detach().clone()
    before_head = head.net[2].weight.detach().clone()
    optimizer = torch.optim.SGD(
        list(trunk.parameters()) + list(head.parameters()), lr=0.1
    )
    predicted = head(trunk(x), actions, detach_features=detached)
    consequence_loss(predicted, targets, mask, actions).backward()
    assert targets.grad is None
    if detached:
        assert trunk.weight.grad is None
    else:
        assert trunk.weight.grad is not None
        assert trunk.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(head.net[2].weight, before_head)
    assert torch.equal(trunk.weight, before_trunk) == detached


def test_loss_weights_components_then_actions_then_transitions():
    actions = torch.tensor([[2, 3], [5, 0]])
    targets, mask = labels(actions)
    loss = consequence_loss(zero_predictions(actions), targets, mask, actions)
    expected = ((math.log(2) + math.log(3)) / 2 + math.log(3)) / 2
    assert loss.item() == pytest.approx(expected)
    assert loss.item() != pytest.approx((math.log(2) + 2 * math.log(3)) / 3)


def test_duplicate_action_counts_once_and_keeps_head_gradient_identical():
    # Different GEMM batch sizes can round differently in float32.
    head = ConsequenceHead(seed=7).double()
    hidden = torch.ones(2, 512, dtype=torch.float64)
    single = torch.tensor([2, 5])
    targets, mask = labels(single)
    original = consequence_loss(head(hidden, single), targets, mask, single)
    original.backward()
    gradients = [p.grad.detach().clone() for p in head.parameters()]
    head.zero_grad(set_to_none=True)
    paired = torch.stack((single, single), dim=1)
    t2, m2 = labels(paired)
    duplicate = consequence_loss(head(hidden, paired), t2, m2, paired)
    duplicate.backward()
    assert duplicate.item() == pytest.approx(original.item())
    for parameter, gradient in zip(head.parameters(), gradients):
        assert torch.allclose(parameter.grad, gradient, atol=1e-12)


def test_duplicate_conflicting_targets_and_masks_fail_loudly():
    actions = torch.tensor([[2, 2]])
    targets, mask = labels(actions)
    predicted = zero_predictions(actions)
    targets[0, 1, 0] = 1
    with pytest.raises(ValueError, match='disagree on targets'):
        consequence_loss(predicted, targets, mask, actions)
    targets.zero_()
    mask[0, 1, 0] = False
    with pytest.raises(ValueError, match='different masks'):
        consequence_loss(predicted, targets, mask, actions)


def test_only_state_dependent_action_components_receive_gradient():
    actions = torch.arange(7)
    targets, mask = labels(actions)
    predicted = zero_predictions(actions)
    expected_mask = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(action_component_masks(actions), expected_mask)
    loss = consequence_loss(predicted, targets, mask, actions)
    assert loss.item() == pytest.approx((math.log(2) + 3 * math.log(3)) / 4)
    loss.backward()
    for field, name in enumerate(FIELDS):
        changed_rows = predicted[name].grad.abs().sum(-1) > 0
        assert torch.equal(changed_rows, expected_mask[:, field])


def test_masked_invalid_targets_and_logits_do_not_poison_loss():
    actions = torch.tensor([[2, 3], [6, 0]])
    targets, mask = labels(actions)
    targets = targets.float()
    mask[0, 1] = False
    effective = mask & action_component_masks(actions)
    targets[~effective] = float('nan')
    predicted = zero_predictions(actions)
    for field, name in enumerate(FIELDS):
        with torch.no_grad():
            predicted[name][~effective[..., field]] = float('nan')
    loss = consequence_loss(predicted, targets, mask, actions)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predicted.values())


def test_active_nonfinite_logits_fail_and_explicit_empty_mask_is_connected():
    actions = torch.tensor([2])
    targets, mask = labels(actions)
    predicted = zero_predictions(actions)
    with torch.no_grad():
        predicted['movement'].fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite active logits'):
        consequence_loss(predicted, targets, mask, actions)
    mask.zero_()
    loss = consequence_loss(predicted, targets, mask, actions)
    assert loss.item() == 0
    loss.backward()
    assert all(bool((p.grad == 0).all()) for p in predicted.values())


@pytest.mark.parametrize('bad', [-1.0, 2.0, 0.5, float('nan'), float('inf')])
def test_invalid_active_targets_are_rejected(bad):
    actions = torch.tensor([2])
    targets, mask = labels(actions)
    targets = targets.float()
    targets[0, 0] = bad
    with pytest.raises(ValueError, match='Invalid active class target'):
        consequence_loss(zero_predictions(actions), targets, mask, actions)


@pytest.mark.parametrize('shape', [(0,), (3, 0), (3, 2)])
def test_empty_or_fully_masked_loss_is_finite_connected_zero(shape):
    actions = torch.zeros(shape, dtype=torch.long)
    targets, mask = labels(actions)
    targets = targets.float().fill_(float('nan'))
    predicted = zero_predictions(actions)
    with torch.no_grad():
        for p in predicted.values():
            p.fill_(float('nan'))
    loss = consequence_loss(predicted, targets, mask, actions)
    assert loss.item() == 0
    loss.backward()
    assert all(
        p.grad is not None and bool((p.grad == 0).all())
        for p in predicted.values()
    )


@pytest.mark.parametrize(
    'actions', [torch.tensor([-1]), torch.tensor([7]), torch.tensor([2.5])]
)
def test_action_indices_must_be_integer_and_in_range(actions):
    with pytest.raises(ValueError, match='Action'):
        action_component_masks(actions)
