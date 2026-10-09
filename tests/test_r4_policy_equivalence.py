"""
R4's detached head must leave the policy update identical to R1.

The unit test that the head passes no gradient to `hidden` is necessary
but not sufficient. R4 adds parameters to the model and terms to the
optimizer, and either can perturb the policy update even with zero
gradient flow into the trunk:

  - a shared optimizer sees extra parameters, and global gradient
    clipping is computed over the whole parameter set, so a large
    auxiliary gradient can rescale the POLICY gradients;
  - constructing the head consumes RNG draws, so an R4 run can
    initialize its policy differently from R1 unless the order is
    controlled;
  - optimizers with state (Adam) can behave differently when their
    parameter list changes.

These check the property that actually matters: on identical data, the
policy parameters after an update are the same with and without a
detached auxiliary head. A failure here means R4 is not a clean control
even though the detach is correct.
"""

import copy

import pytest
import torch
import torch.nn as nn

from algos.explanation_head import ExplanationHead, explanation_loss


class TinyPolicy(nn.Module):
    """
    A stand-in for the shared trunk plus actor, small enough to compare
    parameters exactly.
    """

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = nn.Linear(8, 512)
        self.actor = nn.Linear(512, 7)

    def forward(self, x):
        hidden = torch.relu(self.trunk(x))
        return hidden, self.actor(hidden)


@pytest.fixture
def batch():
    torch.manual_seed(123)
    return {
        'obs': torch.randn(16, 8),
        'advantage': torch.randn(16),
        'target': torch.randn(16, 64),
        'mask': torch.ones(16),
    }


def policy_step(batch, head=None, clip_norm=None, lambda_aux=1.0,
                shared_optimizer=True, seed=0):
    """
    Run one update and return the resulting policy parameters.

    `head=None` is R1. A head with `detach_features=True` is R4. The
    policy is always constructed first from a fixed seed so both arms
    start from identical weights.
    """

    policy = TinyPolicy(seed=seed)
    params = list(policy.parameters())
    if head is not None and shared_optimizer:
        params = params + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    hidden, logits = policy(batch['obs'])
    loss = -(logits.log_softmax(-1).mean(-1) * batch['advantage']).mean()
    if head is not None:
        aux = explanation_loss(
            head(hidden, detach_features=True),
            batch['target'], batch['mask'],
        )
        loss = loss + lambda_aux * aux

    optimizer.zero_grad()
    loss.backward()
    if clip_norm is not None:
        # Deliberately clipping over whatever the optimizer owns, which
        # is how a naive implementation would do it.
        nn.utils.clip_grad_norm_(params, clip_norm)
    optimizer.step()
    return [p.detach().clone() for p in policy.parameters()]


def assert_same(a, b, message):
    for left, right in zip(a, b):
        assert torch.allclose(left, right, atol=0, rtol=0), message


def test_detached_head_leaves_policy_update_identical_without_clipping(
        batch):
    """
    The baseline case: no clipping, so the only risk is gradient flow,
    and the detach handles it.
    """

    head = ExplanationHead(hidden_dim=512, embed_dim=64)
    assert_same(
        policy_step(batch, head=None),
        policy_step(batch, head=head),
        'a detached head changed the policy update with no clipping',
    )


def test_detached_head_leaves_policy_update_identical_with_separate_optimizer(
        batch):
    """
    Giving the head its own optimizer is the construction that is safe
    by design, and it must reproduce R1 exactly.
    """

    head = ExplanationHead(hidden_dim=512, embed_dim=64)
    assert_same(
        policy_step(batch, head=None, clip_norm=0.5),
        policy_step(batch, head=head, clip_norm=0.5,
                    shared_optimizer=False),
        'a separately-optimized detached head must not touch the policy',
    )


def test_global_clipping_over_a_shared_parameter_list_breaks_r4(batch):
    """
    The failure mode this file exists to catch.

    With one optimizer and global clipping, the head's gradients enter
    the total norm, so the policy gradients get rescaled by a factor
    that depends on the auxiliary loss. R4 then differs from R1 for a
    reason that has nothing to do with representation learning.

    This asserts the hazard is REAL, so the implementation is required
    to avoid it rather than assume it away.
    """

    head = ExplanationHead(hidden_dim=512, embed_dim=64)
    r1 = policy_step(batch, head=None, clip_norm=0.01)
    r4 = policy_step(batch, head=head, clip_norm=0.01,
                     lambda_aux=100.0, shared_optimizer=True)

    differs = any(
        not torch.allclose(a, b, atol=0, rtol=0) for a, b in zip(r1, r4)
    )
    assert differs, (
        'expected shared-list global clipping to perturb the policy; if '
        'this no longer holds the guard below can be relaxed'
    )


def test_clipping_only_policy_parameters_restores_equivalence(batch):
    """
    The correction: clip over the policy's parameters only, and R4
    reproduces R1 exactly even with an enormous auxiliary loss.
    """

    def step(with_head):
        policy = TinyPolicy(seed=0)
        head = ExplanationHead(hidden_dim=512, embed_dim=64)
        opt = torch.optim.Adam(
            list(policy.parameters())
            + (list(head.parameters()) if with_head else []),
            lr=1e-3,
        )
        hidden, logits = policy(batch['obs'])
        loss = -(logits.log_softmax(-1).mean(-1)
                 * batch['advantage']).mean()
        if with_head:
            loss = loss + 100.0 * explanation_loss(
                head(hidden, detach_features=True),
                batch['target'], batch['mask'])
        opt.zero_grad()
        loss.backward()
        # Clip the POLICY only -- the head is not part of the norm.
        nn.utils.clip_grad_norm_(policy.parameters(), 0.01)
        opt.step()
        return [p.detach().clone() for p in policy.parameters()]

    assert_same(
        step(with_head=False), step(with_head=True),
        'clipping only policy parameters must make R4 equal to R1',
    )


def test_head_construction_does_not_shift_policy_initialization():
    """
    Building the head consumes RNG draws. If it is constructed before
    the policy, R4 starts from different weights than R1 -- a difference
    that has nothing to do with the control.
    """

    policy_first = TinyPolicy(seed=0)
    ExplanationHead(hidden_dim=512, embed_dim=64)

    ExplanationHead(hidden_dim=512, embed_dim=64)
    policy_second = TinyPolicy(seed=0)

    assert_same(
        list(policy_first.parameters()),
        list(policy_second.parameters()),
        'policy init must be seeded independently of head construction',
    )


def test_auxiliary_loss_magnitude_cannot_move_the_policy_when_detached(
        batch):
    """
    With the head separately optimized, scaling lambda_aux by four
    orders of magnitude must leave the policy update untouched.
    """

    head = ExplanationHead(hidden_dim=512, embed_dim=64)
    small = policy_step(batch, head=head, lambda_aux=1e-3,
                        shared_optimizer=False)
    huge = policy_step(batch, head=copy.deepcopy(head), lambda_aux=1e3,
                       shared_optimizer=False)
    assert_same(small, huge,
                'a detached head must be invariant to lambda_aux')
