"""
The explanation head and its two controls must differ in exactly one way.

R2, R3 and R4 are only interpretable if each isolates a single thing:
R4 changes whether the auxiliary gradient reaches shared features, R3
changes only the target's content, and neither changes the action label,
the query mask or the number of auxiliary samples. These tests pin those
properties directly, because a control that quietly differs in a second
way produces a gap that means nothing.

No policy is trained and no teacher is called.
"""

import numpy as np
import pytest
import torch

from algos.explanation_head import (
    ExplanationHead,
    PhaseShuffler,
    apply_shared_eligibility,
    explanation_loss,
)


@pytest.fixture
def head():
    torch.manual_seed(0)
    return ExplanationHead(hidden_dim=512, embed_dim=64)


# --- The head itself ---

def test_head_maps_trunk_features_to_the_embedding_dimension(head):
    out = head(torch.zeros(8, 512))
    assert out.shape == (8, 64)


def test_head_never_returns_anything_shaped_like_an_action(head):
    """
    The head is auxiliary only. Its output width is the embedding size,
    which is not the action count, so it cannot be wired to the actor by
    accident.
    """

    assert head.embed_dim == 64
    assert head(torch.zeros(2, 512)).shape[-1] != 7


# --- R4: the detached control ---

def test_detached_head_sends_no_gradient_to_shared_features(head):
    """
    R4's whole purpose: the auxiliary loss must not update the trunk.
    """

    hidden = torch.randn(4, 512, requires_grad=True)
    target = torch.randn(4, 64)
    mask = torch.ones(4)

    loss = explanation_loss(head(hidden, detach_features=True),
                            target, mask)
    loss.backward()

    assert hidden.grad is None or torch.allclose(
        hidden.grad, torch.zeros_like(hidden.grad)
    ), 'R4 must not push gradient into the shared representation'


def test_attached_head_does_send_gradient_to_shared_features(head):
    """
    The complement: in R2 the gradient is *supposed* to reach the trunk.
    A control is only meaningful against a treatment that differs.
    """

    hidden = torch.randn(4, 512, requires_grad=True)
    target = torch.randn(4, 64)
    mask = torch.ones(4)

    loss = explanation_loss(head(hidden, detach_features=False),
                            target, mask)
    loss.backward()

    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0


def test_detached_and_attached_share_parameters_and_head_gradients(head):
    """
    R2 and R4 must differ only in gradient routing -- same architecture,
    same parameter count, and the head itself still learns in both.
    """

    target = torch.randn(4, 64)
    mask = torch.ones(4)

    for detached in (True, False):
        head.zero_grad()
        hidden = torch.randn(4, 512, requires_grad=True)
        explanation_loss(head(hidden, detach_features=detached),
                         target, mask).backward()
        grads = [p.grad.abs().sum().item() for p in head.parameters()
                 if p.grad is not None]
        assert grads and sum(grads) > 0, (
            'the head must still train in both arms'
        )


# --- The loss ---

def test_loss_is_zero_for_a_perfectly_predicted_target(head):
    vec = torch.randn(3, 64)
    loss = explanation_loss(vec, vec, torch.ones(3))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_ignores_unmasked_samples():
    """
    Unqueried steps carry no label and must not contribute.
    """

    good = torch.randn(1, 64)
    predicted = torch.cat([good, torch.randn(1, 64)])
    target = torch.cat([good, torch.randn(1, 64)])
    # Only the first sample is valid, and it is perfect.
    loss = explanation_loss(predicted, target, torch.tensor([1.0, 0.0]))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_zero_and_finite_when_nothing_is_labelled():
    """
    A rollout with no queried transition must not produce NaN.
    """

    loss = explanation_loss(torch.randn(4, 64), torch.randn(4, 64),
                            torch.zeros(4))
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_frozen_target_receives_no_gradient(head):
    """
    The text encoder is frozen; the auxiliary loss must never update it.
    """

    target = torch.randn(4, 64, requires_grad=True)
    explanation_loss(head(torch.randn(4, 512)), target,
                     torch.ones(4)).backward()
    assert target.grad is None


# --- R3: the shuffle ---

def test_shuffle_never_returns_a_sample_its_own_target():
    """
    A self match is not a shuffle; it silently turns R3 into R2.
    """

    phases = ['a'] * 6 + ['b'] * 6
    donors, valid = PhaseShuffler(seed=0).donor_indices(phases)
    for i, donor in enumerate(donors):
        if valid[i]:
            assert donor != i


def test_shuffle_donates_only_within_the_declared_phase():
    """
    A global permutation would make R3 a plausibility test rather than a
    content test, so donors must share the recipient's phase.
    """

    phases = ['a'] * 5 + ['b'] * 5
    donors, valid = PhaseShuffler(seed=0).donor_indices(phases)
    for i, donor in enumerate(donors):
        if valid[i]:
            assert phases[donor] == phases[i]


def test_shuffle_is_reproducible_for_a_fixed_seed():
    """
    Drawn once and fixed: the corrupted mapping is a property of the
    dataset, not per-epoch noise the head can average away.
    """

    phases = ['a'] * 8 + ['b'] * 4
    first, _ = PhaseShuffler(seed=7).donor_indices(phases)
    second, _ = PhaseShuffler(seed=7).donor_indices(phases)
    assert np.array_equal(first, second)


def test_singleton_phase_is_masked_rather_than_left_correct():
    """
    With no donor available the sample must be dropped, not quietly
    handed back its own correct target.
    """

    phases = ['a', 'a', 'lonely']
    donors, valid = PhaseShuffler(seed=0).donor_indices(phases)
    assert not valid[2], 'a phase of one has no donor and must be masked'
    assert valid[0] and valid[1]


def test_coverage_reports_the_unmatched_fraction():
    """
    The manifest needs to record how much of the control degenerated.
    """

    shuffler = PhaseShuffler(seed=0)
    shuffler.donor_indices(['a', 'a', 'x', 'y'])
    stats = shuffler.coverage()
    assert stats['unmatched'] == 2
    assert stats['unmatched_fraction'] == pytest.approx(0.5)


def test_shared_eligibility_drops_a_sample_from_every_arm():
    """
    If R3 masks a sample, R2 and R4 must lose it too, or the arms differ
    in auxiliary sample count and the gap is an artefact.
    """

    r3_valid = np.array([True, False, True, True])
    r2_valid = np.array([True, True, True, True])
    shared = apply_shared_eligibility([r2_valid, r3_valid])
    assert list(shared) == [True, False, True, True]
    assert shared.sum() == r3_valid.sum()


# --- Point 1: a NaN on a masked-out row must not poison the update ---

def test_nan_target_on_a_masked_row_does_not_poison_the_loss(head):
    """
    Multiplying a per-row cosine by a 0/1 mask gives NaN * 0 = NaN. Rows
    without a label must be dropped before the cosine is computed.
    """

    predicted = torch.randn(3, 64, requires_grad=True)
    target = torch.randn(3, 64)
    target[1] = float('nan')          # masked-out row carries garbage
    mask = torch.tensor([1.0, 0.0, 1.0])

    loss = explanation_loss(predicted, target, mask)
    assert torch.isfinite(loss), 'a masked NaN target leaked into the loss'

    loss.backward()
    assert torch.isfinite(predicted.grad).all(), (
        'a masked NaN target leaked into the gradients'
    )


def test_nonfinite_target_marked_valid_is_rejected(head):
    """
    A bad target on a row claimed to be valid is a pipeline bug and must
    surface, not be quietly averaged.
    """

    target = torch.randn(2, 64)
    target[0] = float('inf')
    with pytest.raises(ValueError, match='non-finite'):
        explanation_loss(torch.randn(2, 64), target, torch.ones(2))


# --- Point 2: donors must themselves be valid ---

def test_donor_is_never_drawn_from_a_sample_without_a_target():
    """
    Two same-phase rows, one with no target: the valid recipient must not
    receive the missing one. Intersecting recipient masks afterwards
    cannot fix this, because the donation already happened.
    """

    phases = ['key', 'key']
    valid = [True, False]
    donors, usable = PhaseShuffler(seed=0).donor_indices(
        phases, valid=valid)

    assert not usable[1], 'a sample with no target cannot receive one'
    assert not usable[0], (
        'the only eligible donor was itself, so this must be masked '
        'rather than handed a missing target'
    )


def test_valid_recipients_still_pair_when_donors_exist():
    """
    The complement: eligibility must not mask everything.
    """

    phases = ['key'] * 3
    donors, usable = PhaseShuffler(seed=0).donor_indices(
        phases, valid=[True, True, True])
    assert usable.all()
    assert all(donors[i] != i for i in range(3))


# --- Point 3: distinct indices are not distinct targets ---

def test_identical_explanations_are_reported_as_unchanged():
    """
    Two identical texts can swap: the index moves, the supervision does
    not. A control that mostly does this is R2 with extra steps.
    """

    shuffler = PhaseShuffler(seed=0)
    texts = ['collect the key', 'collect the key']
    shuffler.donor_indices(['key', 'key'], targets=texts)

    stats = shuffler.coverage()
    assert stats['unchanged_target'] == 2
    assert stats['unchanged_fraction'] == pytest.approx(1.0)


def test_distinct_explanations_are_not_counted_as_unchanged():
    shuffler = PhaseShuffler(seed=0)
    texts = ['collect the key', 'open the door']
    shuffler.donor_indices(['key', 'key'], targets=texts)
    assert shuffler.coverage()['unchanged_fraction'] == 0.0


def test_mapping_persists_across_calls_for_stable_ids():
    """
    Re-drawing per epoch would make the corruption noise the head can
    average away. The same ids must yield the same permutation.
    """

    shuffler = PhaseShuffler(seed=0)
    phases = ['key'] * 4
    ids = ['a', 'b', 'c', 'd']

    first, _ = shuffler.donor_indices(phases, sample_ids=ids)
    second, _ = shuffler.donor_indices(phases, sample_ids=ids)
    assert np.array_equal(first, second), (
        'a second call must reuse the persisted mapping'
    )
