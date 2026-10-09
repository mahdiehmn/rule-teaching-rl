"""
Action-conditioned immediate-effect prediction.

These categorical labels are a chosen supervision vocabulary. This module
does not generate, validate or correct LLM labels using a simulator. It
consumes encoded targets and explicit validity masks supplied by a caller.
Neither its logits nor the queried action are an input to the policy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from teachers.consequence_targets import EFFECTS, FIELDS, component_mask


ACTIONS = ('left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done')
WIDTHS = tuple(len(EFFECTS[name]) for name in FIELDS)
INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _check_actions(actions):
    """
    Accept one action or several action branches per transition.
    """
    if actions.ndim not in (1, 2) or actions.dtype not in INTEGER_DTYPES:
        raise ValueError('Actions must be an integer [N] or [N,B] tensor')
    if bool(((actions < 0) | (actions >= len(ACTIONS))).any()):
        raise ValueError('Action indices must be in 0..6')


def action_component_masks(actions):
    """
    Return boolean [...,4] masks in FIELDS order, including done=6.
    """
    _check_actions(actions)
    table = torch.tensor(
        [component_mask(i) for i in range(len(ACTIONS))],
        dtype=torch.bool,
        device=actions.device,
    )
    return table[actions.long()]


class ConsequenceHead(nn.Module):
    """
    Predict 2/3/3/2 classes from hidden[N,H] and actions[N] or [N,B].

    The same transition feature is broadcast over its action branches.
    Branch role or teacher endorsement is never supplied to the head.
    Construction initializes CPU parameters with a private seed and restores
    the CPU RNG stream. Moving the head to a device afterwards draws no RNG.
    """

    def __init__(self, hidden_dim=512, seed=0):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError('hidden_dim must be a positive integer')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Head seed must be an integer in [0, 2**63)')
        self.hidden_dim, self.seed = hidden_dim, seed
        # Seed only CPU; torch.manual_seed would also affect CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + 7, 512, device='cpu'),
                nn.ReLU(),
                nn.Linear(512, sum(WIDTHS), device='cpu'),
            )
            for layer, gain in (
                (self.net[0], math.sqrt(2)),
                (self.net[2], 0.01),
            ):
                nn.init.orthogonal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)

    def forward(self, hidden, actions, detach_features=False):
        """
        Return field logits with leading shape equal to actions.shape.
        """
        _check_actions(actions)
        if hidden.ndim != 2 or hidden.shape != (
            actions.shape[0],
            self.hidden_dim,
        ):
            raise ValueError('Hidden features must have shape [N,hidden_dim]')
        if hidden.device != actions.device:
            raise ValueError('Actions and hidden features must share a device')
        if detach_features:
            hidden = hidden.detach()
        if actions.ndim == 2:
            hidden = hidden[:, None, :].expand(-1, actions.shape[1], -1)
        encoded = F.one_hot(actions.long(), num_classes=7).to(hidden.dtype)
        logits = self.net(torch.cat((hidden, encoded), dim=-1))
        return dict(zip(FIELDS, logits.split(WIDTHS, dim=-1)))

    def metadata(self):
        """
        Describe the vocabulary and normalization without model claims.
        """
        return dict(
            effects={name: list(EFFECTS[name]) for name in FIELDS},
            fields=list(FIELDS),
            actions=list(ACTIONS),
            hidden_dim=self.hidden_dim,
            head_seed=self.seed,
            intermediate_dim=512,
            output_dim=sum(WIDTHS),
            loss='active components -> unique actions -> eligible transitions',
        )


def consequence_loss(predicted, targets, mask, actions):
    """
    Masked CE with equal weight per eligible transition.

    targets and boolean mask have shape actions.shape + (4,), in FIELDS
    order. Finite integer-valued floating targets are accepted so invalid
    masked entries may contain NaN; active targets must be class indices.
    All targets are detached before use. Action masks are always enforced.

    Average active components per unique action, then eligible unique actions
    per transition, then eligible transitions. Duplicate action branches must
    have identical effective masks and active targets; otherwise fail rather
    than silently select a preferred branch. The head produces identical
    logits for duplicated actions because it has no branch-role input.
    """
    _check_actions(actions)
    shape = (*actions.shape, len(FIELDS))
    if (
        targets.shape != shape
        or mask.shape != shape
        or mask.dtype != torch.bool
    ):
        raise ValueError('Targets/mask must be [...,4] with a boolean mask')
    if set(predicted) != set(FIELDS):
        raise ValueError('Predictions must contain exactly the four fields')
    for name, width in zip(FIELDS, WIDTHS):
        logits = predicted[name]
        if logits.shape != (*actions.shape, width):
            raise ValueError(f'Wrong logit shape for {name}')
        if logits.device != actions.device:
            raise ValueError('Predictions and actions must share a device')
    if targets.device != actions.device or mask.device != actions.device:
        raise ValueError('Targets, masks and actions must share a device')
    targets = targets.detach()
    valid = mask & action_component_masks(actions)
    # Empty slices keep the graph connected without evaluating NaN * 0.
    zero = sum(logits[..., :0].sum() for logits in predicted.values())
    if not bool(valid.any()):
        return zero
    if targets.is_complex():
        raise ValueError('Active targets must be real class indices')
    if actions.ndim == 1:
        actions, targets, valid = (
            actions[:, None],
            targets[:, None, :],
            valid[:, None, :],
        )
        predicted = {name: p[:, None, :] for name, p in predicted.items()}
    n = actions.shape[0]
    rows = torch.arange(n, device=actions.device)
    transition_sums = zero.expand(n).clone()
    action_counts = torch.zeros(n, device=actions.device, dtype=torch.long)
    for action in (2, 3, 4, 5):
        matches = actions == action
        representative = matches.long().argmax(dim=1)
        present = matches.any(dim=1)
        selected_mask = valid[rows, representative] & present[:, None]
        duplicated = matches.sum(dim=1) > 1
        relevant = matches & duplicated[:, None]
        if bool(
            (relevant[..., None] & (valid != selected_mask[:, None, :])).any()
        ):
            raise ValueError('Duplicate action branches have different masks')
        duplicate_targets = targets[rows, representative][:, None, :]
        if bool(
            (
                relevant[..., None] & valid & (targets != duplicate_targets)
            ).any()
        ):
            raise ValueError('Duplicate action branches disagree on targets')
        component_sums = zero.expand(n).clone()
        for field, (name, width) in enumerate(zip(FIELDS, WIDTHS)):
            chosen_rows = rows[selected_mask[:, field]]
            if chosen_rows.numel() == 0:
                continue
            branches = representative[chosen_rows]
            chosen = targets[chosen_rows, branches, field]
            if (
                not bool(torch.isfinite(chosen).all())
                or bool(((chosen < 0) | (chosen >= width)).any())
                or (
                    chosen.is_floating_point()
                    and not torch.equal(chosen, chosen.trunc())
                )
            ):
                raise ValueError(f'Invalid active class target for {name}')
            logits = predicted[name][chosen_rows, branches]
            if not bool(torch.isfinite(logits).all()):
                raise ValueError(f'Nonfinite active logits for {name}')
            losses = F.cross_entropy(logits, chosen.long(), reduction='none')
            component_sums = component_sums.index_add(0, chosen_rows, losses)
        component_counts = selected_mask.sum(dim=1)
        transition_sums = transition_sums + (
            component_sums / component_counts.clamp_min(1)
        )
        action_counts = action_counts + (component_counts > 0)
    usable = action_counts > 0
    return (transition_sums[usable] / action_counts[usable]).mean()
