"""
The explanation auxiliary head and its two controls (R2, R3, R4).

R2 trains a small head to predict the teacher's explanation embedding
from the student's post-GRU state, so the auxiliary gradient shapes the
features the actor also reads. R3 and R4 are the controls that make an
R2-over-R1 gap interpretable:

  R2  correct explanation embedding, gradients reach shared features
  R3  SHUFFLED explanation embedding, gradients reach shared features
  R4  correct embedding, but the head reads `stop_gradient(h)`

**What these controls can and cannot conclude.** A gap is informative in
one direction only. If R2 beats R3, something about the target's
alignment mattered. If R2 and R3 come out similar, that does **not**
prove the explanation content was irrelevant: the permutation may have
left the target nearly as predictable, the shuffle may have changed
little for repeated texts, the head may have fit both equally, or the
experiment may lack the power to separate them. Matching head
architecture and gradient scale is necessary but does not make a
shuffled target equally learnable, so held-out target prediction, loss
curves and gradient norms must be compared before any content claim.
Read `unchanged_fraction` from the shuffler before reading the arms at
all. The same asymmetry applies to R4: a null says the auxiliary
gradient into shared features did not measurably help *here*, not that
representation shaping is useless. Keep the behavioural comparison with
R1 primary and the content claim narrow.

The head is discarded at inference and never feeds the actor. Only the
protocol's declared objective is implemented here:

    L_total       = L_PPO + beta(t) * L_action + lambda_aux * L_expl
    L_explanation = mean_valid(1 - cosine(g(h_t), stop_gradient(z_t)))

Nothing in this module trains a policy or calls a teacher. It is the
mechanism plus its bookkeeping, so both can be tested without a run.
"""

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal init, matching the convention used across algos/.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ExplanationHead(nn.Module):
    """
    Predict a frozen text embedding from the student's recurrent state.

    The protocol fixes this as an MLP of 512 -> 512 -> embedding
    dimension on the post-GRU state. It is auxiliary only: no output of
    this module ever reaches action selection, and the whole module is
    dropped at evaluation.
    """

    def __init__(self, hidden_dim, embed_dim):
        """
        Build the head for a given trunk width and embedding size.

        `embed_dim` must be the dimension of the actual embedding model
        in use; the protocol requires recording it in the manifest
        because embeddings from a different model are a different
        dataset, not a drop-in substitute.
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, embed_dim), std=0.01),
        )

    def forward(self, hidden, detach_features=False):
        """
        Map trunk features to a predicted embedding.

        `detach_features` is the R4 control. Detaching here rather than
        building a separate module means R2 and R4 share one code path
        and one parameter count, so the only difference between the arms
        is whether the auxiliary gradient reaches the shared trunk.
        """

        if detach_features:
            hidden = hidden.detach()
        return self.net(hidden)


def explanation_loss(predicted, target, mask):
    """
    Masked mean of (1 - cosine similarity) against a frozen target.

    Valid rows are selected BEFORE the cosine is computed. Multiplying a
    per-row cosine by a 0/1 mask afterwards does not work: a NaN target
    on a masked-out row produces NaN * 0 = NaN, which poisons the loss
    and every gradient in the update. Rows without a label are dropped
    from the computation entirely instead.

    A target that is marked valid but is not finite is a bug in the
    caller's pipeline, not something to average over, so it raises
    rather than silently contributing.

    Returns a finite zero (not NaN) when no row is valid, so a rollout
    with no queried transition does not poison the update.
    """

    valid = mask.bool() if mask.dtype != torch.bool else mask
    if not bool(valid.any()):
        # Keep the graph connected so the caller can always call
        # backward() without branching on the label count.
        return predicted.sum() * 0.0

    chosen = predicted[valid]
    frozen = target[valid].detach()
    if not bool(torch.isfinite(frozen).all()):
        raise ValueError(
            'explanation_loss received a non-finite target on a row '
            'marked valid; fix the embedding pipeline rather than '
            'masking the symptom'
        )

    cos = nn.functional.cosine_similarity(chosen, frozen, dim=-1)
    return (1.0 - cos).mean()


def subgoal_loss(predicted, target, mask):
    """Classify the LLM's subgoal field on the same eligible samples."""

    valid = mask.bool()
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    chosen = target[valid].detach()
    if not bool(torch.isfinite(chosen).all()):
        raise ValueError('Nonfinite structured target')
    return nn.functional.cross_entropy(predicted[valid], chosen.argmax(-1))


class PhaseShuffler:
    """
    Build the R3 donor permutation: within-phase, reproducible, fixed.

    The protocol is specific about this, and each rule closes a way the
    control could be silently wrong:

    - **Within a declared coarse phase**, never a global shuffle. A
      global permutation is trivially detectable and would make R3 a
      test of "is this text plausible here" rather than of content.
    - **Training-only donors.** Held-out data must not leak in.
    - **No self matches**, or the "shuffled" sample is the correct one.
    - **Drawn once, not re-shuffled each epoch**, so the corrupted
      mapping is a fixed property of the dataset rather than noise the
      head averages away.
    - **Mask, do not fabricate**, when a phase has no valid donor -- and
      the caller must apply the same eligibility to R2 and R4 so the
      arms see the same number of auxiliary samples.

    Only the target moves. The action label, query mask and PPO sample
    are untouched by construction: this class returns an index mapping
    and never sees them.
    """

    def __init__(self, seed=0):
        """
        Fix the donor draw to a seed so the permutation is reproducible.
        """

        self.rng = np.random.default_rng(seed)
        self.unmatched = 0
        self.matched = 0
        self.unchanged_target = 0
        # Mapping persisted by stable sample id, so asking twice returns
        # the SAME permutation. Re-drawing per epoch would turn the
        # corrupted mapping into noise the head can average away, which
        # is exactly what the protocol forbids.
        self._mapping = {}

    def donor_indices(self, phases, valid=None, sample_ids=None,
                      targets=None):
        """
        Map each sample to a donor drawn from its own phase.

        Parameters
        ----------
        phases: sequence
            Declared coarse task stage per sample. The phase *rule*
            belongs in the manifest, not here; this class only groups by
            whatever label it is handed.
        valid: sequence of bool or None
            Which samples actually carry a usable target. **Donors are
            drawn only from these.** Intersecting recipient masks after
            the fact is not enough: with two same-phase rows where one
            has no target, the remaining valid recipient would otherwise
            be handed the missing one.
        sample_ids: sequence or None
            Stable identifiers. When given, the mapping is remembered
            per id and reused, so a second call returns the same
            permutation instead of drawing a fresh one.
        targets: sequence or None
            The explanation texts or embeddings. Used only to report how
            often a "shuffle" left the supervision unchanged, because
            two identical explanations can swap and change nothing --
            distinct donor indices do not imply distinct targets.

        Returns
        -------
        (donors, usable): donors[i] is the index whose target sample i
        receives; usable[i] is False where no donor existed and the
        auxiliary sample must be masked out.
        """

        phases = list(phases)
        n = len(phases)
        if sample_ids is not None:
            ids = list(sample_ids)
            if len(set(ids)) != len(ids):
                # Duplicated ids silently collapse rows in the persisted
                # mapping, which produced donors that pointed at
                # themselves. Fail loudly: the caller must supply one id
                # per transition, not a content hash.
                raise ValueError(
                    'PhaseShuffler received duplicate sample_ids; ids '
                    'must be unique per transition (a text hash is not '
                    'a transition id)'
                )
        eligible = (
            np.ones(n, dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool)
        )

        buckets = {}
        for i, phase in enumerate(phases):
            # Only samples with a real target may DONATE.
            if eligible[i]:
                buckets.setdefault(phase, []).append(i)

        donors = np.arange(n)
        usable = np.zeros(n, dtype=bool)

        for members in buckets.values():
            if len(members) < 2:
                # No donor but itself, and a self match is not a
                # shuffle. Leave `usable` False; the sweep below counts
                # it once. Incrementing here as well would double-count.
                continue

            order = self._persisted_order(members, sample_ids)
            for offset, i in enumerate(order):
                donors[i] = order[(offset + 1) % len(order)]
                usable[i] = True
                self.matched += 1

        # A sample with no target of its own can never be a recipient.
        usable &= eligible
        self.unmatched += int((eligible & ~usable).sum())

        if targets is not None:
            self._count_unchanged(donors, usable, targets)
        return donors, usable

    def _persisted_order(self, members, sample_ids):
        """
        Return this phase's donor order, drawing it only once.

        Keyed by the members' stable ids so a later call reproduces the
        same rotation rather than re-randomizing the control.
        """

        if sample_ids is None:
            order = list(members)
            self.rng.shuffle(order)
            return order

        key = tuple(sorted(sample_ids[i] for i in members))
        if key not in self._mapping:
            order = list(members)
            self.rng.shuffle(order)
            self._mapping[key] = [sample_ids[i] for i in order]

        by_id = {sample_ids[i]: i for i in members}
        return [by_id[sid] for sid in self._mapping[key]]

    def _count_unchanged(self, donors, usable, targets):
        """
        Count shuffles that changed the index but not the supervision.

        Two identical explanations can swap: the donor index differs and
        the target does not, so the "corrupted" sample is really a
        correct one. Reporting this is the only way to know whether the
        intended relationship actually changed.
        """

        for i, donor in enumerate(donors):
            if not usable[i]:
                continue
            mine, theirs = targets[i], targets[donor]
            same = (
                bool(np.array_equal(np.asarray(mine), np.asarray(theirs)))
                if not isinstance(mine, str) else mine == theirs
            )
            if same:
                self.unchanged_target += 1

    def coverage(self):
        """
        Report what the permutation actually did, for the manifest.

        `unchanged_fraction` is the share of donations that left the
        supervision identical. A control with a high value here is not a
        control: it is R2 wearing a different index.
        """

        total = self.matched + self.unmatched
        return {
            'matched': self.matched,
            'unmatched': self.unmatched,
            'unmatched_fraction': (
                self.unmatched / total if total else 0.0
            ),
            'unchanged_target': self.unchanged_target,
            'unchanged_fraction': (
                self.unchanged_target / self.matched
                if self.matched else 0.0
            ),
        }


def apply_shared_eligibility(valid_masks):
    """
    Intersect the auxiliary masks so every arm sees the same samples.

    When R3 masks a sample for want of a donor, R2 and R4 must drop it
    too. Otherwise the arms differ in how much auxiliary supervision
    they received, and an R2-over-R3 gap could be a sample-count
    artefact rather than a content effect.
    """

    shared = None
    for mask in valid_masks:
        arr = np.asarray(mask, dtype=bool)
        shared = arr if shared is None else (shared & arr)
    return shared
