"""Train auxiliary memory and knowledge targets from observation history."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tyro

from algos import ppo_distill as ppo
from algos.history_targets import HistoryTargets, IGNORE_INDEX


MODES = ('ppo', 'detached', 'memory', 'knowledge', 'combined',
         'permuted', 'image')
FIELDS = {'memory': 6, 'knowledge': 18}
HASH_FORMAT = 'sorted UTF-8 name, NUL, contiguous tensor bytes'


@dataclass
class Args(ppo.Args):
    task: str = 'doorkey_8x8'
    obs_mode: str = 'symbolic'
    agent_view_size: int = 7
    recurrent: bool = True
    dual_value: bool = True
    bonus: str = 'count'
    cuda: bool = False
    gamma: float = 0.999
    guidance: bool = False
    teacher: str = 'oracle'
    teacher_model: str = ''
    query_budget: int = 0
    advice_budget: int = 0
    budget_ledger: str = ''
    distill_coef_start: float = 0.0
    distill_coef_min: float = 0.0
    record_initial_policy: bool = True
    eval_episodes: int = 50
    eval_sampled: bool = False
    memory_mode: str = 'combined'
    # Zero means every vector step in the finite training horizon.
    memory_budget: int = 0
    memory_coef: float = 0.1
    knowledge_coef: float = 0.1
    memory_gradient_interval: int = 50
    memory_audit_vector_steps: int = 256


def policy_hash(agent):
    """
    Match the native trainer's name/NUL/tensor initialization digest.
    """

    state = agent.state_dict() if hasattr(agent, 'state_dict') else agent
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode('utf-8') + b'\0')
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    """
    Keep strict, portable JSON for contracts and final counts.
    """

    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def append_json(path, value):
    """
    Record a complete journal row without retaining open file handles.
    """

    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def opportunity_schedule(args):
    """
    Select complete vector steps independently of policy and donor RNG.
    """

    total = args.num_iterations * args.num_steps
    if (type(args.memory_budget) is not int
            or not 0 <= args.memory_budget <= total):
        raise ValueError('Memory budget counts vector steps within the run')
    selected = None
    if args.memory_budget:
        selected = frozenset(map(int, np.random.default_rng(
            args.seed + 731_009).choice(
                total, args.memory_budget, replace=False)))
    descriptor = dict(
        total_vector_steps=total,
        selected_vector_steps=(total if selected is None else len(selected)),
        selection='all' if selected is None else sorted(selected),
        maximum_transition_opportunities=(
            total if selected is None else len(selected)) * args.num_envs,
        seed=args.seed + 731_009,
    )
    return selected, descriptor


def class_counts(target):
    """
    Count each class per field; ignored fields contribute zero.
    """

    return np.stack([(target == c).sum(axis=0) for c in range(3)], axis=-1)


def paired_targets(targets, active, rng):
    """
    Swap reciprocal cross-environment pairs under symmetric field masks.
    """

    active = np.asarray(active, dtype=bool)
    n = len(active)
    eligible = rng.permutation(np.flatnonzero(active))
    eligible = eligible[:len(eligible) // 2 * 2]
    donors = np.full(n, -1, dtype=np.int64)
    donors[eligible[::2]] = eligible[1::2]
    donors[eligible[1::2]] = eligible[::2]
    result = {'donors': donors}
    lookup = np.maximum(donors, 0)
    for name, width in FIELDS.items():
        target = np.asarray(targets[name])
        mask = np.asarray(targets[name + '_mask'], dtype=bool)
        if target.shape != (n, width) or mask.shape != target.shape:
            raise ValueError('Unexpected history-target shape')
        if np.any(mask & ((target < 0) | (target > 2))):
            raise ValueError('Active history labels must be class indices')
        shared = mask & mask[lookup] & (donors >= 0)[:, None]
        correct = np.where(shared, target, IGNORE_INDEX)
        permuted = np.where(shared, target[lookup], IGNORE_INDEX)
        # Both members of each pair are selected. Symmetric intersection
        # preserves actual class counts, including after sparse masking.
        if not np.array_equal(class_counts(correct), class_counts(permuted)):
            raise ValueError('Pairing changed the effective class marginals')
        result[name] = correct
        result[name + '_permuted'] = permuted
        result[name + '_mask'] = shared
    return result


def batch_normalized_ce(logits, target):
    """
    Divide summed CE by every minibatch transition and fixed field count.
    """

    if (logits.ndim != 3 or logits.shape[-1] != 3
            or target.shape != logits.shape[:2] or target.dtype != torch.long):
        raise ValueError('Expected logits[N,F,3] and integer targets[N,F]')
    if not target.numel():
        raise ValueError('A PPO minibatch must contain transitions')
    return F.cross_entropy(
        logits.transpose(1, 2), target, ignore_index=IGNORE_INDEX,
        reduction='sum') / target.numel()


def parameter_gradients(memory_loss, knowledge_loss, agent):
    """
    Measure weighted auxiliary gradients on actual encoder and GRU weights.
    """

    encoder = list(agent.encoder.parameters())
    core = list(agent.core.parameters())
    parameters = encoder + core
    gradients = [torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True)
        for loss in (memory_loss, knowledge_loss)]
    output = {}
    for label, start, end in (('encoder', 0, len(encoder)),
                              ('core', len(encoder), len(parameters))):
        for name, values in zip(FIELDS, gradients):
            output[name + '_' + label + '_grad_l2'] = math.sqrt(sum(
                float(g.detach().double().square().sum())
                for g in values[start:end] if g is not None))
        total = []
        for left, right in zip(gradients[0][start:end],
                               gradients[1][start:end]):
            if left is not None or right is not None:
                total.append(right if left is None else left if right is None
                             else left + right)
        output['combined_' + label + '_grad_l2'] = math.sqrt(sum(
            float(g.detach().double().square().sum()) for g in total))
    return output


class MemoryAuxiliary:
    """
    Keep history collection separate from minibatch recurrent supervision.
    """

    def initialize(self, agent, run_dir, device, args):
        if (args.memory_mode not in MODES or args.guidance or args.cuda
                or args.explanation != 'none' or args.consequence != 'none'
                or args.advice_replay or args.query_budget
                or args.advice_budget
                or args.budget_ledger or not args.recurrent
                or args.obs_mode != 'symbolic' or args.agent_view_size != 7
                or agent.core is None or args.num_envs < 2
                or args.distill_coef_start or args.distill_coef_min
                or args.imitation_weighting != 'none'):
            raise ValueError('Unsupported observation-history configuration')
        if (not all(math.isfinite(v) and v > 0 for v in
                    (args.memory_coef, args.knowledge_coef))
                or args.memory_gradient_interval < 1
                or args.memory_audit_vector_steps < 0):
            raise ValueError('Invalid memory coefficients or audit intervals')
        self.args, self.agent, self.device = args, agent, device
        self.directory = Path(run_dir) / 'memory'
        self.directory.mkdir()
        self.tracker = HistoryTargets(args.num_envs)
        self.selected, schedule = opportunity_schedule(args)
        self.donor_rng = np.random.default_rng(args.seed + 811_073)
        audit_pool = (schedule['total_vector_steps'] if self.selected is None
                      else np.array(sorted(self.selected)))
        self.audit_steps = frozenset(map(int, np.random.default_rng(
            args.seed + 947_021).choice(
                audit_pool, min(args.memory_audit_vector_steps,
                                schedule['selected_vector_steps']),
                replace=False)))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(args.seed + 617_003)
            self.heads = nn.ModuleDict({
                name: nn.Linear(512, width * 3)
                for name, width in FIELDS.items()
            }).to(device)
        self.optimizer = torch.optim.Adam(
            self.heads.parameters(), lr=args.learning_rate, eps=1e-5)
        self.initial = policy_hash(agent)
        self.counts = Counter()
        self.episodes = np.zeros(args.num_envs, dtype=np.int64)
        self.memory_episodes = set()
        self.knowledge_episodes = set()
        self.confusion = {name: np.zeros((3, 3), dtype=np.int64)
                          for name in FIELDS}
        self.evidence_digest = hashlib.sha256()
        self.iteration = 0
        self.collection_steps = 0
        self.batch = None
        self.active = False
        self.update_index = 0
        self.gradient_logged = False
        self.targets = {name: torch.full(
            (args.num_steps, args.num_envs, width), IGNORE_INDEX,
            dtype=torch.long, device=device) for name, width in FIELDS.items()}
        self.totals = {kind: {name: np.zeros((width, 3), dtype=np.int64)
                             for name, width in FIELDS.items()}
                       for kind in ('raw', 'paired', 'permuted')}
        write_json(self.directory / 'contract.json', dict(
            args=asdict(args), initial_policy_sha256=self.initial,
            policy_hash_format=HASH_FORMAT, schedule=schedule,
            schedule_sha256=hashlib.sha256(json.dumps(
                schedule, sort_keys=True).encode()).hexdigest(),
            target_source='student observation history; no map or LLM',
            normalization='CE sum / minibatch transitions / fixed fields',
            fields=FIELDS, classes_per_field=3,
            donor='reciprocal active-environment pairs at one vector step',
            mask='symmetric recipient/donor field intersection in every mode',
            comparison_scope=('Masks and marginals match counterfactual '
                              'consumers of a recorded rollout; live policies '
                              'can visit different histories.'),
            audit_vector_steps=sorted(self.audit_steps),
            no_evaluation_head_input=True,
        ))

    def _flush_updates(self):
        if self.iteration:
            append_json(self.directory / 'rollouts.jsonl', dict(
                rollout=self.iteration - 1, **dict(self.rollout_counts),
                update_minibatches=self.update_index,
                **dict(self.training_counts),
                training_confusion={name: value.tolist() for name, value in
                                    self.rollout_confusion.items()}))

    def start_collection(self, iteration):
        """
        Clear rollout buffers while preserving observation history.
        """

        if self.iteration and self.collection_steps != self.args.num_steps:
            raise ValueError('Previous rollout missed observation steps')
        self._flush_updates()
        self.iteration = iteration
        self.collection_steps = 0
        self.rollout_counts = Counter()
        self.training_counts = Counter()
        self.rollout_confusion = {name: np.zeros((3, 3), dtype=np.int64)
                                  for name in FIELDS}
        self.update_index = 0
        self.gradient_logged = False
        for value in self.targets.values():
            value.fill_(IGNORE_INDEX)

    def observe(self, step, global_step, next_obs, episode_start, next_done):
        """
        Update history on every frame; select complete paired vector steps.
        """

        expected = ((self.iteration - 1) * self.args.num_steps + step + 1)
        if (step != self.collection_steps
                or global_step != expected * self.args.num_envs):
            raise ValueError('History hook and transition clock disagree')
        images = next_obs.detach().cpu().numpy()
        starts = episode_start.detach().cpu().numpy()
        inactive = next_done.detach().cpu().numpy()
        target = self.tracker.update(images, starts, inactive=inactive)
        self.episodes += np.asarray(starts, dtype=bool)
        self.collection_steps += 1
        self.counts['observed_vector_steps'] += 1
        vector_step = expected - 1
        if self.selected is not None and vector_step not in self.selected:
            return
        paired = paired_targets(target, ~np.asarray(inactive, dtype=bool),
                                self.donor_rng)
        donors = paired['donors']
        local = Counter(
            scheduled_vector_steps=1,
            transition_opportunities=self.args.num_envs,
            inactive_transitions=int(np.asarray(inactive, dtype=bool).sum()),
            paired_transitions=int((donors >= 0).sum()),
            unpaired_active_transitions=int(
                ((donors < 0) & ~np.asarray(inactive, dtype=bool)).sum()),
        )
        digest = self.evidence_digest
        digest.update(np.asarray([vector_step], dtype='<i8').tobytes())
        digest.update(donors.astype('<i8').tobytes())
        for name in FIELDS:
            raw = np.where(target[name + '_mask'], target[name], IGNORE_INDEX)
            correct, permuted = paired[name], paired[name + '_permuted']
            used = permuted if self.args.memory_mode == 'permuted' else correct
            self.targets[name][step] = torch.as_tensor(
                used, device=self.device)
            local['raw_' + name + '_fields'] = int((raw >= 0).sum())
            local['paired_' + name + '_fields'] = int((correct >= 0).sum())
            local['discarded_' + name + '_fields'] = int(
                (raw >= 0).sum() - (correct >= 0).sum())
            local['changed_' + name + '_fields'] = int(
                ((correct != permuted) & (correct >= 0)).sum())
            if name == 'knowledge':
                # Always-unseen padding must not dilute the diagnostic
                # denominator for whether the shuffle changed meaning.
                local['informative_knowledge_fields'] = int(
                    ((correct >= 0)
                     & ((correct != 0) | (permuted != 0))).sum())
            for kind, value in (('raw', raw), ('paired', correct),
                                ('permuted', permuted)):
                self.totals[kind][name] += class_counts(value)
                digest.update(value.astype('<i8').tobytes())
        self.counts.update(local)
        self.rollout_counts.update(local)
        for index in range(self.args.num_envs):
            identity = (index, int(self.episodes[index]))
            if (paired['memory'][index] >= 0).any():
                self.memory_episodes.add(identity)
            if (paired['knowledge'][index] == 1).any():
                self.knowledge_episodes.add(identity)
        self.counts['memory_eligible_episodes'] = len(self.memory_episodes)
        self.counts['knowledge_remembered_episodes'] = len(
            self.knowledge_episodes)
        if vector_step in self.audit_steps:
            append_json(self.directory / 'target_audit.jsonl', dict(
                vector_step=vector_step, episode_ids=[
                    [i, int(e)] for i, e in enumerate(self.episodes)],
                inactive=np.asarray(inactive, dtype=bool).tolist(),
                observations=images.tolist(), donors=donors.tolist(),
                memory_raw=target['memory'].tolist(),
                knowledge_raw=target['knowledge'].tolist(),
                memory_correct=paired['memory'].tolist(),
                memory_permuted=paired['memory_permuted'].tolist(),
                knowledge_correct=paired['knowledge'].tolist(),
                knowledge_permuted=paired['knowledge_permuted'].tolist(),
            ))

    def begin_rollout(self, iteration):
        """
        Prepare optimization only; the collected labels must remain intact.
        """

        if (iteration != self.iteration
                or self.collection_steps != self.args.num_steps):
            raise ValueError('Optimization began before complete collection')
        if self.args.anneal_lr:
            self.optimizer.param_groups[0]['lr'] = self.args.learning_rate * (
                1 - (iteration - 1) / self.args.num_iterations)

    def set_batch(self, mb_inds, newhidden, images):
        """
        Bind targets to the exact sequence-replayed PPO feature order.
        """

        self.batch = (mb_inds, newhidden, images)

    def loss(self):
        """
        Shape the recurrent features without supplying labels to the actor.
        """

        self.active = False
        if self.batch is None:
            raise ValueError('PPO did not bind a minibatch before memory loss')
        indices, hidden, images = self.batch
        self.batch = None
        self.update_index += 1
        if self.args.memory_mode == 'ppo':
            return torch.zeros((), device=self.device)
        image_mode = self.args.memory_mode == 'image'
        features = self.agent._encode(images) if image_mode else hidden
        shared = {
            'memory': self.args.memory_mode in
            ('memory', 'combined', 'permuted', 'image'),
            'knowledge': self.args.memory_mode in
            ('knowledge', 'combined', 'permuted', 'image'),
        }
        losses = {}
        for name, width in FIELDS.items():
            values = features if shared[name] else features.detach()
            logits = self.heads[name](values).reshape(-1, width, 3)
            target = self.targets[name].reshape(-1, width)[indices]
            losses[name] = batch_normalized_ce(logits, target)
            if self.update_index <= self.args.num_minibatches:
                valid = target >= 0
                self.training_counts[name + '_training_fields'] += int(
                    valid.sum())
                self.training_counts[name + '_training_correct'] += int(
                    ((logits.argmax(-1) == target) & valid).sum())
                self.training_counts[name + '_ce_sum'] += float(
                    losses[name].detach()) * target.numel()
                prediction = logits.argmax(-1)
                confusion = torch.bincount(
                    target[valid] * 3 + prediction[valid], minlength=9
                ).reshape(3, 3).detach().cpu().numpy()
                self.rollout_confusion[name] += confusion
                self.confusion[name] += confusion
        memory_loss = self.args.memory_coef * losses['memory']
        knowledge_loss = self.args.knowledge_coef * losses['knowledge']
        self.optimizer.zero_grad()
        self.active = True
        due = (self.iteration == 1 or
               self.iteration % self.args.memory_gradient_interval == 0)
        if due and not self.gradient_logged:
            gradients = parameter_gradients(memory_loss, knowledge_loss,
                                            self.agent)
            append_json(self.directory / 'gradients.jsonl', dict(
                rollout=self.iteration - 1, minibatch=self.update_index - 1,
                mode=self.args.memory_mode, **gradients,
                memory_loss=float(memory_loss.detach()),
                knowledge_loss=float(knowledge_loss.detach()),
                gradient_definition='d(weighted auxiliary loss)/d(parameters)',
            ))
            self.gradient_logged = True
        return memory_loss + knowledge_loss

    def optimizer_step(self):
        """
        Keep head clipping separate from policy and from the other head.
        """

        if self.active:
            for head in self.heads.values():
                nn.utils.clip_grad_norm_(head.parameters(),
                                         self.args.max_grad_norm)
            self.optimizer.step()

    def finish(self, global_step):
        """
        Persist the trained heads, exact counts, and native policy identities.
        """

        self._flush_updates()
        schedule = opportunity_schedule(self.args)[1]
        if (self.counts['observed_vector_steps']
                != schedule['total_vector_steps']
                or self.counts['scheduled_vector_steps']
                != schedule['selected_vector_steps']):
            raise ValueError('History collection missed its frozen horizon')
        write_json(self.directory / 'finished.json', dict(
            global_step=global_step, args=asdict(self.args),
            initial_policy_sha256=self.initial,
            final_policy_sha256=policy_hash(self.agent),
            policy_hash_format=HASH_FORMAT, counts=dict(self.counts),
            class_counts={kind: {name: value.tolist()
                                 for name, value in group.items()}
                          for kind, group in self.totals.items()},
            eligible_episodes=dict(
                memory=[list(pair) for pair in sorted(self.memory_episodes)],
                knowledge_remembered=[list(pair) for pair in
                                      sorted(self.knowledge_episodes)]),
            training_confusion={name: value.tolist()
                                for name, value in self.confusion.items()},
            target_stream_sha256=self.evidence_digest.hexdigest(),
            target_hash_schema=('For each selected vector step: int64le step, '
                                'donor indices, then memory/knowledge '
                                'raw/correct/permuted int64le C-order arrays'),
        ))
        torch.save(self.heads.state_dict(), self.directory / 'heads.pt')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=MemoryAuxiliary())
