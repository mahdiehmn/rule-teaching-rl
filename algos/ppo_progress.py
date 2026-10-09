"""Finite progress-label PPO pilot."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from algos import ppo_babyai as ppo


def runtime_identity():
    from minigrid.envs.babyai.core import verifier
    return dict(packages={name: version(name) for name in
                          ('minigrid', 'gymnasium', 'numpy', 'torch')},
                verifier_sha256=hashlib.sha256(
                    Path(inspect.getfile(verifier)).read_bytes()).hexdigest())


def progress_labels(instruction):
    """Read latched verifier state; do not invoke its mutating verify method."""
    labels = []

    def walk(node, eligible=True, completed=False):
        kind = type(node).__name__
        if kind == 'GoToInstr':
            labels.append(2 if completed else int(eligible))
            return 'goto'
        if kind not in ('BeforeInstr', 'AfterInstr', 'AndInstr'):
            raise ValueError(f'Unsupported progress instruction: {kind}')
        a_done = completed or node.a_done == 'success'
        b_done = completed or node.b_done == 'success'
        a_eligible = eligible and (kind != 'AfterInstr' or b_done)
        b_eligible = eligible and (kind != 'BeforeInstr' or a_done)
        a = walk(node.instr_a, a_eligible, a_done)
        b = walk(node.instr_b, b_eligible, b_done)
        return (kind, a, b)

    tree = walk(instruction)
    if not 2 <= len(labels) <= 4:
        raise ValueError('This pilot requires two to four GoTo clauses')
    return tuple(labels), json.dumps(tree, separators=(',', ':'))


def policy_hash(agent):
    """Stable tensor digest independent of checkpoint container metadata."""
    value = hashlib.sha256()
    state = agent.state_dict() if hasattr(agent, 'state_dict') else agent
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def annotation_schedule(args):
    cutoff = int(args.num_iterations * args.batch_size * args.progress_cutoff)
    if args.progress_budget > cutoff:
        raise ValueError('More annotation opportunities than transitions')
    candidates = np.random.default_rng(args.seed + 91_003).choice(
        cutoff, size=max(args.progress_budget, min(10_000, cutoff)), replace=False)
    return cutoff, set(map(int, candidates[:args.progress_budget]))


def append(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')


@dataclass
class Args(ppo.Args):
    experiment_id: str = ''
    task: str = 'gotoseq_s5r2_sequence'
    obs_mode: str = 'symbolic'
    cuda: bool = False
    gamma: float = 0.999
    eval_episodes: int = 50
    eval_sampled: bool = False
    progress_mode: str = 'correct'
    progress_budget: int = 10_000
    progress_coef: float = 0.1
    progress_cutoff: float = 0.75
    progress_buffer: int = 1024


class ProgressAuxiliary:
    """Training-only head with isolated initialization and donor randomness."""

    def initialize(self, agent, envs, run_dir, device, args):
        self.args, self.device = args, device
        self.run_dir = Path(run_dir)
        self.agent = agent
        agent.capture_auxiliary_features = True
        # Adding a detached head must not alter the policy's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 19_331)
            self.head = torch.nn.Linear(ppo.CORE_HIDDEN, 12).to(device)
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=args.learning_rate, eps=1e-5
        )
        # The low dose is a nested subset of the high-dose opportunities.
        cutoff, self.slots = annotation_schedule(args)
        self.donor_rng = np.random.default_rng(args.seed + 17_023)
        self.buffers = defaultdict(lambda: deque(maxlen=args.progress_buffer))
        self.episodes = np.zeros(args.num_envs, dtype=np.int64)
        self.counts = Counter()
        self.label_counts = Counter()
        self.initial_hash = policy_hash(agent)
        self.last_loss, self.last_accuracy = None, None
        self.last_gradient = None
        self.active = False
        (self.run_dir / 'progress_contract.json').write_text(json.dumps(dict(
            args=vars(args), initial_policy_sha256=self.initial_hash,
            runtime=runtime_identity(),
            opportunities=len(self.slots), cutoff_transition=cutoff,
            schedule_sha256=hashlib.sha256(
                json.dumps(sorted(self.slots)).encode()).hexdigest(),
            target='surface-AST GoTo leaves: inactive=0, eligible=1, completed=2',
            teacher='native verifier, diagnostic reference; no LLM',
            policy_input='local symbolic image and full mission; no labels',
        ), indent=2), encoding='utf-8')

    def begin_rollout(self, iteration):
        self.iteration = iteration
        self.last_gradient = None
        self.targets = torch.full(
            (self.args.batch_size, 4), -100, dtype=torch.long,
            device=self.device
        )
        self.gradient_recorded = False

    def observe(self, step, global_step, envs, next_done):
        for index, env in enumerate(envs.envs):
            terminal = bool(next_done[index].item())
            if not terminal and env.unwrapped.step_count == 0:
                self.episodes[index] += 1
            slot = global_step + index
            if slot not in self.slots:
                continue
            self.counts['scheduled'] += 1
            # NEXT_STEP autoreset does not execute an action on terminal frames.
            if terminal:
                self.counts['autoreset_skipped'] += 1
                continue
            target, tree = progress_labels(env.unwrapped.instrs)
            episode = (index, int(self.episodes[index]))
            self.counts['annotations'] += 1
            self.label_counts.update(map(str, target))
            self.counts['annotations_with_completed_leaf'] += int(2 in target)
            pool = self.buffers[tree]
            donors = [row for row in pool if row['episode'] != episode]
            chosen = None
            if donors:
                chosen = donors[int(self.donor_rng.integers(len(donors)))]
                self.counts['donor_eligible'] += 1
                self.counts['different_donor_target'] += int(
                    chosen['target'] != target
                )
            else:
                self.counts['donor_warmup_withheld'] += 1
            supplied = (chosen['target'] if self.args.progress_mode == 'shuffled'
                        and chosen is not None else target)
            # Both active/reference controls use the same donor-eligibility rule.
            if chosen is not None:
                flat = step * self.args.num_envs + index
                self.targets[flat, :len(target)] = torch.tensor(
                    supplied, device=self.device
                )
            append(self.run_dir / 'progress_annotations.jsonl', dict(
                transition=slot, env=index, episode=episode, tree=tree,
                target=target, supplied=supplied,
                used=chosen is not None,
                donor_transition=chosen['slot'] if chosen is not None else None,
                donor_episode=chosen['episode'] if chosen is not None else None,
            ))
            pool.append(dict(episode=episode, target=target, slot=slot))

    def loss(self, features, indices, ppo_loss):
        self.active = False
        self.optimizer.zero_grad(set_to_none=True)
        labels = self.targets[indices]
        selected = (labels != -100).any(dim=1)
        if self.args.progress_mode == 'none' or not selected.any():
            return features.new_zeros(())
        inputs = features.detach() if self.args.progress_mode == 'detached' else features
        logits = self.head(inputs[selected]).reshape(-1, 4, 3)
        labels = labels[selected]
        valid = labels != -100
        # Average per annotated example, then across annotated examples.
        terms = F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1),
            ignore_index=-100, reduction='none'
        ).reshape(-1, 4)
        auxiliary = ((terms * valid).sum(1) / valid.sum(1)).mean()
        loss = self.args.progress_coef * auxiliary
        self.last_loss = float(auxiliary.detach())
        self.last_accuracy = float((logits.argmax(-1)[valid] == labels[valid]).float().mean())
        self.active = True
        self.counts['auxiliary_minibatch_updates'] += 1
        if not self.gradient_recorded:
            auxiliary_gradient = torch.autograd.grad(
                loss, features, retain_graph=True, allow_unused=True
            )[0]
            ppo_gradient = torch.autograd.grad(
                ppo_loss, features, retain_graph=True, allow_unused=True
            )[0]
            norm_aux = 0.0 if auxiliary_gradient is None else float(auxiliary_gradient.norm())
            norm_ppo = 0.0 if ppo_gradient is None else float(ppo_gradient.norm())
            cosine = None
            if norm_aux > 0 and norm_ppo > 0:
                cosine = float(F.cosine_similarity(
                    auxiliary_gradient.flatten(), ppo_gradient.flatten(), dim=0
                ))
            self.last_gradient = dict(auxiliary_norm=norm_aux,
                                      ppo_norm=norm_ppo, cosine=cosine)
            self.gradient_recorded = True
        return loss

    def optimizer_step(self):
        if self.active:
            # Detached head gradients never participate in policy clipping.
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

    def record_iteration(self, global_step, evaluation):
        append(self.run_dir / 'progress_updates.jsonl', dict(
            global_step=global_step, counts=dict(self.counts),
            leaf_state_counts=dict(self.label_counts),
            last_auxiliary_loss=self.last_loss,
            last_auxiliary_accuracy=self.last_accuracy,
            gradient_at_recurrent_features=self.last_gradient,
        ))

    def finish(self, agent, global_step):
        torch.save(self.head.state_dict(), self.run_dir / 'progress_head.pt')
        (self.run_dir / 'progress_finished.json').write_text(json.dumps(dict(
            global_step=global_step, initial_policy_sha256=self.initial_hash,
            final_policy_sha256=policy_hash(agent), counts=dict(self.counts),
        ), indent=2), encoding='utf-8')


def train(args):
    if (args.task != 'gotoseq_s5r2_sequence' or args.obs_mode != 'symbolic'
            or args.cuda or args.progress_mode not in
            ('none', 'detached', 'correct', 'shuffled')
            or not 0 < args.progress_cutoff <= 1
            or args.progress_coef <= 0 or args.progress_budget < 0
            or args.progress_buffer < 2):
        raise ValueError('Invalid frozen progress-pilot configuration')
    ppo.train(args, auxiliary=ProgressAuxiliary())


if __name__ == '__main__':
    train(tyro.cli(Args))
