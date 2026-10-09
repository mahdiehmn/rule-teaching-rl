"""Explanation formats as offline lesson targets, with explicit timing.

A new auxiliary for
the unchanged `algos.ppo_distill` trainer; `algos.ppo_contrastive` and
the frozen isolation study are not touched.

Replay follows the earlier lesson studies: one minibatch of fixed
historical cases per scheduled rollout, encoder-only (the replay bypasses
the GRU; the student acts with its ordinary recurrent policy). Heads read
the encoded image plus the fixed endorsed/foil action one-hots, exactly
like the reason-code heads, so an explanation-only arm is not free of
action information.

Modes:
    ppo          draws and logs the same replay stream, applies no loss
    actions      endorsed cross-entropy + foil unlikelihood (coef 1)
    explanation  the chosen format's loss only (no explicit action loss)
    combined     actions + explanation
    detached     explanation heads train on detached features (smoke
                 control: the policy must match ppo exactly)

Formats: plain, contrastive, subgoal, consequence and plan (the next
subgoal's object, egocentric direction and distance; privileged when out
of view). Every target is predicted from the CURRENT encoded image plus
the action pair; a privileged target is only partly predictable from it
by design. `format_access` must match the bank's information access
(full_state primary, local_only opt-in) whenever an explanation loss is
applied.

Targets: `aligned` (each case's own explanation) or `permuted` (the
bank's frozen cross-episode bundle permutation). Loss scale comes from a
frozen calibration file and is never tuned on RL outcomes.

Timing: exactly `lesson_exposures` replay updates at iterations
offset + every, offset + 2*every, ... (offset 0, every 4, 1,830 updates
reproduces the isolation study's schedule exactly). Early and late arms
with equal `every` and exposures span equally long intervals and have
identical coefficient integrals; the schedule and integrals are written
to lesson_contract.json and verified.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from algos import ppo_distill as ppo
from algos.ppo_progress import policy_hash
from scripts.explanation_formats import (
    ACCESS, CONSEQUENCE_FIELDS, FORMATS, PLAN_FIELDS, PRIMARY_ACCESS, ROLES,
    SUBGOAL_FIELDS)
from scripts.explanation_screen_panel import file_hash

MODES = ('ppo', 'actions', 'explanation', 'combined', 'detached')
TARGETS = ('aligned', 'permuted')
ACTIONS = 7
FEATURES = 512


@dataclass
class Args(ppo.Args):
    task: str = 'doorkey_8x8'
    obs_mode: str = 'symbolic'
    recurrent: bool = True
    dual_value: bool = True
    bonus: str = 'count'
    cuda: bool = False
    gamma: float = 0.999
    guidance: bool = False
    teacher: str = 'oracle'
    teacher_model: str = ''
    query_budget: int = 0
    budget_ledger: str = ''
    record_initial_policy: bool = True
    eval_episodes: int = 50
    eval_sampled: bool = False
    format_bank: str = ''
    format_bank_sha256: str = ''
    # Information access of the explanation writer; must match the bank
    # whenever an explanation loss is applied. full_state is primary.
    format_access: str = PRIMARY_ACCESS
    lesson_format: str = 'plain'
    lesson_mode: str = 'explanation'
    lesson_targets: str = 'aligned'
    lesson_scale: float = 0.0
    lesson_action_coef: float = 0.0
    lesson_batch: int = 32
    lesson_offset: int = 0
    lesson_every: int = 4
    lesson_exposures: int = 1830


def lesson_schedule(num_iterations, offset, every, exposures):
    """Iterations (1-based) carrying a replay update, at a fixed stride."""
    if offset < 0 or every < 1 or exposures < 1:
        raise ValueError('lesson offset/every/exposures must be positive')
    schedule = [offset + every * (k + 1) for k in range(exposures)]
    if schedule[-1] > num_iterations:
        raise ValueError('lesson schedule runs past the horizon')
    return schedule


def target_tensors(rows, fmt, device):
    """Stack one format's targets; embeddings as floats, categories as ids."""
    if fmt == 'plain':
        return {'plain': torch.tensor(
            [r['plain']['embedding'] for r in rows], device=device).float()}
    if fmt == 'contrastive':
        return {
            'endorsed': torch.tensor(
                [r['contrastive']['endorsed_embedding'] for r in rows],
                device=device).float(),
            'foil': torch.tensor(
                [r['contrastive']['foil_embedding'] for r in rows],
                device=device).float()}
    if fmt == 'subgoal':
        return {name: torch.tensor(
            [values.index(r['subgoal'][name]) for r in rows], device=device)
            for name, values in SUBGOAL_FIELDS}
    if fmt == 'plan':
        return {name: torch.tensor(
            [values.index(r['plan'][name]) for r in rows], device=device)
            for name, values in PLAN_FIELDS}
    if fmt == 'consequence':
        return {f'{role}_{name}': torch.tensor(
            [values.index(r['consequence'][f'{role}_{name}']) for r in rows],
            device=device)
            for role in ROLES for name, values in CONSEQUENCE_FIELDS}
    raise ValueError(f'Unknown format {fmt!r}')


def make_heads(fmt, dimension):
    """One linear head per target, reading features + two action one-hots."""
    width = FEATURES + 2 * ACTIONS
    if fmt == 'plain':
        sizes = {'plain': dimension}
    elif fmt == 'contrastive':
        sizes = {'endorsed': dimension, 'foil': dimension}
    elif fmt == 'subgoal':
        sizes = {name: len(values) for name, values in SUBGOAL_FIELDS}
    elif fmt == 'plan':
        sizes = {name: len(values) for name, values in PLAN_FIELDS}
    else:
        sizes = {f'{role}_{name}': len(values) for role in ROLES
                 for name, values in CONSEQUENCE_FIELDS}
    return torch.nn.ModuleDict(
        {name: torch.nn.Linear(width, size) for name, size in sizes.items()})


def format_loss(fmt, heads, x, targets):
    """Mean over the format's heads: 1 - cosine or cross-entropy."""
    losses = []
    for name, head in heads.items():
        out = head(x)
        if fmt in ('plain', 'contrastive'):
            losses.append(
                (1 - F.cosine_similarity(out, targets[name], dim=-1)).mean())
        else:
            losses.append(F.cross_entropy(out, targets[name]))
    return torch.stack(losses).mean()


def head_input(h, pos, neg):
    return torch.cat((h, F.one_hot(pos, ACTIONS).float(),
                      F.one_hot(neg, ACTIONS).float()), -1)


def action_loss(agent, h, pos, neg):
    """Positive cross-entropy plus foil unlikelihood (lesson studies)."""
    logp = agent.actor(h).log_softmax(-1)
    positive = F.nll_loss(logp, pos)
    negative = -torch.log1p(
        -logp.exp().gather(1, neg[:, None]).clamp(max=1 - 1e-6)).mean()
    return positive + negative


class FormatLessonAuxiliary:
    """Same replay stream in every arm; only the applied losses differ."""

    def initialize(self, agent, run_dir, device, args):
        if (args.lesson_mode not in MODES or args.lesson_targets not in TARGETS
                or args.lesson_format not in FORMATS or args.guidance
                or args.explanation != 'none' or args.consequence != 'none'
                or args.advice_replay or args.task != 'doorkey_8x8'
                or args.obs_mode != 'symbolic' or args.cuda):
            raise ValueError('Unsupported explanation-format configuration')
        uses_explanation = args.lesson_mode in (
            'explanation', 'combined', 'detached')
        uses_actions = args.lesson_mode in ('actions', 'combined')
        if uses_explanation != (args.lesson_scale > 0):
            raise ValueError('lesson_scale must be positive exactly when an '
                             'explanation loss is applied')
        if uses_actions != (args.lesson_action_coef > 0):
            raise ValueError('lesson_action_coef must be positive exactly '
                             'in actions/combined modes')
        path = Path(args.format_bank)
        if file_hash(path) != args.format_bank_sha256:
            raise ValueError('Frozen format-bank hash differs')
        self.bank = json.loads(path.read_text())
        if args.format_access not in ACCESS:
            raise ValueError(f'format_access must be one of {ACCESS}')
        if uses_explanation and (
                self.bank.get('information_access') != args.format_access):
            raise ValueError('Bank information access differs from '
                             'format_access; do not mix access conditions')
        self.args, self.agent, self.device = args, agent, device
        self.directory = Path(run_dir)
        self.rows = self.bank['train']
        self.uses_explanation, self.uses_actions = (
            uses_explanation, uses_actions)
        self.schedule = lesson_schedule(
            args.num_iterations, args.lesson_offset, args.lesson_every,
            args.lesson_exposures)
        self.due_at = set(self.schedule)
        # Same stream in every arm, format and background for a seed.
        self.rng = np.random.default_rng(args.seed + 51_007)
        self.obs = torch.tensor(np.asarray(
            [r['observation'] for r in self.rows]), device=device).float()
        self.pos = torch.tensor([r['positive_action'] for r in self.rows],
                                device=device)
        self.neg = torch.tensor([r['foil_action'] for r in self.rows],
                                device=device)
        fmt = args.lesson_format
        # ppo/actions read only images and action pairs, so the cached
        # categorical lesson corpus can serve as their bank (timing test).
        self.targets = None
        if uses_explanation:
            source = [r['targets'] if args.lesson_targets == 'aligned'
                      else r['permuted']['targets'] for r in self.rows]
            self.targets = target_tensors(source, fmt, device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 81_007)
            self.heads = make_heads(
                fmt, self.bank.get('embedding_dimension') or 1).to(device)
        self.optimizer = torch.optim.Adam(
            self.heads.parameters(), lr=args.learning_rate, eps=1e-5)
        self.shared = list(agent.encoder.parameters())
        self.initial = policy_hash(agent)
        self.updates = self.exposures = self.action_exposures = 0
        self.explanation_integral = self.action_integral = 0.0
        self.active = self.due = False
        self.log = self.directory / 'lesson_updates.jsonl'
        contract = dict(
            args=vars(args), initial_policy_sha256=self.initial,
            format_bank_sha256=args.format_bank_sha256,
            train_cases=len(self.rows), schedule=self.schedule,
            schedule_fraction=[self.schedule[0] / args.num_iterations,
                               self.schedule[-1] / args.num_iterations],
            planned_explanation_integral=(
                args.lesson_scale * len(self.schedule)
                if uses_explanation else 0.0),
            planned_action_integral=(
                args.lesson_action_coef * len(self.schedule)
                if uses_actions else 0.0),
            replay='encoder-only; no stored histories or recurrent states',
            heads='features + endorsed/foil action one-hots',
            information_access=(self.bank.get('information_access')
                                if uses_explanation else 'not applied'),
            request_version=self.bank.get('request_version'),
            request_sha256=self.bank.get('request_sha256'),
            target_representation=(
                'current encoded image + fixed action pair; a privileged '
                'target is only partly predictable from it by design'),
        )
        (self.directory / 'lesson_contract.json').write_text(
            json.dumps(contract, indent=2, default=str), encoding='utf-8')

    def begin_rollout(self, iteration):
        self.iteration = iteration
        self.due = iteration in self.due_at

    def loss(self):
        self.active = False
        if not self.due:
            return torch.zeros((), device=self.device)
        self.due = False
        self.active = True
        ids = self.rng.choice(len(self.rows), self.args.lesson_batch,
                              replace=len(self.rows) < self.args.lesson_batch)
        self.updates += 1
        self.exposures += len(ids)
        record = dict(iteration=self.iteration,
                      ids=[self.rows[i]['case_id'] for i in ids])
        if self.args.lesson_mode == 'ppo':
            self._write(record)
            return torch.zeros((), device=self.device)
        idx = torch.as_tensor(ids, device=self.device)
        h = self.agent._encode(self.obs[idx])
        pos, neg = self.pos[idx], self.neg[idx]
        total = torch.zeros((), device=self.device)
        self.optimizer.zero_grad()
        if self.uses_actions:
            a_loss = action_loss(self.agent, h, pos, neg)
            total = total + self.args.lesson_action_coef * a_loss
            self.action_exposures += len(ids)
            self.action_integral += self.args.lesson_action_coef
            record['action_loss'] = float(a_loss.detach())
        if self.uses_explanation:
            features = h.detach() if self.args.lesson_mode == 'detached' else h
            x = head_input(features, pos, neg)
            targets = {k: v[idx] for k, v in self.targets.items()}
            f_loss = format_loss(self.args.lesson_format, self.heads, x,
                                 targets)
            scaled = self.args.lesson_scale * f_loss
            self.explanation_integral += self.args.lesson_scale
            record['format_loss'] = float(f_loss.detach())
            if self.args.lesson_mode != 'detached':
                grads = torch.autograd.grad(
                    scaled, self.shared, retain_graph=True,
                    allow_unused=True)
                record['shared_grad_norm'] = float(torch.sqrt(sum(
                    (g.detach() ** 2).sum() for g in grads if g is not None)))
            total = total + scaled
        self._write(record)
        return total

    def optimizer_step(self):
        if self.active and self.uses_explanation:
            torch.nn.utils.clip_grad_norm_(
                self.heads.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

    def _write(self, record):
        with self.log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def finish(self, global_step):
        audit = self.bank['audit']
        metrics = {}
        if audit and self.uses_explanation:
            with torch.no_grad():
                obs = torch.tensor(np.asarray(
                    [r['observation'] for r in audit]),
                    device=self.device).float()
                pos = torch.tensor([r['positive_action'] for r in audit],
                                   device=self.device)
                neg = torch.tensor([r['foil_action'] for r in audit],
                                   device=self.device)
                x = head_input(self.agent._encode(obs), pos, neg)
                fmt = self.args.lesson_format
                targets = target_tensors(
                    [r['targets'] for r in audit], fmt, self.device)
                for name, head in self.heads.items():
                    out = head(x)
                    if fmt in ('plain', 'contrastive'):
                        metrics[name] = dict(
                            mean_cosine=float(F.cosine_similarity(
                                out, targets[name], dim=-1).mean()),
                            retrieval_top1=float((F.normalize(out, dim=-1)
                                @ F.normalize(targets[name], dim=-1).T)
                                .argmax(-1).eq(torch.arange(
                                    len(audit), device=self.device))
                                .float().mean()))
                    else:
                        metrics[name] = dict(accuracy=float(
                            out.argmax(-1).eq(targets[name]).float().mean()))
        result = dict(
            global_step=global_step, initial_policy_sha256=self.initial,
            final_policy_sha256=policy_hash(self.agent),
            format_bank_sha256=self.args.format_bank_sha256,
            updates=self.updates, replay_exposures=self.exposures,
            effective_action_exposures=self.action_exposures,
            explanation_integral=self.explanation_integral,
            action_integral=self.action_integral,
            schedule_first=self.schedule[0], schedule_last=self.schedule[-1],
            audit_head_metrics=metrics,
            semantic_correctness='not certified')
        (self.directory / 'lesson_finished.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        torch.save(self.heads.state_dict(),
                   self.directory / 'lesson_heads.pt')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=FormatLessonAuxiliary())
