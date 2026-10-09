"""Code facts alone versus the same facts plus LLM interpretation.

A new auxiliary for the
unchanged `algos.ppo_distill` trainer, following `ppo_lesson_formats`:
same replay stream (seed + 51_007), same encoder-only replay of the fixed
formats-bank cases, same head input (encoded image + endorsed/foil action
one-hots), same schedule arguments.

Arms (every arm draws and logs the identical replay stream):

    ppo                 no auxiliary loss
    facts               facts_scale x mean cross-entropy over the 11 native
                        categorical facts (subgoal, next stage geometry,
                        one-step effects of both actions)
    facts_llm           facts loss + text_scale x mean (1 - cosine) to the
                        case's own LLM sentence embeddings (plain,
                        contrastive, memory), masked where unusable
    facts_llm_shuffled  facts loss (aligned) + the same text loss towards a
                        donor case's sentences (formats-bank donors)

facts_llm - facts measures language added to identical facts; facts_llm -
facts_llm_shuffled tests case alignment at matched loss form and scale;
the shuffle does not guarantee matched target learnability. All heads
are built in every arm from the same
seeded initialization. No online teacher; zero explicit action loss.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from algos import ppo_distill as ppo
from algos import ppo_lesson_formats as formats
from algos.ppo_progress import policy_hash
from scripts.explanation_screen_panel import file_hash

ARMS = ('ppo', 'facts', 'facts_llm', 'facts_llm_shuffled')
TEXT_ARMS = ('facts_llm', 'facts_llm_shuffled')
TEXT_FIELDS = ('plain_reason', 'contrastive_reason', 'memory_hint')
head_input = formats.head_input
WIDTH = formats.FEATURES + 2 * formats.ACTIONS


@dataclass
class Args(formats.Args):
    pilot_arm: str = 'ppo'
    pilot_bank: str = ''
    pilot_bank_sha256: str = ''
    pilot_calibration: str = ''
    pilot_calibration_sha256: str = ''
    facts_scale: float = 0.0
    text_scale: float = 0.0


def bank_tensors(rows, fact_values, device, donor=False):
    """Observations, action pairs, fact ids, text targets and masks."""
    texts = 'donor_embeddings' if donor else 'embeddings'
    dimension = next(len(v) for r in rows for v in r[texts].values()
                     if v is not None)
    out = dict(
        obs=torch.tensor(np.asarray([r['observation'] for r in rows]),
                         device=device).float(),
        pos=torch.tensor([r['positive_action'] for r in rows], device=device),
        neg=torch.tensor([r['foil_action'] for r in rows], device=device),
        facts={f: torch.tensor([values.index(r['facts'][f]) for r in rows],
                               device=device)
               for f, values in fact_values.items()},
        text={}, mask={})
    for field in TEXT_FIELDS:
        vectors = [r[texts][field] for r in rows]
        out['mask'][field] = torch.tensor([v is not None for v in vectors],
                                          device=device)
        out['text'][field] = torch.tensor(
            [v if v is not None else [0.0] * dimension for v in vectors],
            device=device).float()
    return out


def make_heads(fact_values, dimension, seed):
    """Identical seeded heads in every arm (unused heads stay untrained)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 81_007)
        heads = {f'fact_{f}': torch.nn.Linear(WIDTH, len(v))
                 for f, v in fact_values.items()}
        heads.update({f'text_{f}': torch.nn.Linear(WIDTH, dimension)
                      for f in TEXT_FIELDS})
    return torch.nn.ModuleDict(heads)


def facts_loss(heads, x, facts):
    return torch.stack([F.cross_entropy(heads[f'fact_{f}'](x), target)
                        for f, target in facts.items()]).mean()


def text_loss(heads, x, text, mask):
    """Mean over fields of the masked mean (1 - cosine); 0 if none usable."""
    losses = []
    for field in TEXT_FIELDS:
        keep = mask[field]
        if keep.any():
            out = heads[f'text_{field}'](x[keep])
            losses.append((1 - F.cosine_similarity(
                out, text[field][keep], dim=-1)).mean())
    if not losses:
        return x.sum() * 0.0
    return torch.stack(losses).mean()


def validate_args(args):
    if (args.pilot_arm not in ARMS or args.task != 'doorkey_8x8'
            or args.bonus != 'count' or args.obs_mode != 'symbolic'
            or not args.recurrent or not args.dual_value or args.cuda
            or args.guidance or args.query_budget != 0
            or args.lesson_action_coef != 0 or args.lesson_scale != 0
            or args.explanation != 'none' or args.consequence != 'none'
            or args.advice_replay):
        raise ValueError('Unsupported language-pilot configuration')
    for path, expected in ((args.pilot_bank, args.pilot_bank_sha256),
                           (args.pilot_calibration,
                            args.pilot_calibration_sha256)):
        if file_hash(path) != expected:
            raise ValueError('Frozen language-pilot input hash differs')
    calibration = json.loads(Path(args.pilot_calibration).read_text())
    if calibration['bank_sha256'] != args.pilot_bank_sha256:
        raise ValueError('Calibration belongs to another bank')
    facts = 0.0 if args.pilot_arm == 'ppo' else calibration['facts_scale']
    text = calibration['text_scale'] if args.pilot_arm in TEXT_ARMS else 0.0
    if args.facts_scale != facts or args.text_scale != text:
        raise ValueError('Scales must equal the frozen calibration')


class LanguagePilotAuxiliary:
    """Same replay stream in every arm; only the applied losses differ."""

    def initialize(self, agent, run_dir, device, args):
        validate_args(args)
        self.args, self.agent, self.device = args, agent, device
        self.directory = Path(run_dir)
        self.bank = json.loads(Path(args.pilot_bank).read_text())
        self.rows = self.bank['train']
        self.fact_values = self.bank['fact_values']
        self.schedule = formats.lesson_schedule(
            args.num_iterations, args.lesson_offset, args.lesson_every,
            args.lesson_exposures)
        self.due_at = set(self.schedule)
        self.rng = np.random.default_rng(args.seed + 51_007)
        self.data = bank_tensors(self.rows, self.fact_values, device)
        donor = bank_tensors(self.rows, self.fact_values, device, donor=True)
        shuffled = args.pilot_arm == 'facts_llm_shuffled'
        self.text = donor['text'] if shuffled else self.data['text']
        self.mask = donor['mask'] if shuffled else self.data['mask']
        self.heads = make_heads(self.fact_values,
                                self.bank['embedding_dimension'],
                                args.seed).to(device)
        self.optimizer = torch.optim.Adam(
            self.heads.parameters(), lr=args.learning_rate, eps=1e-5)
        self.shared = list(agent.encoder.parameters())
        self.initial = policy_hash(agent)
        self.updates = self.exposures = 0
        self.facts_integral = self.text_integral = 0.0
        self.active = self.due = False
        self.log = self.directory / 'lesson_updates.jsonl'
        contract = dict(
            args=vars(args), initial_policy_sha256=self.initial,
            pilot_bank_sha256=args.pilot_bank_sha256,
            pilot_calibration_sha256=args.pilot_calibration_sha256,
            arm=args.pilot_arm, train_cases=len(self.rows),
            schedule=self.schedule,
            planned_facts_integral=args.facts_scale * len(self.schedule),
            planned_text_integral=args.text_scale * len(self.schedule),
            text_targets=('donor' if shuffled else 'aligned'
                          if args.pilot_arm in TEXT_ARMS else 'none'),
            replay='encoder-only; no stored histories or recurrent states',
            heads='features + endorsed/foil action one-hots',
            semantic_correctness='not certified; LLM errors retained')
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
        if self.args.pilot_arm == 'ppo':
            self._write(record)
            return torch.zeros((), device=self.device)
        idx = torch.as_tensor(ids, device=self.device)
        d = self.data
        self.optimizer.zero_grad()
        x = head_input(self.agent._encode(d['obs'][idx]), d['pos'][idx],
                       d['neg'][idx])
        f = facts_loss(self.heads, x, {k: v[idx] for k, v in
                                       d['facts'].items()})
        total = self.args.facts_scale * f
        self.facts_integral += self.args.facts_scale
        record['facts_loss'] = float(f.detach())
        if self.args.pilot_arm in TEXT_ARMS:
            mask = {k: v[idx] for k, v in self.mask.items()}
            t = text_loss(self.heads, x, {k: v[idx] for k, v in
                                          self.text.items()}, mask)
            total = total + self.args.text_scale * t
            self.text_integral += self.args.text_scale
            record['text_loss'] = float(t.detach())
            record['text_rows'] = {k: int(v.sum()) for k, v in mask.items()}
        grads = torch.autograd.grad(total, self.shared, retain_graph=True,
                                    allow_unused=True)
        record['shared_grad_norm'] = float(torch.sqrt(sum(
            (g.detach() ** 2).sum() for g in grads if g is not None)))
        self._write(record)
        return total

    def optimizer_step(self):
        if self.active and self.args.pilot_arm != 'ppo':
            torch.nn.utils.clip_grad_norm_(
                self.heads.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

    def _write(self, record):
        with self.log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def finish(self, global_step):
        audit = self.bank['audit']
        metrics = {}
        if audit:
            with torch.no_grad():
                d = bank_tensors(audit, self.fact_values, self.device)
                x = head_input(self.agent._encode(d['obs']), d['pos'],
                               d['neg'])
                for f, target in d['facts'].items():
                    metrics[f'fact_{f}'] = float(
                        self.heads[f'fact_{f}'](x).argmax(-1).eq(target)
                        .float().mean())
                for f in TEXT_FIELDS:
                    keep = d['mask'][f]
                    if keep.any():
                        metrics[f'text_{f}_cosine'] = float(
                            F.cosine_similarity(
                                self.heads[f'text_{f}'](x[keep]),
                                d['text'][f][keep], dim=-1).mean())
        result = dict(
            global_step=global_step, arm=self.args.pilot_arm,
            initial_policy_sha256=self.initial,
            final_policy_sha256=policy_hash(self.agent),
            pilot_bank_sha256=self.args.pilot_bank_sha256,
            updates=self.updates, replay_exposures=self.exposures,
            facts_integral=self.facts_integral,
            text_integral=self.text_integral,
            schedule_first=self.schedule[0], schedule_last=self.schedule[-1],
            audit_head_metrics=metrics, audit_heads_trained=(
                self.args.pilot_arm != 'ppo'),
            semantic_correctness='not certified')
        (self.directory / 'lesson_finished.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        torch.save(self.heads.state_dict(),
                   self.directory / 'lesson_heads.pt')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=LanguagePilotAuxiliary())
