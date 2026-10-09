"""Offline LLM-selected correction lessons."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from algos import ppo_distill as ppo
from algos.ppo_progress import policy_hash
from scripts.contrastive_lessons import POSITIVE, NEGATIVE, write
from scripts.explanation_screen_panel import file_hash


MODES = ('labels', 'ordinary', 'contrastive', 'shuffled', 'detached', 'ppo')


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
    lesson_file: str = ''
    lesson_sha256: str = ''
    lesson_mode: str = 'contrastive'
    lesson_every: int = 4
    lesson_batch: int = 32
    lesson_cutoff: float = 0.75
    lesson_action_coef: float = 1.0
    lesson_reason_coef: float = 0.1


class LessonAuxiliary:
    """Same action labels/replay stream; only reason gradients differ."""

    def initialize(self, agent, run_dir, device, args):
        if (
            args.lesson_mode not in MODES
            or args.guidance
            or args.explanation != 'none'
            or args.consequence != 'none'
            or args.advice_replay
            or args.task != 'doorkey_8x8'
            or args.obs_mode != 'symbolic'
            or args.cuda
        ):
            raise ValueError('Unsupported frozen-lesson configuration')
        path = Path(args.lesson_file)
        if file_hash(path) != args.lesson_sha256:
            raise ValueError('Frozen lesson hash differs')
        self.data = json.loads(path.read_text())
        if not self.data['technical_ready']:
            raise ValueError('Lesson integrity gate did not pass')
        self.args, self.agent, self.device = args, agent, device
        self.directory = Path(run_dir)
        self.rng = np.random.default_rng(args.seed + 51_007)
        self.rows = self.data['train']
        self.obs = torch.tensor(
            np.asarray([r['observation'] for r in self.rows]), device=device
        ).float()
        self.pos = torch.tensor(
            [r['positive_action'] for r in self.rows], device=device
        )
        self.neg = torch.tensor(
            [r['foil_action'] for r in self.rows], device=device
        )
        self.ptarget = torch.tensor(
            [r['positive'] for r in self.rows], device=device
        )
        field = (
            'shuffled_negative'
            if args.lesson_mode == 'shuffled'
            else 'negative'
        )
        self.ntarget = torch.tensor(
            [r[field] for r in self.rows], device=device
        )
        # Heads see the same action pair, isolating state-conditioned value.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 81_007)
            self.heads = torch.nn.ModuleList(
                [
                    torch.nn.Linear(526, len(POSITIVE)),
                    torch.nn.Linear(526, len(NEGATIVE)),
                ]
            ).to(device)
        self.optimizer = torch.optim.Adam(
            self.heads.parameters(), lr=args.learning_rate, eps=1e-5
        )
        self.initial = policy_hash(agent)
        self.updates = self.exposures = 0
        self.active = self.due = False
        write(
            self.directory / 'lesson_contract.json',
            dict(
                args=vars(args),
                initial_policy_sha256=self.initial,
                lesson_sha256=args.lesson_sha256,
                replay='encoder-only; no stored histories or recurrent states',
                teacher='oracle action pair + GPT-5-mini structured reason codes',
            ),
        )

    def begin_rollout(self, iteration):
        self.iteration = iteration
        self.due = (
            iteration % self.args.lesson_every == 0
            and iteration
            <= int(self.args.num_iterations * self.args.lesson_cutoff)
        )

    def loss(self):
        """One lesson minibatch per scheduled rollout, never per PPO epoch."""
        self.active = False
        if not self.due:
            return torch.zeros((), device=self.device)
        self.due = False
        self.active = True
        ids = self.rng.choice(
            len(self.rows),
            self.args.lesson_batch,
            replace=len(self.rows) < self.args.lesson_batch,
        )
        if self.args.lesson_mode == 'ppo':
            # Log the paired schedule without consulting labels or images.
            self.updates += 1
            self.exposures += len(ids)
            with (self.directory / 'lesson_updates.jsonl').open('a') as handle:
                handle.write(
                    json.dumps(
                        dict(
                            iteration=self.iteration,
                            ids=[self.rows[i]['case_id'] for i in ids],
                            positive_loss=0.0,
                            negative_loss=0.0,
                            reason_shared_grad_norm=0.0,
                            action_loss=0.0,
                        )
                    )
                    + '\n'
                )
            return torch.zeros((), device=self.device)
        h = self.agent._encode(self.obs[ids])
        logits = self.agent.actor(h)
        # The established replay path is feed-forward, bypassing the GRU.
        # Both signs of action supervision are identical in every arm.
        logp = logits.log_softmax(-1)
        positive_loss = F.nll_loss(logp, self.pos[ids])
        negative_loss = -torch.log1p(
            -logp.exp().gather(1, self.neg[ids, None]).clamp(max=1 - 1e-6)
        ).mean()
        actions = torch.cat(
            (F.one_hot(self.pos[ids], 7), F.one_hot(self.neg[ids], 7)), -1
        ).float()
        pshared = self.args.lesson_mode in (
            'ordinary',
            'contrastive',
            'shuffled',
        )
        nshared = self.args.lesson_mode in ('contrastive', 'shuffled')
        pinput = torch.cat((h if pshared else h.detach(), actions), -1)
        ninput = torch.cat((h if nshared else h.detach(), actions), -1)
        ploss = F.cross_entropy(self.heads[0](pinput), self.ptarget[ids])
        nloss = F.cross_entropy(self.heads[1](ninput), self.ntarget[ids])
        self.optimizer.zero_grad()
        reasons = self.args.lesson_reason_coef * (ploss + nloss)
        if self.args.lesson_mode == 'labels':
            # Identical forward work; no private-head updates in this arm.
            reasons = reasons.detach()
        shared_grad = (
            torch.autograd.grad(
                reasons, h, retain_graph=True, allow_unused=True
            )[0]
            if reasons.requires_grad
            else None
        )
        self.updates += 1
        self.exposures += len(ids)
        with (self.directory / 'lesson_updates.jsonl').open('a') as handle:
            handle.write(
                json.dumps(
                    dict(
                        iteration=self.iteration,
                        ids=[self.rows[i]['case_id'] for i in ids],
                        positive_loss=float(ploss.detach()),
                        negative_loss=float(nloss.detach()),
                        reason_shared_grad_norm=(
                            float(shared_grad.norm())
                            if shared_grad is not None
                            else 0.0
                        ),
                        action_loss=float(
                            (positive_loss + negative_loss).detach()
                        ),
                    )
                )
                + '\n'
            )
        return (
            self.args.lesson_action_coef * (positive_loss + negative_loss)
            + reasons
        )

    def optimizer_step(self):
        """Private clipping keeps detached gradients out of PPO scaling."""
        if self.active and self.args.lesson_mode not in ('labels', 'ppo'):
            for head in self.heads:
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), self.args.max_grad_norm
                )
            self.optimizer.step()

    def finish(self, global_step):
        """Retain policy identities, exposure and held-out code prediction."""
        audit = self.data['audit']
        with torch.no_grad():
            obs = torch.tensor(
                np.asarray([r['observation'] for r in audit]),
                device=self.device,
            ).float()
            h = self.agent._encode(obs)
            actions = torch.tensor(
                [[r['positive_action'], r['foil_action']] for r in audit],
                device=self.device,
            )
            x = torch.cat((h, F.one_hot(actions, 7).flatten(1).float()), -1)
            scores = {}
            for j, field in enumerate(('positive', 'negative')):
                target = torch.tensor(
                    [r[field] for r in audit], device=self.device
                )
                scores[field] = float(
                    (self.heads[j](x).argmax(-1) == target).float().mean()
                )
        write(
            self.directory / 'lesson_finished.json',
            dict(
                global_step=global_step,
                initial_policy_sha256=self.initial,
                final_policy_sha256=policy_hash(self.agent),
                lesson_sha256=self.args.lesson_sha256,
                updates=self.updates,
                replay_exposures=self.exposures,
                applied_action_exposures=(
                    0 if self.args.lesson_mode == 'ppo' else self.exposures
                ),
                audit_code_accuracy=scores,
                semantic_correctness='not certified',
            ),
        )
        torch.save(self.heads.state_dict(), self.directory / 'lesson_heads.pt')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=LessonAuxiliary())
