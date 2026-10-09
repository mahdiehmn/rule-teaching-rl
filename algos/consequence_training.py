"""Simulator-only channel diagnostic.

This explicit oracle-target path does not correct or replace LLM outputs.
It copies DoorKey states and never steps the student's live environment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from teachers.consequence_targets import (
    ConsequenceRecord, resolve_rollout_targets)


def gradient_diagnostics(auxiliary, ppo_loss, action_loss, agent):
    """Compare weighted losses on the same shared encoder/core weights."""
    parameters = list(agent.encoder.parameters())
    if agent.core is not None:
        parameters += list(agent.core.parameters())
    gradients = [torch.autograd.grad(
        value, parameters, retain_graph=True, allow_unused=True)
        for value in (auxiliary, ppo_loss, action_loss)]
    norms = [sum(float(g.detach().double().square().sum())
                 for g in values if g is not None)**.5
             for values in gradients]
    def cosine(index):
        if not norms[0] or not norms[index]:
            return None
        dot = sum(float((a.detach().double()*b.detach().double()).sum())
                  for a,b in zip(gradients[0], gradients[index])
                  if a is not None and b is not None)
        return dot/(norms[0]*norms[index])
    return dict(shared_aux_norm=norms[0], shared_ppo_norm=norms[1],
                shared_action_norm=norms[2], aux_ppo_cosine=cosine(1),
                aux_action_cosine=cosine(2))


class SimulatorConsequences:
    """Keep rollout-local target evidence and immutable epoch buffers."""

    def __init__(self, args, run_dir, device):
        self.args = args
        self.directory = Path(run_dir) / 'consequences'
        self.directory.mkdir()
        shape = (args.num_steps, args.num_envs, 2)
        self.actions = torch.full(shape, 6, dtype=torch.long, device=device)
        self.targets = torch.zeros((*shape, 4), dtype=torch.long, device=device)
        self.mask = torch.zeros_like(self.targets, dtype=torch.bool)
        self.pending = []
        self.counts = dict(consultations=0, eligible=0, changed=0, rollouts=0)
        # None keeps DoorKey on the screen's own strict restorer; any
        # other task is restored by env id through the generic path.
        self.env_id = None
        if args.task != 'doorkey_8x8':
            from envs.registry import TASKS
            self.env_id = TASKS[args.task]

    def collect(self, env, student, target, phase, rollout, step, index):
        """Create exact effects for the endorsed and proposed actions."""
        from scripts.run_consequence_readiness import effect_truth
        # A dense-advice run consults millions of times, and every target
        # costs two simulator copies. Past the budget the head simply
        # stops being fed; the action labels are untouched.
        budget = getattr(self.args, 'consequence_target_budget', 0)
        if budget and self.counts['consultations'] >= budget:
            return
        state = dict(
            full_grid=env.grid.encode().tolist(),
            agent_pos=list(map(int, env.agent_pos)),
            agent_dir=int(env.agent_dir), mission=env.mission,
            carrying=([env.carrying.type, env.carrying.color]
                      if env.carrying else None))
        identity = hashlib.sha256(json.dumps(
            state, sort_keys=True).encode()).hexdigest()
        endorsed = int(np.argmax(target))
        payload = dict(
            action=endorsed,
            recommended=effect_truth(state, endorsed, env_id=self.env_id),
            student=effect_truth(state, student, env_id=self.env_id),
            why_recommended='Simulator exact target; not LLM text.',
            why_student='Simulator exact target; not LLM text.')
        record = ConsequenceRecord(
            f'{rollout}:{step}:{index}', rollout, phase,
            student, payload, identity)
        self.pending.append((record, step, index))

    def resolve(self, rollout):
        """Share masks across conditions; persist donors only once."""
        self.actions.fill_(6)
        self.targets.zero_()
        self.mask.zero_()
        if self.pending:
            resolved = resolve_rollout_targets(
                [v[0] for v in self.pending],
                seed=self.args.seed + 300_001 + rollout)
            selected = (resolved.shuffled if self.args.consequence ==
                        'shuffled' else resolved.correct)
            changed = 0
            with (self.directory/'targets.jsonl').open('a') as handle:
                for i, (record, step, env) in enumerate(self.pending):
                    self.actions[step, env] = torch.as_tensor(
                        resolved.actions[i], device=self.actions.device)
                    self.targets[step, env] = torch.as_tensor(
                        selected[i], device=self.targets.device)
                    self.mask[step, env] = torch.as_tensor(
                        resolved.mask[i], device=self.mask.device)
                    differs = bool(((resolved.correct[i] !=
                                     resolved.shuffled[i]) &
                                    resolved.mask[i]).any())
                    changed += differs
                    # The placebo only differs from the treatment where
                    # the permutation moved something. Training on the
                    # bundles it left alone makes the two arms share
                    # those samples and dilutes the contrast; this
                    # drops them from both arms alike.
                    if getattr(self.args, 'consequence_changed_only',
                               False) and not differs:
                        self.mask[step, env] = False
                    handle.write(json.dumps(dict(
                        sample_id=record.sample_id, phase=record.phase,
                        rollout=rollout, step=step, env=env,
                        query_identity=record.query_identity,
                        action=record.response['action'],
                        student_action=record.student_action,
                        actions=resolved.actions[i].tolist(),
                        correct=resolved.correct[i].tolist(),
                        shuffled=resolved.shuffled[i].tolist(),
                        mask=resolved.mask[i].tolist(),
                        status=resolved.status[i],
                        donor_id=resolved.donor_ids[i]))+'\n')
            self.counts['consultations'] += len(self.pending)
            self.counts['eligible'] += resolved.stats['eligible_consultations']
            self.counts['changed'] += changed
            self.counts['rollouts'] += 1
            with (self.directory/'rollouts.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(
                    rollout=rollout, changed_bundles=changed,
                    **resolved.stats))+'\n')
            (self.directory/'summary.json').write_text(json.dumps(
                dict(source='simulator_exact', mode=self.args.consequence,
                     **self.counts), indent=2)+'\n')
        self.pending.clear()
        return (self.actions.reshape(-1, 2),
                self.targets.reshape(-1, 2, 4), self.mask.reshape(-1, 2, 4))
