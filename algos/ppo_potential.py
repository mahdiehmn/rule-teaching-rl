"""Teacher knowledge through a message-conditioned potential, not actions.

Protocol:
research/potential_channel_protocol_2026-09-27.md.

The unchanged `algos.ppo_distill` trainer plus two opt-in hooks: the
potential is anchored on the true reset states (`on_reset`, before the
first action) and shaping is added to the extrinsic reward after each
step (`shape_reward`).
A rule teacher (teachers/subgoal_potential.py) sends budgeted, possibly
wrong subgoal messages. The student receives only potential-based
shaping on its extrinsic reward, during training. There is no action
advice, auxiliary loss or policy input from the teacher, and evaluation
is teacher-off and unshaped. The teacher's messages come from a private
RNG (seed + 71_003); the policy's initialization and every global random
stream match the same configuration without this auxiliary.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import tyro

from algos import ppo_distill as ppo
from teachers.subgoal_potential import (SUPPORTED, PotentialTeacher,
                                        load_plan)


@dataclass
class Args(ppo.Args):
    potential_scale: float = 1.0
    # Total teacher messages over training; -1 = one at every episode
    # start and stage change (dense).
    potential_queries: int = -1
    potential_wrong_rate: float = 0.0
    # 'zero' (exact invariance, primary) or 'keep' (see subgoal_potential)
    potential_truncation: str = 'zero'
    # Record full shaping traces of the first N episodes per env (tests).
    potential_trace_episodes: int = 0
    # Rule-bank imitation AND shaping in one run (fix wave, 2026-09-29).
    # False keeps the pure potential channel, and every earlier run,
    # exactly: no action advice reaches the student.
    potential_with_guidance: bool = False
    # A frozen subgoal-plan file (e.g. the LLM teacher's,
    # research/stage_plans/llm_plans_20260929.json) and the digest of this
    # task's plan. '' keeps the hand-coded stages and every earlier run.
    potential_plan: str = ''
    potential_plan_sha256: str = ''


def validate_args(args):
    combined = getattr(args, 'potential_with_guidance', False)
    if bool(getattr(args, 'potential_plan', '')) != bool(
            getattr(args, 'potential_plan_sha256', '')):
        raise ValueError('potential_plan and its digest go together')
    if combined and (not args.guidance or args.teacher != 'rule_bank'):
        raise ValueError('potential_with_guidance pairs shaping with a '
                         'rule-bank teacher only')
    if (args.task not in SUPPORTED or (args.guidance and not combined)
            or args.query_budget != 0 or args.explanation != 'none'
            or args.consequence != 'none' or args.advice_replay
            or args.potential_scale <= 0
            or not 0 <= args.potential_wrong_rate <= 1
            or args.potential_queries < -1
            or args.potential_truncation not in ('zero', 'keep')
            or args.potential_trace_episodes < 0):
        raise ValueError('Unsupported potential-channel configuration')


class PotentialAuxiliary:
    """Adds shaping through the trainer's reward hook; no loss, no params."""

    def initialize(self, agent, run_dir, device, args):
        validate_args(args)
        self.args, self.device = args, device
        self.directory = Path(run_dir)
        self.teacher = PotentialTeacher(
            args.task, args.num_envs, args.gamma,
            scale=args.potential_scale, queries=args.potential_queries,
            wrong_rate=args.potential_wrong_rate, seed=args.seed + 71_003,
            truncation=args.potential_truncation,
            trace_episodes=args.potential_trace_episodes,
            plan=(load_plan(args.potential_plan, args.task,
                            args.potential_plan_sha256)
                  if args.potential_plan else None))
        (self.directory / 'potential_contract.json').write_text(json.dumps(
            dict(args=vars(args), channel='potential over (state, message)',
                 terminal_potential=0.0, evaluation='teacher-off, unshaped',
                 message_rng_seed=args.seed + 71_003), indent=2,
            default=str), encoding='utf-8')

    def on_reset(self, sync_envs, global_step):
        self.teacher.begin_all([env.unwrapped for env in sync_envs.envs],
                               global_step)

    def shape_reward(self, sync_envs, reset_only, done, global_step):
        unwrapped = [env.unwrapped for env in sync_envs.envs]
        return self.teacher.shape(unwrapped, reset_only, done, global_step)

    def begin_rollout(self, iteration):
        pass

    def loss(self):
        return torch.zeros((), device=self.device)

    def optimizer_step(self):
        pass

    def finish(self, global_step):
        summary = dict(global_step=global_step, **self.teacher.summary())
        (self.directory / 'potential_summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
        with (self.directory / 'potential_queries.jsonl').open(
                'w', encoding='utf-8') as handle:
            for row in self.teacher.log:
                handle.write(json.dumps(row) + '\n')
        if self.teacher.trace:
            with (self.directory / 'potential_trace.jsonl').open(
                    'w', encoding='utf-8') as handle:
                for row in self.teacher.trace:
                    handle.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=PotentialAuxiliary())
