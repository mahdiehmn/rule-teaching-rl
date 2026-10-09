"""Freeze explanation-format loss scales from shared-parameter gradients.

Run before any RL outcome is read; the
output file is an input to training, never revised afterwards.

CE losses over a few categories and cosine losses over 1,536-d
embeddings have different units and head counts, so a common scalar
such as 0.1 does not give matched learning influence. For each training
seed's ACTUAL initial policy (torch.manual_seed(seed) -> vector envs ->
Agent, as in algos.ppo_distill) and freshly initialized heads (seed +
81,007, as in training), on one frozen training-only minibatch, this
measures the norm of each unscaled format loss's gradient with respect
to the shared encoder parameters. The reference is the running isolation
study's aligned arm: 0.1 x (positive-code CE + foil-code CE) through heads
of the same shape, on the same cases.

    primary scale      median over seeds of  |g_reference| / |g_format|
    sensitivity scale  3 x primary; a separately labelled value that
                       does not double the grid unless approved

The explicit action loss (coefficient 1) is reported for context only.
"""

import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from algos import ppo_distill as ppo
from algos.nets import is_symbolic_mode
from algos.ppo_lesson_formats import (
    Args, action_loss, format_loss, head_input, make_heads, target_tensors)
from algos.ppo_progress import policy_hash
from scripts.explanation_formats import FORMATS
from scripts.explanation_screen_panel import file_hash

CALIBRATION_SALT = 'explanation_formats_20260925:calibration_minibatch'
SENSITIVITY_FACTOR = 3.0
CODE_COEF = 0.1


def initial_agent(seed):
    """The agent algos.ppo_distill would build for this seed."""
    args = Args(seed=seed)
    torch.manual_seed(seed)
    envs = gym.vector.SyncVectorEnv([
        ppo.make_thunk(args.task, seed, i, obs_mode=args.obs_mode,
                       agent_view_size=args.agent_view_size,
                       obs_tile_size=args.obs_tile_size,
                       obs_target_size=args.obs_target_size)
        for i in range(args.num_envs)])
    agent = ppo.Agent(envs, symbolic=is_symbolic_mode(args.obs_mode),
                      recurrent=args.recurrent, dual_value=args.dual_value)
    envs.close()
    return agent


def calibration_rows(bank, lessons, size):
    """Frozen minibatch: bank train cases that also carry cached codes."""
    codes = {r['case_id']: r for r in lessons['train']}
    rows = [r for r in bank['train'] if r['case_id'] in codes]
    rows.sort(key=lambda r: hashlib.sha256(
        (CALIBRATION_SALT + ':' + r['case_id']).encode()).hexdigest())
    rows = rows[:size]
    if len(rows) < size:
        raise ValueError(f'Only {len(rows)} calibration cases available')
    return rows, [codes[r['case_id']] for r in rows]


def shared_norm(loss, shared):
    grads = torch.autograd.grad(loss, shared, allow_unused=True)
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads
                                if g is not None)))


def measure(seed, rows, code_rows, dimension):
    agent = initial_agent(seed)
    shared = list(agent.encoder.parameters())
    obs = torch.tensor(np.asarray([r['observation'] for r in rows])).float()
    pos = torch.tensor([r['positive_action'] for r in rows])
    neg = torch.tensor([r['foil_action'] for r in rows])
    norms = {'initial_policy_sha256': policy_hash(agent)}
    for fmt in FORMATS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 81_007)
            heads = make_heads(fmt, dimension)
        x = head_input(agent._encode(obs), pos, neg)
        loss = format_loss(fmt, heads, x, target_tensors(
            [r['targets'] for r in rows], fmt, 'cpu'))
        norms[fmt] = shared_norm(loss, shared)
    from scripts.contrastive_lessons import NEGATIVE, POSITIVE
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 81_007)
        code_heads = torch.nn.ModuleList([
            torch.nn.Linear(526, len(POSITIVE)),
            torch.nn.Linear(526, len(NEGATIVE))])
    x = head_input(agent._encode(obs), pos, neg)
    reference = CODE_COEF * (
        F.cross_entropy(code_heads[0](x), torch.tensor(
            [r['positive'] for r in code_rows]))
        + F.cross_entropy(code_heads[1](x), torch.tensor(
            [r['negative'] for r in code_rows])))
    norms['reference_codes'] = shared_norm(reference, shared)
    norms['actions_coef1'] = shared_norm(
        action_loss(agent, agent._encode(obs), pos, neg), shared)
    return norms


def calibrate(bank_path, lessons_path, seeds, size=64):
    bank = json.loads(Path(bank_path).read_text())
    lessons = json.loads(Path(lessons_path).read_text())
    rows, code_rows = calibration_rows(bank, lessons, size)
    dimension = bank['embedding_dimension']
    per_seed = {str(s): measure(s, rows, code_rows, dimension)
                for s in seeds}
    ratios = {f: [m['reference_codes'] / m[f] for m in per_seed.values()]
              for f in FORMATS}
    primary = {f: st.median(r) for f, r in ratios.items()}
    return dict(
        author='mahdiehmn', date='2026-09-25',
        method='shared encoder-parameter gradient norm on a frozen '
               'training-only minibatch at each seed\'s initial policy',
        reference=f'isolation aligned codes, {CODE_COEF} x (positive CE + '
                  'foil CE), same head input and cases',
        bank_sha256=file_hash(Path(bank_path)),
        lessons_sha256=file_hash(Path(lessons_path)),
        calibration_cases=[r['case_id'] for r in rows], seeds=list(seeds),
        norms=per_seed, primary_scale=primary,
        sensitivity_scale={f: SENSITIVITY_FACTOR * v
                           for f, v in primary.items()},
        sensitivity_factor=SENSITIVITY_FACTOR,
        ratio_spread={f: [min(r), max(r)] for f, r in ratios.items()},
        frozen_before_rl=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--lessons', required=True,
                        help='contrastive_lessons_20260924_v1/lessons.json')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[12_300_000 + 100 * i for i in range(5)])
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Calibration output exists; it is frozen once written')
    result = calibrate(args.bank, args.lessons, args.seeds, args.size)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(primary_scale=result['primary_scale'],
                          ratio_spread=result['ratio_spread']), indent=2))


if __name__ == '__main__':
    main()
