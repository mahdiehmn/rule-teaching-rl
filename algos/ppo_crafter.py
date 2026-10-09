"""PPO on symbolic Crafter, optionally imitating a frozen rule bank.

A compact CleanRL-style
PPO with the same teaching channel as the MiniGrid students: the student
always acts by its own policy; on steps where the frozen rule bank fires
(scripts/crafter_rules_v2_20260930.advise over the student's predicates)
its action is a cross-entropy target, averaged over labelled steps, with
weight `distill_start` decaying linearly to `distill_min` by
`distill_decay_end` of training and zero after `distill_off` (the paper's
schedule; the defaults are its tenth, .1 -> .001). No LLM call is made.

Evaluation is teacher-free on held-out worlds (never in the training
pool): sampled actions, `eval_episodes` episodes capped at `eval_cap`
steps, every `eval_every` steps; the primary measure is mean achievements
per episode, its normalized area over training (`auc`) and Crafter's
score (geometric mean of achievement success rates).

    python -m algos.ppo_crafter --seed 1 --rule-bank <bank.json> --out <dir>
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from envs import crafter_symbolic as cs
from scripts import crafter_rules_pilot_20260928 as pilot
from scripts import crafter_rules_v2_20260930 as c2


@dataclass
class Args:
    seed: int = 1
    total_steps: int = 1_000_000
    num_envs: int = 16
    num_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rule_bank: str = ''
    distill_start: float = 0.1
    distill_min: float = 0.001
    distill_decay_end: float = 0.5
    distill_off: float = 0.75
    pool_size: int = 200
    world_seed0: int = 40_000_000
    eval_every: int = 50_000
    eval_episodes: int = 10
    eval_cap: int = 3000
    eval_seed0: int = 41_000_000
    out: str = 'results/crafter_runs/run'
    # The stronger, exploration-driven student (fix-wave-style Crafter wave,
    # research/crafter_vulcan_protocol_2026-10-02.md): an episodic count
    # bonus count_coef / sqrt(n) over the student's own 9x7 view, n its
    # visits this episode, as MiniGrid's count student counts its own view.
    # Added to the training reward (this trainer has one value head);
    # evaluation and the reported achievements use the game's reward only.
    # 0 (the default) is the plain student, unchanged.
    count_coef: float = 0.0


def layer(module, gain=np.sqrt(2)):
    nn.init.orthogonal_(module.weight, gain)
    nn.init.constant_(module.bias, 0.0)
    return module


class Net(nn.Module):
    """Embedding + two convolutions over the 9x7 view, an MLP with the
    inventory vector, and policy and value heads."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(cs.KINDS), 16)
        self.conv = nn.Sequential(
            layer(nn.Conv2d(16, 32, 3, padding=1)), nn.ReLU(),
            layer(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(), nn.Flatten())
        size = 32 * cs.GRID_SHAPE[0] * cs.GRID_SHAPE[1] + cs.VECTOR_SIZE
        self.body = nn.Sequential(layer(nn.Linear(size, 256)), nn.ReLU(),
                                  layer(nn.Linear(256, 256)), nn.ReLU())
        self.pi = layer(nn.Linear(256, cs.N_ACTIONS), 0.01)
        self.v = layer(nn.Linear(256, 1), 1.0)

    def forward(self, grid, vector):
        x = self.embed(grid).permute(0, 3, 1, 2)
        h = self.body(torch.cat([self.conv(x), vector], 1))
        return self.pi(h), self.v(h).squeeze(-1)


def distill_coef(args, fraction):
    if not args.rule_bank or fraction >= args.distill_off:
        return 0.0
    if fraction >= args.distill_decay_end:
        return args.distill_min
    span = fraction / args.distill_decay_end
    return args.distill_start + span * (args.distill_min - args.distill_start)


def view_key(observation):
    """The count student's state key: its 9x7 view of kinds, nothing else."""
    return observation[0].tobytes()


def stack(observations):
    grids = torch.as_tensor(np.stack([o[0] for o in observations]))
    vectors = torch.as_tensor(np.stack([o[1] for o in observations]))
    return grids, vectors


def evaluate(net, pool, args):
    """Teacher-free episodes on the held-out worlds, sampled actions."""
    rng = torch.Generator().manual_seed(args.seed)
    episodes = []
    for world in pool:
        env = cs.CrafterSymbolic([world], seed=0)
        obs = env.reset()
        steps, done = 0, False
        while not done and steps < args.eval_cap:
            with torch.no_grad():
                logits, _ = net(*stack([obs]))
            action = int(torch.multinomial(F.softmax(logits, -1), 1,
                                           generator=rng))
            obs, _, done, _ = env.step(action)
            steps += 1
        episodes.append(dict(steps=steps, achievements=env.achievements(),
                             died=bool(env.env._player.health <= 0)))
    score, rates = pilot.crafter_score(episodes)
    return dict(mean_achievements=float(np.mean(
        [len(e['achievements']) for e in episodes])), score=score,
        rates=rates, mean_steps=float(np.mean([e['steps']
                                               for e in episodes])),
        deaths=int(sum(e['died'] for e in episodes)))


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'args.json').write_text(json.dumps(asdict(args), indent=1))
    started = time.time()
    # The bank names its observer: v2 banks read the view and inventory
    # (pilot.observe, unchanged), v3 banks also the agent's own unlocked
    # achievements (crafter_rules_v3_20261001.observe_mem).
    from scripts.crafter_rules_v3_20261001 import load as load_bank
    rules, pre, observe = (load_bank(args.rule_bank) if args.rule_bank
                           else ([], False, pilot.observe))
    pool = cs.world_pool(args.world_seed0 + 1000 * args.seed + i
                         for i in range(args.pool_size))
    eval_pool = cs.world_pool(args.eval_seed0 + i
                              for i in range(args.eval_episodes))
    envs = [cs.CrafterSymbolic(pool, seed=args.seed * 1000 + i)
            for i in range(args.num_envs)]
    obs = [e.reset() for e in envs]
    # Episodic visit counts of each env's own view (count student only).
    visits = [{view_key(o): 1} for o in obs]
    bonus_total = 0.0
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
                                 eps=1e-5)
    batch = args.num_envs * args.num_steps
    minibatch = batch // args.num_minibatches
    updates = args.total_steps // batch
    shape = (args.num_steps, args.num_envs)
    grids = torch.zeros(shape + cs.GRID_SHAPE, dtype=torch.long)
    vectors = torch.zeros(shape + (cs.VECTOR_SIZE,))
    actions = torch.zeros(shape, dtype=torch.long)
    labels = torch.full(shape, -1, dtype=torch.long)
    logprobs, rewards, dones, values = (torch.zeros(shape) for _ in range(4))
    step, next_eval, evaluations = 0, args.eval_every, []
    labelled = asked = 0
    ep_log = (out / 'episodes.jsonl').open('w')
    ev_log = (out / 'evaluations.jsonl').open('w')
    done_now = torch.zeros(args.num_envs)
    for update in range(updates):
        fraction = step / args.total_steps
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * (1 - update / updates)
        coef = distill_coef(args, fraction)
        for t in range(args.num_steps):
            g, v = stack(obs)
            grids[t], vectors[t], dones[t] = g, v, done_now
            with torch.no_grad():
                logits, value = net(g, v)
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions[t], logprobs[t], values[t] = (action, dist.log_prob(
                action), value)
            if coef > 0:
                for i, env in enumerate(envs):
                    a, _status = c2.advise(rules, observe(env.env), pre)
                    labels[t, i] = -1 if a is None else a
                    asked += 1
                    labelled += a is not None
            else:
                labels[t] = -1
            new_done = torch.zeros(args.num_envs)
            for i, env in enumerate(envs):
                obs[i], reward, done, _ = env.step(int(action[i]))
                if args.count_coef > 0 and not done:
                    key = view_key(obs[i])
                    n = visits[i].get(key, 0) + 1
                    visits[i][key] = n
                    bonus = args.count_coef / np.sqrt(n)
                    reward += bonus
                    bonus_total += bonus
                rewards[t, i] = reward
                if done:
                    ep_log.write(json.dumps(dict(
                        step=step, steps=env.env._step,
                        achievements=env.achievements())) + '\n')
                    obs[i] = env.reset()
                    visits[i] = {view_key(obs[i]): 1}
                    new_done[i] = 1
            done_now = new_done
            step += args.num_envs
        with torch.no_grad():
            _, next_value = net(*stack(obs))
        advantages = torch.zeros(shape)
        last = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nonterminal, nxt = 1 - done_now, next_value
            else:
                nonterminal, nxt = 1 - dones[t + 1], values[t + 1]
            delta = rewards[t] + args.gamma * nxt * nonterminal - values[t]
            last = delta + args.gamma * args.gae_lambda * nonterminal * last
            advantages[t] = last
        returns = advantages + values
        b = dict(grid=grids.reshape((-1,) + cs.GRID_SHAPE),
                 vector=vectors.reshape(-1, cs.VECTOR_SIZE),
                 action=actions.reshape(-1), logprob=logprobs.reshape(-1),
                 adv=advantages.reshape(-1), ret=returns.reshape(-1),
                 label=labels.reshape(-1))
        for _ in range(args.update_epochs):
            order = torch.randperm(batch)
            for start in range(0, batch, minibatch):
                idx = order[start:start + minibatch]
                logits, value = net(b['grid'][idx], b['vector'][idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(b['action'][idx])
                         - b['logprob'][idx]).exp()
                adv = b['adv'][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(
                    1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((value - b['ret'][idx]) ** 2).mean()
                loss = (pg + args.vf_coef * v_loss
                        - args.ent_coef * dist.entropy().mean())
                mask = b['label'][idx] >= 0
                if coef > 0 and mask.any():
                    loss = loss + coef * F.cross_entropy(
                        logits[mask], b['label'][idx][mask])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()
        if step >= next_eval or update == updates - 1:
            result = dict(step=step, **evaluate(net, eval_pool, args))
            evaluations.append(result)
            ev_log.write(json.dumps(result) + '\n')
            ev_log.flush()
            ep_log.flush()
            next_eval += args.eval_every
    ep_log.close()
    ev_log.close()
    wall = time.time() - started
    summary = dict(
        status='completed', args=asdict(args), global_step=step,
        wall_time_sec=wall, sps=step / wall,
        auc=float(np.mean([e['mean_achievements'] for e in evaluations])),
        final=evaluations[-1], teacher=dict(
            asked=asked, labelled=labelled, llm_calls=0,
            rule_bank=args.rule_bank),
        count_bonus=dict(coef=args.count_coef, total=float(bonus_total)))
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=1))
    torch.save(net.state_dict(), out / 'final_model.pt')
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for f in fields(Args):
        cli.add_argument('--' + f.name.replace('_', '-'), type=type(f.default),
                         default=f.default)
    summary = train(Args(**vars(cli.parse_args())))
    print(json.dumps(dict(auc=summary['auc'], sps=round(summary['sps']),
                          final=summary['final']['mean_achievements'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
