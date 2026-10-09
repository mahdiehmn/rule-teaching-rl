"""Do the frozen rule banks still give correct labels on larger maps?

Running the rules AS A POLICY on larger maps mostly fails (DoorKey
16x16 .34; KeyCorridor S4R3/S5R3/S6R3 .03/.01/.00), but a teacher's job is
to label the states the student visits, and MultiRoom's checked rules
finished no episode alone yet sped learning up a lot. So this measures, on
each map size, the precision (labels on an optimal action) and coverage
(share of states labelled) of the unchanged banks, against the exact
oracles: DoorKey's (conditional_advice_check.optimal) and KeyCorridor's
Dijkstra over pose, key and doors (conditional_rules_keycorridor.optimal).

States are sampled like the panels: a layout's oracle route is replayed to
a random point, half the time followed by a short random walk over the
useful actions; states with no route to success are skipped. Seeds
37,000,000+ are used nowhere else.

    python -m scripts.larger_map_labels_20261001 [--layouts N]
"""

import argparse
import importlib
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers the MiniGrid and BabyAI ids)
import numpy as np

from scripts import conditional_rules_keycorridor as kc
from scripts import conditional_rules_v3 as v3
from teachers.minigrid.rule_bank import OBSERVERS, STATEFUL, load_bank

SEED0 = 37_000_000
DK_BANK = 'research/rule_banks/v3_20260928/blind_strict.json'
KC_MEM = ('research/rule_banks/keycorridor_mem_20260929/'
          'self_checked_pooled_valid.json')
KC_V1 = 'research/rule_banks/keycorridor_20260928/scoped_valid.json'
CASES = (
    ('DoorKey-8x8', 'MiniGrid-DoorKey-8x8-v0', 'doorkey', (DK_BANK,)),
    ('DoorKey-16x16', 'MiniGrid-DoorKey-16x16-v0', 'doorkey', (DK_BANK,)),
    ('KeyCorridor-S3R3', 'BabyAI-KeyCorridorS3R3-v0', 'keycorridor',
     (KC_MEM, KC_V1)),
    ('KeyCorridor-S4R3', 'BabyAI-KeyCorridorS4R3-v0', 'keycorridor',
     (KC_MEM, KC_V1)),
    ('KeyCorridor-S5R3', 'BabyAI-KeyCorridorS5R3-v0', 'keycorridor',
     (KC_MEM, KC_V1)),
    ('KeyCorridor-S6R3', 'BabyAI-KeyCorridorS6R3-v0', 'keycorridor',
     (KC_MEM, KC_V1)))
WALK = (0, 1, 2, 3, 4, 5)
# The KeyCorridor oracle searches pose x key x opened doors; on the
# largest maps a route takes 20-100 s, so fewer layouts there.
LAYOUTS = {'KeyCorridor-S5R3': 20, 'KeyCorridor-S6R3': 12}
PREFER = (2, 5, 3, 4, 0, 1)        # forward, toggle, pickup, drop, turns


_ORACLES = {}


def oracle(kind, env_id):
    """The exact optimal-action set; DoorKey's BFS is built per map size."""
    if kind != 'doorkey':
        return kc.optimal
    if env_id not in _ORACLES:
        from envs.state import extract_doorkey_state
        from teachers.minigrid.bfs_solver import MiniGridBFSTeacher
        bfs = MiniGridBFSTeacher(env_id=env_id)
        _ORACLES[env_id] = lambda u: tuple(bfs.optimal_actions(
            extract_doorkey_state(u)))
    return _ORACLES[env_id]


def fresh(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def route(env_id, seed, optimal):
    env = fresh(env_id, seed)
    u, actions = env.unwrapped, []
    for _ in range(u.max_steps):
        opt = optimal(u)
        if not opt:
            break
        a = next(a for a in PREFER if a in opt)
        actions.append(a)
        _, reward, done, trunc, _ = env.step(a)
        if done or trunc:
            env.close()
            return actions if done and reward > 0 else None
    env.close()
    return None


def states(env_id, kind, layouts, per_layout, rng):
    """(env, optimal actions) pairs; the env is left at the sampled state."""
    optimal = oracle(kind, env_id)
    for i in range(layouts):
        seed = SEED0 + i
        actions = route(env_id, seed, optimal)
        if not actions:
            continue
        for _ in range(per_layout):
            env = fresh(env_id, seed)
            done = False
            for a in actions[:int(rng.integers(len(actions)))]:
                env.step(a)
            if rng.random() < .5:
                for _ in range(int(rng.integers(1, 7))):
                    _, _, done, trunc, _ = env.step(int(rng.choice(WALK)))
                    if done or trunc:
                        break
            opt = () if done else optimal(env.unwrapped)
            if opt:
                yield env, opt
            env.close()


def observer(bank_path):
    name = json.loads(Path(bank_path).read_text(encoding='utf-8')).get(
        'observer', 'doorkey_v3')
    module, fn = OBSERVERS[name]
    return getattr(importlib.import_module(module), fn), name in STATEFUL


def measure(layouts=60, per_layout=8, seed=0):
    rng = np.random.default_rng(seed)
    result = {}
    for label, env_id, kind, banks in CASES:
        loaded = {b: (load_bank(b)[1], *observer(b)) for b in banks}
        counts = {b: dict(labelled=0, correct=0) for b in banks}
        total = 0
        for env, opt in states(env_id, kind, LAYOUTS.get(label, layouts),
                               per_layout, rng):
            u = env.unwrapped
            img = u.gen_obs()['image']
            total += 1
            for b, (rules, observe, stateful) in loaded.items():
                action, status = v3.advise(
                    rules, observe(img, u) if stateful else observe(img))
                if status == 'advised':
                    counts[b]['labelled'] += 1
                    counts[b]['correct'] += action in opt
        result[label] = {Path(b).parent.name + '/' + Path(b).stem: dict(
            states=total, coverage=c['labelled'] / max(total, 1),
            precision=c['correct'] / c['labelled'] if c['labelled'] else None,
            correct=c['correct'], labelled=c['labelled'])
            for b, c in counts.items()}
        for bank, r in result[label].items():
            print(f"{label:17s} {bank:56s} states {r['states']:4d}  "
                  f"coverage {r['coverage']:.2f}  precision "
                  f"{(r['precision'] or 0):.3f}", flush=True)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--layouts', type=int, default=60)
    cli.add_argument('--per-layout', type=int, default=8)
    cli.add_argument('--out', type=Path, default=Path(
        'docs/assets/rule_speed_2026-09-30/larger_map_labels.json'))
    args = cli.parse_args()
    result = measure(args.layouts, args.per_layout)
    args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
