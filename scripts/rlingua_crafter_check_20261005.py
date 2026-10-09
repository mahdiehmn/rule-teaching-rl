"""Check A: can an LLM write a complete Crafter controller (RLingua recipe)?

Controller-level check only (no RL
training): GPT-5-mini writes a complete controller by RLingua's recipe
(Appendix B.A: phases from a fill-in template, then code for the exact
inputs, then automatic feedback for up to three rounds, keeping the best),
given the same Crafter task text our Crafter rules were written from
(scripts/crafter_rules_pilot_20260928.TASK_TEXT). Two information settings:

  full  the controller reads the whole 64x64 world (materials, creatures
        with positions, the player's position, inventory and vitals);
  view  it reads only what our student sees: the 9x7 local view, the
        inventory and vitals, facing, daylight, sleeping, plus the action
        actually executed and a memory of its own.

Feedback uses 8 worlds (seeds 41,000,000+); view controllers are also run
with a random action replacing theirs at 25% of the steps and selected on
those episodes first. The selected controller plays 20 fresh worlds
(seeds 42,000,000+) by itself, up to Crafter's 10,000 steps; we report
achievements unlocked per episode and Crafter's score, next to a random
policy on the same worlds. A development check, not a training input.

  python -m scripts.rlingua_crafter_check_20261005 --variant full
"""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time
import traceback

from scripts import crafter_rules_pilot_20260928 as pilot
from scripts import rlingua_controllers_20261005 as rc
from teachers.minigrid.rlingua_controller import load_controller

OUT = rc.OUT / 'checks_20261005' / 'crafter'
FEEDBACK_SEED = 41_000_000
EVAL_SEED = 42_000_000
N_FEEDBACK = 8
N_EVAL = 20
MAX_STEPS = 10_000
WALL_CAP = 300.0             # seconds per episode before it is cut short
MIX = 0.25
ROUNDS = 3
WORKERS = 5
DIRS = {(-1, 0): 'left', (1, 0): 'right', (0, -1): 'up', (0, 1): 'down'}

TASK = (pilot.TASK_TEXT
        .replace('You are the TEACHER for a Crafter student. ', '')
        .replace('The student should', 'The player should'))
INPUTS = {
    'full': (
        'def controller(state) -> int\n'
        'The controller input is one variable, `state`, a dict with:\n'
        "1. state['size'] = (64, 64); state['map'][y][x] is the material at "
        '(x, y): one of water, grass, stone, path, sand, tree, lava, coal, '
        'iron, diamond, table, furnace. x grows right, y grows down.\n'
        "2. state['objects']: a list of dicts {'kind': one of cow, zombie, "
        "skeleton, arrow, plant; 'pos': (x, y); 'health': int; 'ripe': "
        'bool (plants)}.\n'
        "3. state['player']: {'pos': (x, y), 'facing': 'left', 'right', "
        "'up' or 'down', 'inventory': a dict of the 16 counts health, "
        'food, drink, energy, sapling, wood, stone, coal, iron, diamond, '
        "wood_pickaxe, stone_pickaxe, iron_pickaxe, wood_sword, "
        "stone_sword, iron_sword; 'achievements': the list of unlocked "
        "achievement names; 'sleeping': bool}.\n"
        "4. state['daylight']: from 0 (night) to 1 (day)."),
    'view': (
        'def controller(obs, memory) -> int\n'
        'The controller inputs are two variables:\n'
        "1. obs['view']: the player's own 9x7 view, 7 rows of 9 names (row "
        '0 at the top, column 0 on the left; the player is at row 3, '
        "column 4 and shown as 'player'). Each cell is a material (water, "
        'grass, stone, path, sand, tree, lava, coal, iron, diamond, table, '
        'furnace), a creature or plant (cow, zombie, skeleton, arrow, '
        "plant, plant_ripe), or 'edge' outside the world.\n"
        "2. obs['inventory']: a dict of the 16 counts health, food, drink, "
        'energy, sapling, wood, stone, coal, iron, diamond, wood_pickaxe, '
        'stone_pickaxe, iron_pickaxe, wood_sword, stone_sword, '
        "iron_sword. obs['facing']: 'left', 'right', 'up' or 'down'. "
        "obs['daylight']: from 0 (night) to 1 (day). obs['sleeping']: "
        'bool.\n'
        "3. obs['last_action']: the action that was actually executed at "
        'the previous step (None at the first step).\n'
        '4. memory: a dict that persists between calls within one episode '
        'and is empty at its start; you may store anything in it.\n'
        'The agent is given nothing else: no map beyond its view, no '
        'position.\n'
        'IMPORTANT: during learning another policy also controls the agent. '
        'Your controller is called at every step, but the action it returns '
        'is executed only at some steps; at the others the other policy '
        "acts. Never assume your previous output was executed: obs"
        "['last_action'] tells you what was actually executed, so keep your "
        'memory consistent with it and with the current view.'),
}


def phases_prompt():
    return (
        'Think about you are an expert who would like to finish a task with '
        'an agent in a 2D game. Please provide a step-by-step description '
        'of how the agent should act in order to finish this task.\n'
        '[Start of the General Task Description]\n'
        f'{TASK}\n{pilot.ACTION_TEXT}'
        '[End of the General Task Description]\n'
        '[Start of the Question]\n'
        "How many phases can the agent's behaviour be divided into?\n"
        '[End of the Question]\n'
        '[Start of the Template]\n'
        'Phase [NUM]: The agent should [CHOICE: explore to find, go to, '
        'collect, place, make, drink, eat, sleep, defeat] [OBJ].\n'
        '[End of the Template]\n'
        'Rules:\n'
        '1. Please use the above template to answer, but do not include '
        '"[Start of the Template]" and "[End of the Template]" in your '
        'response.\n'
        '2. If you see phrases like [NUM], replace the entire phrase with an '
        'integer.\n'
        '3. If you see phrases like [OBJ], replace the entire phrase with an '
        'object or a place.\n'
        '4. If you see phrases like [CHOICE: choice1, choice2, ...], you '
        'should replace the entire phrase with one of the choices listed.\n'
        '5. You do not need to ensure that the task is finished '
        'successfully.')


def code_prompt(variant):
    return (
        'Can you write the detailed code for the controller of this '
        'process? You can write the staged controller in a nested if-else '
        'statement if necessary.\n' + INPUTS[variant] + '\n'
        'You must consider all the following information to make the task '
        'less ambiguous:\n'
        '1. The output of the controller must be one integer action from: '
        + pilot.ACTION_TEXT +
        '2. The controller is called once per time step; an episode lasts '
        'up to 10,000 steps or until the player dies, so keep the player '
        'alive and avoid repeating useless actions.\n'
        '3. You may use the Python standard library modules collections, '
        'heapq, math, random and itertools, and nothing else. Keep each '
        'call fast: it runs hundreds of thousands of times.\n'
        '4. You only need to write the detailed controller code and do not '
        'need to provide examples. Put the complete code, defining the '
        'function `controller` with exactly the signature above, in one '
        '```python code block.')


def full_state(env):
    world, player = env._world, env._player
    lut = [world._mat_names.get(i) for i in range(
        max(world._mat_names) + 1)]
    grid = [[lut[i] for i in row] for row in world._mat_map.T.tolist()]
    objects = []
    for obj in world.objects:
        kind = type(obj).__name__.lower()
        if kind == 'player':
            continue
        objects.append({'kind': kind, 'pos': (int(obj.pos[0]),
                                              int(obj.pos[1])),
                        'health': int(getattr(obj, 'health', 0)),
                        'ripe': bool(getattr(obj, 'ripe', False))})
    return {'size': tuple(int(v) for v in world.area), 'map': grid,
            'objects': objects,
            'player': {'pos': (int(player.pos[0]), int(player.pos[1])),
                       'facing': DIRS[tuple(int(v) for v in player.facing)],
                       'inventory': dict(player.inventory),
                       'achievements': sorted(
                           k for k, v in player.achievements.items() if v),
                       'sleeping': bool(player.sleeping)},
            'daylight': float(world.daylight)}


def view_obs(env, last):
    world, player = env._world, env._player
    px, py = (int(v) for v in player.pos)
    view = [['player' if (dx, dy) == (0, 0) else
             pilot.cell_kind(world, (px + dx, py + dy))
             for dx in range(-pilot.HALF_W, pilot.HALF_W + 1)]
            for dy in range(-pilot.HALF_H, pilot.HALF_H + 1)]
    return {'view': view, 'inventory': dict(player.inventory),
            'facing': DIRS[tuple(int(v) for v in player.facing)],
            'daylight': float(world.daylight),
            'sleeping': bool(player.sleeping), 'last_action': last}


def episode(job):
    """One world; job = (code or None for random, variant, seed, mix)."""
    code, variant, seed, mix = job
    fn = None if code is None else load_controller(code)
    rng = random.Random(seed * 7 + 3)
    env = pilot.make_env(seed)
    memory, last, steps, error, invalid = {}, None, 0, None, 0
    tail, spent, start, done, cut = deque(maxlen=200), 0.0, \
        time.perf_counter(), False, False
    while not done and steps < MAX_STEPS:
        if fn is None:
            action = rng.randrange(len(pilot.ACTIONS))
        else:
            t0 = time.perf_counter()
            try:
                action = (fn(full_state(env)) if variant == 'full'
                          else fn(view_obs(env, last), memory))
                action = int(action)
            except Exception as exc:
                frames = traceback.extract_tb(exc.__traceback__)
                line = next((f.lineno for f in reversed(frames)
                             if f.filename == '<rlingua_controller>'), None)
                error = f'{type(exc).__name__}: {exc} (line {line})'[:240]
                break
            spent += time.perf_counter() - t0
            if not 0 <= action < len(pilot.ACTIONS):
                invalid += 1
                action = 0
            if mix and rng.random() < mix:
                action = rng.randrange(len(pilot.ACTIONS))
        tail.append(action)
        _, _, done, _ = env.step(action)
        last = action
        steps += 1
        if time.perf_counter() - start > WALL_CAP:
            cut = True
            break
    unlocked = sorted(k for k, v in env._player.achievements.items() if v)
    return dict(seed=seed, steps=steps, died=bool(env._player.health <= 0),
                achievements=unlocked, error=error, invalid=invalid,
                cut=cut, ms_per_step=spent / max(1, steps) * 1e3,
                tail=dict(Counter(pilot.ACTIONS[a] for a in tail)))


def play(pool, code, variant, seeds, mix=0.0):
    return list(pool.map(episode, [(code, variant, s, mix) for s in seeds]))


def describe(eps):
    n = len(eps)
    counts = Counter(a for e in eps for a in e['achievements'])
    mean = sum(len(e['achievements']) for e in eps) / n
    text = (f'it unlocked {mean:.1f} of 22 achievements per world on '
            f'average ({", ".join(f"{a} {c}/{n}" for a, c in sorted(counts.items(), key=lambda x: -x[1])) or "none"}); '
            f'it died in {sum(e["died"] for e in eps)} of {n} worlds '
            f'(survived {sum(e["steps"] for e in eps) / n:.0f} steps on '
            'average).')
    errors = [e for e in eps if e['error']]
    if errors:
        text += (f' In {len(errors)} worlds the code raised an error, for '
                 f'example: {errors[0]["error"]}.')
    cut = [e for e in eps if e['cut']]
    if cut:
        text += (f' In {len(cut)} worlds it was too slow to finish (about '
                 f'{cut[0]["ms_per_step"]:.0f} ms per step).')
    if eps[0]['tail']:
        text += (' In a typical world its last 200 actions were '
                 f'{eps[0]["tail"]}.')
    return mean, text


def feedback(alone, mixed=None):
    _, text = describe(alone)
    msg = f'We ran your controller alone in {len(alone)} worlds: {text}'
    if mixed is not None:
        _, m_text = describe(mixed)
        msg += (f' In {len(mixed)} other worlds, as during learning, another '
                f'policy acted instead of it at about {MIX:.0%} of the steps '
                f"(obs['last_action'] reported what was executed): {m_text}")
    return (msg + ' Can you fine-tune the code above by incorporating this '
            'information? Return the complete updated code in one ```python '
            'block.')


def summary(eps):
    score, rates = pilot.crafter_score(eps)
    return dict(episodes=len(eps), score=round(score, 2),
                mean_achievements=round(sum(len(e['achievements'])
                                            for e in eps) / len(eps), 2),
                deaths=sum(e['died'] for e in eps),
                mean_steps=round(sum(e['steps'] for e in eps) / len(eps)),
                errors=sum(bool(e['error']) for e in eps),
                cut=sum(e['cut'] for e in eps),
                ms_per_step=round(sum(e['ms_per_step'] for e in eps)
                                  / len(eps), 3),
                rates={k: v for k, v in rates.items() if v})


def run(dotenv, variant):
    OUT.mkdir(parents=True, exist_ok=True)
    client = rc.client_from(dotenv)
    log, rounds = [], []
    messages = [{'role': 'user', 'content': phases_prompt()}]
    reply, _ = rc.ask(client, messages, log)
    messages += [{'role': 'assistant', 'content': reply},
                 {'role': 'user', 'content': code_prompt(variant)}]
    seeds = range(FEEDBACK_SEED, FEEDBACK_SEED + N_FEEDBACK)
    best = None
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for round_ in range(ROUNDS + 1):
            reply, usd = rc.ask(client, messages, log)
            try:
                code = rc.extract_code(reply)
                load_controller(code)
                alone = play(pool, code, variant, seeds)
                mixed = (play(pool, code, variant, seeds, MIX)
                         if variant == 'view' else None)
            except Exception as exc:
                code, score = None, (-1.0, -1.0)
                note = (f'Your reply could not be used: '
                        f'{type(exc).__name__}: {exc}. Return the complete '
                        'code defining `controller` in one ```python block.')
            else:
                a_mean, _ = describe(alone)
                m_mean = describe(mixed)[0] if mixed else None
                score = (a_mean,) if mixed is None else (m_mean, a_mean)
                note = feedback(alone, mixed)
            rounds.append(dict(round=round_, score=score, feedback=note,
                               code_sha256=None if code is None
                               else rc.sha(code)))
            print(f'crafter {variant} round {round_}: score {score} '
                  f'(spent ${sum(c["usd"] for c in log):.3f})', flush=True)
            if code is not None and (best is None or score > best[0]):
                best = (score, code, round_)
            if round_ == ROUNDS:
                break
            messages += [{'role': 'assistant', 'content': reply},
                         {'role': 'user', 'content': note}]
        score, code, round_ = best
        (OUT / f'crafter_{variant}.py').write_text(code, encoding='utf-8')
        eval_seeds = range(EVAL_SEED, EVAL_SEED + N_EVAL)
        result = dict(variant=variant, selected_round=round_,
                      feedback_score=score,
                      alone=summary(play(pool, code, variant, eval_seeds)),
                      random_policy=summary(play(pool, None, variant,
                                                 eval_seeds)))
        if variant == 'view':
            result['mixed'] = summary(play(pool, code, variant, eval_seeds,
                                           MIX))
    result['usd'] = round(sum(c['usd'] for c in log), 4)
    (OUT / f'crafter_{variant}_receipt.json').write_text(json.dumps(dict(
        result, messages=messages, calls=log, rounds=rounds,
        controller_sha256=rc.sha(code)), indent=1), encoding='utf-8')
    print(f'crafter {variant} RESULT {json.dumps(result)}', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--variant', choices=('full', 'view'), required=True)
    cli.add_argument('--dotenv', type=Path,
                     default=Path('.env'))
    args = cli.parse_args()
    run(args.dotenv, args.variant)


if __name__ == '__main__':
    main()
