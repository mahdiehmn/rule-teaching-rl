"""Check B: do full-map example situations help the LLM write a working
complete controller from the student's view?

Our rule banks were written by an LLM that
was shown example situations with the full map ("you see the full state;
the student sees only its 7x7 view"). RLingua's recipe shows its LLM only
the task description and test-episode feedback. This check removes that
difference for the view controllers: the same RLingua recipe
(scripts/rlingua_controllers_20261005.py: phases, code, automatic feedback
alone and with another policy acting, up to three rounds, selection on the
mixed episodes first), with one added message before the code request:
36 example situations, each with the full map (which the agent never sees)
and the agent's own 7x7 view. Situations come from fresh layouts (seeds
40,000,000+) paused after 0-24 steps of the full-state controller. The
selected controller is scored on the same 100 layouts as the others
(seeds 37,000,000+), alone and with another policy at 25% of the steps.
Development check, not a training input.

  python -m scripts.rlingua_examples_check_20261005 --tasks dk mr kc
"""

import argparse
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from scripts import rlingua_controllers_20261005 as rc
from scripts.conditional_rules_v3 import render_ascii_map
from teachers.minigrid.rlingua_controller import RLinguaController

OUT = rc.OUT / 'checks_20261005' / 'examples'
SEED = 40_000_000
N_EXAMPLES = 36
CHAR = {0: '?', 1: '.', 2: '#', 3: '.', 5: 'k', 6: 'o', 7: 'b', 8: 'G',
        9: '~', 10: '^'}
DOOR = {0: '/', 1: 'D', 2: 'L'}
COLOR = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
NAME = {4: 'door', 5: 'key', 6: 'ball', 7: 'box'}


def view_text(u):
    """The agent's 7x7 view as text, row 0 farthest ahead."""
    img = u.gen_obs()['image']
    rows, objects = [], []
    for j in range(7):
        row = ''
        for i in range(7):
            obj, color, state = (int(v) for v in img[i][j])
            if (i, j) == (3, 6):
                row += '^'
                continue
            row += DOOR[state] if obj == 4 else CHAR.get(obj, '?')
            if obj in NAME:
                kind = NAME[obj] + (
                    f' ({("open", "closed", "locked")[state]})'
                    if obj == 4 else '')
                objects.append(f'{COLOR[color]} {kind} at column {i}, '
                               f'row {j}')
        rows.append(row)
    carrying = ('nothing' if u.carrying is None else
                f'{u.carrying.color} {u.carrying.type}')
    return '\n'.join(rows), objects, carrying


def situations(task, key):
    """N_EXAMPLES fresh layouts, each paused mid-task, rendered."""
    rng = np.random.default_rng(SEED)
    full = RLinguaController(rc.OUT / f'{key}_full.py', 'full')
    texts = []
    for k in range(N_EXAMPLES):
        seed = SEED + k
        env = build_env(task, seed=seed, obs_mode='symbolic')
        env.reset(seed=seed)
        u = env.unwrapped
        full.reset(0)
        for _ in range(int(rng.integers(0, 25))):
            _, _, term, trunc, _ = env.step(full.act(0, u))
            if term or trunc:
                env.reset(seed=seed)
                full.reset(0)
                break
        view, objects, carrying = view_text(u)
        texts.append(
            f'SITUATION {k + 1}:\nFULL MAP (the agent never sees this; '
            f'arrow = agent):\n{render_ascii_map(u)}\n'
            f"AGENT'S 7x7 VIEW (what obs['image'] shows; row 0 is farthest "
            f"ahead, '^' is the agent facing up, '?' unseen, '#' wall, "
            f"'.' floor, 'D' closed door, 'L' locked door, '/' open door, "
            f"'k' key, 'o' ball, 'b' box, 'G' goal):\n{view}\n"
            f'VISIBLE OBJECTS: {"; ".join(objects) or "none"}\n'
            f'CARRYING: {carrying}\n')
        env.close()
    return texts


def run(dotenv, key):
    task, text = rc.TASKS[key]
    OUT.mkdir(parents=True, exist_ok=True)
    client = rc.client_from(dotenv)
    log, rounds = [], []
    messages = [{'role': 'user', 'content': rc.phases_prompt(text)}]
    reply, _ = rc.ask(client, messages, log)
    examples = situations(task, key)
    messages += [
        {'role': 'assistant', 'content': reply},
        {'role': 'user', 'content': (
            'Before writing the code, here are example situations from '
            'this task. Each shows the full map, which the agent never '
            "sees, and the agent's own 7x7 view, which is all your "
            'controller will receive (with obs[\'last_action\'] and your '
            'memory). Use them to understand what the agent can and cannot '
            'see.\n\n' + '\n'.join(examples) + '\n' + rc.code_prompt('view'))}]
    seeds = range(rc.FEEDBACK_SEED, rc.FEEDBACK_SEED + rc.FEEDBACK_EPISODES)
    best = None
    for round_ in range(rc.ROUNDS + 1):
        reply, usd = rc.ask(client, messages, log)
        mixed = None
        try:
            code = rc.extract_code(reply)
            rc.load_controller(code)
            wins, records = rc.run_episodes(code, task, 'view', seeds)
            mixed, m_records = rc.run_episodes(code, task, 'view', seeds,
                                               mix=rc.MIX)
        except Exception as exc:
            code, wins = None, -1
            note = (f'Your reply could not be used: {type(exc).__name__}: '
                    f'{exc}. Return the complete code defining `controller` '
                    'in one ```python block.')
        else:
            note = rc.feedback(wins, records, (mixed, m_records))
        score = (-1, -1) if code is None else (mixed, wins)
        rounds.append(dict(round=round_, wins=wins, mixed_wins=mixed,
                           code_sha256=None if code is None else rc.sha(code),
                           feedback=note))
        print(f'{key}_view+examples round {round_}: {wins}/20 alone, '
              f'{mixed}/20 mixed', flush=True)
        if code is not None and (best is None or score > best[0]):
            best = (score, code, round_)
        if score[0] >= rc.TARGET * rc.FEEDBACK_EPISODES or \
                round_ == rc.ROUNDS:
            break
        messages += [{'role': 'assistant', 'content': reply},
                     {'role': 'user', 'content': note}]
    score, code, round_ = best
    (OUT / f'{key}_view.py').write_text(code, encoding='utf-8')
    eval_seeds = range(rc.EVAL_SEED, rc.EVAL_SEED + rc.EVAL_EPISODES)
    alone, _ = rc.run_episodes(code, task, 'view', eval_seeds)
    mixed, _ = rc.run_episodes(code, task, 'view', eval_seeds, mix=rc.MIX)
    result = dict(task=task, selected_round=round_,
                  feedback_alone=score[1], feedback_mixed=score[0],
                  success_alone=alone / rc.EVAL_EPISODES,
                  success_mixed=mixed / rc.EVAL_EPISODES,
                  usd=round(sum(c['usd'] for c in log), 4))
    (OUT / f'{key}_view_receipt.json').write_text(json.dumps(dict(
        result, messages=messages, calls=log, rounds=rounds,
        controller_sha256=rc.sha(code)), indent=1), encoding='utf-8')
    print(f'{key}_view+examples RESULT {json.dumps(result)}', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tasks', nargs='+', default=['dk', 'mr', 'kc'])
    cli.add_argument('--dotenv', type=Path,
                     default=Path('.env'))
    args = cli.parse_args()
    for key in args.tasks:
        run(args.dotenv, key)


if __name__ == '__main__':
    main()
