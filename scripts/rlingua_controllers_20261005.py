"""Generate RLingua controllers for our MiniGrid tasks with GPT-5-mini.

RLingua (Chen et al., RA-L 2024) extracts
a COMPLETE rule-based controller from an LLM by prompting (Appendix B.A):
(1) ask for the task's phases with a fill-in template, (2) ask for the
controller code given the exact input variables and constraints, (3) refine
with feedback. We follow that recipe with the same model and the same task
knowledge our rule banks were written from. RLingua's step (3) uses a
non-expert's feedback; here the feedback is generated automatically from
test episodes (success count, raised errors, where the agent got stuck),
which is reproducible and involves no researcher judgement.

Two information settings, one controller each per task:
  full  the controller reads the complete simulator state (RLingua's
        setting: its controllers read the full robot/object state);
  view  the controller reads exactly the student's 7x7 view, plus a
        memory dict of its own (information-matched to our rules).

Feedback episodes use layouts from seed 36,000,000 upward; the reported
standalone success uses 100 layouts from seed 37,000,000 upward. Neither
block is used by training (training seeds are 14.5M-14.7M + 100r) or by the
teacher-free evaluation (training seed + 50,000).

A controller that is correct but too slow for training (it is called up to
about 250,000 times per run) gets speed-only feedback: its decisions are
not criticised, only its time per call (`speed`). A stored speed reply can
be installed later under a stated per-call limit after re-checking its
success and speed (`accept`). Every reply, accepted or not, is kept in the
receipt.

  python -m scripts.rlingua_controllers_20261005 generate --dotenv <.env>
  python -m scripts.rlingua_controllers_20261005 evaluate
  python -m scripts.rlingua_controllers_20261005 speed --name kc_full
  python -m scripts.rlingua_controllers_20261005 accept --name kc_full \\
      --attempt 3 --limit-ms 60
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import traceback

from envs.registry import build_env
from teachers.minigrid.rlingua_controller import (ACTIONS, full_state,
                                                  load_controller, view_obs)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/rlingua_controllers_20261005'
MODEL = 'gpt-5-mini'
EFFORT = 'medium'
PRICE = (0.25, 2.00)            # USD per 1M input / output tokens
BUDGET = 3.00                   # hard stop for the whole generation
ROUNDS = 3                      # feedback rounds after the first code
FEEDBACK_EPISODES = 20
FEEDBACK_SEED = 36_000_000
EVAL_EPISODES = 100
EVAL_SEED = 37_000_000
TARGET = 0.95                   # stop refining once feedback success >= this

ACTION_TEXT = (
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (the object '
    'in front), 4 drop (the carried object onto the empty cell in front), '
    '5 toggle (open/close the door in front; unlocking a locked door needs '
    'the key of the same colour), 6 done (does nothing useful here). A '
    'wall, a closed or locked door, a key, a ball or a box in front blocks '
    'forward.')
TASKS = {
    'dk': ('doorkey_8x8',
           'The agent must reach the green goal square. A wall with one '
           'LOCKED door splits the grid into two rooms; the agent and the '
           'key of the door\'s colour start in one room, the goal is in the '
           'other. The agent carries at most one object: it must pick up the '
           'key, face the locked door and toggle it to unlock and open it, '
           'then go through the doorway to the goal. Reaching the goal ends '
           'the episode with success.'),
    'mr': ('multiroom_n6',
           'Six rooms are connected in a chain by doors; the agent must '
           'reach the green goal square in the last room. Doors are never '
           'locked: toggle opens a closed door in front (and closes an open '
           'one). Reaching the goal ends the episode with success.'),
    'kc': ('keycorridor_s3r3',
           'A corridor has small side rooms behind doors. The agent must '
           'pick up the ball. The ball is behind a LOCKED door; the key of '
           'the same colour is in another room, possibly behind a closed '
           'door. The agent carries at most one object: it must carry the '
           'key to unlock the door (toggle while facing it), then drop the '
           'key on an empty cell in front before it can pick up the ball. '
           'Picking up the ball ends the episode with success.'),
}
INPUTS = {
    'full': (
        'def controller(state) -> int\n'
        'The controller input is one variable, `state`, a dict with:\n'
        "1. state['grid']: a list of rows; state['grid'][y][x] is None for "
        "empty floor, or a dict {'type': one of 'wall', 'door', 'key', "
        "'ball', 'box', 'goal', 'lava'; 'color': one of 'red', 'green', "
        "'blue', 'purple', 'yellow', 'grey'}; doors also have 'is_open' and "
        "'is_locked' (booleans).\n"
        "2. state['width'], state['height']: the grid size; x grows east "
        '(to the right), y grows south (down).\n'
        "3. state['agent_pos']: (x, y) of the agent (its own cell holds "
        'whatever it stands on, normally None or an open door).\n'
        "4. state['agent_dir']: 0 facing east (+x), 1 south (+y), 2 west "
        '(-x), 3 north (-y).\n'
        "5. state['carrying']: None, or {'type', 'color'} of the object the "
        'agent holds.\n'
        "6. state['mission']: the mission text."),
    'view': (
        'def controller(obs, memory) -> int\n'
        'The controller inputs are two variables:\n'
        "1. obs['image']: the agent's own 7x7 egocentric view, a nested list "
        'image[i][j] = [object, color, state] with i = column 0..6 from '
        'left to right and j = row 0..6 from top (far) to bottom (near). '
        'The agent is always at image[3][6] facing up (towards j = 0); the '
        'cell directly in front is image[3][5]. The agent cannot see '
        'through walls or closed doors: cells it cannot see have object 0.\n'
        '   object: 0 unseen, 1 empty, 2 wall, 3 floor, 4 door, 5 key, '
        '6 ball, 7 box, 8 goal, 9 lava, 10 agent.\n'
        '   color: 0 red, 1 green, 2 blue, 3 purple, 4 yellow, 5 grey.\n'
        '   state (doors only): 0 open, 1 closed, 2 locked.\n'
        '   image[3][6] shows the object the agent carries (object 1 = '
        'nothing carried).\n'
        "2. obs['last_action']: the action that was actually executed at "
        'the previous step of this episode (None at its first step).\n'
        '3. memory: a dict that persists between calls within one episode '
        'and is empty at the start of each episode; you may store anything '
        'in it (for example what has been done so far).\n'
        'The agent is given nothing else: no map, no position, no '
        'direction.\n'
        'IMPORTANT: during learning another policy also controls the agent. '
        'Your controller is called at every step, but the action it returns '
        'is executed only at some steps; at the others the other policy '
        "acts. Never assume your previous output was executed: obs"
        "['last_action'] tells you what was actually executed, so keep your "
        'memory consistent with it and with the current view, and re-plan '
        'from the current view whenever needed.'),
}
# Feedback for a view controller also tests it as it will be used: at
# each step a random action is executed instead of its own with this
# probability (seeded per episode).
MIX = 0.25


def phases_prompt(task_text):
    """RLingua Appendix B.A, first user turn, adapted to a grid world."""
    return (
        'Think about you are an expert who would like to finish a task with '
        'an agent in a grid world. Please provide a step-by-step description '
        'of how the agent should act in order to finish this task.\n'
        '[Start of the General Task Description]\n'
        f'{task_text}\n{ACTION_TEXT}\n'
        '[End of the General Task Description]\n'
        '[Start of the Question]\n'
        "How many phases can the agent's behaviour be divided into?\n"
        '[End of the Question]\n'
        '[Start of the Template]\n'
        'Phase [NUM]: The agent should [CHOICE: explore to find, go to, '
        'pick up, drop, open, unlock, go through] [OBJ].\n'
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
    """RLingua Appendix B.A, second user turn."""
    return (
        'Can you write the detailed code for the controller of this '
        'process? You can write the staged controller in a nested if-else '
        'statement if necessary.\n' + INPUTS[variant] + '\n'
        'You must consider all the following information to make the task '
        'less ambiguous:\n'
        '1. The output of the controller must be one integer action from: '
        + ACTION_TEXT + '\n'
        '2. forward moves one cell in the facing direction only if that '
        'cell is empty floor, an open door or the goal.\n'
        '3. The controller is called once per time step and the episode has '
        'a limited number of steps, so avoid repeating useless actions.\n'
        '4. You may use the Python standard library modules collections, '
        'heapq, math, random and itertools, and nothing else.\n'
        '5. You only need to write the detailed controller code and do not '
        'need to provide examples. Put the complete code, defining the '
        'function `controller` with exactly the signature above, in one '
        '```python code block.')


def extract_code(text):
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, flags=re.S)
    if not blocks:
        raise ValueError('reply contains no code block')
    return max(blocks, key=len).strip() + '\n'


def run_episodes(code, task, variant, seeds, mix=0.0):
    """Run the controller (source `code`, a fresh instance per episode);
    with `mix` > 0 a random action is executed instead of the
    controller's with that probability at each step. Return (successes,
    per-episode records)."""
    records, wins = [], 0
    for seed in seeds:
        fn = load_controller(code)
        rng = random.Random(seed * 7 + 1)
        env = build_env(task, seed=seed, obs_mode='symbolic')
        env.reset(seed=seed)
        u = env.unwrapped
        memory, history, error, invalid = {}, [], None, 0
        success, last = False, None
        for _ in range(u.max_steps):
            try:
                action = (fn(full_state(u)) if variant == 'full'
                          else fn(view_obs(u, last), memory))
                action = int(action)
            except Exception as exc:
                tb = traceback.extract_tb(exc.__traceback__)
                line = next((f.lineno for f in reversed(tb)
                             if f.filename == '<rlingua_controller>'), None)
                error = f'{type(exc).__name__}: {exc} (line {line})'[:240]
                break
            if not 0 <= action < len(ACTIONS):
                invalid += 1
                action = 0
            if mix and rng.random() < mix:
                action = rng.randrange(6)      # the other policy acts
            front = u.grid.get(*u.front_pos)
            history.append(dict(pos=[int(u.agent_pos[0]),
                                     int(u.agent_pos[1])],
                                dir=int(u.agent_dir), action=action,
                                front=None if front is None else front.type,
                                carrying=None if u.carrying is None
                                else u.carrying.type))
            last = action
            _, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                success = bool(terminated and reward > 0)
                break
        env.close()
        wins += success
        records.append(dict(seed=seed, success=success, error=error,
                            invalid=invalid, steps=len(history),
                            tail=history[-20:]))
    return wins, records


def feedback(wins, records, mixed=None):
    """Non-expert feedback written from test episodes, without judgement.
    `mixed` = (wins, records) of the episodes in which another policy
    acted at some steps (view controllers); its failures are described."""
    n = len(records)
    lines = [f'We tested your controller in {n} episodes with different '
             f'random layouts. It succeeded in {wins} of {n}.']
    if mixed is not None:
        m_wins, m_records = mixed
        lines = [f'We tested your controller in {n} episodes with different '
                 f'random layouts in which it controlled the agent alone: it '
                 f'succeeded in {wins} of {n}. In {len(m_records)} other '
                 'episodes, as during learning, another policy acted instead '
                 f'of it at about {MIX:.0%} of the steps (obs[\'last_action\'] '
                 'reported what was executed): it succeeded in '
                 f'{m_wins} of {len(m_records)}. What follows describes those '
                 'episodes.']
        records = m_records
    errors = [r for r in records if r['error']]
    if errors:
        lines.append(f'In {len(errors)} episodes the code raised an error, '
                     f'for example: {errors[0]["error"]}.')
    invalid = sum(r['invalid'] for r in records)
    if invalid:
        lines.append(f'It returned {invalid} invalid actions (not an '
                     'integer 0-6).')
    stuck = [r for r in records if not r['success'] and not r['error']]
    if stuck:
        tail = stuck[0]['tail']
        acts = Counter(ACTIONS[t['action']] for t in tail)
        cells = {tuple(t['pos']) for t in tail}
        fronts = Counter(str(t['front']) for t in tail)
        carrying = tail[-1]['carrying'] if tail else None
        lines.append(
            f'In {len(stuck)} episodes the agent ran out of steps. In a '
            f'typical one, during its last {len(tail)} steps it stayed '
            f'within {len(cells)} cells, used the actions '
            f'{dict(acts.most_common())}, mostly with '
            f'{fronts.most_common(1)[0][0]} in front, carrying {carrying}.')
    lines.append('Can you fine-tune the code above by incorporating this '
                 'information? Return the complete updated code in one '
                 '```python block.')
    return ' '.join(lines)


def client_from(dotenv):
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import dotenv_values
    from openai import OpenAI
    key = (dotenv_values(dotenv, encoding='utf-8-sig', interpolate=False)
           .get('OPENAI_API_KEY') or '')
    if not key.strip():
        raise ValueError('OPENAI_API_KEY is missing')
    return OpenAI(api_key=key.strip(), timeout=600, max_retries=0)


def ask(client, messages, log):
    t0 = time.perf_counter()
    r = client.responses.create(model=MODEL, input=messages,
                                reasoning={'effort': EFFORT},
                                max_output_tokens=25_000)
    usd = (r.usage.input_tokens / 1e6 * PRICE[0]
           + r.usage.output_tokens / 1e6 * PRICE[1])
    log.append(dict(model=r.model, status=r.status,
                    tokens_in=r.usage.input_tokens,
                    tokens_out=r.usage.output_tokens, usd=round(usd, 6),
                    seconds=round(time.perf_counter() - t0, 1)))
    if r.status != 'completed':
        raise RuntimeError(f'response status {r.status}')
    return r.output_text, usd


def sha(text):
    return hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest()


def generate(dotenv, tasks, variants):
    client = client_from(dotenv)
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    feedback_seeds = range(FEEDBACK_SEED, FEEDBACK_SEED + FEEDBACK_EPISODES)
    for key in tasks:
        task, text = TASKS[key]
        for variant in variants:
            name = f'{key}_{variant}'
            if (OUT / f'{name}.py').exists():
                print(f'{name}: frozen already, skipped')
                continue
            log, rounds = [], []
            messages = [{'role': 'user', 'content': phases_prompt(text)}]
            reply, usd = ask(client, messages, log)
            spent += usd
            messages += [{'role': 'assistant', 'content': reply},
                         {'role': 'user', 'content': code_prompt(variant)}]
            best = None
            for round_ in range(ROUNDS + 1):
                if spent > BUDGET:
                    raise RuntimeError(f'budget ${BUDGET} exceeded')
                reply, usd = ask(client, messages, log)
                spent += usd
                mixed_wins = None
                try:
                    code = extract_code(reply)
                    load_controller(code)
                    wins, records = run_episodes(code, task, variant,
                                                 feedback_seeds)
                    if variant == 'view':
                        mixed_wins, mixed_records = run_episodes(
                            code, task, variant, feedback_seeds, mix=MIX)
                except Exception as exc:
                    code, wins, records = None, -1, []
                    note = (f'Your reply could not be used: {type(exc).__name__}'
                            f': {exc}. Return the complete code defining '
                            '`controller` in one ```python block.')
                else:
                    note = feedback(wins, records, None if mixed_wins is None
                                    else (mixed_wins, mixed_records))
                # A view controller is chosen by how it does when another
                # policy also acts (as in training), then alone.
                score = (wins,) if mixed_wins is None else (mixed_wins, wins)
                rounds.append(dict(round=round_, wins=wins,
                                   mixed_wins=mixed_wins,
                                   code_sha256=None if code is None
                                   else sha(code), feedback=note))
                print(f'{name} round {round_}: {wins}/{FEEDBACK_EPISODES}'
                      + ('' if mixed_wins is None else
                         f', mixed {mixed_wins}/{FEEDBACK_EPISODES}')
                      + f' (spent ${spent:.3f})')
                if code is not None and (best is None or score > best[0]):
                    best = (score, code, round_)
                if score[0] >= TARGET * FEEDBACK_EPISODES or round_ == ROUNDS:
                    break
                messages += [{'role': 'assistant', 'content': reply},
                             {'role': 'user', 'content': note}]
            if best is None:
                raise RuntimeError(f'{name}: no usable controller')
            score, code, round_ = best
            shown = (f'{score[0]}/{FEEDBACK_EPISODES} feedback episodes'
                     if len(score) == 1 else
                     f'{score[1]}/{FEEDBACK_EPISODES} alone, {score[0]}/'
                     f'{FEEDBACK_EPISODES} with another policy acting')
            header = (f'# RLingua controller {name} ({task}, {variant} '
                      f'information). Written by {MODEL} following RLingua '
                      f'Appendix B.A;\n# selected round {round_} '
                      f'({shown}). '
                      'Generated by scripts/rlingua_controllers_20261005.py.'
                      '\n')
            (OUT / f'{name}.py').write_text(header + code, encoding='utf-8')
            (OUT / f'{name}_receipt.json').write_text(json.dumps(dict(
                task=task, variant=variant, model=MODEL, effort=EFFORT,
                generated=datetime.now(timezone.utc).isoformat(
                    timespec='seconds'),
                messages=messages, calls=log, rounds=rounds,
                selected_round=round_, selected_wins=score[-1],
                selected_mixed_wins=None if len(score) == 1 else score[0],
                controller_sha256=sha(header + code)), indent=1),
                encoding='utf-8')
    print(f'total spent ${spent:.4f}')
    return spent


def refine(dotenv, name, extra=3, below=0.5, note=None):
    """Extra feedback rounds for a view controller whose success with
    another policy acting stays below `below` on the feedback episodes
    (one rule for every view controller): continue its conversation from
    the selected version for up to `extra` rounds and keep the best of
    all its rounds. With `note` (a defect a reviewer found), that note is
    the first feedback instead, whatever the controller's success."""
    key, variant = name.split('_')
    task, _ = TASKS[key]
    defect_note = note
    path, receipt_path = OUT / f'{name}.py', OUT / f'{name}_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    best_score = (receipt['selected_mixed_wins'], receipt['selected_wins'])
    if variant != 'view' or (note is None and
                             best_score[0] >= below * FEEDBACK_EPISODES):
        print(f'{name}: no refinement under the rule')
        return False
    seeds = range(FEEDBACK_SEED, FEEDBACK_SEED + FEEDBACK_EPISODES)
    code = path.read_text(encoding='utf-8').split('\n', 2)[2]
    if sha(code) != receipt['rounds'][receipt['selected_round']][
            'code_sha256']:
        raise RuntimeError(f'{name}: file differs from its selected round')
    wins, records = run_episodes(code, task, variant, seeds)
    m_wins, m_records = run_episodes(code, task, variant, seeds, mix=MIX)
    messages = receipt['messages'] + [
        {'role': 'assistant', 'content': f'```python\n{code}```'},
        {'role': 'user', 'content': note or feedback(wins, records,
                                                     (m_wins, m_records))}]
    client = client_from(dotenv)
    rounds = receipt['rounds']
    best = (best_score, code, receipt['selected_round'])
    for _ in range(extra):
        reply, usd = ask(client, messages, receipt['calls'])
        round_ = len(rounds)
        mixed_wins = None
        try:
            new = extract_code(reply)
            load_controller(new)
            wins, records = run_episodes(new, task, variant, seeds)
            mixed_wins, m_records = run_episodes(new, task, variant, seeds,
                                                 mix=MIX)
        except Exception as exc:
            new, wins = None, -1
            note = (f'Your reply could not be used: {type(exc).__name__}: '
                    f'{exc}. Return the complete code defining `controller` '
                    'in one ```python block.')
        else:
            note = feedback(wins, records, (mixed_wins, m_records))
        rounds.append(dict(round=round_, wins=wins, mixed_wins=mixed_wins,
                           refinement=True, usd=round(usd, 6),
                           code_sha256=None if new is None else sha(new),
                           feedback=note))
        print(f'{name} refinement round {round_}: {wins}/'
              f'{FEEDBACK_EPISODES}, mixed {mixed_wins}/{FEEDBACK_EPISODES}'
              f' (${usd:.4f})')
        score = (-1, -1) if new is None else (mixed_wins, wins)
        if new is not None and score > best[0]:
            best = (score, new, round_)
        messages += [{'role': 'assistant', 'content': reply}]
        if score[0] >= TARGET * FEEDBACK_EPISODES:
            break
        messages += [{'role': 'user', 'content': note}]
    score, code, round_ = best
    history = receipt.get('refinements', [])
    if isinstance(receipt.get('refinement'), dict):   # first format
        history.append(receipt.pop('refinement'))
    history.append(dict(extra=extra, note=defect_note) if defect_note else
                   dict(rule=f'mixed success below {below:.0%} on the '
                        'feedback episodes', extra=extra))
    receipt['refinements'] = history
    receipt['messages'] = messages
    if round_ != receipt['selected_round']:
        header = (f'# RLingua controller {name} ({task}, {variant} '
                  f'information). Written by {MODEL} following RLingua '
                  f'Appendix B.A;\n# selected round {round_} '
                  f'({score[1]}/{FEEDBACK_EPISODES} alone, {score[0]}/'
                  f'{FEEDBACK_EPISODES} with another policy acting). '
                  'Generated by scripts/rlingua_controllers_20261005.py.\n')
        path.write_text(header + code, encoding='utf-8')
        receipt.update(selected_round=round_, selected_wins=score[1],
                       selected_mixed_wins=score[0],
                       controller_sha256=sha(header + code))
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    return True


def time_per_call(code, task, variant, steps=300, seed=FEEDBACK_SEED):
    """Mean seconds per controller call along one feedback layout."""
    fn = load_controller(code)
    env = build_env(task, seed=seed, obs_mode='symbolic')
    env.reset(seed=seed)
    u, memory, total, n, last = env.unwrapped, {}, 0.0, 0, None
    while n < steps:
        t0 = time.perf_counter()
        action = (fn(full_state(u)) if variant == 'full'
                  else fn(view_obs(u, last), memory))
        total += time.perf_counter() - t0
        n += 1
        last = int(action) % 6
        _, _, terminated, truncated, _ = env.step(last)
        if terminated or truncated:
            seed += 1
            env.reset(seed=seed)
            fn, memory, last = load_controller(code), {}, None
    env.close()
    return total / n


def speed_round(dotenv, name, limit_ms=10.0, rounds=3):
    """Speed-only feedback for a correct but slow controller (training calls
    it hundreds of thousands of times). Up to `rounds` replies; the first
    that keeps the success target and runs under `limit_ms` per call
    replaces the controller, otherwise the slow controller stays."""
    key, variant = name.split('_')
    task, _ = TASKS[key]
    path = OUT / f'{name}.py'
    receipt_path = OUT / f'{name}_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    attempts = receipt.get('speed_rounds', [])
    if 'speed_round' in receipt:          # an earlier single, rejected try
        attempts.append(dict(receipt.pop('speed_round'),
                             note='reply text not stored'))
    old_code = path.read_text(encoding='utf-8')
    old_ms = time_per_call(old_code, task, variant, steps=100) * 1e3
    seeds = range(FEEDBACK_SEED, FEEDBACK_SEED + FEEDBACK_EPISODES)
    old_wins = receipt['selected_wins']
    need = TARGET * FEEDBACK_EPISODES
    note = (f'Your controller succeeded in {old_wins} of {FEEDBACK_EPISODES} '
            f'test episodes, but it is far too slow: about {old_ms:.0f} ms '
            'per call, and it is called once per time step, hundreds of '
            'thousands of times. Please make it at least 100 times faster '
            'while keeping its decisions, for example by not copying or '
            'serializing the whole grid inside the search. Return the '
            'complete updated code in one ```python block.')
    messages = receipt['messages'] + [
        {'role': 'assistant', 'content': f'```python\n{old_code}```'},
        {'role': 'user', 'content': note}]
    client = client_from(dotenv)
    chosen = None
    for round_ in range(rounds):
        reply, usd = ask(client, messages, receipt['calls'])
        try:
            code = extract_code(reply)
            load_controller(code)
            wins, _ = run_episodes(code, task, variant, seeds)
            ms = time_per_call(code, task, variant) * 1e3
        except Exception as exc:
            code, wins, ms = None, -1, None
            follow = (f'Your reply could not be used: {type(exc).__name__}: '
                      f'{exc}. Return the complete code defining '
                      '`controller` in one ```python block.')
        else:
            follow = speed_follow(wins, ms, old_wins, limit_ms)
        ok = code is not None and wins >= need and ms <= limit_ms
        attempts.append(dict(round=len(attempts), wins=wins, ms=ms,
                             usd=round(usd, 6), accepted=ok, reply=reply,
                             feedback=None if ok else follow,
                             code_sha256=None if code is None else sha(code)))
        print(f'{name} speed round {round_}: {wins}/{FEEDBACK_EPISODES}, '
              f'{old_ms:.1f} -> {ms} ms/call (${usd:.4f})')
        messages += [{'role': 'assistant', 'content': reply}]
        if ok:
            chosen = (code, wins, ms)
            break
        messages += [{'role': 'user', 'content': follow}]
    receipt['speed_rounds'] = attempts
    receipt['speed_baseline'] = dict(feedback=note, wins=old_wins,
                                     ms=old_ms, sha256=sha(old_code),
                                     code=old_code)
    if chosen:
        install(path, receipt, messages, *chosen)
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    return chosen is not None


def speed_follow(wins, ms, old_wins, limit_ms):
    return (f'This version succeeded in {wins} of {FEEDBACK_EPISODES} test '
            f'episodes and takes about {ms:.1f} ms per call. It must succeed '
            f'as often as before ({old_wins} of {FEEDBACK_EPISODES}) and take '
            f'under {limit_ms:.0f} ms per call. Return the complete updated '
            'code in one ```python block.')


def install(path, receipt, messages, code, wins, ms):
    name = path.stem
    header = (f'# RLingua controller {name} ({receipt["task"]}, '
              f'{receipt["variant"]} information). Written by {MODEL} '
              f'following RLingua Appendix B.A;\n# selected round '
              f'{receipt["selected_round"]} ({receipt["selected_wins"]}/'
              f'{FEEDBACK_EPISODES}), then speed-only feedback '
              f'({wins}/{FEEDBACK_EPISODES}, {ms:.2f} ms/call). '
              'Generated by scripts/rlingua_controllers_20261005.py.\n')
    path.write_text(header + code, encoding='utf-8')
    receipt['messages'] = messages
    receipt['controller_sha256'] = sha(header + code)


def accept_speed(name, index, limit_ms):
    """Install a stored speed-round reply under a stated per-call limit,
    after re-checking its success and speed (no API call)."""
    key, variant = name.split('_')
    task, _ = TASKS[key]
    path = OUT / f'{name}.py'
    receipt_path = OUT / f'{name}_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    attempts, base = receipt['speed_rounds'], receipt['speed_baseline']
    old_code = path.read_text(encoding='utf-8')
    if sha(old_code) != base['sha256']:
        raise RuntimeError(f'{name} changed since its speed rounds')
    base['code'] = old_code
    messages = receipt['messages'] + [
        {'role': 'assistant', 'content': f'```python\n{old_code}```'},
        {'role': 'user', 'content': base['feedback']}]
    first = next(i for i, a in enumerate(attempts) if a.get('reply'))
    for i in range(first, index + 1):
        a = attempts[i]
        messages.append({'role': 'assistant', 'content': a['reply']})
        if i < index:
            messages.append({'role': 'user', 'content': a.get('feedback') or
                             speed_follow(a['wins'], a['ms'], base['wins'],
                                          10.0)})
    code = extract_code(attempts[index]['reply'])
    seeds = range(FEEDBACK_SEED, FEEDBACK_SEED + FEEDBACK_EPISODES)
    wins, _ = run_episodes(code, task, variant, seeds)
    ms = time_per_call(code, task, variant) * 1e3
    print(f'{name} attempt {index}: {wins}/{FEEDBACK_EPISODES}, '
          f'{ms:.1f} ms/call (limit {limit_ms} ms)')
    if wins < TARGET * FEEDBACK_EPISODES or ms > limit_ms:
        raise RuntimeError('attempt does not meet the stated limits')
    attempts[index].update(accepted=True, accepted_limit_ms=limit_ms,
                           recheck=dict(wins=wins, ms=ms))
    install(path, receipt, messages, code, wins, ms)
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding='utf-8')


# The family's larger map, served by the same controller in training.
LARGER = {'dk': 'doorkey_16x16', 'mr': 'multiroom_n10',
          'kc': 'keycorridor_s4r3'}


def evaluate(tasks, variants, larger=False):
    """Success on 100 fresh layouts (reported, not tuned on): alone, and
    for a view controller also with another policy acting at MIX of the
    steps. `larger` scores the family's larger map instead (the
    full-state KeyCorridor controller is too slow there and is not run on
    it in training, so it is skipped)."""
    target = OUT / 'standalone_success.json'
    result = (json.loads(target.read_text(encoding='utf-8'))
              if target.exists() else {})
    for key in tasks:
        task = LARGER[key] if larger else TASKS[key][0]
        for variant in variants:
            name = f'{key}_{variant}'
            path = OUT / f'{name}.py'
            if not path.exists() or (larger and name == 'kc_full'):
                continue
            code = path.read_text(encoding='utf-8')
            seeds = range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES)
            wins, records = run_episodes(code, task, variant, seeds)
            row = dict(task=task, variant=variant,
                       success=wins / EVAL_EPISODES,
                       errors=sum(bool(r['error']) for r in records),
                       invalid=sum(r['invalid'] for r in records),
                       controller_sha256=sha(code))
            if variant == 'view':
                m_wins, _ = run_episodes(code, task, variant, seeds, mix=MIX)
                row['mixed_success'] = m_wins / EVAL_EPISODES
                row['mix'] = MIX
            result[f'{name}@{task}' if larger else name] = row
            print(name, task, row)
    target.write_text(json.dumps(result, indent=1, sort_keys=True) + '\n',
                      encoding='utf-8')
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('generate', 'evaluate', 'speed',
                                        'accept', 'refine', 'defect'))
    cli.add_argument('--note-file', type=Path,
                     help='defect: the reviewer-found defect to report')
    cli.add_argument('--name', help='controller for the speed round')
    cli.add_argument('--attempt', type=int, help='speed attempt to accept')
    cli.add_argument('--limit-ms', type=float, default=10.0)
    cli.add_argument('--dotenv', type=Path,
                     default=Path('.env'))
    cli.add_argument('--tasks', nargs='+', default=list(TASKS))
    cli.add_argument('--variants', nargs='+', default=['full', 'view'])
    cli.add_argument('--larger', action='store_true',
                     help="evaluate on the families' larger maps")
    cli.add_argument('--no-eval', action='store_true',
                     help='generate only (lets generations run in parallel)')
    args = cli.parse_args()
    if args.action == 'speed':
        speed_round(args.dotenv, args.name, limit_ms=args.limit_ms)
        return
    if args.action == 'accept':
        accept_speed(args.name, args.attempt, args.limit_ms)
        return
    if args.action == 'refine':
        refine(args.dotenv, args.name)
        return
    if args.action == 'defect':
        refine(args.dotenv, args.name, extra=2,
               note=args.note_file.read_text(encoding='utf-8').strip())
        return
    if args.action == 'generate':
        generate(args.dotenv, args.tasks, args.variants)
        if args.no_eval:
            return
    evaluate(args.tasks, args.variants, larger=args.larger)


if __name__ == '__main__':
    main()
