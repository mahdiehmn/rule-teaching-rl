"""Offline feasibility check: can one conditional teacher rule label many states?

No training. The only API use is a small GPT-5-mini collection
through scripts/collect_prompt_reliability_20260927.py.

The candidate idea: a teacher consultation returns a CONDITIONAL rule
("WHEN <observable condition> PREFER <action> UNLESS <exception>")
instead of one action label. If the rule is right wherever its
condition holds, one paid consultation labels many student states,
which is the budget argument.

Everything checkable is computed by code:

- Rules may use only predicates the student can observe in its own 7x7
  view: the front cell, the carried item, and whether and where the key,
  door and goal are visible.
- Fresh DoorKey-8x8 states come from unseen layouts (seed 13,600,000+),
  varying position, facing, inventory and door state.
- Optimal action sets come from the existing BFS oracle.

Per rule, the check reports:

- coverage: fresh states where the rule applies;
- precision: of those, the fraction where the preferred action is
  optimal;
- exception validity;
- whether it is correct at its own consultation state.

Comparators:

- hand-written reference rules (comparison only);
- exact action replay (the consultation's label reused only on identical
  observations);
- permuted rules (conditions paired with another consultation's action).

Pipeline:

  --export    consultation states + requests.json/manifest (0 calls)
  (collect)   collector --split all (about 36 calls)
  --evaluate  offline scoring and a summary table
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from envs.state import extract_doorkey_state
from minigrid.core.world_object import Key
from teachers.minigrid.bfs_solver import MiniGridBFSTeacher
from teachers.minigrid.llm_general import render_ascii_map

STUDY = 'conditional_advice_check_20260927_v1'
CONDITION = 'conditional_rule_v1'
MODEL = 'gpt-5-mini-2025-08-07'
CONSULT_SEED, EVAL_SEED = 13_500_000, 13_600_000
ACTIONS = ('turn_left', 'turn_right', 'forward', 'pickup', 'drop', 'toggle',
           'done')
FRONT = ('empty', 'wall', 'key', 'door_locked', 'door_closed', 'door_open',
         'goal')
DIRS = ('ahead', 'left', 'right', 'none')
FIELDS = {'front': FRONT, 'carrying': ('nothing', 'key'),
          'key_dir': DIRS, 'door_dir': DIRS, 'goal_dir': DIRS}
HAND_RULES = (       # our own programming: the reference to beat
    ({'front': 'key', 'carrying': 'nothing'}, 3, None),
    ({'front': 'door_locked', 'carrying': 'key'}, 5, None),
    ({'front': 'door_closed'}, 5, None),
    ({'front': 'door_open'}, 2, None),
    ({'front': 'goal'}, 2, None),
    ({'carrying': 'nothing', 'key_dir': 'left'}, 0, None),
    ({'carrying': 'nothing', 'key_dir': 'right'}, 1, None),
    ({'carrying': 'key', 'door_dir': 'left'}, 0, ('front', 'door_open')),
    ({'carrying': 'key', 'door_dir': 'right'}, 1, ('front', 'door_open')),
    ({'goal_dir': 'left', 'front': 'empty'}, 0, None),
    ({'goal_dir': 'right', 'front': 'empty'}, 1, None),
)
ORACLE = MiniGridBFSTeacher(env_id='MiniGrid-DoorKey-8x8-v0')


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


# ------------------------------------------------------------ observation

def observe(u):
    """Observable predicates from the student's own 7x7 view."""
    img = u.gen_obs()['image']
    kind, _, state = map(int, img[3][5])
    front = {1: 'empty', 2: 'wall', 5: 'key', 8: 'goal'}.get(kind, 'empty')
    if kind == 4:
        front = ('door_open', 'door_closed', 'door_locked')[state]
    pred = dict(front=front, carrying=(
        'key' if isinstance(u.carrying, Key) else 'nothing'))
    for name, idx in (('key', 5), ('door', 4), ('goal', 8)):
        where = np.argwhere(img[:, :, 0] == idx)
        if len(where) == 0:
            pred[f'{name}_dir'] = 'none'
            continue
        x, y = where[0]
        forward, right = 6 - int(y), int(x) - 3
        pred[f'{name}_dir'] = ('ahead' if abs(forward) >= abs(right)
                               else 'right' if right > 0 else 'left')
    return pred


def image_key(u):
    return hashlib.sha256(u.gen_obs()['image'].tobytes()
                          + bytes([isinstance(u.carrying, Key)])).hexdigest()


def optimal(u):
    return tuple(ORACLE.optimal_actions(extract_doorkey_state(u)))


# ------------------------------------------------------------------ states

def _cells(u, kind):
    return [(x, y) for x in range(u.width) for y in range(u.height)
            if (c := u.grid.get(x, y)) is not None and c.type == kind]


def make_state(seed, rng, phase=None):
    """One fresh, valid DoorKey state with a chosen or random phase."""
    env = build_env('doorkey_8x8', seed=seed, obs_mode='symbolic')
    env.reset(seed=seed)
    u = env.unwrapped
    phase = phase or rng.choice(['key', 'door', 'goal'])
    (kx, ky), (dx, dy) = _cells(u, 'key')[0], _cells(u, 'door')[0]
    door = u.grid.get(dx, dy)
    if phase in ('door', 'goal'):
        u.grid.set(kx, ky, None)
        u.carrying = Key(door.color)
    if phase == 'goal':
        door.is_locked, door.is_open = False, True
    # reachable side: the start side unless the door is open
    side = [(x, y) for x in range(1, u.width - 1)
            for y in range(1, u.height - 1)
            if u.grid.get(x, y) is None and
            (phase == 'goal' or x < dx)]
    target = {'key': (kx, ky), 'door': (dx, dy),
              'goal': _cells(u, 'goal')[0]}[phase]
    near = [(x, y) for (x, y) in side
            if abs(x - target[0]) + abs(y - target[1]) == 1]
    if near and rng.random() < .35:             # enrich facing an object
        x, y = near[rng.integers(len(near))]
        u.agent_pos = np.array((x, y))
        vec = (target[0] - x, target[1] - y)
        u.agent_dir = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[vec]
    else:
        x, y = side[rng.integers(len(side))]
        u.agent_pos = np.array((x, y))
        u.agent_dir = int(rng.integers(4))
    return env, phase


def fresh_states(n, seed0, balanced=False):
    rng = np.random.default_rng(seed0)
    out = []
    for k in range(n):
        phase = ['key', 'door', 'goal'][k % 3] if balanced else None
        env, phase = make_state(seed0 + k, rng, phase)
        u = env.unwrapped
        out.append(dict(seed=seed0 + k, phase=phase, pred=observe(u),
                        image=image_key(u), optimal=optimal(u),
                        full_map=render_ascii_map(u),
                        pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                              int(u.agent_dir)]))
        env.close()
    return out


# ------------------------------------------------------------------- rules

def applies(rule, pred):
    condition, _, exception = rule
    if any(v != 'any' and pred[k] != v for k, v in condition.items()):
        return False
    return not (exception and pred[exception[0]] == exception[1])


def score(rule, states):
    applicable = [s for s in states if applies(rule, s['pred'])]
    correct = sum(rule[1] in s['optimal'] for s in applicable)
    condition, action, exception = rule
    blocked = [s for s in states if exception and all(
        v == 'any' or s['pred'][k] == v for k, v in condition.items())
        and s['pred'][exception[0]] == exception[1]]
    return dict(coverage=len(applicable), correct=correct,
                precision=correct / len(applicable) if applicable else None,
                exception_cases=len(blocked),
                exception_valid=(sum(action not in s['optimal']
                                     for s in blocked) / len(blocked)
                                 if blocked else None))


# ------------------------------------------------------------------ prompt

VOCAB = (
    'OBSERVABLE PREDICATES (computed from the student\'s own 7x7 view; the '
    'student can check nothing else):\n'
    '- front: the cell directly ahead: empty, wall, key, door_locked, '
    'door_closed, door_open, goal\n'
    '- carrying: nothing or key\n'
    '- key_dir / door_dir / goal_dir: where that object appears in the '
    'student view (ahead = forward offset dominates, ties ahead; left; '
    'right) or none if not visible\n'
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (the cell in '
    'front, without moving), 4 drop, 5 toggle (open/unlock the door in '
    'front; unlocking needs the matching key), 6 done. A key or a closed '
    'door in front blocks forward.\n')


def prompt(state):
    return (
        'You are the TEACHER for a DoorKey student (reach the green goal; '
        'the yellow key unlocks the locked door). You see the full map; the '
        'student sees only a 7x7 window ahead of it.\n'
        f'FULL MAP (x east, y south; arrow = agent):\n{state["full_map"]}\n'
        f'Agent pose (x, y, dir 0=east 1=south 2=west 3=north): '
        f'{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB +
        'The student can ask you rarely. Give ONE reusable rule it can apply '
        'on its own in other situations: WHEN a condition over the '
        'observable predicates holds, PREFER an action, UNLESS an optional '
        'exception predicate holds. The rule must be correct NOW and should '
        'stay correct in as many other states where its condition holds as '
        'possible; choose the most general condition that remains reliable. '
        'Use "any" for predicates the rule does not need. If no rule over '
        'these predicates is reliable here, set abstain=true.\n')


def schema():
    cond = {k: {'type': 'string', 'enum': ['any', *v]}
            for k, v in FIELDS.items()}
    return dict(
        type='object', additionalProperties=False,
        required=['abstain', 'condition', 'action', 'exception_field',
                  'exception_value', 'rationale'],
        properties=dict(
            abstain={'type': 'boolean'},
            condition=dict(type='object', additionalProperties=False,
                           required=list(FIELDS), properties=cond),
            action={'type': 'integer', 'enum': list(range(7))},
            exception_field={'type': 'string',
                             'enum': ['none', *FIELDS]},
            exception_value={'type': 'string', 'enum': sorted(
                {'none', *[v for vs in FIELDS.values() for v in vs]})},
            rationale={'type': 'string', 'maxLength': 240}))


def request(state):
    return dict(model=MODEL, store=False, service_tier='default',
                max_output_tokens=4096, reasoning={'effort': 'low'},
                input=[dict(role='user', content=prompt(state))],
                text={'format': dict(type='json_schema', strict=True,
                                     name='conditional_rule_v1',
                                     schema=schema())})


def export(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    consult = fresh_states(36, CONSULT_SEED, balanced=True)
    rows = []
    for k, state in enumerate(consult):
        body = request(state)
        rows.append(dict(case_id=f'consult_{k:03d}', condition=CONDITION,
                         split='consult', model=MODEL, request=body,
                         request_sha256=digest(body)))
    out.mkdir(parents=True)
    (out / 'consult_states.json').write_text(json.dumps(consult, indent=1))
    (out / 'requests.json').write_text(json.dumps(rows, indent=1))
    (out / 'manifest.json').write_text(json.dumps(dict(
        study=STUDY, requests_sha256=digest(rows), cases=len(rows),
        model=MODEL, api_calls=0), indent=1))
    (out / 'example_prompt.txt').write_text(prompt(consult[0]))
    print(f'Exported {len(rows)} consultation requests; 0 calls')


def parse(answer):
    if not answer or answer['abstain']:
        return None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exception = (None if answer['exception_field'] == 'none' or
                 answer['exception_value'] == 'none' else
                 (answer['exception_field'], answer['exception_value']))
    return condition, int(answer['action']), exception


def pooled(rules, states):
    scores = [score(r, states) for r in rules]
    cov = sum(s['coverage'] for s in scores)
    cor = sum(s['correct'] for s in scores)
    return dict(rules=len(rules), mean_coverage=cov / max(1, len(rules)),
                pooled_precision=cor / cov if cov else None,
                correct_labels_per_rule=cor / max(1, len(rules)))


def evaluate(out, n_eval):
    out = Path(out)
    consult = json.loads((out / 'consult_states.json').read_text())
    replies = {json.loads(line)['case_id']: json.loads(line) for line in
               (out / 'replies.jsonl').read_text().splitlines()}
    states = fresh_states(n_eval, EVAL_SEED)
    rules, rows, abstained, invalid = [], [], 0, 0
    for k, state in enumerate(consult):
        row = replies.get(f'consult_{k:03d}')
        if not row or row['response_status'] != 'completed':
            invalid += 1
            continue
        rule = parse(row['answer'])
        if rule is None:
            abstained += 1
            continue
        own = dict(applies=applies(rule, state['pred']),
                   optimal_here=rule[1] in state['optimal'])
        rules.append(rule)
        rows.append(dict(case=k, phase=state['phase'],
                         rule=[rule[0], ACTIONS[rule[1]], rule[2]],
                         rationale=row['answer']['rationale'], own=own,
                         **score(rule, states)))
    # exact action replay: the consultation's own optimal label, reused
    # only where the observation is identical
    replay = []
    for state in consult:
        same = [s for s in states if s['image'] == state['image']]
        label = state['optimal'][0]
        replay.append(dict(coverage=len(same), correct=sum(
            label in s['optimal'] for s in same)))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rules))
    while len(rules) > 1 and np.any(order == np.arange(len(rules))):
        order = rng.permutation(len(rules))
    permuted = [(rules[i][0], rules[j][1], rules[i][2])
                for i, j in enumerate(order)]
    hand_keys = {(tuple(sorted(c.items())), a) for c, a, _ in HAND_RULES}
    duplicates = sum((tuple(sorted(c.items())), a) in hand_keys
                     for c, a, _ in rules)
    cov = sum(r['coverage'] for r in replay)
    summary = dict(
        study=STUDY, eval_states=len(states),
        eval_phases=dict(Counter(s['phase'] for s in states)),
        consultations=len(consult), invalid=invalid, abstained=abstained,
        llm=pooled(rules, states),
        llm_correct_at_own_state=sum(r['own']['applies'] and
                                     r['own']['optimal_here'] for r in rows),
        llm_rules_equal_to_a_hand_rule=duplicates,
        permuted=pooled(permuted, states),
        hand_written=pooled(list(HAND_RULES), states),
        action_replay=dict(mean_coverage=cov / len(replay),
                           pooled_precision=(sum(r['correct'] for r in replay)
                                             / cov if cov else None)),
        per_rule=rows)
    (out / 'evaluation.json').write_text(json.dumps(summary, indent=1))
    printable = {k: v for k, v in summary.items() if k != 'per_rule'}
    print(json.dumps(printable, indent=1))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('action', choices=('export', 'evaluate', 'hand'))
    p.add_argument('--out', type=Path,
                   default=Path('results') / STUDY)
    p.add_argument('--n-eval', type=int, default=3000)
    args = p.parse_args()
    if args.action == 'export':
        export(args.out)
    elif args.action == 'evaluate':
        evaluate(args.out, args.n_eval)
    else:                        # hand-written reference only, no replies
        states = fresh_states(args.n_eval, EVAL_SEED)
        print(json.dumps(dict(
            eval_states=len(states),
            eval_phases=dict(Counter(s['phase'] for s in states)),
            hand_written=pooled(list(HAND_RULES), states),
            per_rule=[dict(rule=[c, ACTIONS[a], e], **score((c, a, e),
                                                            states))
                      for c, a, e in HAND_RULES]), indent=1))


if __name__ == '__main__':
    main()
