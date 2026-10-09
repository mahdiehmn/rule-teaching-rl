"""Scoped LLM rules with a blind check on MultiRoom-N6: a second environment.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The method is frozen from DoorKey, so
nothing here is tuned on MultiRoom:

- same GPT-5-mini model, low effort, one attempt per request;
- same rule form and three-valued scope semantics (conditions observed,
  exceptions observed false, unknown never assumed, conflicts abstain);
- same self-check and blind check on up to 5 pool situations per rule;
- same selection: keep a rule only if every blind answer equals its
  action (strict).

Only the environment-specific parts are new:

- student predicates from the 7x7 view: front cell, the nearest visible
  door and its state, the nearest visible CLOSED door and the goal;
- states along the shortest route through the six rooms (doors opened
  as the route opens them), a third of them from each stretch, half
  followed by a short random walk;
- an exact optimal-action oracle (Dijkstra over position and heading;
  entering a closed door costs toggle + forward). It only SCORES; it
  never writes, repairs or selects a rule.

Layouts: 300 distinct reset maps split 36 / 132 / 132 by a salted hash
into consultation, development and confirmation. The experience pool
comes from the consultation layouts only.

Pipeline:
  build           panels and consultation requests (0 calls)
  (collect)       consult_requests -> consult_replies
  export-checks   self-check and blind-check requests (0 calls)
  (collect)       refine_requests -> refine_replies,
                  blind_requests -> blind_replies
  compare         score every method on development or confirmation
  banks           frozen banks for the learning study (0 calls)
"""

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from scripts import conditional_rules_v3 as v3
from teachers.minigrid.llm_general import render_ascii_map

STUDY = 'conditional_rules_multiroom_20260928'
TASK = 'multiroom_n6'
MODEL = v3.MODEL
ACTIONS = v3.v1.ACTIONS
UNKNOWN = v3.UNKNOWN
FRONT = ('empty', 'wall', 'door_closed', 'door_open', 'goal', 'unseen')
OBJECTS = ('door', 'closed_door', 'goal')
ATTRS = v3.ATTRS
FIELDS = {'front': FRONT, 'door_state': ('closed', 'open')}
for _o in OBJECTS:
    FIELDS[f'{_o}_visible'] = ('yes', 'no')
    FIELDS.update({f'{_o}_{a}': vals for a, vals in ATTRS.items()})
SPLIT = dict(consult=36, develop=132, confirm=132)
PER_LAYOUT = 12
PANEL_SEED0 = 20_000_000
K_SITUATIONS = v3.K_SITUATIONS
LEFT, RIGHT, FORWARD, TOGGLE = 0, 1, 2, 5
DX, DY = (1, 0, -1, 0), (0, 1, 0, -1)
PREFER = (FORWARD, TOGGLE, LEFT, RIGHT)
STAGES = ('early', 'middle', 'late')
BANKS = Path('research/rule_banks/multiroom_20260928')
OUT = Path('results') / STUDY


def digest(value):
    return v3.digest(value)


# ---------------------------------------------------------------- observe

def observe_mr(img):
    """Student predicates from the 7x7 view (agent at (3, 6), facing up)."""
    kind, _, state = map(int, img[3][5])
    front = {0: 'unseen', 1: 'empty', 2: 'wall', 8: 'goal'}.get(kind,
                                                                  'empty')
    if kind == 4:
        front = 'door_open' if state == 0 else 'door_closed'
    seen = dict(door=[], closed_door=[], goal=[])
    for x in range(7):
        for y in range(7):
            if (x, y) == (3, 6):
                continue
            k = int(img[x][y][0])
            if k == 4:
                seen['door'].append((x, y))
                if int(img[x][y][2]) != 0:
                    seen['closed_door'].append((x, y))
            elif k == 8:
                seen['goal'].append((x, y))
    pred = dict(front=front, door_state=UNKNOWN)
    for name, where in seen.items():
        if not where:
            pred[f'{name}_visible'] = 'no'
            pred.update({f'{name}_{a}': UNKNOWN for a in ATTRS})
            continue
        x, y = min(where, key=lambda c: (6 - c[1] + abs(c[0] - 3),
                                         6 - c[1], c[0]))
        fwd, right = 6 - y, x - 3
        pred.update({f'{name}_visible': 'yes', f'{name}_fwd': str(fwd),
                     f'{name}_right': str(right),
                     f'{name}_side': ('left' if right < 0 else 'right'
                                      if right > 0 else 'center'),
                     f'{name}_ahead': 'beside' if fwd == 0 else 'ahead'})
        if name == 'door':
            pred['door_state'] = 'open' if int(img[x][y][2]) == 0 \
                else 'closed'
    return pred


# ----------------------------------------------------------------- oracle

def _standable(u, x, y):
    if not (0 <= x < u.width and 0 <= y < u.height):
        return False
    c = u.grid.get(x, y)
    return c is None or c.type == 'door'


def distances(u):
    """Fewest actions to the goal from every (x, y, dir), doors as now."""
    gx, gy = next((x, y) for x in range(u.width) for y in range(u.height)
                  if (c := u.grid.get(x, y)) is not None
                  and c.type == 'goal')
    heap = [(1, gx - DX[d], gy - DY[d], d) for d in range(4)
            if _standable(u, gx - DX[d], gy - DY[d])]
    heapq.heapify(heap)
    dist = {}
    while heap:
        v, x, y, d = heapq.heappop(heap)
        if (x, y, d) in dist:
            continue
        dist[(x, y, d)] = v
        for before in ((d + 1) % 4, (d - 1) % 4):      # left / right
            if (x, y, before) not in dist:
                heapq.heappush(heap, (v + 1, x, y, before))
        px, py = x - DX[d], y - DY[d]                   # forward
        if _standable(u, px, py) and (px, py, d) not in dist:
            c = u.grid.get(x, y)
            closed = c is not None and c.type == 'door' and not c.is_open
            heapq.heappush(heap, (v + (2 if closed else 1), px, py, d))
    return dist


def optimal(u):
    """Every action on some shortest route to the goal (scoring only)."""
    dist = distances(u)
    x, y, d = int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)
    here = dist.get((x, y, d))
    if here is None:
        return ()
    inf = float('inf')
    out = []
    if dist.get((x, y, (d - 1) % 4), inf) + 1 == here:
        out.append(LEFT)
    if dist.get((x, y, (d + 1) % 4), inf) + 1 == here:
        out.append(RIGHT)
    fx, fy = x + DX[d], y + DY[d]
    c = u.grid.get(fx, fy)
    if c is not None and c.type == 'goal':
        out.append(FORWARD)
    elif c is None or (c.type == 'door' and c.is_open):
        if dist.get((fx, fy, d), inf) + 1 == here:
            out.append(FORWARD)
    elif c.type == 'door' and dist.get((fx, fy, d), inf) + 2 == here:
        out.append(TOGGLE)
    return tuple(sorted(out))


# ----------------------------------------------------------------- panels

def make_env(seed):
    env = build_env(TASK, seed=seed, obs_mode='symbolic')
    env.reset(seed=seed)
    return env


def layout_id(u):
    cells = [(x, y, c.type) for x in range(u.width)
             for y in range(u.height) if (c := u.grid.get(x, y)) is not None]
    return digest(sorted(cells))[:16]


def route(seed):
    """The oracle's shortest route from the reset state, or None."""
    env = make_env(seed)
    actions = []
    for _ in range(env.unwrapped.max_steps):
        opt = optimal(env.unwrapped)
        if not opt:
            break
        a = next(a for a in PREFER if a in opt)
        actions.append(a)
        _, reward, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            env.close()
            return actions if terminated and reward > 0 else None
    env.close()
    return None


def native_key(u):
    return digest([u.grid.encode().tolist(),
                   [int(u.agent_pos[0]), int(u.agent_pos[1])],
                   int(u.agent_dir)])


def teacher_facts(u):
    facts = []
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None or c.type not in ('door', 'goal'):
                continue
            row = dict(kind=c.type, color=c.color, x=x, y=y)
            if c.type == 'door':
                row['state'] = 'open' if c.is_open else 'closed'
            facts.append(row)
    return facts


def state_record(u, panel, layout, seed, stage):
    img = u.gen_obs()['image']
    return dict(panel=panel, layout=layout, seed=seed, phase=stage,
                native=native_key(u), pred=observe_mr(img),
                image=hashlib.sha256(img.tobytes()).hexdigest(),
                optimal=optimal(u), full_map=render_ascii_map(u),
                facts=teacher_facts(u),
                pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                      int(u.agent_dir)])


def make_state(seed, actions, rng, stage=None):
    """A state along the route, in a given stretch, maybe walked off it."""
    stage = stage or STAGES[int(rng.integers(3))]
    n, k = len(actions), STAGES.index(stage)
    lo, hi = k * n // 3, max(k * n // 3 + 1, (k + 1) * n // 3)
    while True:
        env = make_env(seed)
        for a in actions[:int(rng.integers(lo, hi))]:
            env.step(a)
        done = False
        if rng.random() < .5:
            for _ in range(int(rng.integers(1, 7))):
                _, _, done, _, _ = env.step(int(rng.choice(
                    [LEFT, RIGHT, FORWARD, TOGGLE])))
                if done:
                    break
        if not done:
            return env, stage
        env.close()


def all_layouts(n=sum(SPLIT.values()), limit=5_000):
    """The first n distinct solvable reset maps from the panel seeds."""
    found = {}
    for s in range(PANEL_SEED0, PANEL_SEED0 + limit):
        env = make_env(s)
        lid = layout_id(env.unwrapped)
        env.close()
        if lid in found:
            continue
        actions = route(s)
        if actions:
            found[lid] = dict(seed=s, route=actions)
        if len(found) == n:
            return found
    raise ValueError(f'Found only {len(found)} layouts')


def build_panels(seed=0):
    layouts = all_layouts()
    order = sorted(layouts, key=lambda lid: digest([STUDY, lid]))
    cut = np.cumsum([SPLIT[k] for k in ('consult', 'develop', 'confirm')])
    parts = dict(consult=order[:cut[0]], develop=order[cut[0]:cut[1]],
                 confirm=order[cut[1]:])
    rng = np.random.default_rng(seed)
    panels = dict(consult=[], pool=[], develop=[], confirm=[])
    for k, lid in enumerate(parts['consult']):
        item = layouts[lid]
        env, stage = make_state(item['seed'], item['route'], rng,
                                STAGES[k % 3])
        panels['consult'].append(state_record(
            env.unwrapped, 'consult', lid, item['seed'], stage))
        env.close()
    for name, source in (('pool', parts['consult']),
                         ('develop', parts['develop']),
                         ('confirm', parts['confirm'])):
        seen = set()
        for lid in source:
            item = layouts[lid]
            for _ in range(PER_LAYOUT):
                env, stage = make_state(item['seed'], item['route'], rng)
                rec = state_record(env.unwrapped, name, lid, item['seed'],
                                   stage)
                env.close()
                if rec['native'] not in seen:
                    seen.add(rec['native'])
                    panels[name].append(rec)
    used = {s['native'] for s in panels['consult'] + panels['pool']}
    for name in ('develop', 'confirm'):
        before = len(panels[name])
        panels[name] = [s for s in panels[name] if s['native'] not in used]
        panels[f'{name}_excluded_shared_worlds'] = before - len(panels[name])
    panels['route_lengths'] = {lid: len(v['route'])
                               for lid, v in layouts.items()}
    return panels


# ---------------------------------------------------------------- prompts

TASK_TEXT = (
    'You are the TEACHER for a MultiRoom student: six rooms in a chain, '
    'connected by doors; the student must reach the green goal (G) in the '
    'last room. Doors are never locked: toggle opens a closed door in front '
    '(and closes an open one).')

ACTION_TEXT = (
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup, 4 drop, '
    '5 toggle (open/close the door in front), 6 done. pickup, drop and '
    'done do nothing useful here. A wall or closed door in front blocks '
    'forward.\n')

VOCAB = (
    'STUDENT-OBSERVABLE PREDICATES (computed only from the student\'s own '
    '7x7 view; the student is at the bottom centre facing up and cannot '
    'see through walls or closed doors):\n'
    '- front: the cell directly ahead: empty, wall, door_closed, '
    'door_open, goal, unseen\n'
    '- door_visible: yes or no (any door, open or closed, in view); the '
    'door_* attributes and door_state (open/closed) describe the NEAREST '
    'visible door\n'
    '- closed_door_visible: yes or no; the closed_door_* attributes '
    'describe the nearest visible CLOSED door\n'
    '- goal_visible: yes or no; the goal_* attributes describe the goal\n'
    '- for a VISIBLE object: *_fwd = cells ahead (0 = same row, up to 6), '
    '*_right = cells to the right (negative = left, -3..3), *_side = '
    'left/center/right, *_ahead = beside (fwd 0) or ahead. These '
    'attributes, and door_state, are UNKNOWN when no such object is '
    'visible.\n' + ACTION_TEXT)

POSE_TEXT = ('Agent pose (x, y, dir 0=east 1=south 2=west 3=north; x grows '
             'east, y grows south): ')
CONVENTIONS = ('Each agent pose is [x, y, dir] with dir 0=east 1=south 2=west '
               '3=north; x grows east, y grows south.\n')


def consult_prompt(state):
    return (
        'PROMPT VERSION: scoped_rule_mr_v1\n' + TASK_TEXT +
        ' You see the full state; the student sees only its 7x7 view.\n'
        f'FULL MAP (arrow = agent, D = door, G = goal):\n'
        f'{state["full_map"]}\n'
        f'OBJECT FACTS (teacher-visible): {json.dumps(state["facts"])}\n'
        f'{POSE_TEXT}{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB + v3.SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


def _situations(situations, with_pred):
    lines = []
    for i, s in enumerate(situations):
        lines.append(
            f'SITUATION {i}:\nFULL MAP:\n{s["full_map"]}\n'
            f'OBJECT FACTS: {json.dumps(s["facts"])}\n'
            f'Agent pose: {s["pose"]}\n' +
            (f'Student observes: {json.dumps(s["pred"])}\n'
             if with_pred else ''))
    return '\n'.join(lines)


def refine_prompt(rule, situations):
    return (
        'PROMPT VERSION: scope_check_mr_v1\n' + TASK_TEXT +
        f' Earlier you gave the student this reusable rule:\n'
        f'{v3.rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. Using the full state, judge for EACH '
        'situation whether the rule\'s action is the best action there, and '
        'give the best action. Then return a REFINED rule that keeps the '
        'rule\'s action wherever it is best but excludes situations where it '
        'is not (add conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + VOCAB + v3.SEMANTICS + CONVENTIONS
        + _situations(situations, True))


def blind_prompt(situations):
    return (
        'PROMPT VERSION: blind_check_mr_v1\n' + TASK_TEXT +
        ' For EACH situation below, give the single best next action for the '
        'agent, using the full state.\n' + ACTION_TEXT + CONVENTIONS
        + _situations(situations, False))


def _rule_schema(with_now):
    cond = {k: {'type': 'string', 'enum': ['any', *v]}
            for k, v in FIELDS.items()}
    exception = dict(type='object', additionalProperties=False,
                     required=['field', 'value'], properties=dict(
                         field={'type': 'string', 'enum': list(FIELDS)},
                         value={'type': 'string', 'enum': sorted(
                             {v for vs in FIELDS.values() for v in vs})}))
    props = dict(
        abstain={'type': 'boolean'},
        condition=dict(type='object', additionalProperties=False,
                       required=list(FIELDS), properties=cond),
        action={'type': 'integer', 'enum': list(range(7))},
        exceptions=dict(type='array', maxItems=2, items=exception),
        rationale={'type': 'string', 'maxLength': 240})
    if with_now:
        props = dict(action_now={'type': 'integer', 'enum': list(range(7))},
                     **props)
    return dict(type='object', additionalProperties=False,
                required=list(props), properties=props)


def refine_schema():
    schema = _rule_schema(False)
    schema['properties'] = dict(
        verdicts=v3.refine_schema()['properties']['verdicts'],
        **schema['properties'])
    schema['required'] = ['verdicts', *schema['required']]
    return schema


# ------------------------------------------------------------ build steps

def export(out, rows, name):
    (out / f'{name}.json').write_text(json.dumps(rows, indent=1))
    (out / f'{name}_manifest.json').write_text(json.dumps(dict(
        study=STUDY, stage=name, requests_sha256=digest(rows),
        cases=len(rows), model=MODEL, api_calls=0), indent=1))
    if rows:
        (out / f'example_{name}_prompt.txt').write_text(
            rows[0]['request']['input'][0]['content'])


def request(case_id, stage, prompt, schema, version):
    b = v3.body(prompt, schema, version)
    return dict(case_id=case_id, condition=f'mr_{stage}', split=stage,
                model=MODEL, request=b, request_sha256=digest(b))


def build(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    panels = build_panels()
    rows = [request(f'consult_{k:03d}', 'consult', consult_prompt(s),
                    _rule_schema(True), 'scoped_rule_mr_v1')
            for k, s in enumerate(panels['consult'])]
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(panels))
    export(out, rows, 'consult_requests')
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v)
                      for k, v in panels.items()}))


def parse(answer, with_now=True):
    now = int(answer['action_now']) if with_now else None
    if answer['abstain']:
        return now, None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in FIELDS[e['field']])
    return now, (condition, int(answer['action']), exceptions)


def consult_rules(out=OUT):
    panels = json.loads((Path(out) / 'panels.json').read_text())
    replies = v3.read_replies(Path(out) / 'consult_replies.jsonl')
    parsed = []
    for k, state in enumerate(panels['consult']):
        row = replies.get(f'consult_{k:03d}')
        if not row or row['response_status'] != 'completed':
            parsed.append((state, None, None, 'missing_or_incomplete'))
            continue
        now, rule = parse(row['answer'])
        parsed.append((state, now, rule, 'rule' if rule else 'abstained'))
    return panels, parsed


def export_checks(out=OUT):
    """Self-check and blind-check requests on the SAME pool situations."""
    out = Path(out)
    panels, parsed = consult_rules(out)
    rng = np.random.default_rng(7)
    refine, blind, plan = [], [], []
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        matches = [s for s in panels['pool']
                   if v3.executable(rule, s['pred'])]
        if not matches:
            plan.append(dict(case=k, situations=0))
            continue
        pick = [matches[i] for i in sorted(rng.choice(
            len(matches), min(K_SITUATIONS, len(matches)), replace=False))]
        refine.append(request(f'refine_{k:03d}', 'refine',
                              refine_prompt(rule, pick), refine_schema(),
                              'scope_check_mr_v1'))
        blind.append(request(f'blind_{k:03d}', 'blind', blind_prompt(pick),
                             v3.blind_schema(), 'blind_check_mr_v1'))
        plan.append(dict(case=k, situations=len(pick),
                         pool_matches=len(matches),
                         natives=[s['native'] for s in pick]))
    (out / 'refine_plan.json').write_text(json.dumps(plan, indent=1))
    export(out, refine, 'refine_requests')
    export(out, blind, 'blind_requests')
    print(f'{len(blind)} rules checked twice (self-check shows the rule, '
          f"blind check does not); {sum(p['situations'] == 0 for p in plan)}"
          ' rules had no pool match')


# ----------------------------------------------------------------- methods

def replay_table(parsed):
    replay = {}
    for state, now, _, _ in parsed:
        if now is not None:
            replay.setdefault(state['image'], set()).add(now)
    return replay


def refined_rules(out, parsed):
    replies = v3.read_replies(Path(out) / 'refine_replies.jsonl')
    refined, status = [], Counter()
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        row = replies.get(f'refine_{k:03d}')
        if row is None:
            refined.append(rule)
            status['kept_unchecked'] += 1
        elif row['response_status'] != 'completed':
            status['check_incomplete_dropped'] += 1
        else:
            _, new = parse(row['answer'], with_now=False)
            status['refined' if new else 'withdrawn'] += 1
            if new:
                refined.append(new)
    return refined, dict(status)


def shuffled_rules(raw, seed=20260928):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw))
    while len(raw) > 1 and np.any(order == np.arange(len(raw))):
        order = rng.permutation(len(raw))
    return [(raw[i][0], raw[j][1], raw[i][2]) for i, j in enumerate(order)]


def random_subsets(raw, size, n=5):
    return [sorted(np.random.default_rng([20260929, r]).choice(
        len(raw), size, replace=False).tolist()) for r in range(n)]


def compare(out=OUT, confirm=False):
    out = Path(out)
    panels, parsed = consult_rules(out)
    states = panels['confirm' if confirm else 'develop']
    raw = [r for _, _, r, _ in parsed if r]

    def rules_policy(rules):
        return lambda s: v3.advise(rules, s['pred'])
    replay = replay_table(parsed)

    def replay_policy(s):
        acts = replay.get(s['image'], set())
        return ((next(iter(acts)), 'advised') if len(acts) == 1 else
                (None, 'conflict' if acts else 'no_rule'))
    methods = dict(
        llm_action_now_replay=v3.coverage(replay_policy, states),
        llm_rules_raw=v3.coverage(rules_policy(raw), states),
        llm_rules_shuffled_actions=v3.coverage(
            rules_policy(shuffled_rules(raw)), states))
    result = dict(
        panel='confirm' if confirm else 'develop', states=len(states),
        consult_status=dict(Counter(st for *_, st in parsed)),
        action_now_correct_at_own_state=sum(
            now is not None and now in s['optimal']
            for s, now, _, _ in parsed),
        rule_action_correct_at_own_state=sum(
            r is not None and v3.executable(r, s['pred'])
            and r[1] in s['optimal'] for s, _, r, _ in parsed),
        rule_counts=dict(raw=len(raw), dedup=len(v3.dedupe(raw))))
    if (out / 'refine_replies.jsonl').exists():
        refined, status = refined_rules(out, parsed)
        methods['llm_rules_self_checked'] = v3.coverage(
            rules_policy(refined), states)
        result['self_check_status'] = status
    if (out / 'blind_replies.jsonl').exists():
        for strict in (False, True):
            kept, status = v3.blind_filtered(out, parsed, panels, strict)
            name = 'llm_rules_blind_strict' if strict else 'llm_rules_blind'
            methods[name] = v3.coverage(rules_policy(kept), states)
            result[f'{name}_status'] = status
        subsets = [v3.coverage(rules_policy([raw[i] for i in pick]), states)
                   for pick in random_subsets(raw, len(kept))]
        methods['llm_rules_random_subset_mean'] = {
            key: float(np.mean([m[key] or 0 for m in subsets]))
            for key in ('advised', 'correct', 'incorrect', 'precision')}
    result['methods'] = methods
    result['cost'] = {stage: v3.receipts(out / f'{stage}_replies.raw.jsonl')
                      for stage in ('consult', 'refine', 'blind')
                      if (out / f'{stage}_replies.raw.jsonl').exists()}
    calls = {'llm_action_now_replay': 'consult', 'llm_rules_raw': 'consult',
             'llm_rules_shuffled_actions': 'consult',
             'llm_rules_random_subset_mean': 'consult',
             'llm_rules_self_checked': 'refine', 'llm_rules_blind': 'blind',
             'llm_rules_blind_strict': 'blind'}
    result['correct_per_call'] = {
        name: m['correct'] / (result['cost']['consult']['calls'] + (
            result['cost'][calls[name]]['calls']
            if calls[name] != 'consult' else 0))
        for name, m in methods.items() if 'consult' in result['cost']}
    name = 'compare_confirm.json' if confirm else 'compare_develop.json'
    (out / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


# ------------------------------------------------------------------ banks

def build_banks(out=OUT, banks=BANKS):
    """Frozen learning banks from the collected replies (no calls)."""
    from scripts.run_rule_bank_pilot_20260928 import (receipts_cost,
                                                      rule_json)
    out, banks = Path(out), Path(banks)
    panels, parsed = consult_rules(out)
    raw = [r for _, _, r, _ in parsed if r]
    kept, status = v3.blind_filtered(out, parsed, panels, strict=True)
    replay = {state['image']: now for state, now, _, _ in parsed
              if now is not None}
    consult = receipts_cost(out / 'consult_replies.raw.jsonl')
    common = dict(study=STUDY, task=TASK, observer='multiroom_v1',
                  model=MODEL, source_replies_sha256={
                      stage: hashlib.sha256((out / f'{stage}_replies.jsonl')
                                            .read_bytes()).hexdigest()
                      for stage in ('consult', 'blind')})
    made = dict(
        replay=dict(mode='replay', replay=replay, cost=dict(
            consult=consult), **common),
        scoped=dict(mode='scoped', rules=[rule_json(r) for r in raw],
                    cost=dict(consult=consult), **common),
        shuffled=dict(mode='scoped', rules=[rule_json(r) for r in
                                            shuffled_rules(raw)],
                      cost=dict(consult=consult), **common),
        blind_strict=dict(mode='scoped', rules=[rule_json(r) for r in kept],
                          blind_check_status=status, cost=dict(
                              consult=consult, blind_check=receipts_cost(
                                  out / 'blind_replies.raw.jsonl')),
                          **common))
    for r, pick in enumerate(random_subsets(raw, len(kept))):
        made[f'random_subset_r{r}'] = dict(
            mode='scoped', rules=[rule_json(raw[i]) for i in pick],
            raw_rule_indices=pick, cost=dict(consult=consult), **common)
    banks.mkdir(parents=True, exist_ok=True)
    for name, bank in made.items():
        path = banks / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        path.write_text(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(rules_raw=len(raw), rules_blind_strict=len(kept),
                          replay_views=len(replay), blind_status=status),
                     indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('action', choices=('build', 'export-checks', 'compare',
                                      'banks'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    if args.action == 'compare':
        compare(args.out, args.confirm)
    else:
        {'build': build, 'export-checks': export_checks,
         'banks': build_banks}[args.action](args.out)


if __name__ == '__main__':
    main()
