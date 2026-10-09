"""Scoped LLM rules with a blind check on KeyCorridor-S3R3: a third task.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? KeyCorridor matters because GPT-5-mini
ACTION advice collapsed a working Count-PPO student here: a privileged
teacher can ask for different actions in views the student cannot tell
apart. Rules are functions of the student's own view, so they cannot.

The method is frozen from DoorKey and unchanged from MultiRoom
(scripts/conditional_rules_multiroom.py): same model, rule form, scope
semantics, 36 consultations, self-check and blind check on up to 5 pool
situations per rule, and the strict keep rule. Only the task text, the
student predicates, the state generator and the scoring oracle are new:

- student predicates from the 7x7 view: front cell, carrying, and the
  nearest visible door (with its state), locked door, key and ball;
- states along the oracle's route (a third from each stretch), half
  followed by a short random walk over all six useful actions;
- an exact optimal-action oracle: Dijkstra over pose, key position (floor
  cell or carried) and whether the locked door is still locked; entering
  a closed door costs toggle + forward, the key must be carried to unlock
  and dropped on an empty cell before the ball can be picked up. It only
  SCORES; it never writes, repairs or selects a rule.

Pipeline (as MultiRoom): build, collect consult, export-checks, collect
refine and blind, compare, banks.
"""

import argparse
from collections import Counter
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from scripts import conditional_rules_multiroom as mr
from scripts import conditional_rules_v3 as v3
from teachers.minigrid.llm_general import render_ascii_map

STUDY = 'conditional_rules_keycorridor_20260928'
TASK = 'keycorridor_s3r3'
MODEL = v3.MODEL
ACTIONS = v3.v1.ACTIONS
UNKNOWN = v3.UNKNOWN
FRONT = ('empty', 'wall', 'key', 'ball', 'door_locked', 'door_closed',
         'door_open', 'unseen')
OBJECTS = ('door', 'locked_door', 'key', 'ball')
ATTRS = v3.ATTRS
FIELDS = {'front': FRONT, 'carrying': ('nothing', 'key'),
          'door_state': ('locked', 'closed', 'open')}
for _o in OBJECTS:
    FIELDS[f'{_o}_visible'] = ('yes', 'no')
    FIELDS.update({f'{_o}_{a}': vals for a, vals in ATTRS.items()})
SPLIT = mr.SPLIT
PER_LAYOUT = mr.PER_LAYOUT
PANEL_SEED0 = 21_000_000
K_SITUATIONS = v3.K_SITUATIONS
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = 0, 1, 2, 3, 4, 5
DX, DY = mr.DX, mr.DY
PREFER = (FORWARD, TOGGLE, PICKUP, DROP, LEFT, RIGHT)
WALK = (LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE)
STAGES = mr.STAGES
BANKS = Path('research/rule_banks/keycorridor_20260928')
OUT = Path('results') / STUDY
CARRIED = 'carried'
SUCCESS = 'success'


def digest(value):
    return v3.digest(value)


# ---------------------------------------------------------------- observe

def observe_kc(img):
    """Student predicates from the 7x7 view (agent at (3, 6), facing up)."""
    kind, _, state = map(int, img[3][5])
    front = {0: 'unseen', 1: 'empty', 2: 'wall', 5: 'key',
             6: 'ball'}.get(kind, 'empty')
    if kind == 4:
        front = ('door_open', 'door_closed', 'door_locked')[state]
    seen = dict(door=[], locked_door=[], key=[], ball=[])
    for x in range(7):
        for y in range(7):
            if (x, y) == (3, 6):
                continue
            k, s = int(img[x][y][0]), int(img[x][y][2])
            if k == 4:
                seen['door'].append((x, y))
                if s == 2:
                    seen['locked_door'].append((x, y))
            elif k == 5:
                seen['key'].append((x, y))
            elif k == 6:
                seen['ball'].append((x, y))
    pred = dict(front=front, door_state=UNKNOWN,
                carrying='key' if int(img[3][6][0]) == 5 else 'nothing')
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
            pred['door_state'] = ('open', 'closed', 'locked')[
                int(img[x][y][2])]
    return pred


# ----------------------------------------------------------------- oracle

def world(u):
    """Door states, ball cell, and the key's cell (or carried)."""
    doors, ball, key = {}, None, None
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None:
                continue
            if c.type == 'door':
                doors[(x, y)] = ('locked' if c.is_locked else 'open'
                                 if c.is_open else 'closed')
            elif c.type == 'ball':
                ball = (x, y)
            elif c.type == 'key':
                key = (x, y)
    if u.carrying is not None and u.carrying.type == 'key':
        key = CARRIED
    return doors, ball, key


def abstract_state(u):
    """(x, y, dir, key cell or carried, doors opened since now)."""
    _, _, key = world(u)
    return (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), key,
            frozenset())


def moves(u, doors, ball, s):
    """(action, cost, next abstract state or SUCCESS) from state s.

    Doors stay open once opened (closing one is never useful), so the
    state records which doors this plan has opened; a door the route
    passes twice is toggled once.
    """
    x, y, d, key, opened = s
    out = [(LEFT, 1, (x, y, (d - 1) % 4, key, opened)),
           (RIGHT, 1, (x, y, (d + 1) % 4, key, opened))]
    f = (x + DX[d], y + DY[d])
    if f == ball:
        if key != CARRIED:
            out.append((PICKUP, 1, SUCCESS))
        return out
    if key == f:
        out.append((PICKUP, 1, (x, y, d, CARRIED, opened)))
        return out
    if f in doors:
        state = 'open' if f in opened else doors[f]
        if state == 'open':
            out.append((FORWARD, 1, (*f, d, key, opened)))
        elif state == 'closed' or key == CARRIED:   # open or unlock
            out.append((TOGGLE, 1, (x, y, d, key, opened | {f})))
        return out
    c = u.grid.get(*f)
    if c is None or c.type == 'key':     # the key is tracked by `key` alone
        out.append((FORWARD, 1, (*f, d, key, opened)))
        if key == CARRIED:
            out.append((DROP, 1, (x, y, d, f, opened)))
    return out


def cost_to_go(u, doors, ball, start):
    """Fewest actions from an abstract state to picking up the ball."""
    tie = itertools.count()
    heap, done = [(0, next(tie), start)], set()
    while heap:
        v, _, s = heapq.heappop(heap)
        if s == SUCCESS:
            return v
        if s in done:
            continue
        done.add(s)
        for _, c, t in moves(u, doors, ball, s):
            if t == SUCCESS or t not in done:
                heapq.heappush(heap, (v + c, next(tie), t))
    return float('inf')


def optimal(u):
    """Every action on some shortest route to the ball (scoring only)."""
    doors, ball, _ = world(u)
    s = abstract_state(u)
    here = cost_to_go(u, doors, ball, s)
    if here == float('inf'):
        return ()
    return tuple(sorted({a for a, c, t in moves(u, doors, ball, s)
                         if c + (0 if t == SUCCESS else
                                 cost_to_go(u, doors, ball, t)) == here}))


# ----------------------------------------------------------------- panels

def make_env(seed):
    env = build_env(TASK, seed=seed, obs_mode='symbolic')
    env.reset(seed=seed)
    return env


def layout_id(u):
    cells = [(x, y, c.type, c.color) for x in range(u.width)
             for y in range(u.height) if (c := u.grid.get(x, y)) is not None]
    return digest([sorted(cells), [int(v) for v in u.agent_pos],
                   int(u.agent_dir)])[:16]


def route(seed):
    """The oracle's route from the reset state, checked by the real env."""
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
                   int(u.agent_dir), u.carrying is not None])


def teacher_facts(u):
    facts = []
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None or c.type not in ('door', 'key', 'ball'):
                continue
            row = dict(kind=c.type, color=c.color, x=x, y=y)
            if c.type == 'door':
                row['state'] = ('locked' if c.is_locked else 'open'
                                if c.is_open else 'closed')
            facts.append(row)
    if u.carrying is not None:
        facts.append(dict(kind=u.carrying.type, color=u.carrying.color,
                          location='carried by the agent'))
    return facts


def state_record(u, panel, layout, seed, stage):
    img = u.gen_obs()['image']
    return dict(panel=panel, layout=layout, seed=seed, phase=stage,
                native=native_key(u), pred=observe_kc(img),
                image=hashlib.sha256(img.tobytes()).hexdigest(),
                optimal=optimal(u), full_map=render_ascii_map(u),
                facts=teacher_facts(u),
                pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                      int(u.agent_dir)])


def make_state(seed, actions, rng, stage=None):
    """A solvable state along the route, maybe walked off it."""
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
                _, _, done, _, _ = env.step(int(rng.choice(WALK)))
                if done:
                    break
        if not done and optimal(env.unwrapped):
            return env, stage
        env.close()


def all_layouts(n=sum(SPLIT.values()), limit=20_000):
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
    'You are the TEACHER for a KeyCorridor student: a corridor with small '
    'side rooms behind doors. The student must pick up the ball. The ball '
    'is behind a LOCKED door; the key of the same colour is in another '
    'room, possibly behind a closed door. The agent carries at most one '
    'object: it must carry the key to unlock the door (toggle while facing '
    'it), then drop the key on an empty cell in front before it can pick '
    'up the ball.')

ACTION_TEXT = (
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (the object '
    'in front), 4 drop (the carried object onto the empty cell in front), '
    '5 toggle (open/close the door in front; unlocking the locked door '
    'needs the key), 6 done (does nothing here). A wall, a closed or locked '
    'door, a key or a ball in front blocks forward.\n')

VOCAB = (
    'STUDENT-OBSERVABLE PREDICATES (computed only from the student\'s own '
    '7x7 view; the student is at the bottom centre facing up and cannot '
    'see through walls or closed doors):\n'
    '- front: the cell directly ahead: empty, wall, key, ball, door_locked, '
    'door_closed, door_open, unseen\n'
    '- carrying: nothing or key\n'
    '- door_visible: yes or no (any door in view); the door_* attributes '
    'and door_state (locked/closed/open) describe the NEAREST visible door\n'
    '- locked_door_visible: yes or no; locked_door_* describe the nearest '
    'visible LOCKED door\n'
    '- key_visible / ball_visible: yes or no (the carried key is not '
    '"visible"); key_* and ball_* describe that object\n'
    '- for a VISIBLE object: *_fwd = cells ahead (0 = same row, up to 6), '
    '*_right = cells to the right (negative = left, -3..3), *_side = '
    'left/center/right, *_ahead = beside (fwd 0) or ahead. These '
    'attributes, and door_state, are UNKNOWN when no such object is '
    'visible.\n' + ACTION_TEXT)


def consult_prompt(state):
    return (
        'PROMPT VERSION: scoped_rule_kc_v1\n' + TASK_TEXT +
        ' You see the full state; the student sees only its 7x7 view.\n'
        f'FULL MAP (arrow = agent, D = door, k = key, o = ball):\n'
        f'{state["full_map"]}\n'
        f'OBJECT FACTS (teacher-visible): {json.dumps(state["facts"])}\n'
        f'{mr.POSE_TEXT}{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB + v3.SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


def refine_prompt(rule, situations):
    return (
        'PROMPT VERSION: scope_check_kc_v1\n' + TASK_TEXT +
        f' Earlier you gave the student this reusable rule:\n'
        f'{v3.rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. Using the full state, judge for EACH '
        'situation whether the rule\'s action is the best action there, and '
        'give the best action. Then return a REFINED rule that keeps the '
        'rule\'s action wherever it is best but excludes situations where it '
        'is not (add conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + VOCAB + v3.SEMANTICS + mr.CONVENTIONS
        + mr._situations(situations, True))


def blind_prompt(situations):
    return (
        'PROMPT VERSION: blind_check_kc_v1\n' + TASK_TEXT +
        ' For EACH situation below, give the single best next action for the '
        'agent, using the full state.\n' + ACTION_TEXT + mr.CONVENTIONS
        + mr._situations(situations, False))


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
    return dict(case_id=case_id, condition=f'kc_{stage}', split=stage,
                model=MODEL, request=b, request_sha256=digest(b))


def build(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    panels = build_panels()
    rows = [request(f'consult_{k:03d}', 'consult', consult_prompt(s),
                    _rule_schema(True), 'scoped_rule_kc_v1')
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
                              'scope_check_kc_v1'))
        blind.append(request(f'blind_{k:03d}', 'blind', blind_prompt(pick),
                             v3.blind_schema(), 'blind_check_kc_v1'))
        plan.append(dict(case=k, situations=len(pick),
                         pool_matches=len(matches),
                         natives=[s['native'] for s in pick]))
    (out / 'refine_plan.json').write_text(json.dumps(plan, indent=1))
    export(out, refine, 'refine_requests')
    export(out, blind, 'blind_requests')
    print(f'{len(blind)} rules checked twice (self-check shows the rule, '
          f"blind check does not); {sum(p['situations'] == 0 for p in plan)}"
          ' rules had no pool match')


# ------------------------------------------------------ compare and banks

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


def compare(out=OUT, confirm=False):
    out = Path(out)
    panels, parsed = consult_rules(out)
    states = panels['confirm' if confirm else 'develop']
    raw = [r for _, _, r, _ in parsed if r]

    def rules_policy(rules):
        return lambda s: v3.advise(rules, s['pred'])
    replay = mr.replay_table(parsed)

    def replay_policy(s):
        acts = replay.get(s['image'], set())
        return ((next(iter(acts)), 'advised') if len(acts) == 1 else
                (None, 'conflict' if acts else 'no_rule'))
    methods = dict(
        llm_action_now_replay=v3.coverage(replay_policy, states),
        llm_rules_raw=v3.coverage(rules_policy(raw), states),
        llm_rules_shuffled_actions=v3.coverage(
            rules_policy(mr.shuffled_rules(raw)), states))
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
                   for pick in mr.random_subsets(raw, len(kept))]
        methods['llm_rules_random_subset_mean'] = {
            key: float(np.mean([m[key] or 0 for m in subsets]))
            for key in ('advised', 'correct', 'incorrect', 'precision')}
    result['methods'] = methods
    result['cost'] = {stage: v3.receipts(out / f'{stage}_replies.raw.jsonl')
                      for stage in ('consult', 'refine', 'blind')
                      if (out / f'{stage}_replies.raw.jsonl').exists()}
    extra = {'llm_rules_self_checked': 'refine', 'llm_rules_blind': 'blind',
             'llm_rules_blind_strict': 'blind'}
    if 'consult' in result['cost']:
        result['correct_per_call'] = {
            name: m['correct'] / (result['cost']['consult']['calls'] + (
                result['cost'][extra[name]]['calls'] if name in extra
                else 0)) for name, m in methods.items()}
    name = 'compare_confirm.json' if confirm else 'compare_develop.json'
    (out / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


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
    common = dict(study=STUDY, task=TASK, observer='keycorridor_v1',
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
                                            mr.shuffled_rules(raw)],
                      cost=dict(consult=consult), **common),
        blind_strict=dict(mode='scoped', rules=[rule_json(r) for r in kept],
                          blind_check_status=status, cost=dict(
                              consult=consult, blind_check=receipts_cost(
                                  out / 'blind_replies.raw.jsonl')),
                          **common))
    for r, pick in enumerate(mr.random_subsets(raw, len(kept))):
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
