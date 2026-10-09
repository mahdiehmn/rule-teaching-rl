"""Scoped conditional rules on DoorKey: observable predicates, abstention, panels.

The earlier check (scripts/conditional_advice_check.py) and its 36 replies
are kept unchanged and serve as a control.

1. Predicates (v2), computed ONLY from the student's 7x7 symbolic view:
   - the front cell;
   - the carried item, read from the agent's own cell;
   - for the floor key, the door and the goal: exact relative offsets
     (`*_fwd` 0..6, `*_right` -3..3) and the coarse side / ahead values
     derived from them;
   - the visible door state.
   Everything outside the view is `hidden`. Nothing is read from the
   teacher's map.
2. Aliasing diagnostic on a development panel: how often identical
   predicate tuples carry disjoint optimal action sets, under v1, v2 and
   the exact view. The oracle is offline only, never available to a
   learner.
3. Scope. A rule is executable only if every condition is observed true
   and every exception observed false. A condition on a hidden fact is
   unmet, not guessed. When executable rules disagree, the student
   abstains; no planner resolves the conflict.
4. Panels are disjoint by LAYOUT identity (the reset map), not by seed.
   All 300 DoorKey-8x8 layouts are split by a salted hash: 36 for
   consultation, 132 for development evaluation and 132 for untouched
   confirmation.
   Physical states are deduplicated. `export` freezes the v2 prompt,
   schema, model and 36 consultations; nothing is called here.
5. `compare` scores hand-written rules, same-LLM exact-view replay, the
   v1 direct rules and, when collected, the v2 scoped rules. It reports
   unique correct and incorrect (state, action) coverage, abstentions,
   conflicts, cost, and raw and deduplicated rule counts. Confirmation
   layouts stay untouched unless `--confirm` is given after the prompt is
   frozen.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from minigrid.core.world_object import Key
from scripts import conditional_advice_check as v1
from teachers.minigrid.llm_general import render_ascii_map

STUDY = 'conditional_rules_v2_20260927'
CONDITION = 'scoped_rule_v2'
MODEL = v1.MODEL
ACTIONS = v1.ACTIONS
FRONT = (*v1.FRONT, 'unseen')
FWD = ('hidden', *[str(k) for k in range(7)])
RIGHT = ('hidden', *[str(k) for k in range(-3, 4)])
SIDE = ('hidden', 'left', 'center', 'right')
AHEAD = ('hidden', 'beside', 'ahead')          # beside = same row (fwd 0)
FIELDS = {'front': FRONT, 'carrying': ('nothing', 'key'),
          'door_state': ('hidden', 'locked', 'closed', 'open')}
for _obj in ('key', 'door', 'goal'):
    FIELDS.update({f'{_obj}_fwd': FWD, f'{_obj}_right': RIGHT,
                   f'{_obj}_side': SIDE, f'{_obj}_ahead': AHEAD})
MAX_EXCEPTIONS = 2
HAND_RULES_V2 = (   # hand-written comparison rules; scored by compare only
    ({'front': 'key', 'carrying': 'nothing'}, 3, ()),
    ({'front': 'door_locked', 'carrying': 'key'}, 5, ()),
    ({'front': 'door_closed'}, 5, ()),
    ({'front': 'door_open'}, 2, ()),
    ({'front': 'goal'}, 2, ()),
    ({'carrying': 'nothing', 'key_fwd': '0', 'key_side': 'left'}, 0, ()),
    ({'carrying': 'nothing', 'key_fwd': '0', 'key_side': 'right'}, 1, ()),
    ({'carrying': 'key', 'door_fwd': '0', 'door_side': 'left'}, 0, ()),
    ({'carrying': 'key', 'door_fwd': '0', 'door_side': 'right'}, 1, ()),
    ({'goal_fwd': '0', 'goal_side': 'left'}, 0, ()),
    ({'goal_fwd': '0', 'goal_side': 'right'}, 1, ()),
)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------- observe

def observe_v2(img):
    """Predicates from the 7x7x3 view alone (agent at (3, 6), facing up)."""
    kind, _, state = map(int, img[3][5])
    front = {0: 'unseen', 1: 'empty', 2: 'wall', 5: 'key',
             8: 'goal'}.get(kind, 'empty')
    if kind == 4:
        front = ('door_open', 'door_closed', 'door_locked')[state]
    pred = dict(front=front,
                carrying='key' if int(img[3][6][0]) == 5 else 'nothing',
                door_state='hidden')
    for name, idx in (('key', 5), ('door', 4), ('goal', 8)):
        where = [(int(x), int(y)) for x, y in np.argwhere(
            img[:, :, 0] == idx) if (int(x), int(y)) != (3, 6)]
        if not where:
            pred.update({f'{name}_fwd': 'hidden', f'{name}_right': 'hidden',
                         f'{name}_side': 'hidden', f'{name}_ahead': 'hidden'})
            continue
        x, y = where[0]
        fwd, right = 6 - y, x - 3
        pred.update({f'{name}_fwd': str(fwd), f'{name}_right': str(right),
                     f'{name}_side': ('left' if right < 0 else 'right'
                                      if right > 0 else 'center'),
                     f'{name}_ahead': 'beside' if fwd == 0 else 'ahead'})
        if name == 'door':
            pred['door_state'] = ('open', 'closed', 'locked')[
                int(img[x][y][2])]
    return pred


# ------------------------------------------------------------------- rules

def executable(rule, pred):
    """Conditions observed true and exceptions observed false."""
    condition, _, exceptions = rule
    if any(pred[k] != v for k, v in condition.items()):
        return False
    return not any(pred[k] == v for k, v in exceptions)


def advise(rules, pred):
    """One action, or an abstention with its reason; never a planner."""
    actions = {r[1] for r in rules if executable(r, pred)}
    if not actions:
        return None, 'no_rule'
    if len(actions) > 1:
        return None, 'conflict'
    return actions.pop(), 'advised'


def v1_rule(rule):
    """v1 rules keep v1 semantics (their own vocabulary) as the control."""
    return rule


# ------------------------------------------------------------------ panels

def layout_id(u):
    """Identity of a reset map: walls, door, key and goal positions."""
    cells = [(x, y, c.type) for x in range(u.width)
             for y in range(u.height)
             if (c := u.grid.get(x, y)) is not None]
    return digest(sorted(cells))[:16]


def all_layouts(base=14_000_000, expected=300, limit=50_000):
    """Every DoorKey-8x8 layout, each with the first seed producing it.

    DoorKey-8x8 has exactly 300 reset maps (4 wall columns x 5 door rows
    x the key cells left of the wall); a scan finds all of them.
    """
    found = {}
    for s in range(base, base + limit):
        env = build_env('doorkey_8x8', seed=s, obs_mode='symbolic')
        env.reset(seed=s)
        found.setdefault(layout_id(env.unwrapped), s)
        env.close()
        if len(found) == expected:
            return found
    raise ValueError(f'Found only {len(found)} layouts')


def build_panels(sizes=(('consult', 36), ('develop', 132),
                        ('confirm', 132)), states_per_layout=12, seed=0):
    """Layout-disjoint panels of deduplicated constructed states.

    The 300 layouts are ordered by a salted hash and split, so panels
    never share a reset map. Consultation takes one phase-balanced state
    per layout; the other panels take up to `states_per_layout` states
    per layout (v1's constructed states: phase, position, facing,
    adjacency enrichment), deduplicated by physical signature.
    """
    layouts = all_layouts()
    order = sorted(layouts, key=lambda lid: digest([STUDY, lid]))
    if sum(n for _, n in sizes) != len(order):
        raise ValueError('Panel sizes must partition the 300 layouts')
    rng = np.random.default_rng(seed)
    panels, start = {}, 0
    for name, count in sizes:
        states, seen = [], set()
        for k, lid in enumerate(order[start:start + count]):
            per = 1 if name == 'consult' else states_per_layout
            for _ in range(per):
                phase = (['key', 'door', 'goal'][k % 3]
                         if name == 'consult' else None)
                env, phase = v1.make_state(layouts[lid], rng, phase)
                u = env.unwrapped
                sig = digest([lid, render_ascii_map(u), int(u.agent_dir),
                              isinstance(u.carrying, Key)])
                if sig not in seen:
                    seen.add(sig)
                    img = u.gen_obs()['image']
                    states.append(dict(
                        panel=name, layout=lid, seed=layouts[lid],
                        phase=phase, signature=sig, v1=v1.observe(u),
                        v2=observe_v2(img),
                        image=hashlib.sha256(img.tobytes()).hexdigest(),
                        optimal=v1.optimal(u),
                        full_map=render_ascii_map(u),
                        pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                              int(u.agent_dir)]))
                env.close()
        start += count
        panels[name] = states
    return panels


# -------------------------------------------------------- item 2: aliasing

def aliasing(states, key):
    groups = defaultdict(list)
    for s in states:
        groups[key(s)].append(set(s['optimal']))
    conflicted = best = 0
    for sets in groups.values():
        if len(sets) > 1 and not set.intersection(*sets):
            conflicted += len(sets)
        counts = Counter(a for opt in sets for a in opt)
        best += max(counts.values())
    return dict(groups=len(groups), states=len(states),
                states_in_disjoint_groups=conflicted,
                disjoint_fraction=conflicted / len(states),
                single_action_oracle_bound=best / len(states))


def aliasing_report(states):
    return {
        'v1_predicates': aliasing(states, lambda s: json.dumps(
            s['v1'], sort_keys=True)),
        'v2_coarse_only': aliasing(states, lambda s: json.dumps(
            {k: v for k, v in s['v2'].items()
             if not k.endswith(('_fwd', '_right'))}, sort_keys=True)),
        'v2_full': aliasing(states, lambda s: json.dumps(s['v2'],
                                                        sort_keys=True)),
        'exact_view': aliasing(states, lambda s: s['image']),
    }


# --------------------------------------------------------- item 4: prompt

VOCAB = (
    'OBSERVABLE PREDICATES (computed only from the student\'s own 7x7 view; '
    'the student is at the bottom centre facing up and cannot see through '
    'walls or closed doors):\n'
    '- front: the cell directly ahead: empty, wall, key, door_locked, '
    'door_closed, door_open, goal, unseen\n'
    '- carrying: nothing or key (a carried key is not on the floor)\n'
    '- door_state: locked, closed, open, or hidden if the door is not '
    'visible\n'
    '- for the floor key, the door and the goal: *_fwd = cells ahead '
    '(0 = same row as the agent, up to 6), *_right = cells to the right '
    '(negative = left, -3..3), *_side = left/center/right, *_ahead = '
    'beside (fwd 0) or ahead; every value is hidden when that object is '
    'not visible\n'
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (cell in '
    'front, without moving), 4 drop, 5 toggle (open/unlock the door in '
    'front; unlocking needs the key), 6 done (does nothing here). A key, '
    'wall or closed door in front blocks forward.\n')

RULE_SEMANTICS = (
    'RULE SEMANTICS (enforced by code): the student applies your rule only '
    'where EVERY condition you set is observed exactly, and NONE of your '
    'exceptions is observed. A condition on something hidden is not met. '
    'If two rules recommend different actions in one state the student '
    'abstains. So scope matters: a rule that is too broad will be applied '
    'in states where it is wrong; a rule that is too narrow helps nowhere '
    'else. Choose the broadest scope over which the action stays optimal, '
    'and add exceptions (at most 2) for observable cases where it would '
    'not. Use "any" for fields the rule does not need. If no rule over '
    'these predicates is reliable here, set abstain=true.\n')


def prompt(state):
    return (
        'PROMPT VERSION: scoped_rule_v2\n'
        'You are the TEACHER for a DoorKey student (reach the green goal; '
        'the key unlocks the locked door). You see the full map; the '
        'student sees only its 7x7 view.\n'
        f'FULL MAP (x east, y south; arrow = agent):\n{state["full_map"]}\n'
        f'Agent pose (x, y, dir 0=east 1=south 2=west 3=north): '
        f'{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["v2"])}\n'
        + VOCAB + RULE_SEMANTICS +
        'Give ONE reusable rule the student can apply on its own: WHEN '
        'conditions, PREFER an action, UNLESS exceptions. It must be correct '
        'in the current state.\n')


def schema():
    cond = {k: {'type': 'string', 'enum': ['any', *v]}
            for k, v in FIELDS.items()}
    exception = dict(type='object', additionalProperties=False,
                     required=['field', 'value'], properties=dict(
                         field={'type': 'string', 'enum': list(FIELDS)},
                         value={'type': 'string', 'enum': sorted(
                             {v for vs in FIELDS.values() for v in vs})}))
    return dict(
        type='object', additionalProperties=False,
        required=['abstain', 'condition', 'action', 'exceptions',
                  'rationale'],
        properties=dict(
            abstain={'type': 'boolean'},
            condition=dict(type='object', additionalProperties=False,
                           required=list(FIELDS), properties=cond),
            action={'type': 'integer', 'enum': list(range(7))},
            exceptions=dict(type='array', maxItems=MAX_EXCEPTIONS,
                            items=exception),
            rationale={'type': 'string', 'maxLength': 240}))


def request(state):
    return dict(model=MODEL, store=False, service_tier='default',
                max_output_tokens=4096, reasoning={'effort': 'low'},
                input=[dict(role='user', content=prompt(state))],
                text={'format': dict(type='json_schema', strict=True,
                                     name=CONDITION, schema=schema())})


def parse(answer):
    """(rule, status); cross-field invalid exceptions reject the rule."""
    if answer['abstain']:
        return None, 'abstained'
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = []
    for e in answer['exceptions']:
        if e['value'] not in FIELDS[e['field']]:
            return None, 'invalid_exception'
        exceptions.append((e['field'], e['value']))
    return (condition, int(answer['action']), tuple(exceptions)), 'rule'


# -------------------------------------------------------- item 5: compare

def coverage(policy, states, view):
    """Unique (state, action) coverage of an advice policy on states."""
    out = Counter()
    for s in states:
        action, status = policy(s[view])
        out[status] += 1
        if action is not None:
            out['correct' if action in s['optimal'] else 'incorrect'] += 1
    n = len(states)
    advised = out['correct'] + out['incorrect']
    return dict(states=n, advised=advised, correct=out['correct'],
                incorrect=out['incorrect'], abstain_no_rule=out['no_rule'],
                abstain_conflict=out['conflict'],
                precision=out['correct'] / advised if advised else None,
                correct_fraction=out['correct'] / n)


def replay_policy(consult_rules):
    """Same-LLM exact-view replay: the consultation's action, same view."""
    table = defaultdict(set)
    for state, rule in consult_rules:
        if rule is not None:
            table[state['image']].add(rule[1])

    def policy(image):
        acts = table.get(image, set())
        if not acts:
            return None, 'no_rule'
        if len(acts) > 1:
            return None, 'conflict'
        return next(iter(acts)), 'advised'
    return policy


def dedupe(rules):
    seen, out = set(), []
    for r in rules:
        key = json.dumps([sorted(r[0].items()), r[1], sorted(r[2])])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_v1_rules(v1_dir):
    """The frozen v1 rules; v1 exception = a single (field, value)."""
    consult = json.loads((v1_dir / 'consult_states.json').read_text())
    replies = {json.loads(line)['case_id']: json.loads(line) for line in
               (v1_dir / 'replies.jsonl').read_text().splitlines()}
    rules = []
    for k in range(len(consult)):
        rule = v1.parse(replies[f'consult_{k:03d}']['answer'])
        if rule is not None:
            c, a, e = rule
            rules.append((c, a, (e,) if e else ()))
    return rules


def compare(out, v1_dir, confirm=False):
    out = Path(out)
    panels = json.loads((out / 'panels.json').read_text())
    states = panels['confirm' if confirm else 'develop']
    v1_rules = load_v1_rules(Path(v1_dir))
    methods = {}

    def rules_policy(rules):
        return lambda pred: advise(rules, pred)
    methods['handwritten_v2'] = coverage(rules_policy(HAND_RULES_V2),
                                         states, 'v2')
    methods['v1_direct_rules_raw'] = coverage(rules_policy(v1_rules),
                                              states, 'v1')
    methods['v1_direct_rules_dedup'] = coverage(
        rules_policy(dedupe(v1_rules)), states, 'v1')
    result = dict(panel='confirm' if confirm else 'develop',
                  layouts=len({s['layout'] for s in states}),
                  states=len(states), v1_rule_count=dict(
                      raw=len(v1_rules), dedup=len(dedupe(v1_rules))))
    replies_path = out / 'replies.jsonl'
    if replies_path.exists():
        consult = panels['consult']
        replies = {json.loads(line)['case_id']: json.loads(line) for line in
                   replies_path.read_text().splitlines()}
        parsed, statuses = [], Counter()
        for k, state in enumerate(consult):
            row = replies.get(f'consult_{k:03d}')
            if not row or row['response_status'] != 'completed':
                statuses['missing_or_failed'] += 1
                parsed.append((state, None))
                continue
            rule, status = parse(row['answer'])
            statuses[status] += 1
            parsed.append((state, rule))
        v2_rules = [r for _, r in parsed if r is not None]
        methods['v2_scoped_rules_raw'] = coverage(rules_policy(v2_rules),
                                                  states, 'v2')
        methods['v2_scoped_rules_dedup'] = coverage(
            rules_policy(dedupe(v2_rules)), states, 'v2')
        methods['same_llm_exact_view_replay'] = coverage(
            replay_policy([(s, r) for s, r in parsed]), states, 'image')
        result.update(
            v2_reply_status=dict(statuses),
            v2_rule_count=dict(raw=len(v2_rules),
                               dedup=len(dedupe(v2_rules))),
            v2_correct_at_own_state=sum(
                r is not None and executable(r, s['v2']) and
                r[1] in s['optimal'] for s, r in parsed))
    result['methods'] = methods
    name = 'compare_confirm.json' if confirm else 'compare_develop.json'
    (out / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------- CLI

def build(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    panels = build_panels()
    diag = aliasing_report(panels['develop'])
    consult = panels['consult']
    rows = []
    for k, state in enumerate(consult):
        body = request(state)
        rows.append(dict(case_id=f'consult_{k:03d}', condition=CONDITION,
                         split='consult', model=MODEL, request=body,
                         request_sha256=digest(body)))
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(panels))
    (out / 'aliasing_develop.json').write_text(json.dumps(diag, indent=1))
    (out / 'requests.json').write_text(json.dumps(rows, indent=1))
    (out / 'manifest.json').write_text(json.dumps(dict(
        study=STUDY, requests_sha256=digest(rows), cases=len(rows),
        model=MODEL, prompt_version='scoped_rule_v2',
        panel_layouts={k: len({s['layout'] for s in v})
                       for k, v in panels.items()},
        panel_states={k: len(v) for k, v in panels.items()},
        panels_sha256=digest(panels), api_calls=0,
        confirmation='untouched until the prompt is frozen'), indent=1))
    (out / 'example_prompt.txt').write_text(prompt(consult[0]))
    print(json.dumps(dict(panels={k: len(v) for k, v in panels.items()},
                          aliasing_develop=diag), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('action', choices=('build', 'compare'))
    p.add_argument('--out', type=Path, default=Path('results') / STUDY)
    p.add_argument('--v1-dir', type=Path, default=Path('results') /
                   'conditional_advice_check_20260927_v1')
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    if args.action == 'build':
        build(args.out)
    else:
        compare(args.out, args.v1_dir, args.confirm)


if __name__ == '__main__':
    main()
