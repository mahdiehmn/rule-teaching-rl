"""Scoped conditional rules on DoorKey (v3): interface and scope-check refinement.

Development only; the confirmation layouts stay untouched.

Interface:

1. Three-valued scope. Visibility (`*_visible` = yes/no) is observed.
   Object attributes (`*_fwd`, `*_right`, `*_side`, `*_ahead`,
   `door_state`) are `unknown` when the object is not in view.
   - A condition on an unknown attribute is unmet: the schema cannot
     even request `unknown`.
   - An exception whose attribute is unknown blocks the rule: it cannot
     be confirmed false, so the student abstains.
   - Conflicting executable rules still make the student abstain.
2. Visibility is separated from unknown attributes, as above.
3. Native physical keys: hash(grid.encode(), pose, inventory). Panels
   drop development/confirmation states whose native world also occurs
   in the consultation or experience pools. Carrying the key erases its
   reset position, so reset-layout disjointness alone is not enough.
4. The teacher's input lists the native object facts, including the door
   state, even when the door is hidden from the student.

Each consultation returns `action_now`, the LLM's point advice (the
action-only control at the same call), plus one scoped rule.

Scope-check refinement: for each rule, the student
samples up to K situations from its own experience pool where the rule
is executable. The pool comes from the consultation layouts only, never
development or confirmation. The student sends them in ONE extra call.
The teacher judges its rule in each situation, with the full state, and
returns a refined rule or abstains. No oracle is used; only the LLM
judges. The refinement calls count in the budget.

Pipeline:
  build           panels, experience pool, consultation requests (0 calls)
  (collect)       collector --split all        -> replies.jsonl
  export-refine   refinement requests from those replies (0 calls)
  (collect)       collector --split all        -> refine_replies.jsonl
  compare         development scoring (oracle used only to score)
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from minigrid.core.world_object import Key
from scripts import conditional_advice_check as v1
from scripts import conditional_rules_v2 as v2
from teachers.minigrid.llm_general import render_ascii_map

STUDY = 'conditional_rules_v3_20260928'
MODEL = v1.MODEL
OBJECTS = ('key', 'door', 'goal')
FWD = tuple(str(k) for k in range(7))
RIGHT = tuple(str(k) for k in range(-3, 4))
ATTRS = {'fwd': FWD, 'right': RIGHT, 'side': ('left', 'center', 'right'),
         'ahead': ('beside', 'ahead')}
FIELDS = {'front': v2.FRONT, 'carrying': ('nothing', 'key'),
          'door_state': ('locked', 'closed', 'open')}
for _o in OBJECTS:
    FIELDS[f'{_o}_visible'] = ('yes', 'no')
    FIELDS.update({f'{_o}_{a}': vals for a, vals in ATTRS.items()})
ATTRIBUTE_FIELDS = {'door_state'} | {f'{o}_{a}' for o in OBJECTS
                                     for a in ATTRS}
UNKNOWN = 'unknown'
K_SITUATIONS = 5
POOL_PER_LAYOUT = 12


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------- observe

def observe_v3(img):
    """Visibility observed; attributes unknown when not in view."""
    base = v2.observe_v2(img)
    pred = dict(front=base['front'], carrying=base['carrying'])
    for o in OBJECTS:
        seen = base[f'{o}_fwd'] != 'hidden'
        pred[f'{o}_visible'] = 'yes' if seen else 'no'
        for a in ATTRS:
            pred[f'{o}_{a}'] = base[f'{o}_{a}'] if seen else UNKNOWN
    pred['door_state'] = (base['door_state'] if base['door_state'] !=
                          'hidden' else UNKNOWN)
    return pred


def native_key(u):
    return digest([u.grid.encode().tolist(),
                   [int(u.agent_pos[0]), int(u.agent_pos[1])],
                   int(u.agent_dir), isinstance(u.carrying, Key)])


def teacher_facts(u):
    facts = []
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None or c.type not in ('key', 'door', 'goal'):
                continue
            row = dict(kind=c.type, color=c.color, x=x, y=y)
            if c.type == 'door':
                row['state'] = ('locked' if c.is_locked else 'open'
                                if c.is_open else 'closed')
            facts.append(row)
    if isinstance(u.carrying, Key):
        facts.append(dict(kind='key', color=u.carrying.color,
                          location='carried by the agent'))
    return facts


# ------------------------------------------------------------------ rules

def executable(rule, pred):
    """Three-valued: unknown never meets a condition or clears an exception."""
    condition, _, exceptions = rule
    if any(pred[k] != v for k, v in condition.items()):
        return False
    for k, v in exceptions:
        if pred[k] == UNKNOWN or pred[k] == v:
            return False
    return True


def advise(rules, pred):
    actions = {r[1] for r in rules if executable(r, pred)}
    if not actions:
        return None, 'no_rule'
    if len(actions) > 1:
        return None, 'conflict'
    return actions.pop(), 'advised'


# ----------------------------------------------------------------- panels

def state_record(u, panel, layout, seed, phase):
    img = u.gen_obs()['image']
    return dict(panel=panel, layout=layout, seed=seed, phase=phase,
                native=native_key(u), v3=observe_v3(img),
                image=hashlib.sha256(img.tobytes()).hexdigest(),
                optimal=v1.optimal(u), full_map=render_ascii_map(u),
                facts=teacher_facts(u),
                pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                      int(u.agent_dir)])


def build_panels(seed=0):
    """Same 300-layout split as v2; native-world dedupe and exclusion."""
    layouts = v2.all_layouts()
    order = sorted(layouts, key=lambda lid: digest([v2.STUDY, lid]))
    parts = dict(consult=order[:36], develop=order[36:168],
                 confirm=order[168:])
    rng = np.random.default_rng(seed)
    panels = dict(consult=[], pool=[], develop=[], confirm=[])
    for k, lid in enumerate(parts['consult']):
        env, phase = v1.make_state(layouts[lid], rng,
                                   ['key', 'door', 'goal'][k % 3])
        panels['consult'].append(state_record(env.unwrapped, 'consult',
                                              lid, layouts[lid], phase))
        env.close()
    for name, source in (('pool', parts['consult']),
                         ('develop', parts['develop']),
                         ('confirm', parts['confirm'])):
        seen = set()
        for lid in source:
            for _ in range(POOL_PER_LAYOUT):
                env, phase = v1.make_state(layouts[lid], rng)
                rec = state_record(env.unwrapped, name, lid, layouts[lid],
                                   phase)
                env.close()
                if rec['native'] not in seen:
                    seen.add(rec['native'])
                    panels[name].append(rec)
    used = {s['native'] for s in panels['consult'] + panels['pool']}
    for name in ('develop', 'confirm'):
        before = len(panels[name])
        panels[name] = [s for s in panels[name] if s['native'] not in used]
        panels[f'{name}_excluded_shared_worlds'] = before - len(panels[name])
    return panels


# ---------------------------------------------------------------- prompts

VOCAB = (
    'STUDENT-OBSERVABLE PREDICATES (computed only from the student\'s own '
    '7x7 view; the student is at the bottom centre facing up and cannot '
    'see through walls or closed doors):\n'
    '- front: the cell directly ahead: empty, wall, key, door_locked, '
    'door_closed, door_open, goal, unseen\n'
    '- carrying: nothing or key\n'
    '- key_visible / door_visible / goal_visible: yes or no (whether that '
    'object is in the student view; the carried key is not "visible")\n'
    '- for a VISIBLE object: *_fwd = cells ahead (0 = same row, up to 6), '
    '*_right = cells to the right (negative = left, -3..3), *_side = '
    'left/center/right, *_ahead = beside (fwd 0) or ahead; door_state = '
    'locked/closed/open. These attributes are UNKNOWN when the object is '
    'not visible.\n'
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (cell in front, '
    'without moving), 4 drop, 5 toggle (open/unlock the door in front; '
    'unlocking needs the key), 6 done (does nothing here). A key, wall or '
    'closed door in front blocks forward.\n')

SEMANTICS = (
    'RULE SEMANTICS (enforced by code):\n'
    '- The student applies a rule only where EVERY condition is observed '
    'exactly. A condition on an attribute of an object that is not visible '
    'is never met.\n'
    '- An exception (at most 2) blocks the rule where it is observed true. '
    'If an exception\'s attribute is unknown (object not visible), the '
    'student cannot rule it out and does NOT apply the rule.\n'
    '- If two rules recommend different actions, the student abstains.\n'
    'SCOPE: the student will reuse your rule in EVERY situation matching '
    'its conditions, not only this one. A rule is only useful if its action '
    'is best in all of those situations. Conditioning only on a coarse fact '
    '(for example only on the front cell) usually makes a rule wrong in '
    'many situations; add the conditions or exceptions that make it hold, '
    'or abstain.\n')


def consult_prompt(state):
    return (
        'PROMPT VERSION: scoped_rule_v3\n'
        'You are the TEACHER for a DoorKey student (reach the green goal; '
        'the key unlocks the locked door). You see the full state; the '
        'student sees only its 7x7 view.\n'
        f'FULL MAP (x east, y south; arrow = agent):\n{state["full_map"]}\n'
        f'OBJECT FACTS (teacher-visible): {json.dumps(state["facts"])}\n'
        f'Agent pose (x, y, dir 0=east 1=south 2=west 3=north): '
        f'{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["v3"])}\n'
        + VOCAB + SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


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


def consult_schema():
    return _rule_schema(True)


def refine_schema():
    verdict = dict(type='object', additionalProperties=False,
                   required=['situation', 'rule_action_is_best',
                             'best_action'],
                   properties=dict(
                       situation={'type': 'integer'},
                       rule_action_is_best={'type': 'boolean'},
                       best_action={'type': 'integer',
                                    'enum': list(range(7))}))
    schema = _rule_schema(False)
    schema['properties'] = dict(
        verdicts=dict(type='array', items=verdict), **schema['properties'])
    schema['required'] = ['verdicts', *schema['required']]
    return schema


def body(prompt, schema, name):
    return dict(model=MODEL, store=False, service_tier='default',
                max_output_tokens=8192, reasoning={'effort': 'low'},
                input=[dict(role='user', content=prompt)],
                text={'format': dict(type='json_schema', strict=True,
                                     name=name, schema=schema)})


def rule_text(rule):
    condition, action, exceptions = rule
    return json.dumps(dict(when=condition, prefer=v1.ACTIONS[action],
                           action=action, unless=[list(e) for e in
                                                  exceptions]))


def refine_prompt(rule, situations):
    lines = []
    for i, s in enumerate(situations):
        lines.append(
            f'SITUATION {i}:\nFULL MAP:\n{s["full_map"]}\n'
            f'OBJECT FACTS: {json.dumps(s["facts"])}\n'
            f'Agent pose: {s["pose"]}\n'
            f'Student observes: {json.dumps(s["v3"])}\n')
    return (
        'PROMPT VERSION: scope_check_v3\n'
        'You are the TEACHER for a DoorKey student (reach the green goal; '
        'the key unlocks the locked door). Earlier you gave the student this '
        f'reusable rule:\n{rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. Using the full state, judge for EACH '
        'situation whether the rule\'s action is the best action there, and '
        'give the best action. Then return a REFINED rule that keeps the '
        'rule\'s action wherever it is best but excludes situations where it '
        'is not (add conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + VOCAB + SEMANTICS + '\n'.join(lines))


BLIND_VERSION = 'blind_check_v3'
BLIND_MIN_AGREEMENT = .5      # keep a rule if > half its checks agree


def blind_prompt(situations):
    lines = []
    for i, s in enumerate(situations):
        lines.append(
            f'SITUATION {i}:\nFULL MAP:\n{s["full_map"]}\n'
            f'OBJECT FACTS: {json.dumps(s["facts"])}\n'
            f'Agent pose: {s["pose"]}\n')
    return (
        f'PROMPT VERSION: {BLIND_VERSION}\n'
        'You are the TEACHER for a DoorKey student (reach the green goal; '
        'the key unlocks the locked door). For EACH situation below, give '
        'the single best next action for the agent, using the full state.\n'
        'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (cell in '
        'front, without moving), 4 drop, 5 toggle (open/unlock the door in '
        'front; unlocking needs the key), 6 done (does nothing here). A key, '
        'wall or closed door in front blocks forward. Agent dir 0=east '
        '1=south 2=west 3=north; x grows east, y grows south.\n'
        + '\n'.join(lines))


def blind_schema():
    answer = dict(type='object', additionalProperties=False,
                  required=['situation', 'best_action'],
                  properties=dict(situation={'type': 'integer'},
                                  best_action={'type': 'integer',
                                               'enum': list(range(7))}))
    return dict(type='object', additionalProperties=False,
                required=['answers'],
                properties=dict(answers=dict(type='array', items=answer)))


# ------------------------------------------------------------ build steps

def export(out, rows, name):
    (out / f'{name}.json').write_text(json.dumps(rows, indent=1))
    (out / f'{name}_manifest.json').write_text(json.dumps(dict(
        study=STUDY, stage=name, requests_sha256=digest(rows),
        cases=len(rows), model=MODEL, api_calls=0), indent=1))


def build(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    panels = build_panels()
    rows = []
    for k, state in enumerate(panels['consult']):
        b = body(consult_prompt(state), consult_schema(), 'scoped_rule_v3')
        rows.append(dict(case_id=f'consult_{k:03d}', condition='v3_consult',
                         split='consult', model=MODEL, request=b,
                         request_sha256=digest(b)))
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(panels))
    export(out, rows, 'consult_requests')
    (out / 'example_consult_prompt.txt').write_text(
        rows[0]['request']['input'][0]['content'])
    print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                      for k, v in panels.items()}))


def parse(answer, with_now=True):
    now = int(answer['action_now']) if with_now else None
    if answer['abstain']:
        return now, None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in FIELDS[e['field']])
    return now, (condition, int(answer['action']), exceptions)


def read_replies(path):
    return {json.loads(line)['case_id']: json.loads(line)
            for line in Path(path).read_text().splitlines()}


def consult_rules(out):
    panels = json.loads((Path(out) / 'panels.json').read_text())
    replies = read_replies(Path(out) / 'consult_replies.jsonl')
    parsed = []
    for k, state in enumerate(panels['consult']):
        row = replies.get(f'consult_{k:03d}')
        if not row or row['response_status'] != 'completed':
            parsed.append((state, None, None, 'missing_or_incomplete'))
            continue
        now, rule = parse(row['answer'])
        parsed.append((state, now, rule, 'rule' if rule else 'abstained'))
    return panels, parsed


def export_blind(out):
    """Blind point checks on the SAME situations, without the rule."""
    out = Path(out)
    panels = json.loads((out / 'panels.json').read_text())
    pool = {s['native']: s for s in panels['pool']}
    rows = []
    for item in json.loads((out / 'refine_plan.json').read_text()):
        if not item['situations']:
            continue
        sits = [pool[n] for n in item['natives']]
        b = body(blind_prompt(sits), blind_schema(), BLIND_VERSION)
        rows.append(dict(case_id=f"blind_{item['case']:03d}",
                         condition='v3_blind', split='blind', model=MODEL,
                         request=b, request_sha256=digest(b)))
    export(out, rows, 'blind_requests')
    (out / 'example_blind_prompt.txt').write_text(
        rows[0]['request']['input'][0]['content'])
    print(f'{len(rows)} blind-check requests (rule never shown)')


def blind_filtered(out, parsed, panels, strict=False):
    """Keep a rule if its action agrees with the blind point answers."""
    out = Path(out)
    replies = read_replies(out / 'blind_replies.jsonl')
    plan = {p['case']: p for p in json.loads(
        (out / 'refine_plan.json').read_text())}
    kept, status = [], Counter()
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        row = replies.get(f'blind_{k:03d}')
        if row is None or row['response_status'] != 'completed':
            status['unchecked_dropped'] += 1
            continue
        n = plan[k]['situations']
        answers = {a['situation']: a['best_action']
                   for a in row['answer']['answers']
                   if 0 <= a['situation'] < n}
        agree = sum(answers.get(i) == rule[1] for i in range(n))
        ok = (agree == n) if strict else (agree / n > BLIND_MIN_AGREEMENT)
        status['kept' if ok else 'dropped'] += 1
        if ok:
            kept.append(rule)
    return kept, dict(status)


def export_refine(out):
    out = Path(out)
    panels, parsed = consult_rules(out)
    rng = np.random.default_rng(7)
    rows, plan = [], []
    for k, (state, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        matches = [s for s in panels['pool']
                   if executable(rule, s['v3'])]
        if not matches:
            plan.append(dict(case=k, situations=0))
            continue
        pick = [matches[i] for i in sorted(rng.choice(
            len(matches), min(K_SITUATIONS, len(matches)), replace=False))]
        b = body(refine_prompt(rule, pick), refine_schema(),
                 'scope_check_v3')
        rows.append(dict(case_id=f'refine_{k:03d}', condition='v3_refine',
                         split='refine', model=MODEL, request=b,
                         request_sha256=digest(b)))
        plan.append(dict(case=k, situations=len(pick),
                         pool_matches=len(matches),
                         natives=[s['native'] for s in pick]))
    export(out, rows, 'refine_requests')
    (out / 'refine_plan.json').write_text(json.dumps(plan, indent=1))
    if rows:
        (out / 'example_refine_prompt.txt').write_text(
            rows[0]['request']['input'][0]['content'])
    print(f'{len(rows)} refinement requests; '
          f"{sum(p['situations'] == 0 for p in plan)} rules had no pool "
          'match (kept unrefined)')


# ----------------------------------------------------------------- compare

def coverage(policy, states):
    out = Counter()
    for s in states:
        action, status = policy(s)
        out[status] += 1
        if action is not None:
            out['correct' if action in s['optimal'] else 'incorrect'] += 1
    advised = out['correct'] + out['incorrect']
    return dict(states=len(states), advised=advised,
                correct=out['correct'], incorrect=out['incorrect'],
                abstain_conflict=out['conflict'],
                precision=out['correct'] / advised if advised else None)


def compare(out, confirm=False):
    out = Path(out)
    panels, parsed = consult_rules(out)
    states = panels['confirm' if confirm else 'develop']
    rules = [r for _, _, r, _ in parsed if r]
    methods = {'handwritten': coverage(
        lambda s: advise(list(hand_rules_v3()), s['v3']), states)}
    replay = {}
    for state, now, _, _ in parsed:
        if now is not None:
            replay.setdefault(state['image'], set()).add(now)

    def replay_policy(s):
        acts = replay.get(s['image'], set())
        return ((next(iter(acts)), 'advised') if len(acts) == 1 else
                (None, 'conflict' if acts else 'no_rule'))
    methods['llm_action_now_replay'] = coverage(replay_policy, states)
    methods['llm_rules_raw'] = coverage(lambda s: advise(rules, s['v3']),
                                        states)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rules))
    shuffled = [(rules[i][0], rules[j][1], rules[i][2])
                for i, j in enumerate(order)]
    methods['llm_rules_shuffled_actions'] = coverage(
        lambda s: advise(shuffled, s['v3']), states)
    result = dict(
        panel='confirm' if confirm else 'develop', states=len(states),
        consult_status=dict(Counter(st for *_, st in parsed)),
        action_now_correct_at_own_state=sum(
            now is not None and now in s['optimal']
            for s, now, _, _ in parsed),
        rule_action_correct_at_own_state=sum(
            r is not None and executable(r, s['v3']) and
            r[1] in s['optimal'] for s, _, r, _ in parsed))
    refine_path = out / 'refine_replies.jsonl'
    if refine_path.exists():
        replies = read_replies(refine_path)
        refined, statuses = [], Counter()
        for k, (_, _, rule, _) in enumerate(parsed):
            if rule is None:
                continue
            row = replies.get(f'refine_{k:03d}')
            if row is None:                    # no pool match: unrefined
                refined.append(rule)
                statuses['kept_unchecked'] += 1
            elif row['response_status'] != 'completed':
                statuses['refine_incomplete_dropped'] += 1
            else:
                _, new = parse(row['answer'], with_now=False)
                statuses['refined' if new else 'refine_abstained'] += 1
                if new:
                    refined.append(new)
        methods['llm_rules_refined'] = coverage(
            lambda s: advise(refined, s['v3']), states)
        result['refine_status'] = dict(statuses)
    if (out / 'blind_replies.jsonl').exists():
        for strict in (False, True):
            kept, status = blind_filtered(out, parsed, panels, strict)
            name = 'llm_rules_blind_strict' if strict else 'llm_rules_blind'
            methods[name] = coverage(lambda s: advise(kept, s['v3']), states)
            result[f'{name}_status'] = status
    result['methods'] = methods
    result['rule_counts'] = dict(raw=len(rules), dedup=len(dedupe(rules)))
    if refine_path.exists():
        result['rule_counts'].update(refined=len(refined),
                                     refined_dedup=len(dedupe(refined)))
    result['cost'] = {stage: receipts(out / f'{stage}_replies.raw.jsonl')
                      for stage in ('consult', 'refine', 'blind')
                      if (out / f'{stage}_replies.raw.jsonl').exists()}
    calls = sum(c['calls'] for c in result['cost'].values())
    result['per_call'] = {
        name: dict(correct=m['correct'] / calls,
                   incorrect=m['incorrect'] / calls)
        for name, m in methods.items() if calls and name != 'handwritten'}
    name = 'compare_confirm.json' if confirm else 'compare_develop.json'
    (out / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


def dedupe(rules):
    seen, out = set(), []
    for r in rules:
        key = json.dumps([sorted(r[0].items()), r[1], sorted(r[2])])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def receipts(path):
    ends = [json.loads(line) for line in Path(path).read_text().splitlines()
            if json.loads(line)['event'] == 'END']
    return dict(calls=len(ends), completed=sum(
        e['response_status'] == 'completed' for e in ends),
        dollars=sum(e.get('dollars') or 0 for e in ends))


def hand_rules_v3():
    """v2's hand-written rules, restated in v3 fields (same meaning)."""
    for condition, action, exceptions in v2.HAND_RULES_V2:
        yield condition, action, exceptions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('action', choices=('build', 'export-refine',
                                      'export-blind', 'compare'))
    p.add_argument('--out', type=Path, default=Path('results') / STUDY)
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    {'build': build, 'export-refine': export_refine,
     'export-blind': export_blind}.get(
        args.action, lambda o: compare(o, args.confirm))(args.out)


if __name__ == '__main__':
    main()
