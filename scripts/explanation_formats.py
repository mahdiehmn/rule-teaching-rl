"""Explanation formats for one fixed historical DoorKey lesson bank.

Design only: this module builds
requests, parses replies, checks what can be checked and freezes a bank.
It makes no API call, reservation or training decision.

The 256 cases, their 192/64 episode-disjoint split and their fixed
endorsed/foil action pairs come unchanged from the contrastive lesson
panel. The action pair is chosen by a full-state shortest-path planner,
as in every other experiment; the student always sees only its own
partial view. One teacher reply per case supplies five formats, each with
an explicit unknown option:

    plain        why the endorsed action helps            -> text embedding
    contrastive  why it helps relative to the foil, and   -> two embeddings
                 why the foil is worse
    subgoal      current target object, relation and       -> 3 categories
                 completion predicate
    consequence  immediate one-step change under each      -> 6 categories
                 supplied action: movement, inventory, door
    plan         the NEXT subgoal after the current one:   -> 3 categories
                 its object, egocentric direction and
                 distance from the agent now (privileged
                 whenever that object is out of view)

Information access (versioned in every request and bank identity):

    full_state   PRIMARY. The writer sees the true full map, pose,
                 inventory and objects, plus a mask of the cells the
                 student can see, and may explain beyond that mask:
                 hidden objects and later subgoals are allowed.
    local_only   Opt-in comparison. The writer sees only the student's
                 7x7 cells. Never the default.

Checks are separate: consequences one step ahead against native snapshot
restoration; the current subgoal against the state-derived phase; the
plan against the full state's actual object positions (map access, not
realized future events). A free-text `future_plan` is logged for the
independent audit and never trained. Prose is not machine-certified.

Target representation: every target is predicted from the CURRENT
encoded image plus the fixed action pair (encoder-only replay; historical
cases carry no stored histories). A privileged plan target is therefore
only partly predictable from the image by design; history-conditioned
targets belong to the separate memory branch.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np

REQUEST_VERSION = 'explanation_formats_request_v2'
ACCESS = ('full_state', 'local_only')
PRIMARY_ACCESS = 'full_state'
FORMATS = ('plain', 'contrastive', 'subgoal', 'consequence', 'plan')
UNKNOWN = 'unknown'
SUBGOAL_TARGET = ('key', 'door', 'goal', UNKNOWN)
SUBGOAL_RELATION = ('approach', 'pick_up', 'unlock_or_open', 'pass_through',
                    'reach', UNKNOWN)
SUBGOAL_PREDICATE = ('carrying_key', 'door_open', 'agent_on_goal', UNKNOWN)
MOVEMENT = ('move_forward', 'turn_left', 'turn_right', 'no_move', UNKNOWN)
INVENTORY = ('pick_up_key', 'drop_key', 'unchanged', UNKNOWN)
DOOR = ('unlock_and_open', 'open', 'close', 'unchanged', UNKNOWN)
PLAN_TARGET = ('key', 'door', 'goal', 'none', UNKNOWN)
PLAN_DIRECTION = ('ahead', 'left', 'right', 'behind', 'none', UNKNOWN)
PLAN_DISTANCE = ('near', 'medium', 'far', 'none', UNKNOWN)
SUBGOAL_FIELDS = (('target', SUBGOAL_TARGET), ('relation', SUBGOAL_RELATION),
                  ('predicate', SUBGOAL_PREDICATE))
CONSEQUENCE_FIELDS = (('movement', MOVEMENT), ('inventory', INVENTORY),
                      ('door', DOOR))
PLAN_FIELDS = (('next_target', PLAN_TARGET),
               ('next_direction', PLAN_DIRECTION),
               ('next_distance', PLAN_DISTANCE))
ROLES = ('endorsed', 'foil')
PHASE_SUBGOAL = {
    'seek_key': ('key', 'carrying_key'),
    'unlock_door': ('door', 'door_open'),
    'reach_target': ('goal', 'agent_on_goal'),
}
PHASE_NEXT = {'seek_key': 'door', 'unlock_door': 'goal',
              'reach_target': 'none'}
OBJECT_INDEX = {'key': 5, 'door': 4, 'goal': 8}
DIRECTIONS = ('east', 'south', 'west', 'north')
DIR_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
# Salt shared by both access conditions: on the same included cases the
# permuted control then uses the same donors in both.
PERMUTATION_SALT = 'explanation_formats_20260925:global_bundle_cycle'
ACTION_NAMES = ('turn left', 'turn right', 'move forward', 'pick up',
                'drop', 'toggle', 'done')
NEAR, MEDIUM = 3, 7   # Manhattan distance: near <= 3, medium <= 7


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _check_access(access):
    if access not in ACCESS:
        raise ValueError(f'information access must be one of {ACCESS}')


# ------------------------------------------------------------------ request

def response_schema():
    """Strict schema; every field required, every category has unknown.

    Identical for both access conditions, so only the prompt differs.
    """
    props = {
        'plain_prose': {'type': 'string'},
        'contrastive_endorsed': {'type': 'string'},
        'contrastive_foil': {'type': 'string'},
    }
    for name, values in SUBGOAL_FIELDS:
        props[f'subgoal_{name}'] = {'type': 'string', 'enum': list(values)}
    for role in ROLES:
        for name, values in CONSEQUENCE_FIELDS:
            props[f'consequence_{role}_{name}'] = {
                'type': 'string', 'enum': list(values)}
    for name, values in PLAN_FIELDS:
        props[f'plan_{name}'] = {'type': 'string', 'enum': list(values)}
    props['future_plan'] = {'type': 'string'}
    return dict(type='object', properties=props, required=list(props),
                additionalProperties=False)


def visible_cells(case):
    from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

    obs = np.asarray(case['state']['local_obs'])
    cells = []
    for x in range(7):
        for y in range(7):
            kind, color, status = map(int, obs[x, y])
            cells.append([x, y, IDX_TO_OBJECT[kind], IDX_TO_COLOR[color],
                          status])
    return cells


def full_state_view(case):
    """Full map, objects, pose and the student-visibility mask map."""
    from envs.state import extract_generic_state
    from scripts.explanation_screen_panel import restore
    from teachers.evidence import student_view_map
    from teachers.minigrid.llm_general import render_ascii_map

    env = restore(case['state'])
    try:
        full = render_ascii_map(env)
        x, y, direction, carrying, objects = extract_generic_state(env)
        mask = student_view_map(env, full)
    finally:
        env.close()
    lines = [f'- {color} {kind} at ({ox}, {oy})'
             + (f', {state}' if state else '')
             for kind, color, ox, oy, state in objects]
    return dict(full_map=full, student_map=mask, objects=lines,
                agent=(x, y), facing=DIRECTIONS[direction],
                carrying=carrying or 'nothing')


COMMON_RULES = (
    'DoorKey: reach the green goal. A key unlocks the same-colour locked '
    'door. Actions: 0 turn left, 1 turn right, 2 move forward, 3 pick up, '
    '4 drop, 5 toggle, 6 done.\n')

FIELD_GUIDE = (
    'Return, in one JSON object:\n'
    '- plain_prose: one sentence on why the endorsed action helps.\n'
    '- contrastive_endorsed: one sentence on why the endorsed action is '
    'better THAN THE FOIL; contrastive_foil: one sentence on why the foil '
    'is worse.\n'
    '- subgoal_target / subgoal_relation / subgoal_predicate: the object '
    'the endorsed action serves now, the relation it moves toward, and '
    'the predicate true when that subgoal is complete.\n'
    '- consequence_<endorsed|foil>_<movement|inventory|door>: the '
    'IMMEDIATE one-step change if that action is taken now.\n'
    '- plan_next_target: the object of the subgoal that comes AFTER the '
    'current one ("none" if the current subgoal is the last). '
    'plan_next_direction: where that object is relative to the agent\'s '
    'CURRENT position and facing: ahead, behind, left or right, decided '
    'by the larger of the forward and sideways offsets (ties count as '
    'ahead/behind); "none" if there is no next target. '
    'plan_next_distance: Manhattan distance from the agent now: near '
    f'(<= {NEAR}), medium ({NEAR + 1}-{MEDIUM}), far (>= {MEDIUM + 1}); '
    '"none" if there is no next target.\n'
    '- future_plan: one sentence on the steps after this one. It is '
    'recorded for review and never scored as a one-step consequence.\n'
    'Use "unknown" (the whole string, or the enum value) only where you '
    'cannot justify an answer.\n')


def prompt_text(case, access=PRIMARY_ACCESS):
    """The writer's view: full state (primary) or the student's cells."""
    _check_access(access)
    endorsed, foil = case['positive_action'], case['foil_action']
    pair = (f'A full-map planner FIXED the action pair: endorsed action '
            f'{endorsed} ({ACTION_NAMES[endorsed]}), foil action {foil} '
            f'({ACTION_NAMES[foil]}). Do not propose other actions; explain '
            'this pair.\n')
    if access == 'local_only':
        return (
            'You explain a teaching example using ONLY the student\'s local '
            'observation below. ' + COMMON_RULES
            + 'The agent is at local (3,6) facing (3,5); coordinates are '
            'egocentric; unseen cells are unknown. At (3,6) the image '
            'encodes the carried object, or empty. Door state 0 open, '
            '1 closed, 2 locked. Do not invent hidden routes or earlier '
            'progress.\n' + pair + FIELD_GUIDE
            + json.dumps(dict(local_cells=visible_cells(case),
                              endorsed_action=endorsed, foil_action=foil)))
    view = full_state_view(case)
    return (
        'You are the TEACHER and see the full state. The student sees only '
        'a 7x7 window ahead of it and not through walls or closed doors. '
        + COMMON_RULES
        + 'Coordinates: x grows east, y grows south, (0, 0) is top-left. '
        'Map: # wall, . floor, D door, k key, G goal; the arrow '
        '(> v < ^) is the agent.\n'
        f'FULL MAP:\n{view["full_map"]}\n'
        'Objects:\n' + '\n'.join(view['objects']) + '\n'
        f'The agent is at ({view["agent"][0]}, {view["agent"][1]}) facing '
        f'{view["facing"]}, carrying {view["carrying"]}.\n'
        'STUDENT VIEW MASK (the same map; ? = a cell the student cannot '
        f'see now):\n{view["student_map"]}\n'
        'Explain with everything you know, INCLUDING objects the student '
        'cannot see and later subgoals; your explanation need not stay '
        'inside the student\'s view.\n' + pair + FIELD_GUIDE)


def request(case, access=PRIMARY_ACCESS, model='gpt-5-mini', effort='low',
            max_output_tokens=4096):
    body = dict(
        model=model, max_output_tokens=max_output_tokens,
        service_tier='default', store=False,
        input=[{'role': 'user', 'content': prompt_text(case, access)}],
        text={'format': dict(type='json_schema', name='explanation_formats',
                             strict=True, schema=response_schema())},
    )
    if effort:
        body['reasoning'] = {'effort': effort}
    return body


def request_identity(panel, access=PRIMARY_ACCESS, **kwargs):
    """Version, access and a hash over every request body in the bank."""
    _check_access(access)
    digest = hashlib.sha256()
    for case in panel['cases']:
        digest.update(canonical(request(case, access, **kwargs)).encode())
    return dict(request_version=REQUEST_VERSION, information_access=access,
                request_sha256=digest.hexdigest(),
                cases=len(panel['cases']))


# -------------------------------------------------------------------- parse

def _known_text(text):
    text = (text or '').strip()
    return bool(text) and text.lower().rstrip('.') != UNKNOWN


def parse_reply(answer):
    """Validate one reply and split it into per-format targets.

    Returns {format: target or None}; None marks unknown/unusable for that
    format. Raises on schema violations, which are recorded as failures.
    """
    schema = response_schema()
    if set(answer) != set(schema['properties']):
        raise ValueError('Reply fields differ from the schema')
    for key, spec in schema['properties'].items():
        if not isinstance(answer[key], str):
            raise ValueError(f'{key} is not a string')
        if 'enum' in spec and answer[key] not in spec['enum']:
            raise ValueError(f'{key} is outside its enum')
    out = {}
    out['plain'] = ({'text': answer['plain_prose'].strip()}
                    if _known_text(answer['plain_prose']) else None)
    both = (answer['contrastive_endorsed'], answer['contrastive_foil'])
    out['contrastive'] = (
        {'endorsed_text': both[0].strip(), 'foil_text': both[1].strip()}
        if all(_known_text(t) for t in both) else None)
    subgoal = {name: answer[f'subgoal_{name}'] for name, _ in SUBGOAL_FIELDS}
    out['subgoal'] = None if UNKNOWN in subgoal.values() else subgoal
    consequence = {
        f'{role}_{name}': answer[f'consequence_{role}_{name}']
        for role in ROLES for name, _ in CONSEQUENCE_FIELDS}
    out['consequence'] = (None if UNKNOWN in consequence.values()
                          else consequence)
    plan = {name: answer[f'plan_{name}'] for name, _ in PLAN_FIELDS}
    out['plan'] = None if UNKNOWN in plan.values() else plan
    out['future_plan'] = answer['future_plan'].strip()
    return out


# ------------------------------------------------------- checkable ground

def true_consequences(case):
    """One-step effects of both supplied actions from native restoration."""
    from scripts.explanation_screen_panel import (
        consequence, restore, snapshot)

    env = restore(case['state'])
    try:
        before = snapshot(env)
    finally:
        env.close()
    result = {}
    for role, action in zip(ROLES, (case['positive_action'],
                                    case['foil_action'])):
        after = consequence(case['state'], int(action))
        result[f'{role}_movement'] = _movement(before, after)
        result[f'{role}_inventory'] = _inventory(before, after)
        result[f'{role}_door'] = _door(before, after)
    return result


def _movement(before, after):
    if after['position'] != before['position']:
        return 'move_forward'
    turn = (after['direction'] - before['direction']) % 4
    return {0: 'no_move', 1: 'turn_right', 3: 'turn_left'}.get(turn,
                                                               'no_move')


def _inventory(before, after):
    if before['carrying'] == after['carrying']:
        return 'unchanged'
    return 'pick_up_key' if after['carrying'] != 'nothing' else 'drop_key'


def _door(before, after):
    b, a = before['door_state'], after['door_state']
    if a == b:
        return 'unchanged'
    if b == 'locked' and a == 'open':
        return 'unlock_and_open'
    return 'open' if a == 'open' else 'close'


def egocentric(agent, facing, target):
    """(forward, right) offsets of `target` from the agent's pose."""
    fx, fy = DIR_VECTORS[facing]
    rx, ry = -fy, fx
    vx, vy = target[0] - agent[0], target[1] - agent[1]
    return vx * fx + vy * fy, vx * rx + vy * ry


def plan_truth(case):
    """Next subgoal's object, direction and distance from the full state.

    Map access, not realized future events: positions are where the
    objects are now; the next subgoal follows the phase rule.
    """
    nxt = PHASE_NEXT[case['phase']]
    if nxt == 'none':
        return dict(next_target='none', next_direction='none',
                    next_distance='none', next_hidden=False)
    grid = np.asarray(case['state']['full_grid'])
    found = np.argwhere(grid[:, :, 0] == OBJECT_INDEX[nxt])
    if len(found) != 1:
        raise ValueError(f'Expected one {nxt} in the full state')
    target = tuple(map(int, found[0]))
    forward, right = egocentric(case['state']['agent_pos'],
                                case['state']['agent_dir'], target)
    if abs(forward) >= abs(right):
        direction = 'ahead' if forward > 0 else 'behind'
    else:
        direction = 'right' if right > 0 else 'left'
    distance = abs(forward) + abs(right)
    local = np.asarray(case['state']['local_obs'])
    return dict(next_target=nxt, next_direction=direction,
                next_distance=('near' if distance <= NEAR else
                               'medium' if distance <= MEDIUM else 'far'),
                next_hidden=not bool(
                    (local[:, :, 0] == OBJECT_INDEX[nxt]).any()))


def score_case(case, parsed):
    """Machine-checkable agreement; prose is left to the human audit."""
    scores = {}
    if parsed['consequence'] is not None:
        truth = true_consequences(case)
        scores['consequence'] = {
            k: parsed['consequence'][k] == truth[k] for k in truth}
        scores['consequence_truth'] = truth
    if parsed['subgoal'] is not None:
        target, predicate = PHASE_SUBGOAL[case['phase']]
        scores['subgoal'] = dict(
            target=parsed['subgoal']['target'] == target,
            predicate=parsed['subgoal']['predicate'] == predicate)
    truth = plan_truth(case)
    scores['plan_next_hidden'] = truth['next_hidden']
    if parsed['plan'] is not None:
        scores['plan'] = {name: parsed['plan'][name] == truth[name]
                          for name, _ in PLAN_FIELDS}
        scores['plan_truth'] = {name: truth[name] for name, _ in PLAN_FIELDS}
    return scores


# ------------------------------------------------------------------- freeze

def bundle_permutation(rows):
    """Hash-sort then one cyclic shift, across episodes, blind to targets.

    Same construction as the explanation-isolation study, with its own
    salt. A case's whole bundle (every target of every format) moves to
    another case together, so every format's marginal is preserved.
    """
    order = sorted(range(len(rows)), key=lambda i: sha256_text(
        PERMUTATION_SALT + ':' + rows[i]['case_id']))
    donors = {i: order[(j + 1) % len(order)] for j, i in enumerate(order)}
    for i, j in donors.items():
        if rows[i]['episode'] == rows[j]['episode']:
            raise ValueError('A donor cannot come from the same episode')
    return donors


def known_cases(panel, replies):
    """Case ids whose reply is valid and known in every format."""
    known = set()
    for case in panel['cases']:
        reply = replies.get(case['case_id'], {})
        if reply.get('status') != 'valid':
            continue
        parsed = parse_reply(reply['answer'])
        if all(parsed[f] is not None for f in FORMATS):
            known.add(case['case_id'])
    return known


def freeze_bank(panel, replies, embed, access=PRIMARY_ACCESS,
                identity=None, restrict_to=None):
    """Freeze one access condition's bank; `embed(texts)` -> unit vectors.

    `replies` maps case_id -> {'status': 'valid'|..., 'answer': {...}}.
    Only cases known in ALL formats enter, so formats are compared on
    identical cases. `restrict_to` (e.g. the cases known under BOTH access
    conditions) holds the included set, and hence the permuted donors,
    fixed across an access comparison. Audit cases never train or donate.
    """
    _check_access(access)
    identity = identity or request_identity(panel, access)
    if identity['information_access'] != access:
        raise ValueError('Request identity is for a different access')
    rows, audit, coverage = [], [], Counter()
    for case in panel['cases']:
        reply = replies.get(case['case_id'], {})
        if reply.get('status') != 'valid':
            coverage['invalid'] += 1
            continue
        parsed = parse_reply(reply['answer'])
        known = [f for f in FORMATS if parsed[f] is not None]
        for f in known:
            coverage[f'{case["split"]}_{f}'] += 1
        item = dict(
            case_id=case['case_id'], split=case['split'],
            episode=case['episode_group'], phase=case['phase'],
            observation=case['state']['local_obs'],
            positive_action=case['positive_action'],
            foil_action=case['foil_action'],
            targets={f: parsed[f] for f in FORMATS},
            future_plan=parsed['future_plan'],
            checks=score_case(case, parsed),
        )
        if len(known) != len(FORMATS):
            coverage[f'{case["split"]}_excluded_by_intersection'] += 1
        elif restrict_to is not None and case['case_id'] not in restrict_to:
            coverage[f'{case["split"]}_excluded_by_pairing'] += 1
        else:
            (rows if case['split'] == 'train' else audit).append(item)
    texts = sorted({t for r in rows + audit for t in _texts(r)})
    vectors = dict(zip(texts, embed(texts))) if texts else {}
    for r in rows + audit:
        _attach_embeddings(r, vectors)
    donors = bundle_permutation(rows)
    for i, r in enumerate(rows):
        donor = rows[donors[i]]
        r['permuted'] = dict(donor_id=donor['case_id'],
                             targets=deepcopy(donor['targets']))
    changed = {f: sum(canonical(r['targets'][f])
                      != canonical(r['permuted']['targets'][f])
                      for r in rows) / max(1, len(rows)) for f in FORMATS}
    hidden = [r['checks']['plan_next_hidden'] for r in rows]
    return dict(
        study=f'explanation_formats_20260925_{access}',
        source_panel=panel.get('study'),
        information_access=access,
        request_version=identity['request_version'],
        request_sha256=identity['request_sha256'],
        formats=list(FORMATS), train=rows, audit=audit,
        coverage=dict(coverage), train_cases=len(rows),
        audit_cases=len(audit), paired_restriction=restrict_to is not None,
        permutation_salt=PERMUTATION_SALT,
        permuted_changed_fraction=changed,
        train_plan_next_hidden_fraction=(sum(hidden) / len(hidden)
                                         if hidden else None),
        embedding_dimension=(len(next(iter(vectors.values())))
                             if vectors else None),
        target_representation=(
            'current encoded image + fixed action pair; encoder-only '
            'replay; no recurrent history'),
        semantic_review='pending independent audit',
        claim='Model-written explanations of a fixed action pair; '
              'consequences one-step checked, plan checked against the '
              'full state, prose not certified',
    )


def _texts(row):
    t = row['targets']
    return [t['plain']['text'], t['contrastive']['endorsed_text'],
            t['contrastive']['foil_text']]


def _vector(values):
    # Six decimals keep a 1,536-d unit vector's cosine exact to ~1e-6
    # while keeping the frozen bank a manageable JSON file.
    return [round(float(v), 6) for v in values]


def _attach_embeddings(row, vectors):
    t = row['targets']
    t['plain']['embedding'] = _vector(vectors[t['plain']['text']])
    c = t['contrastive']
    c['endorsed_embedding'] = _vector(vectors[c['endorsed_text']])
    c['foil_embedding'] = _vector(vectors[c['foil_text']])


def audit_sheet(bank):
    """One row per audit case for the independent language/plan check."""
    sheet = []
    for r in bank['audit']:
        t = r['targets']
        sheet.append(dict(
            case_id=r['case_id'], phase=r['phase'],
            information_access=bank['information_access'],
            endorsed=r['positive_action'], foil=r['foil_action'],
            plain=t['plain']['text'],
            contrastive_endorsed=t['contrastive']['endorsed_text'],
            contrastive_foil=t['contrastive']['foil_text'],
            subgoal=canonical(t['subgoal']), plan=canonical(t['plan']),
            plan_next_hidden=r['checks']['plan_next_hidden'],
            future_plan=r['future_plan'],
            reviewer_plain_supported='', reviewer_contrastive_supported='',
            reviewer_subgoal_supported='', reviewer_future_plan_supported='',
            reviewer_notes=''))
    return sheet


# ------------------------------------------------ second-teacher bound

def checkable_units(panel, replies, audit_labels=None):
    """case_id -> {unit: correct?} over machine- or human-checked units.

    Units: the six one-step consequence fields, current subgoal target
    and predicate, the three plan fields, plus optional independent
    audit judgments {(case_id, format): bool}. An unknown or invalid
    answer is recorded as None (abstained), never as correct.
    """
    units = {}
    for case in panel['cases']:
        row = replies.get(case['case_id'], {})
        entry = {}
        parsed = (parse_reply(row['answer'])
                  if row.get('status') == 'valid' else None)
        truth = true_consequences(case)
        for key in truth:
            entry[f'consequence_{key}'] = (
                None if parsed is None or parsed['consequence'] is None
                else parsed['consequence'][key] == truth[key])
        target, predicate = PHASE_SUBGOAL[case['phase']]
        sub = None if parsed is None else parsed['subgoal']
        entry['subgoal_target'] = (None if sub is None
                                   else sub['target'] == target)
        entry['subgoal_predicate'] = (None if sub is None
                                      else sub['predicate'] == predicate)
        plan = None if parsed is None else parsed['plan']
        truth_plan = plan_truth(case)
        for name, _ in PLAN_FIELDS:
            entry[f'plan_{name}'] = (None if plan is None
                                     else plan[name] == truth_plan[name])
        for (case_id, fmt), ok in (audit_labels or {}).items():
            if case_id == case['case_id']:
                entry[f'audit_{fmt}'] = bool(ok)
        units[case['case_id']] = entry
    return units


def complementarity(units_a, units_b):
    """Upper bound on what choosing between two teachers could buy.

    Over units checked for both teachers: each teacher's accuracy (an
    abstention counts as not correct), the best single teacher, the
    ideal per-unit chooser (correct if either is), and where exactly one
    is right. The chooser is an oracle bound, not a deployable selector.
    """
    pairs = [(a[u], units_b[c][u]) for c, a in units_a.items()
             if c in units_b for u in a if u in units_b[c]]
    n = len(pairs)
    if not n:
        return dict(units=0)
    ok_a = [a is True for a, _ in pairs]
    ok_b = [b is True for _, b in pairs]
    acc_a, acc_b = sum(ok_a) / n, sum(ok_b) / n
    return dict(
        units=n,
        coverage_a=sum(a is not None for a, _ in pairs) / n,
        coverage_b=sum(b is not None for _, b in pairs) / n,
        accuracy_a=acc_a, accuracy_b=acc_b,
        best_single=max(acc_a, acc_b),
        oracle_chooser=sum(x or y for x, y in zip(ok_a, ok_b)) / n,
        only_a_right=sum(x and not y for x, y in zip(ok_a, ok_b)) / n,
        only_b_right=sum(y and not x for x, y in zip(ok_a, ok_b)) / n,
        both_wrong=sum(not x and not y for x, y in zip(ok_a, ok_b)) / n,
    )
