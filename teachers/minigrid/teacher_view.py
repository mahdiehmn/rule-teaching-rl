"""Teacher-view execution of a frozen rule bank (fix-wave addendum 6).

Protocol:
research/fix_wave_protocol_2026-09-29.md, addendum 6.

The question: does it matter that a rule's conditions are checked on what
the student can see? The rules, the predicate names and the matcher stay
exactly as they are; only the predicates' values change. Here every object
predicate is computed from the full map, in the agent's own frame: every
object on the grid counts as visible, at its true forward and rightward
offset, whether or not the student can see it (beyond its 7x7 window,
behind a wall, or behind the agent). So a rule can now fire on facts the
student cannot check.

Unchanged from the student's observer: `front` (the cell ahead is always
in view), `carrying`, and the KeyCorridor unlocking memory. An object
behind the agent takes ahead='behind' and a negative fwd, values no rule
was written with, so a rule that tests them does not fire, while a rule
that tests only the side does. An object absent from the grid (a carried
key, no closed door left) is not visible, as for the student.
"""

import importlib

UNKNOWN = 'unknown'
DOOR_STATES = ('open', 'closed', 'locked')
# Each student vocabulary: its own observer (for the unchanged fields), its
# object categories as (grid type, door states admitted or None), and how
# it names a door's state.
SCHEMES = {
    'doorkey_v3': (
        ('scripts.conditional_rules_v3', 'observe_v3', False),
        {'key': ('key', None), 'door': ('door', None),
         'goal': ('goal', None)}, DOOR_STATES),
    'multiroom_v1': (
        ('scripts.conditional_rules_multiroom', 'observe_mr', False),
        {'door': ('door', None), 'closed_door': ('door', (1, 2)),
         'goal': ('goal', None)}, ('open', 'closed', 'closed')),
    'keycorridor_v1': (
        ('scripts.conditional_rules_keycorridor', 'observe_kc', False),
        {'door': ('door', None), 'locked_door': ('door', (2,)),
         'key': ('key', None), 'ball': ('ball', None)}, DOOR_STATES),
    'keycorridor_mem_v1': (
        ('scripts.conditional_rules_keycorridor_mem', 'observe_kc_mem', True),
        {'door': ('door', None), 'locked_door': ('door', (2,)),
         'key': ('key', None), 'ball': ('ball', None)}, DOOR_STATES),
}
SUFFIX = '_teacher_view'


def objects(u):
    """(type, door state 0/1/2, fwd, right) of every object on the grid.

    fwd and right are the offsets in the agent's frame, the same ones the
    student's 7x7 view encodes (fwd = 6 - y, right = x - 3 there). The
    agent's own cell is skipped, as the student's observers skip it.
    """
    u = u.unwrapped
    ax, ay = (int(v) for v in u.agent_pos)
    dx, dy = (int(v) for v in u.dir_vec)
    rx, ry = (int(v) for v in u.right_vec)
    found = []
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None or (x, y) == (ax, ay) or c.type not in (
                    'door', 'key', 'ball', 'goal'):
                continue
            state = ((2 if c.is_locked else 0 if c.is_open else 1)
                     if c.type == 'door' else 0)
            ox, oy = x - ax, y - ay
            found.append((c.type, state, ox * dx + oy * dy,
                          ox * rx + oy * ry))
    return found


def nearest(candidates):
    """The student's choice order where it applies: Manhattan distance,
    then nearer ahead, then leftmost; objects behind come after those
    ahead at the same distance."""
    return min(candidates, key=lambda o: (abs(o[2]) + abs(o[3]), o[2] < 0,
                                          abs(o[2]), o[3]))


def attributes(fwd, right):
    return dict(fwd=str(fwd), right=str(right),
                side='left' if right < 0 else 'right' if right > 0
                else 'center',
                ahead='beside' if fwd == 0 else 'ahead' if fwd > 0
                else 'behind')


def observe(scheme, image, u):
    """The scheme's predicates with every object read from the full map."""
    (module, name, stateful), categories, door_names = SCHEMES[scheme]
    student = getattr(importlib.import_module(module), name)
    pred = dict(student(image, u) if stateful else student(image))
    found = objects(u)
    for cat, (kind, states) in categories.items():
        pool = [o for o in found if o[0] == kind
                and (states is None or o[1] in states)]
        if not pool:
            pred[f'{cat}_visible'] = 'no'
            for a in ('fwd', 'right', 'side', 'ahead'):
                pred[f'{cat}_{a}'] = UNKNOWN
            if cat == 'door':
                pred['door_state'] = UNKNOWN
            continue
        o = nearest(pool)
        pred[f'{cat}_visible'] = 'yes'
        pred.update({f'{cat}_{a}': v
                     for a, v in attributes(o[2], o[3]).items()})
        if cat == 'door':
            pred['door_state'] = door_names[o[1]]
    return pred


def observe_doorkey(image, u):
    return observe('doorkey_v3', image, u)


def observe_multiroom(image, u):
    return observe('multiroom_v1', image, u)


def observe_keycorridor(image, u):
    return observe('keycorridor_v1', image, u)


def observe_keycorridor_mem(image, u):
    return observe('keycorridor_mem_v1', image, u)


# Registered in teachers.minigrid.rule_bank.OBSERVERS under these names.
OBSERVERS = {
    'doorkey_v3' + SUFFIX: (__name__, 'observe_doorkey'),
    'multiroom_v1' + SUFFIX: (__name__, 'observe_multiroom'),
    'keycorridor_v1' + SUFFIX: (__name__, 'observe_keycorridor'),
    'keycorridor_mem_v1' + SUFFIX: (__name__, 'observe_keycorridor_mem'),
}
