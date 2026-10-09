"""
Student-checkable evidence for teacher advice.

A privileged teacher acts on the whole map; a student with a 7x7 window
can only imitate what its own information determines. Identical student
observations can carry different teacher actions (label aliasing), and
injected aliasing measurably slows learning
(`research/alias_injection_finding_2026-09-22.md`). Whether aliasing is
what destroys the S3R3 count student is a CANDIDATE explanation, not an
established cause: a recurrent student's history can differ between
identical images, and two different teacher actions can both be valid.

Every teacher here already says WHY it acts. The claim a teacher sends
is the object its decision rests on, named the way a sentence would name
it: its type and colour ("the yellow key"). The DoorKey oracle names its
phase target, the MultiRoom planner the next door or the goal, the
BabyAI bot the object on top of its subgoal stack, and an evidence-citing
LLM names one in its structured reply. Positions ("explore towards
(3, 5)") are not objects and cannot be checked, so they are not claims.

The check is STUDENT-SIDE by construction: `student_verifies(image,
claim)` reads only the student's own symbolic observation (the 7x7x3
MiniGrid encoding of type, colour and state, with occluded cells marked
unseen and the carried object in the agent's cell) and the claim. No
coordinate conversion, pose or hidden map content enters the decision, so
changing anything the student cannot see cannot change acceptance
(tested). A claim naming an object the student cannot see -- because it
is out of view, occluded, or does not exist (a hallucination) -- is
rejected. The check does not certify that the advised action is correct.

Gating is decided at the END of each rollout, from per-label verdicts
recorded during it, so that the placebo can match the gate's delivered
count exactly:

none        every usable label is delivered (all previous experiments)
visible     only labels whose claimed evidence the student can see
random      the same NUMBER of labels as `visible` would deliver in this
            rollout, drawn uniformly from the rollout's usable labels,
            blind to evidence (its own generator) -- same dose and timing
            at rollout granularity, no evidence information
inverse     only labels whose evidence the student cannot see
consistent  explanation-free baseline: only while the student's exact
            observation has never carried two teacher actions. It needs
            repeat consultation at identical observations, which a dense
            free teacher supplies and a budgeted teacher does not.
"""

import hashlib
import re
from copy import deepcopy

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

GATE_MODES = ('none', 'visible', 'random', 'inverse', 'consistent')
EVIDENCE_TEACHERS = ('oracle', 'door_bfs', 'bot', 'llm_general')
EVIDENCE_OBJECTS = ('key', 'door', 'goal', 'ball', 'box', 'none')
EVIDENCE_COLORS = tuple(COLOR_TO_IDX) + ('none',)
CITATION_MODES = ('', 'cite', 'cite_view')
_POSITION = re.compile(r'\((\d+),\s*(\d+)\)')
_INTERACT = ('OpenSubgoal', 'PickupSubgoal', 'DropSubgoal', 'CloseSubgoal')


# --------------------------------------------------------------------
# Student side: only the student's observation and the teacher's claim.
# --------------------------------------------------------------------

def student_image(unwrapped):
    """The student's own symbolic observation (7x7x3 type/colour/state)."""

    return np.asarray(unwrapped.gen_obs()['image'])


def student_verifies(image, claim):
    """
    Can the student see an object matching the teacher's claim?

    `claim` is (object_type, colour) or None. Reads nothing but `image`.
    """

    if claim is None:
        return False
    kind, color = claim
    if kind not in OBJECT_TO_IDX or color not in COLOR_TO_IDX:
        return False
    image = np.asarray(image)
    return bool(np.any((image[..., 0] == OBJECT_TO_IDX[kind])
                       & (image[..., 1] == COLOR_TO_IDX[color])))


# --------------------------------------------------------------------
# Teacher side: which object the teacher's own reason names.
# --------------------------------------------------------------------

def evidence_cell(teacher_kind, teacher, unwrapped, advice):
    """
    The absolute cell the teacher's reason names, or None.

    Teacher-side bookkeeping only (it reads the teacher's map); the
    student never uses the cell. Read immediately after `recommend` for
    the same state.
    """

    if teacher_kind == 'oracle':
        from envs.state import extract_doorkey_state
        _action, target, _key, _door = teacher._solve(
            extract_doorkey_state(unwrapped))
        return (int(target[0]), int(target[1]))
    if teacher_kind == 'door_bfs':
        match = _POSITION.search(getattr(advice, 'explanation', '') or '')
        return (int(match.group(1)), int(match.group(2))) if match else None
    if teacher_kind == 'bot':
        return _bot_evidence(teacher, unwrapped)
    if teacher_kind == 'llm_general':
        cell = (getattr(advice.cost, 'metadata', None) or {}).get(
            'evidence_cell')
        return None if cell is None else (int(cell[0]), int(cell[1]))
    raise ValueError(f'No evidence rule for teacher {teacher_kind!r}')


def teacher_claim(teacher_kind, teacher, unwrapped, advice):
    """
    The message the teacher sends: (object_type, colour), or None.

    A free planner names the object at its evidence cell. An LLM's claim
    is exactly what it wrote -- its own type and colour words -- so a
    hallucinated object stays hallucinated and the student rejects it.
    """

    if teacher_kind == 'llm_general':
        metadata = getattr(advice.cost, 'metadata', None) or {}
        kind = metadata.get('evidence_object')
        color = metadata.get('evidence_color')
        if not kind or kind == 'none' or not color or color == 'none':
            return None
        return (str(kind), str(color))
    cell = evidence_cell(teacher_kind, teacher, unwrapped, advice)
    if cell is None:
        return None
    x, y = cell
    if not (0 <= x < unwrapped.width and 0 <= y < unwrapped.height):
        return None
    obj = unwrapped.grid.get(x, y)
    if obj is None or obj.type not in EVIDENCE_OBJECTS:
        return None
    return (obj.type, obj.color)


def _bot_evidence(teacher, unwrapped):
    from minigrid.core.world_object import WorldObj
    from minigrid.utils.baby_ai_bot import ObjDesc

    bot = getattr(teacher, '_bot', None)
    if bot is None or not bot.stack:
        return None
    subgoal = bot.stack[-1]
    name = type(subgoal).__name__
    if name in _INTERACT:
        return tuple(int(v) for v in unwrapped.front_pos)
    if name != 'GoNextToSubgoal':
        return None
    datum = subgoal.datum
    if isinstance(datum, ObjDesc):
        _obj, pos = bot._find_obj_pos(datum, subgoal.reason == 'PutNext')
    elif isinstance(datum, WorldObj):
        pos = datum.cur_pos
    else:
        pos = datum
    return None if pos is None else (int(pos[0]), int(pos[1]))


# --------------------------------------------------------------------
# LLM citation request (opt-in; see llm_general.evidence_citations).
# --------------------------------------------------------------------

def view_cells(unwrapped):
    """Absolute cells the agent can see now, for the 'cite_view' prompt."""

    _grid, vis_mask = unwrapped.gen_obs_grid()
    out = set()
    ax, ay = unwrapped.agent_pos
    reach = unwrapped.agent_view_size
    for x in range(max(0, ax - reach), min(unwrapped.width, ax + reach + 1)):
        for y in range(max(0, ay - reach),
                       min(unwrapped.height, ay + reach + 1)):
            rel = unwrapped.relative_coords(x, y)
            if rel is not None and vis_mask[rel[0], rel[1]]:
                out.add((x, y))
    return out


def student_view_map(unwrapped, ascii_map):
    """`ascii_map` with every cell the student cannot see drawn as '?'."""

    cells = view_cells(unwrapped)
    return '\n'.join(
        ''.join(ch if (x, y) in cells else '?' for x, ch in enumerate(row))
        for y, row in enumerate(ascii_map.split('\n')))


def evidence_schema(base):
    """Add one cited evidence object to a structured teacher response."""

    result = deepcopy(base)
    schema = result['format']['schema']
    schema['properties'].update({
        'evidence_object': {'type': 'string',
                            'enum': list(EVIDENCE_OBJECTS)},
        'evidence_color': {'type': 'string',
                           'enum': list(EVIDENCE_COLORS)},
        'evidence_x': {'type': 'integer'},
        'evidence_y': {'type': 'integer'},
    })
    schema['required'] += ['evidence_object', 'evidence_color',
                           'evidence_x', 'evidence_y']
    return result


def evidence_prompt(unwrapped, ascii_map, show_view):
    """
    Ask for the object the advice rests on.

    'cite' does not show the student's view: the teacher explains from
    what it knows and the student checks. 'cite_view' shows the view --
    the teacher-side condition. The screen found 'cite' separates better.
    """

    view = ''
    if show_view:
        view = ('\nThe student sees only a 7x7 window ahead and not '
                'through walls or closed doors. This is the map with '
                'every cell it cannot see now replaced by ?:\n'
                f'{student_view_map(unwrapped, ascii_map)}\n')
    return (f'{view}\nAlso name the single object that is your main '
            'evidence for this action: the object the action is heading '
            'for or acting on (evidence_object, evidence_color, and its '
            'map cell evidence_x, evidence_y). Use "none", "none", -1, -1 '
            'only if no object justifies it.')


# --------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------

class EvidenceGate:
    """
    Record a verdict per usable label, then select at rollout end.

    `verdict` is called once per usable label as it arrives and never
    changes the teacher's answer or its record. `finalize` turns the
    rollout's usable-label mask into the mask the action loss sees. The
    placebo draws from its own generator so the policy's random stream is
    identical across modes.
    """

    def __init__(self, mode, teacher_kind, seed, keep_fraction=1.0):
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError('keep_fraction must lie in (0, 1]')
        if mode not in GATE_MODES:
            raise ValueError(f'evidence gate must be one of {GATE_MODES}')
        if mode not in ('none', 'consistent') and (
                teacher_kind not in EVIDENCE_TEACHERS):
            raise ValueError(
                f'teacher {teacher_kind!r} states no checkable evidence')
        self.mode = mode
        self.teacher_kind = teacher_kind
        self.keep_fraction = float(keep_fraction)
        self.rng = np.random.default_rng(seed)
        # observation digest -> bitmask of teacher actions seen there
        self.actions_seen = {}
        self.totals = dict(usable=0, verified=0, accepted=0, rejected=0,
                           used=0, no_claim=0, contradicted=0)

    def verdict(self, teacher, unwrapped, advice):
        """True if this label passes the mode's evidence rule."""

        self.totals['usable'] += 1
        if self.mode == 'none':
            return True  # dose-only selection reads no evidence
        image = student_image(unwrapped)
        if self.mode == 'consistent':
            key = hashlib.blake2b(np.ascontiguousarray(image).tobytes(),
                                  digest_size=12).digest()
            mask = self.actions_seen.get(key, 0) | (1 << int(advice.action))
            self.actions_seen[key] = mask
            ok = (mask & (mask - 1)) == 0  # one action ever seen here
            self.totals['contradicted'] += int(not ok)
            return ok
        claim = teacher_claim(self.teacher_kind, teacher, unwrapped, advice)
        self.totals['no_claim'] += int(claim is None)
        ok = student_verifies(image, claim)
        self.totals['verified'] += int(ok)
        return ok

    def finalize(self, usable, verdicts, used_for_training):
        """
        Mask of labels the action loss will see for this rollout.

        `usable` and `verdicts` are same-shaped 0/1 arrays over the
        rollout; `used_for_training` says whether the distillation
        coefficient is positive at this update.
        """

        usable = np.asarray(usable) > 0
        passed = usable & (np.asarray(verdicts) > 0)
        # Dose: with keep_fraction f < 1 every mode delivers at most
        # round(f * usable) labels this rollout, drawn from its own pool.
        # visible and random then deliver the same count whenever enough
        # labels pass, so dose and timing match and only selection by
        # evidence differs.
        cap = (None if self.keep_fraction >= 1.0
               else int(round(self.keep_fraction * usable.sum())))
        if self.mode in ('visible', 'consistent'):
            keep = self._draw(passed, cap)
        elif self.mode == 'inverse':
            keep = self._draw(usable & ~passed, cap)
        elif self.mode == 'random':
            count = int(passed.sum()) if cap is None else min(
                cap, int(passed.sum()))
            keep = self._draw(usable, count)
        else:
            keep = self._draw(usable, cap)
        row = dict(usable=int(usable.sum()), passed=int(passed.sum()),
                   cap=cap, accepted=int(keep.sum()),
                   rejected=int(usable.sum() - keep.sum()),
                   used=int(keep.sum()) if used_for_training else 0)
        self.totals['accepted'] += row['accepted']
        self.totals['rejected'] += row['rejected']
        self.totals['used'] += row['used']
        return keep, row

    def _draw(self, pool, count):
        """`count` labels uniformly from `pool` (all of it if None)."""
        pool = np.asarray(pool, dtype=bool)
        if count is None or count >= pool.sum():
            return pool.copy()
        chosen = self.rng.choice(np.flatnonzero(pool), size=count,
                                 replace=False)
        keep = np.zeros(pool.size, dtype=bool)
        keep[chosen] = True
        return keep.reshape(pool.shape)

    def stats(self):
        t = self.totals
        return {
            'evidence_gate': self.mode,
            'evidence_keep_fraction': self.keep_fraction,
            'evidence_usable': int(t['usable']),
            'evidence_verified': int(t['verified']),
            'evidence_no_claim': int(t['no_claim']),
            'evidence_accepted': int(t['accepted']),
            'evidence_rejected': int(t['rejected']),
            'evidence_used_in_training': int(t['used']),
            'evidence_accept_rate': (t['accepted'] / t['usable']
                                     if t['usable'] else None),
            'consistency_observations': len(self.actions_seen),
            'consistency_contradicted': int(t['contradicted']),
        }
