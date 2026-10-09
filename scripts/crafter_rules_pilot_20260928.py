"""Crafter rule-teaching pilot, offline stage (exploratory, time-boxed).

Contract, frozen
before any reply: research/crafter_rules_pilot_protocol_2026-09-28.md.

Question: can an explanation make one teacher consultation useful across
multiple situations, beyond simply replaying its action label? The
MiniGrid studies say yes for three grid tasks. This pilot asks whether
the same method writes useful rules in a different environment family,
Crafter, before any learning run is built. Only the environment-specific
parts are new; the teacher model, the rule form and the rule semantics
(conditional_rules_v3.executable/advise) are unchanged:

- student predicates read from what Crafter shows the agent: its 9x7
  local view, inventory, vitals and daylight;
- a privileged teacher view: a 17x13 map, exact inventory, vitals and
  achievements;
- states from a short random walk, sometimes after a jump to a random
  walkable cell, with a tech-tree phase injected (inventory, and a table
  or furnace placed when the phase needs one). Every recipe is
  deterministic in its world seed.

  build     panels, 36 consultation requests (0 calls)
  (collect) scripts/collect_prompt_reliability_20260927.py --split all
  compare   held-out coverage and effect, rules-as-policy rollouts
            against a random policy, and the frozen go/no-go decision
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import conditional_rules_v3 as v3

STUDY = 'crafter_rules_pilot_20260928'
OUT = Path('results') / STUDY
MODEL = v3.MODEL
UNKNOWN = v3.UNKNOWN

# World seeds from the frozen contract: consultation, held-out states
# and rollout episodes never overlap.
CONSULT_SEED0, N_CONSULT = 30_000_000, 36
HELD_OUT_SEED0, N_HELD_OUT = 30_100_000, 600
ROLLOUT_SEED0, N_ROLLOUT = 30_200_000, 30
ROLLOUT_CAP = 3000

# The student's local view is 9 cells wide and 7 tall, centred on the
# player (crafter.Env renders exactly this window above the inventory).
HALF_W, HALF_H = 4, 3
# The teacher sees a wider window, 17 by 13 cells.
MAP_HALF_W, MAP_HALF_H = 8, 6

ACTIONS = ('noop', 'move_left', 'move_right', 'move_up', 'move_down',
           'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
           'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe',
           'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword',
           'make_iron_sword')
NOOP, MOVES = 0, (1, 2, 3, 4)
MATERIALS = ('water', 'grass', 'stone', 'path', 'sand', 'tree', 'lava',
             'coal', 'iron', 'diamond', 'table', 'furnace')
CREATURES = ('cow', 'zombie', 'skeleton', 'arrow', 'plant')
FRONT = MATERIALS + CREATURES + ('plant_ripe', 'edge')
DIRS = ('left', 'right', 'up', 'down')
FACING = {(-1, 0): 'left', (1, 0): 'right', (0, -1): 'up', (0, 1): 'down'}
TARGETS = ('tree', 'water', 'stone', 'coal', 'iron', 'diamond', 'table',
           'furnace', 'lava', 'cow', 'zombie', 'skeleton', 'plant')
COUNTS = {'wood': ('0', '1', '2+'), 'stone': ('0', '1-3', '4+'),
          'coal': ('0', '1+'), 'iron': ('0', '1+'),
          'diamond': ('0', '1+'), 'sapling': ('0', '1+')}
TOOLS = ('wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword',
         'stone_sword', 'iron_sword')
VITALS = ('health', 'food', 'drink', 'energy')
YES_NO = ('yes', 'no')

# Every predicate the student computes, with the values a rule may name.
# Direction and distance of a type that is not in view are 'unknown',
# which no condition can request (three-valued semantics, as in v3).
FIELDS = {'front': FRONT, 'facing': DIRS, 'daylight': ('day', 'night'),
          'near_table': YES_NO, 'near_furnace': YES_NO}
for _t in TARGETS:
    FIELDS[f'{_t}_visible'] = YES_NO
    FIELDS[f'{_t}_dir'] = DIRS
    FIELDS[f'{_t}_dist'] = ('adjacent', 'near', 'far')
FIELDS.update(COUNTS)
FIELDS.update({t: YES_NO for t in TOOLS})
FIELDS.update({v: ('low', 'mid', 'high') for v in VITALS})

# Map symbols for the teacher's privileged view.
SYMBOL = {'grass': '.', 'sand': ':', 'path': '_', 'water': '~',
          'stone': '#', 'tree': 'T', 'lava': '!', 'coal': 'c',
          'iron': 'i', 'diamond': 'd', 'table': 't', 'furnace': 'f',
          'cow': 'C', 'zombie': 'Z', 'skeleton': 'S', 'arrow': '*',
          'plant': 'p', 'plant_ripe': 'P', 'edge': ' '}
PLAYER = {'left': '<', 'right': '>', 'up': '^', 'down': 'v'}

# Tech-tree phases injected into sampled states. Numbers are inclusive
# inventory ranges; 'table' and 'furnace' say where one is placed:
# 'adjacent' puts it inside the 3x3 crafting area, 'view' puts it in
# the local view but outside that area.
PHASES = {
    'start': {},
    'wood_1': {'wood': (1, 1)},
    'wood_2': {'wood': (2, 4)},
    'table_adjacent': {'wood': (1, 3), 'table': 'adjacent'},
    'table_in_view': {'wood': (1, 3), 'table': 'view'},
    'wood_pickaxe': {'wood_pickaxe': (1, 1), 'wood': (0, 2)},
    'stone_some': {'wood_pickaxe': (1, 1), 'wood': (0, 2),
                   'stone': (1, 3)},
    'stone_many': {'wood_pickaxe': (1, 1), 'wood': (1, 2),
                   'stone': (4, 6), 'table': 'adjacent'},
    'furnace': {'wood_pickaxe': (1, 1), 'stone_pickaxe': (1, 1),
                'wood': (1, 2), 'coal': (1, 2), 'stone': (0, 3),
                'table': 'adjacent', 'furnace': 'adjacent'},
    'iron': {'wood_pickaxe': (1, 1), 'stone_pickaxe': (1, 1),
             'wood': (1, 2), 'coal': (1, 2), 'iron': (1, 1),
             'table': 'adjacent', 'furnace': 'adjacent'},
}
# Achievements implied by a phase, shown to the teacher only.
IMPLIED = {'wood': 'collect_wood', 'stone': 'collect_stone',
           'coal': 'collect_coal', 'iron': 'collect_iron',
           'wood_pickaxe': 'make_wood_pickaxe',
           'stone_pickaxe': 'make_stone_pickaxe',
           'table': 'place_table', 'furnace': 'place_furnace'}

# Thresholds of the frozen decision rule (protocol section "Decision").
GO_ACHIEVEMENT_GAIN, NO_GO_ACHIEVEMENT_GAIN = 1.5, 0.5
GO_TABLE, GO_PICKAXE = .40, .20
GO_COVERAGE, GO_EFFECT, NO_GO_COVERAGE = .20, .70, .05


# -------------------------------------------------------------- the world

def make_env(seed):
    """
    Create and reset a Crafter world that skips image rendering.

    The pilot reads the world directly, and drawing the 64x64 image is
    most of the cost of a step, so the observation is replaced by None.
    """

    env = stable_env_class()(seed=seed)
    # Replace the rendered observation; reset() and step() call _obs().
    env._obs = lambda: None
    env.reset()
    return env


_STABLE_ENV = None


def stable_env_class():
    """
    A crafter.Env whose despawning does not depend on memory layout.

    Crafter keeps each chunk's objects in a set, which orders them by
    memory address, and despawns one picked by index, so a world
    depends on whatever the process allocated before. Sorting the
    candidates by position fixes the order (no two objects share a
    cell). The override lives on a subclass, not on the instance, so a
    deep copy of an environment balances its own world.
    """

    global _STABLE_ENV
    if _STABLE_ENV is None:
        import crafter

        class StableEnv(crafter.Env):

            def _balance_object(self, chunk, objs, *args):
                ordered = sorted(objs, key=lambda o: tuple(
                    int(v) for v in o.pos))
                return super()._balance_object(chunk, ordered, *args)

        _STABLE_ENV = StableEnv
    return _STABLE_ENV


def cell_kind(world, pos):
    """
    Name what occupies a world cell: a creature first, else a material.
    """

    from crafter import objects
    material, obj = world[pos]
    # Cells outside the 64x64 world read as None.
    if material is None:
        return 'edge'
    if obj is not None and not isinstance(obj, objects.Player):
        name = type(obj).__name__.lower()
        if name == 'plant' and obj.ripe:
            return 'plant_ripe'
        return name
    return material


def bucket(item, count):
    """
    Map an inventory count onto the values the rules can name.
    """

    if item == 'wood':
        return '0' if count == 0 else '1' if count == 1 else '2+'
    if item == 'stone':
        return '0' if count == 0 else '1-3' if count <= 3 else '4+'
    return '0' if count == 0 else '1+'


def vital(level):
    """
    Map a vital (0-9) onto low, mid or high.
    """

    return 'low' if level <= 3 else 'mid' if level <= 6 else 'high'


def observe(env):
    """
    Compute the student's predicates from what Crafter shows it.

    Everything here is visible in the agent's own observation: the 9x7
    local view (including the player's facing sprite and the darkness
    of night) and the inventory bar.
    """

    world, player = env._world, env._player
    px, py = (int(v) for v in player.pos)
    # Collect the offsets of every visible cell, grouped by what fills it.
    seen = {}
    for dx in range(-HALF_W, HALF_W + 1):
        for dy in range(-HALF_H, HALF_H + 1):
            if dx == 0 and dy == 0:
                continue
            kind = cell_kind(world, (px + dx, py + dy))
            kind = 'plant' if kind == 'plant_ripe' else kind
            seen.setdefault(kind, []).append((dx, dy))
    fx, fy = (int(v) for v in player.facing)
    pred = {'front': cell_kind(world, (px + fx, py + fy)),
            'facing': FACING[(fx, fy)],
            'daylight': 'day' if world.daylight >= .5 else 'night'}
    # A table or furnace counts for crafting inside the 3x3 around the
    # player, which is exactly what crafter's Player._make checks.
    near, _ = world.nearby(player.pos, 1)
    pred['near_table'] = 'yes' if 'table' in near else 'no'
    pred['near_furnace'] = 'yes' if 'furnace' in near else 'no'
    # For each type, describe the nearest visible instance: the first
    # move toward it along its longer axis, and how far it is.
    for target in TARGETS:
        spots = seen.get(target, [])
        if not spots:
            pred.update({f'{target}_visible': 'no', f'{target}_dir': UNKNOWN,
                         f'{target}_dist': UNKNOWN})
            continue
        dx, dy = min(spots, key=lambda d: (abs(d[0]) + abs(d[1]),
                                           abs(d[1]), d))
        if abs(dx) >= abs(dy):
            direction = 'left' if dx < 0 else 'right'
        else:
            direction = 'up' if dy < 0 else 'down'
        steps = abs(dx) + abs(dy)
        pred.update({f'{target}_visible': 'yes',
                     f'{target}_dir': direction,
                     f'{target}_dist': ('adjacent' if steps == 1 else
                                        'near' if steps <= 3 else 'far')})
    inventory = player.inventory
    pred.update({item: bucket(item, inventory[item]) for item in COUNTS})
    pred.update({t: 'yes' if inventory[t] > 0 else 'no' for t in TOOLS})
    pred.update({v: vital(inventory[v]) for v in VITALS})
    return pred


# ------------------------------------------------------ teacher's view

def teacher_map(env):
    """
    Render the teacher's 17x13 map around the player as text.
    """

    world, player = env._world, env._player
    px, py = (int(v) for v in player.pos)
    rows = []
    for dy in range(-MAP_HALF_H, MAP_HALF_H + 1):
        row = []
        for dx in range(-MAP_HALF_W, MAP_HALF_W + 1):
            if dx == 0 and dy == 0:
                row.append(PLAYER[FACING[tuple(int(v) for v in
                                               player.facing)]])
            else:
                row.append(SYMBOL[cell_kind(world, (px + dx, py + dy))])
        rows.append(''.join(row))
    return '\n'.join(rows)


def teacher_facts(env):
    """
    List the exact state only the teacher sees.
    """

    player = env._player
    near, _ = env._world.nearby(player.pos, 1)
    return dict(
        inventory={k: int(v) for k, v in player.inventory.items() if v},
        achievements=sorted(k for k, v in player.achievements.items()
                            if v > 0),
        daylight=round(float(env._world.daylight), 2),
        crafting_area=sorted(m for m in near if m),
        sleeping=bool(player.sleeping))


# ------------------------------------------------------- state sampling

def free_cells(env, low, high):
    """
    Placeable cells whose Chebyshev distance from the player is in
    [low, high] and that lie inside the student's view.
    """

    world, player = env._world, env._player
    px, py = (int(v) for v in player.pos)
    out = []
    for dx in range(-HALF_W, HALF_W + 1):
        for dy in range(-HALF_H, HALF_H + 1):
            ring = max(abs(dx), abs(dy))
            if not low <= ring <= high:
                continue
            material, obj = world[(px + dx, py + dy)]
            if obj is None and material in ('grass', 'sand', 'path'):
                out.append((px + dx, py + dy))
    return out


def jump(env, rng):
    """
    Move the player to a random free walkable cell, so that sampled
    states also show stone, coal, iron and water, as later in an
    episode.
    """

    world, player = env._world, env._player
    for _ in range(400):
        pos = (int(rng.integers(5, 59)), int(rng.integers(5, 59)))
        material, obj = world[pos]
        if obj is None and material in ('grass', 'sand', 'path'):
            world.move(player, np.array(pos))
            return True
    return False


def inject(env, rng, phase):
    """
    Give the player the inventory of a tech-tree phase, and place a
    table or furnace in view when the phase needs one.
    """

    player = env._player
    placed = {}
    for key, value in PHASES[phase].items():
        if key in ('table', 'furnace'):
            ring = (1, 1) if value == 'adjacent' else (2, 3)
            cells = [c for c in free_cells(env, *ring)
                     if key != 'furnace' or c not in placed.values()]
            if cells:
                pos = cells[int(rng.integers(len(cells)))]
                env._world[pos] = key
                placed[key] = pos
            continue
        low, high = value
        player.inventory[key] = int(rng.integers(low, high + 1))
    # Mark the achievements this history implies (teacher view only).
    for key, name in IMPLIED.items():
        if player.inventory.get(key, 0) > 0 or key in placed:
            player.achievements[name] = max(player.achievements[name], 1)
    return {k: [int(v) for v in p] for k, p in placed.items()}


def vary(env, rng):
    """
    Independently lower some vitals and sometimes move the clock to
    night, so rules see hungry, thirsty, tired and dark states too.
    """

    player = env._player
    changes = {}
    for name, chance in (('drink', .2), ('food', .2), ('energy', .2),
                         ('health', .15)):
        if rng.uniform() < chance:
            player.inventory[name] = int(rng.integers(1, 4))
            changes[name] = player.inventory[name]
    if rng.uniform() < .3:
        # Daylight is lowest when (step / 300) % 1 is near 0.7.
        env._step = 300 * int(rng.integers(0, 3)) + int(
            rng.integers(165, 255))
        env._update_time()
        changes['night'] = True
    return changes


def make_state(seed):
    """
    Rebuild the sampled state of one world seed, deterministically.

    Returns the environment and a JSON record of the recipe, the
    student's predicates and the teacher's view.
    """

    for attempt in range(8):
        rng = np.random.default_rng([seed, attempt])
        env = make_env(seed)
        jumped = bool(rng.uniform() < .5) and jump(env, rng)
        # Walk mostly by moving; other actions add chance events.
        walk = int(rng.integers(0, 121))
        for _ in range(walk):
            action = (int(rng.choice(MOVES)) if rng.uniform() < .8
                      else int(rng.integers(len(ACTIONS))))
            env.step(action)
            if env._player.health <= 0:
                break
        # Discard a walk that killed the player and try again.
        if env._player.health <= 0:
            continue
        # Wake the player: a sleeping player ignores every action, so
        # a state sampled mid-sleep cannot show which action helps.
        env._player.sleeping = False
        phase = list(PHASES)[int(rng.integers(len(PHASES)))]
        placed = inject(env, rng, phase)
        changes = vary(env, rng)
        return env, dict(seed=seed, attempt=attempt, jumped=jumped,
                         walk=walk, phase=phase, placed=placed,
                         variations=changes, pred=observe(env),
                         full_map=teacher_map(env),
                         facts=teacher_facts(env))
    raise RuntimeError(f'No living state for seed {seed}')


# --------------------------------------------------------------- prompts

TASK_TEXT = (
    'You are the TEACHER for a Crafter student. Crafter is a 2D world '
    'seen from above. The student should unlock as many of 22 '
    'achievements as possible and stay alive: collect wood, place a '
    'table, make pickaxes and swords, collect stone, coal, iron and '
    'diamond, place a furnace, drink water, eat cows and ripe plants, '
    'collect and place saplings, sleep and wake up, defeat zombies and '
    'skeletons.\nMECHANICS: the player faces left, right, up or down. A '
    'move action turns the player that way and steps only if that cell is '
    'grass, sand or path and free (stepping into lava kills). "do" acts on '
    'the FACING cell: a tree gives wood (no tool); stone and coal need a '
    'wood pickaxe; iron needs a stone pickaxe; diamond needs an iron '
    'pickaxe; water gives drink; grass gives a sapling with 10% chance; it '
    'attacks a cow, zombie or skeleton (a killed cow gives food); it eats a '
    'ripe plant. place_table needs 2 wood; place_furnace needs 4 stone and '
    'a table nearby; place_stone needs 1 stone; place_plant needs a sapling '
    'and grass in front. A placed item goes into the facing cell, which '
    'must be free grass, sand or path (stone may also go on water or '
    'lava). make_* needs a table nearby (iron tools also a furnace), where '
    'nearby means inside the 3x3 area around the player: wood pickaxe and '
    'wood sword cost 1 wood; stone pickaxe and stone sword cost 1 wood and '
    '1 stone; iron pickaxe and iron sword cost 1 wood, 1 coal and 1 iron. '
    'Food, drink and energy fall over time; health falls while any of them '
    'is zero and recovers otherwise. Sleeping restores energy; zombies '
    'appear on grass, mostly at night.')

ACTION_TEXT = (
    'ACTIONS: ' + ', '.join(f'{i} {a}' for i, a in enumerate(ACTIONS)) +
    '.\n')

VOCAB = (
    'STUDENT-OBSERVABLE PREDICATES (computed only from the student\'s '
    'own 9x7 view, 4 cells left and right and 3 up and down, and its '
    'inventory bar):\n'
    '- front: what fills the facing cell (a material, a creature, plant or '
    'plant_ripe, or edge)\n'
    '- facing: left, right, up or down; daylight: day or night\n'
    '- near_table, near_furnace: yes if one is inside the 3x3 crafting area\n'
    '- for each of ' + ', '.join(TARGETS) + ': <type>_visible yes/no; '
    '<type>_dir = the first move toward the NEAREST visible one (along '
    'its longer axis); <type>_dist = adjacent (1 step), near (2-3) or far '
    '(4+). _dir and _dist are unknown when the type is not visible.\n'
    '- inventory: wood 0/1/2+, stone 0/1-3/4+, coal, iron, diamond and '
    'sapling 0/1+; tools ' + ', '.join(TOOLS) + ' yes/no\n'
    '- vitals ' + ', '.join(VITALS) + ': low (0-3), mid (4-6), high '
    '(7-9)\n' + ACTION_TEXT)


def consult_prompt(state):
    """
    The consultation prompt: the same structure as the MiniGrid tasks.
    """

    return (
        'PROMPT VERSION: scoped_rule_crafter_v1\n' + TASK_TEXT +
        ' You see a 17x13 map; the student sees only its 9x7 view.\n'
        'MAP (rows top to bottom; the player is the arrow at the centre, '
        'pointing where it faces; . grass, : sand, _ path, ~ water, # stone, '
        'T tree, ! lava, c coal, i iron, d diamond, t table, f furnace, C '
        'cow, Z zombie, S skeleton, * arrow, p plant, P ripe plant):\n'
        f'{state["full_map"]}\n'
        f'EXACT STATE (teacher-visible): {json.dumps(state["facts"])}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB + v3.SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


def rule_schema():
    """
    The structured-output schema: every predicate as an optional
    condition, at most two exceptions, and one of the 17 actions.
    """

    condition = {k: {'type': 'string', 'enum': ['any', *v]}
                 for k, v in FIELDS.items()}
    exception = dict(type='object', additionalProperties=False,
                     required=['field', 'value'], properties=dict(
                         field={'type': 'string', 'enum': list(FIELDS)},
                         value={'type': 'string', 'enum': sorted(
                             {v for vs in FIELDS.values() for v in vs})}))
    actions = {'type': 'integer', 'enum': list(range(len(ACTIONS)))}
    props = dict(
        action_now=actions, abstain={'type': 'boolean'},
        condition=dict(type='object', additionalProperties=False,
                       required=list(FIELDS), properties=condition),
        action=actions,
        exceptions=dict(type='array', maxItems=2, items=exception),
        rationale={'type': 'string', 'maxLength': 240})
    return dict(type='object', additionalProperties=False,
                required=list(props), properties=props)


# ------------------------------------------------------------------ build

def build(out=OUT):
    """
    Sample every panel and write the consultation requests (0 calls).
    """

    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory; panels are frozen')
    consult = [make_state(CONSULT_SEED0 + k)[1] for k in range(N_CONSULT)]
    held_out = [make_state(HELD_OUT_SEED0 + k)[1]
                for k in range(N_HELD_OUT)]
    rows = []
    for k, state in enumerate(consult):
        body = v3.body(consult_prompt(state), rule_schema(),
                       'scoped_rule_crafter_v1')
        rows.append(dict(case_id=f'consult_{k:03d}',
                         condition='crafter_consult', split='consult',
                         model=MODEL, request=body,
                         request_sha256=v3.digest(body)))
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(dict(
        consult=consult, held_out=held_out,
        rollout_seeds=[ROLLOUT_SEED0 + k for k in range(N_ROLLOUT)])))
    (out / 'consult_requests.json').write_text(json.dumps(rows, indent=1))
    (out / 'consult_requests_manifest.json').write_text(json.dumps(dict(
        study=STUDY, stage='consult_requests', cases=len(rows), model=MODEL,
        api_calls=0, requests_sha256=v3.digest(rows)), indent=1))
    (out / 'example_consult_prompt.txt').write_text(
        rows[0]['request']['input'][0]['content'])
    phases = Counter(s['phase'] for s in consult)
    print(f'{len(rows)} consultation requests; phases {dict(phases)}; '
          f'{len(held_out)} held-out states')


# ------------------------------------------------------- rules and effect

def parse(answer):
    """
    Turn one reply into (action_now, rule or None), dropping 'any' and
    exception values a field cannot take.
    """

    now = int(answer['action_now'])
    if answer['abstain']:
        return now, None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in FIELDS[e['field']])
    return now, (condition, int(answer['action']), exceptions)


def rule_text(rule):
    """
    A readable rule with Crafter's action names (v3.rule_text knows
    only MiniGrid's seven actions).
    """

    condition, action, exceptions = rule
    return json.dumps(dict(when=condition, prefer=ACTIONS[action],
                           unless=[list(e) for e in exceptions]))


def consult_rules(out=OUT):
    """
    Pair each consultation state with its reply's action and rule.
    """

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


def signature(env, front):
    """
    What one step can change for the player: position, facing,
    inventory, achievements, sleep, the world's materials, and the
    health of the creature that was in front.
    """

    player = env._player
    return (tuple(int(v) for v in player.pos),
            tuple(int(v) for v in player.facing),
            tuple(sorted(player.inventory.items())),
            tuple(sorted(player.achievements.items())),
            bool(player.sleeping), env._world._mat_map.tobytes(),
            None if front is None else front.health)


def effective(env, action):
    """
    Whether one step of `action` differs from one step of noop.

    Both branches start from identical copies, random state included,
    so any difference comes from the action itself.
    """

    player = env._player
    target = tuple(int(v) for v in player.pos + np.array(player.facing))
    _, front = env._world[target]
    outcomes = []
    for chosen in (action, NOOP):
        memo = {}
        branch = copy.deepcopy(env, memo)
        branch.step(chosen)
        outcomes.append(signature(branch, None if front is None
                                  else memo[id(front)]))
    return outcomes[0] != outcomes[1]


# --------------------------------------------------------------- rollouts

def rollout(seed, rules=None):
    """
    Play one episode: the rule action where a rule fires, else (and
    always for the random baseline) a uniformly random action.
    """

    env = make_env(seed)
    rng = np.random.default_rng([seed, 1])
    fired = steps = 0
    done = False
    while not done and steps < ROLLOUT_CAP:
        action = None
        if rules is not None:
            action, _ = v3.advise(rules, observe(env))
        if action is None:
            action = int(rng.integers(len(ACTIONS)))
        else:
            fired += 1
        _, _, done, _ = env.step(action)
        steps += 1
    unlocked = sorted(k for k, v in env._player.achievements.items() if v)
    return dict(seed=seed, steps=steps, fired=fired,
                died=bool(env._player.health <= 0), achievements=unlocked)


def crafter_score(episodes):
    """
    Crafter's score: the geometric mean of per-achievement success
    rates in percent (Hafner, 2022), plus the rates themselves.
    """

    from crafter import constants
    rates = {a: 100 * float(np.mean([a in e['achievements']
                                     for e in episodes]))
             for a in constants.achievements}
    score = float(np.exp(np.mean(np.log(1 + np.array(
        list(rates.values()))))) - 1)
    return score, rates


def summary(episodes):
    score, rates = crafter_score(episodes)
    return dict(episodes=len(episodes), score=score,
                mean_achievements=float(np.mean(
                    [len(e['achievements']) for e in episodes])),
                mean_steps=float(np.mean([e['steps'] for e in episodes])),
                deaths=int(sum(e['died'] for e in episodes)),
                rule_share=float(np.mean([e['fired'] / e['steps']
                                          for e in episodes])),
                rates=rates)


def decide(coverage, effect, rules_run, random_run):
    """
    Apply the frozen go/no-go rule of the contract.
    """

    gain = rules_run['mean_achievements'] - random_run['mean_achievements']
    table = rules_run['rates']['place_table'] / 100
    pickaxe = rules_run['rates']['make_wood_pickaxe'] / 100
    if gain < NO_GO_ACHIEVEMENT_GAIN or coverage < NO_GO_COVERAGE:
        verdict = 'NO-GO'
    elif (gain >= GO_ACHIEVEMENT_GAIN and table >= GO_TABLE
          and pickaxe >= GO_PICKAXE and coverage >= GO_COVERAGE
          and effect >= GO_EFFECT):
        verdict = 'GO'
    else:
        verdict = 'INCONCLUSIVE'
    return dict(verdict=verdict, achievement_gain=gain,
                place_table=table, make_wood_pickaxe=pickaxe,
                coverage=coverage, effective_share=effect)


def compare(out=OUT):
    """
    Score the collected rules: held-out coverage and effect, then
    rules-as-policy against random rollouts, then the decision.
    """

    out = Path(out)
    panels, parsed = consult_rules(out)
    rules = [r for _, _, r, _ in parsed if r]
    # Exploratory, added after reading the replies and outside the
    # frozen decision: a rule with no condition fires in every state
    # and conflicts with every other rule, so the bank without such
    # rules is scored on a separate line.
    banks = {'raw': rules, 'without_empty': [r for r in rules if r[0]]}
    # Regenerate each held-out state with the deterministic generator
    # and label it with each bank. The build-time records in
    # panels.json came from Crafter's unpatched, address-dependent
    # despawning and cannot be rebuilt; nothing was computed on them.
    labels = {name: Counter() for name in banks}
    regenerated = []
    for seed in (record['seed'] for record in panels['held_out']):
        env, record = make_state(seed)
        regenerated.append(record)
        for name, bank in banks.items():
            action, status = v3.advise(bank, record['pred'])
            labels[name][status] += 1
            if action is not None:
                labels[name]['effective' if effective(env, action)
                             else 'no_effect'] += 1
    # Check that a state rebuilt later in the same process matches.
    for record in regenerated[:3]:
        if make_state(record['seed'])[1] != record:
            raise ValueError(f"State {record['seed']} is not deterministic")
    (out / 'held_out_regenerated.json').write_text(json.dumps(dict(
        note='Held-out states regenerated with the deterministic despawn '
             'patch; panels.json keeps the build-time records only for '
             'provenance.', states=regenerated)))

    def shares(counts):
        advised = counts['effective'] + counts['no_effect']
        return (advised / len(regenerated),
                counts['effective'] / advised if advised else 0.0)

    # Play the rollout seeds with no rules, the raw bank, and the bank
    # without empty rules (when it differs).
    seeds = panels['rollout_seeds']
    random_run = summary([rollout(s) for s in seeds])
    runs = {'raw': summary([rollout(s, rules) for s in seeds])}
    runs['without_empty'] = (
        runs['raw'] if len(banks['without_empty']) == len(rules)
        else summary([rollout(s, banks['without_empty']) for s in seeds]))
    result = dict(
        consult_status=dict(Counter(st for *_, st in parsed)),
        rules=len(rules),
        rule_text=[rule_text(r) for r in rules],
        action_now=dict(Counter(ACTIONS[n] for _, n, _, _ in parsed
                                if n is not None)),
        held_out=dict(states=len(regenerated), **labels['raw']),
        random=random_run, rules_as_policy=runs['raw'],
        decision=decide(*shares(labels['raw']), runs['raw'], random_run),
        exploratory_without_empty_rules=dict(
            rules=len(banks['without_empty']),
            held_out=dict(labels['without_empty']),
            rules_as_policy=runs['without_empty'],
            decision_if_it_were_primary=decide(
                *shares(labels['without_empty']), runs['without_empty'],
                random_run)))
    (out / 'compare.json').write_text(json.dumps(result, indent=1))
    for name in banks:
        coverage, effect = shares(labels[name])
        print(f"{name:14s} rules {len(banks[name]):2d}; held-out coverage "
              f"{coverage:.2f}, effective {effect:.2f}")
    for name, run in (('random', random_run), ('rules raw', runs['raw']),
                      ('rules w/o empty', runs['without_empty'])):
        print(f"{name:16s} score {run['score']:5.2f}  achievements "
              f"{run['mean_achievements']:.2f}  table "
              f"{run['rates']['place_table']:.0f}%  wood pickaxe "
              f"{run['rates']['make_wood_pickaxe']:.0f}%  deaths "
              f"{run['deaths']}/{run['episodes']}  rule share "
              f"{run['rule_share']:.2f}")
    d = result['decision']
    e = result['exploratory_without_empty_rules'][
        'decision_if_it_were_primary']
    print(f"DECISION (frozen, raw rules): {d['verdict']} "
          f"(gain {d['achievement_gain']:+.2f})")
    print(f"exploratory, without empty rules: {e['verdict']} "
          f"(gain {e['achievement_gain']:+.2f}); not the decision")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('build', 'compare'))
    cli.add_argument('--out', type=Path, default=OUT)
    args = cli.parse_args()
    {'build': build, 'compare': compare}[args.action](args.out)


if __name__ == '__main__':
    main()
