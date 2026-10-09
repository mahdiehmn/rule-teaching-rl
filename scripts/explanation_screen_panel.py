"""Freeze a common-state DoorKey explanation diagnostic without API calls."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Key

from envs.phases import s3r3_phase
from envs.state import extract_doorkey_state, extract_generic_state
from teachers.minigrid.bfs_solver import MiniGridBFSTeacher
from teachers.minigrid.llm_general import render_ascii_map


PHASES = ('seek_key', 'unlock_door', 'reach_target')
FORMATS = ('prose', 'structured', 'contrastive')
ACTIONS = ('LEFT', 'RIGHT', 'FORWARD', 'PICKUP', 'DROP', 'TOGGLE', 'DONE')
STATE_FIELDS = ('full_grid', 'agent_pos', 'agent_dir', 'carrying', 'mission')
ENV_ID = 'MiniGrid-DoorKey-8x8-v0'
SELECTION_SALT = 'aleph-explanation-screen-20260915-v1'


def object_hash(value):
    """Hash JSON identically across machines and dictionary ordering."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path):
    """Stream a file hash without loading a large journal twice."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore(state):
    """Restore physics; reject an inconsistent saved student observation."""
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=0)
    encoded = np.asarray(state['full_grid'])
    if encoded.shape != (8, 8, 3):
        raise ValueError('Screen requires an 8x8 DoorKey state')
    env.grid, _ = Grid.decode(encoded.astype(np.uint8))
    env.agent_pos = tuple(state['agent_pos'])
    env.agent_dir = int(state['agent_dir'])
    env.mission = state['mission']
    held = state['carrying']
    if held is not None and held[0] != 'key':
        raise ValueError('Unexpected DoorKey inventory')
    env.carrying = Key(held[1]) if held else None
    # Episode age is absent. Only physical effects and goal entry count.
    env.step_count = 0
    if not np.array_equal(env.grid.encode(), encoded):
        raise ValueError('Grid did not round-trip')
    if 'local_obs' in state and not np.array_equal(
            env.gen_obs()['image'], state['local_obs']):
        raise ValueError('Saved local observation does not match the state')
    return env


def snapshot(env, goal_reached=False):
    """Encode directly checkable one-action outcomes, without reward/time."""
    doors = [cell for cell in env.grid.grid
             if cell is not None and cell.type == 'door']
    if len(doors) != 1:
        raise ValueError('Screen requires exactly one DoorKey door')
    door = doors[0]
    return {
        'position': [int(x) for x in env.agent_pos],
        'direction': int(env.agent_dir),
        'carrying': env.carrying.color if env.carrying else 'nothing',
        'door_state': ('open' if door.is_open else
                       'locked' if door.is_locked else 'closed'),
        'goal_reached': bool(goal_reached),
    }


def consequence(state, action):
    """Use an independent restored simulator for each candidate action."""
    env = restore(state)
    try:
        _, _, terminated, _, _ = env.step(action)
        cell = env.grid.get(*env.agent_pos)
        reached = terminated and cell is not None and cell.type == 'goal'
        return snapshot(env, reached)
    finally:
        env.close()


def critical_stratum(state):
    """Identify three prerequisite transitions before looking at text."""
    x, y = state['agent_pos']
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[state['agent_dir']]
    obj, color, status = state['full_grid'][x + dx][y + dy]
    held = state['carrying']
    if obj == 5 and held is None:
        return 'key_front'
    if obj == 4 and status == 2 and held:
        from minigrid.core.constants import COLOR_TO_IDX
        if COLOR_TO_IDX[held[1]] == color:
            return 'unlock_front'
    if obj == 4 and status == 1:
        return 'closed_unlocked_front'
    if obj == 8:
        return 'goal_front'
    return 'other'


def read_candidates(corpus):
    """Read only the completed named cohort, excluding duplicate exports."""
    corpus = Path(corpus)
    files = sorted(corpus.glob(
        'phase1_native_aleph_20260915_v1_r[0-4]/code/results/runs/'
        '*/explanations/collected_explanations.jsonl'))
    if len(files) != 30:
        raise ValueError(f'Need all 30 delivered-text journals; got '
                         f'{len(files)} in {corpus}')
    candidates, sources = [], {}
    for path in files:
        relative = path.relative_to(corpus).as_posix()
        sources[relative] = file_hash(path)
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith('\n'):
                    raise ValueError(f'Incomplete journal: {relative}')
                row = json.loads(line)
                state = {k: row[k] for k in STATE_FIELDS}
                world_id = object_hash(state)
                state.update(local_obs=row['local_obs'])
                episode = [path.parents[1].name, row['env'], row['episode']]
                candidates.append({
                    'world_id': world_id, 'state': state,
                    'phase': row['phase'], 'episode_group': episode,
                    'sample_id': row['sample_id'], 'source': relative,
                    'source_line': line_number,
                    'student_action': row['student_action'],
                    'old_teacher_action': row['teacher_action'],
                    'old_text': row['text'],
                    'stratum': critical_stratum(state),
                })
    if len(candidates) != 10660:
        raise ValueError('Expected the frozen 10,660-record delivered corpus')
    return candidates, sources


def select_cases(candidates):
    """Select 12 states per phase, including two key transitions per phase."""
    ranked = sorted(candidates, key=lambda row: (
        object_hash([SELECTION_SALT, row['world_id']]), row['sample_id']))
    selected, worlds, episodes = [], set(), set()

    def take(phase, stratum, number):
        """Choose distinct worlds and episodes without an outcome filter."""
        found = 0
        for row in ranked:
            episode = tuple(row['episode_group'])
            if (row['phase'] != phase or row['world_id'] in worlds
                    or episode in episodes
                    or (stratum and row['stratum'] != stratum)):
                continue
            selected.append(dict(row))
            worlds.add(row['world_id'])
            episodes.add(episode)
            found += 1
            if found == number:
                return
        raise ValueError(f'Insufficient unique states for {phase}/{stratum}')

    for phase, stratum in zip(PHASES, (
            'key_front', 'unlock_front', 'goal_front')):
        take(phase, stratum, 2)
    take('reach_target', 'closed_unlocked_front', 2)
    for phase in PHASES:
        count = sum(row['phase'] == phase for row in selected)
        take(phase, None, 12 - count)
    return sorted(selected, key=lambda row: row['world_id'])


def build_panel(corpus):
    """Freeze paired actions and simulator truth before requesting text."""
    candidates, sources = read_candidates(corpus)
    cases = select_cases(candidates)
    oracle = MiniGridBFSTeacher(ENV_ID)
    for index, case in enumerate(cases):
        env = restore(case['state'])
        try:
            if s3r3_phase(env) != case['phase']:
                raise ValueError('Recorded phase and restored physics differ')
            critical_actions = {'key_front': 3, 'unlock_front': 5,
                                'closed_unlocked_front': 5, 'goal_front': 2}
            reference = critical_actions.get(case['stratum'])
            if reference is None:
                reference = oracle.peek_action(extract_doorkey_state(env))
            student = case['student_action']
            # Keep disagreement examples; use a fixed cyclic alternative
            # for agreement. Neither choice uses generated explanations.
            alternative = (student if student != reference else
                           (reference + 1) % 6)
            case.update(case_id=f'S{index + 1:02}',
                        reference_action=reference,
                        alternative_action=alternative,
                        unchanged_baseline=snapshot(env))
        finally:
            env.close()
        case['truth'] = {
            'reference': consequence(case['state'], reference),
            'alternative': consequence(case['state'], alternative),
        }
        truth = case['truth']['reference']
        required = {
            'key_front': truth['carrying'] != 'nothing',
            'unlock_front': truth['door_state'] == 'open',
            'closed_unlocked_front': truth['door_state'] == 'open',
            'goal_front': truth['goal_reached'],
        }
        if not required.get(case['stratum'], True):
            raise ValueError('Critical pair omitted the intended interaction')
    return {
        'version': SELECTION_SALT, 'source_sha256': sources,
        'corpus_records': len(candidates), 'cases': cases,
        'phase_counts': dict(Counter(c['phase'] for c in cases)),
        'history': 'state-only; old six-action teacher history unavailable',
    }


def response_schema(kind):
    """Keep the physics questions identical across explanation formats."""
    effect_fields = {
        'position': {'type': 'array', 'items': {'type': 'integer'},
                     'minItems': 2, 'maxItems': 2},
        'direction': {'type': 'integer', 'enum': [0, 1, 2, 3]},
        'carrying': {'type': 'string', 'enum': [
            'nothing', 'red', 'green', 'blue', 'purple', 'yellow', 'grey']},
        'door_state': {'type': 'string', 'enum': ['open', 'closed', 'locked']},
        'goal_reached': {'type': 'boolean'},
    }

    def closed(fields):
        """Require every field and refuse silent schema additions."""
        return {'type': 'object', 'properties': fields,
                'required': list(fields), 'additionalProperties': False}

    fields = {name: closed(effect_fields)
              for name in ('reference', 'alternative')}
    text = {'type': 'string', 'minLength': 1, 'maxLength': 1200}
    if kind == 'prose':
        fields['rationale'] = text
    elif kind == 'structured':
        fields.update(subgoal=text, prerequisite=text,
                      completion_condition=text, invalidation_condition=text)
    elif kind == 'contrastive':
        fields.update(why_reference=text, why_not_alternative=text)
    else:
        raise ValueError(f'Unknown format: {kind}')
    return closed(fields)


def make_request(case, kind):
    """Render the full-map state without leaking simulator answers."""
    env = restore(case['state'])
    try:
        state = extract_generic_state(env)
        objects = state[-1]
        rendered = render_ascii_map(env)
    finally:
        env.close()
    styles = {
        'prose': 'Give one short ordinary-language rationale.',
        'structured': ('Separate the immediate subgoal, its prerequisite, '
                       'completion condition and invalidation condition.'),
        'contrastive': ('Contrast the reference action with the alternative: '
                        'why the reference and why not the alternative. '
                        'If both are reasonable, say so.'),
    }
    state = case['state']
    ref, alt = case['reference_action'], case['alternative_action']
    prompt = (
        'Explain two fixed actions in MiniGrid-DoorKey-8x8-v0. '
        'Do not choose or change the actions. Do not assume the reference '
        'is optimal: describe limitations honestly.\n'
        'This is a state-only diagnostic with no previous actions supplied. '
        'You see the full map. The student sees a local 7x7 symbolic view. '
        'Do not assume the student has already seen distant objects.\n'
        'Coordinates are (x,y), x east/right, y south/down. '
        'Directions: 0 east, 1 south, 2 west, 3 north.\n'
        f'Mission: {state["mission"]}\n'
        f'Agent position: {state["agent_pos"]}; '
        f'direction: {state["agent_dir"]}; carrying: {state["carrying"]}.\n'
        'Map: # wall, . floor, D door, k key, G goal; arrow is agent.\n'
        f'{rendered}\n'
        f'Objects (kind,color,x,y,door_state): {objects}\n'
        'Actions: 0 LEFT, 1 RIGHT, 2 FORWARD, 3 PICKUP, 4 DROP, '
        '5 TOGGLE, 6 DONE. Turning changes direction only. Forward moves '
        'one cell onto floor, an open door, or the goal; walls, keys and '
        'closed doors block it. Pickup needs an empty hand and a key '
        'directly in front. Drop needs a held key and empty floor in front. '
        'Toggle affects only a door directly in front. A locked door needs '
        'a matching-color held key; toggle then unlocks AND opens it, '
        'retaining the key. An unlocked door toggles open/closed. '
        'DONE has no effect in DoorKey.\n'
        'DoorKey success requires ENTERING the goal tile. Standing adjacent '
        'and facing it is NOT success. The BabyAI object-GoTo adjacency '
        'rule does not apply. Ignore episode time limits.\n'
        f'Reference: {ref} {ACTIONS[ref]}. '
        f'Alternative: {alt} {ACTIONS[alt]}.\n'
        'For EACH action independently from this same starting state, '
        'predict the state immediately after exactly one action: position, '
        'direction, carrying (key color or nothing), the single door state, '
        'and whether the goal was reached. Do not describe a whole plan '
        'as if it happened in one step.\n'
        f'{styles[kind]} Keep prose concise, grounded in this state, and '
        'mention whether a needed fact concerns a distant object. '
        'Return only the required JSON.'
    )
    return {'input': [{'role': 'user', 'content': prompt}],
            'text': {'format': {'type': 'json_schema', 'name': 'explanation',
                                'schema': response_schema(kind),
                                'strict': True}}}
