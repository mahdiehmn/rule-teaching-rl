"""Frozen local-view correction lessons."""

from collections import Counter, deque
from functools import lru_cache
import json

import numpy as np

from scripts import explanation_screen_panel as source


STUDY = 'contrastive_lessons_20260924_v1'
POSITIVE = (
    'approach_key',
    'acquire_key',
    'approach_passage',
    'unlock_passage',
    'open_passage',
    'approach_goal',
    'unknown',
)
NEGATIVE = (
    'blocked_route',
    'missing_key',
    'irrelevant_interaction',
    'premature_drop',
    'turn_away_from_target',
    'delay_available_subgoal',
    'unknown',
)
DEFINITIONS = {
    'blocked_route': 'forward is blocked by an observed object or door',
    'missing_key': 'trying to unlock an observed locked door without its key',
    'irrelevant_interaction': 'pickup/drop/toggle has no useful visible target',
    'premature_drop': 'dropping the held key before an observed locked door',
    'turn_away_from_target': 'turning away from a visible relevant target',
    'delay_available_subgoal': 'delaying an immediately available useful step',
    'unknown': 'the local snapshot cannot justify a specific reason',
}


def write(path, value):
    """Keep artifacts deterministic and reject nonfinite JSON."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding='utf-8'
    )


def geometry(state):
    """DoorKey geometry without any policy, reward or episode-age state."""
    grid = np.asarray(state['full_grid'])
    doors = np.argwhere(grid[:, :, 0] == 4)
    goals = np.argwhere(grid[:, :, 0] == 8)
    keys = np.argwhere(grid[:, :, 0] == 5)
    if len(doors) != 1 or len(goals) != 1 or len(keys) > 1:
        raise ValueError('Expected one-door, one-key DoorKey geometry')
    door = tuple(map(int, doors[0]))
    key = tuple(map(int, keys[0])) if len(keys) else (-1, -1)
    held = state['carrying'] is not None
    if held == bool(len(keys)):
        raise ValueError('Expected exactly one key, carried or on grid')
    walls = tuple(map(tuple, np.argwhere(grid[:, :, 0] == 2).tolist()))
    door_state = int(grid[door][2])
    dynamic = (
        *state['agent_pos'],
        state['agent_dir'],
        *key,
        int(held),
        door_state,
    )
    return (walls, door, tuple(map(int, goals[0]))), dynamic


def transition(layout, state, action):
    """Exact single-key DoorKey physics, including drop and door closure."""
    walls, door, goal = layout
    x, y, direction, kx, ky, held, door_state = state
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
    front = (x + dx, y + dy)
    empty = (
        front not in walls
        and front != (kx, ky)
        and front != door
        and front != goal
    )
    if action == 0:
        direction = (direction - 1) % 4
    elif action == 1:
        direction = (direction + 1) % 4
    elif action == 2 and (
        front not in walls
        and front != (kx, ky)
        and (front != door or door_state == 0)
    ):
        x, y = front
    elif action == 3 and front == (kx, ky) and not held:
        held, kx, ky = 1, -1, -1
    elif action == 4 and held and empty:
        held, (kx, ky) = 0, front
    elif action == 5 and front == door:
        if door_state == 2 and held:
            door_state = 0
        elif door_state != 2:
            door_state = 1 - door_state
    return x, y, direction, kx, ky, held, door_state


@lru_cache(maxsize=8192)
def distance(layout, start):
    """Shortest completion without dropping; dropping is never necessary here.

    A foil drop is applied first, so its new key location is respected.
    Within this single-key task, carrying does not obstruct any action needed
    for success. Further drops cannot shorten a completion path.
    """
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        state, depth = queue.popleft()
        if state[:2] == layout[2]:
            return depth
        for action in (0, 1, 2, 3, 5):
            nxt = transition(layout, state, action)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return 10000


def reference(state, foil):
    """Reject tied alternatives; record a strict completion-distance gap."""
    layout, start = geometry(state)
    ds = [distance(layout, transition(layout, start, a)) for a in range(7)]
    best = min(ds)
    if best >= 10000 or ds[foil] <= best:
        return None
    positive = ds.index(best)
    return dict(
        positive_action=positive,
        foil_action=foil,
        distance_after=ds,
        strict_gap=ds[foil] - best,
    )


def make_panel(corpus):
    """Freeze 192 train / 64 audit cases, disjoint by source episode."""
    candidates, sources = source.read_candidates(corpus)
    candidates.sort(
        key=lambda r: source.object_hash(
            [STUDY, r['world_id'], r['sample_id']]
        )
    )
    cases, seen, episodes = [], set(), set()
    counts = Counter()
    for row in candidates:
        episode = tuple(row['episode_group'])
        phase = row['phase']
        # Do not ask the local teacher to justify a hidden target.
        target_type = {'seek_key': 5, 'unlock_door': 4, 'reach_target': 8}[
            phase
        ]
        target_visible = bool(
            (
                np.asarray(row['state']['local_obs'])[:, :, 0] == target_type
            ).any()
        )
        # One case per episode makes the audit split episode-disjoint.
        if (
            row['world_id'] in seen
            or episode in episodes
            or not target_visible
            or counts[phase] >= (86 if phase == 'seek_key' else 85)
        ):
            continue
        env = source.restore(row['state'])
        env.close()
        ref = reference(row['state'], int(row['student_action']))
        if ref is None:
            continue
        i = len(cases)
        case = {
            k: v
            for k, v in row.items()
            if k not in ('old_text', 'old_teacher_action', 'stratum')
        }
        case.update(
            ref,
            case_id=f'lesson_{i:03d}',
            split='audit' if i % 4 == 3 else 'train',
            relevant_target_visible=True,
        )
        cases.append(case)
        seen.add(row['world_id'])
        episodes.add(episode)
        counts[phase] += 1
        if len(cases) == 256:
            break
    if len(cases) != 256:
        raise ValueError(f'Only {len(cases)} distinct-episode strict mistakes')
    return dict(
        study=STUDY,
        cases=cases,
        source_hashes=sources,
        phases=dict(counts),
        history_available=False,
        scope='Local snapshot rationale; offline historical lessons',
    )


def request(case):
    """Only the learner-visible image and fixed action pair enter the prompt."""
    from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

    obs = np.asarray(case['state']['local_obs'])
    visible = []
    for x in range(7):
        for y in range(7):
            kind, color, status = map(int, obs[x, y])
            visible.append(
                [x, y, IDX_TO_OBJECT[kind], IDX_TO_COLOR[color], status]
            )
    properties = {
        'positive': {'type': 'string', 'enum': list(POSITIVE)},
        'negative': {'type': 'string', 'enum': list(NEGATIVE)},
        'positive_reason': {'type': 'string'},
        'negative_reason': {'type': 'string'},
    }
    schema = dict(
        type='object',
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )
    prompt = (
        'Explain this DoorKey correction using ONLY the local observation. '
        'Goal: reach the green goal. A key unlocks the same-color locked '
        'door. Actions 0 left,1 right,2 forward,3 pickup,4 drop,5 toggle,6 done. '
        'The agent is at local (3,6), faces (3,5). Unseen cells are unknown. '
        'At (3,6) the image encodes the carried object, or empty if none. '
        'Door state 0=open,1=closed unlocked,2=locked. Coordinates are '
        'egocentric. Do not invent hidden routes or earlier progress. '
        'A full-state reference fixes the action pair, but you cannot see '
        'its map. Select unknown when its reason cannot be justified here. '
        'Return one positive purpose and one reason against the foil, with '
        'one short factual sentence each. Negative category definitions: '
        + json.dumps(DEFINITIONS)
        + '\n'
        + json.dumps(
            dict(
                local_cells=visible,
                endorsed=case['positive_action'],
                student_proposed=case['foil_action'],
            )
        )
    )
    return dict(
        model='gpt-5-mini',
        reasoning={'effort': 'low'},
        max_output_tokens=8192,
        service_tier='default',
        store=False,
        input=[{'role': 'user', 'content': prompt}],
        text={
            'format': dict(
                type='json_schema',
                name='correction_reason',
                strict=True,
                schema=schema,
            )
        },
    )


def freeze_lessons(panel, rows):
    """Same inclusion/donors in every arm; audit examples never train."""
    by_id = {r['case_id']: r for r in rows}
    if len(rows) != 256 or len(by_id) != 256:
        raise ValueError('Need all 256 attempts, with unique identities')
    usable, audit = [], []
    for case in panel['cases']:
        row = by_id[case['case_id']]
        if row.get('status') != 'valid':
            continue
        a = row['answer']
        if a['positive'] == 'unknown' or a['negative'] == 'unknown':
            continue
        item = dict(
            case_id=case['case_id'],
            observation=case['state']['local_obs'],
            positive_action=case['positive_action'],
            foil_action=case['foil_action'],
            episode=case['episode_group'],
            positive=POSITIVE.index(a['positive']),
            negative=NEGATIVE.index(a['negative']),
            positive_reason=a['positive_reason'],
            negative_reason=a['negative_reason'],
        )
        (usable if case['split'] == 'train' else audit).append(item)
    # Permute only within fixed action pairs, across source episodes.
    # Do not select donors by their explanation values.
    groups = {}
    for i, row in enumerate(usable):
        groups.setdefault(
            (row['positive_action'], row['foil_action']), []
        ).append(i)
    kept = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for j, i in enumerate(indices):
            row = dict(usable[i])
            donor = usable[indices[(j + 1) % len(indices)]]
            row.update(
                donor_id=donor['case_id'], shuffled_negative=donor['negative']
            )
            kept.append(row)
    kept.sort(key=lambda r: r['case_id'])
    changed = sum(r['negative'] != r['shuffled_negative'] for r in kept)
    # Readiness is technical, NOT a certification of rationale correctness.
    ready = (
        len(kept) >= 96
        and len(audit) >= 32
        and len({r['positive'] for r in kept}) >= 3
        and len({r['negative'] for r in kept}) >= 3
        and changed >= 0.2 * len(kept)
    )
    return dict(
        study=STUDY,
        train=kept,
        audit=audit,
        changed_foil_fraction=changed / max(1, len(kept)),
        technical_ready=ready,
        semantic_review='pending',
        claim='Exploratory model-selected labels, not validated reasons',
    )
