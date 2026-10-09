"""Ask the LLM teacher for each task's subgoal plan; compare with the shaping teacher.

The fix wave's shaping arms (scripts/run_fix_wave_20260929.py) reward the
student for progress through a subgoal plan, e.g. key -> locked door ->
ball on KeyCorridor. teachers/subgoal_potential.py encodes that plan by
hand. This script asks the SAME LLM that wrote the rule banks
(GPT-5-mini, the dated snapshot, low reasoning effort, strict JSON
schema, Standard service) to write the plan itself, from the mission
text and the game mechanics only, in the vocabulary the potential can
measure (targets, completion conditions, repetition). Distractor options
are included, and the plan's length and order are the LLM's choice.

If the LLM's plan equals the teacher's, the shaping runs already follow
an LLM-written plan: the LLM names the subgoals, the environment measures
progress through them. That is a thinner LLM contribution than the rule
banks, and the record says so. If they differ, the difference is the
finding, and the fix wave did not test the LLM's plan.

Usage:
    python -m scripts.llm_stage_plan_20260929 --dry-run
    python -m scripts.llm_stage_plan_20260929 --collect --credential-file <.env>
    python -m scripts.llm_stage_plan_20260929 --check
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/stage_plans/stage_plans_20260929.json'
MODEL = 'gpt-5-mini-2025-08-07'
SAMPLES = 5
PRICE = (0.25, 2.00)          # USD per 1M input / output tokens (gpt-5-mini)
MAX_OUTPUT = 4096

TARGETS = ('key', 'locked_door', 'next_room_door', 'ball', 'goal', 'box')
DONE = ('carrying_key', 'door_unlocked', 'entered_next_room',
        'task_complete')
REPEAT = ('once', 'each_room_before_the_goal_room')

# What teachers/subgoal_potential.TaskStages measures, as a plan in the
# same vocabulary: its targets, its completion predicates, its repetition.
REFERENCE = {
    'doorkey_8x8': [('key', 'carrying_key', 'once'),
                    ('locked_door', 'door_unlocked', 'once'),
                    ('goal', 'task_complete', 'once')],
    'keycorridor_s3r3': [('key', 'carrying_key', 'once'),
                         ('locked_door', 'door_unlocked', 'once'),
                         ('ball', 'task_complete', 'once')],
    'multiroom_n6': [('next_room_door', 'entered_next_room',
                      'each_room_before_the_goal_room'),
                     ('goal', 'task_complete', 'once')],
}

MECHANICS = (
    'The world is a MiniGrid grid. The agent moves with actions turn left, '
    'turn right, move forward, pick up, drop and toggle. Toggle opens or '
    'closes the door in front of the agent; a LOCKED door opens only if the '
    'agent is carrying a key of the same colour. The agent can carry one '
    'object at a time. Walls and closed doors block movement.')

TASKS = {
    'doorkey_8x8': (
        'DoorKey-8x8. An 8x8 grid is split by a wall into two rooms. The '
        'agent starts in one room. The only passage between the rooms is a '
        'locked door. The green goal square is in the other room. '
        "Mission text: 'use the key to open the door and then get to the "
        "goal'. The episode succeeds when the agent reaches the goal."),
    'keycorridor_s3r3': (
        'KeyCorridor-S3R3. A corridor runs through the map with small rooms '
        'on its sides, each behind a door. The ball is inside a room whose '
        'door is locked. The key for that door lies in one of the other '
        "rooms. Mission text: 'pick up the <colour> ball'. The episode "
        'succeeds when the agent picks up the ball.'),
    'multiroom_n6': (
        'MultiRoom-N6. Up to six rooms are connected one after another in a '
        'chain; each room has a closed (not locked) door leading to the next '
        'room. The agent starts in the first room and the green goal square '
        "is in the last room. Mission text: 'traverse the rooms to get to "
        "the goal'. The episode succeeds when the agent reaches the goal."),
}

SCHEMA = {
    'type': 'object',
    'properties': {
        'plan': {
            'type': 'array', 'minItems': 1, 'maxItems': 8,
            'items': {
                'type': 'object',
                'properties': {
                    'target': {'type': 'string', 'enum': list(TARGETS)},
                    'done_when': {'type': 'string', 'enum': list(DONE)},
                    'repeat': {'type': 'string', 'enum': list(REPEAT)},
                },
                'required': ['target', 'done_when', 'repeat'],
                'additionalProperties': False,
            },
        },
        'rationale': {'type': 'string'},
    },
    'required': ['plan', 'rationale'],
    'additionalProperties': False,
}


def prompt(task):
    return (
        'PROMPT VERSION: stage_plan_v1\n'
        'You are the TEACHER for a reinforcement-learning student. Write the '
        'ordered SUBGOAL PLAN the student should follow to accomplish the '
        "mission. During training the student is rewarded for progress "
        'through your plan: for each step, getting closer to its target, and '
        'completing it.\n\n'
        f'TASK. {TASKS[task]}\n\nMECHANICS. {MECHANICS}\n\n'
        'VOCABULARY. Each step has:\n'
        '- target: key (the key that opens the locked door), locked_door, '
        'next_room_door (the door leading out of the current room toward '
        'the goal), ball, goal (the green goal square), box\n'
        '- done_when: carrying_key (the agent holds the key), door_unlocked '
        '(the target door has been unlocked), entered_next_room (the agent '
        'has passed into the next room), task_complete (the mission is '
        'accomplished and the episode ends)\n'
        '- repeat: once, or each_room_before_the_goal_room (repeat the step '
        'for every room until the room that contains the goal)\n\n'
        'Use only these options, in the order the student should follow '
        'them, with as few steps as the mission needs. Give a one-sentence '
        'rationale.')


def body(task):
    return dict(model=MODEL, store=False, service_tier='default',
                max_output_tokens=MAX_OUTPUT, reasoning={'effort': 'low'},
                input=[dict(role='user', content=prompt(task))],
                text={'format': dict(type='json_schema', strict=True,
                                     name='stage_plan', schema=SCHEMA)})


def as_tuples(plan):
    return [(s['target'], s['done_when'], s['repeat']) for s in plan]


def dollars(tokens_in, tokens_out):
    return tokens_in / 1e6 * PRICE[0] + tokens_out / 1e6 * PRICE[1]


def bound():
    """Worst case: every call spends the whole output ceiling."""
    worst_in = max(len(prompt(t)) // 3 + 400 for t in TASKS)
    return len(TASKS) * SAMPLES * dollars(worst_in, MAX_OUTPUT)


def collect(credential_file):
    try:
        # Verify TLS against the operating system's certificate store, as
        # git does. On a machine whose HTTPS traffic is re-signed by a
        # local root, Python's bundled certificates fail every call with
        # CERTIFICATE_VERIFY_FAILED (the first attempt here did, at $0).
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import dotenv_values
    from openai import OpenAI
    key = (dotenv_values(credential_file, encoding='utf-8-sig',
                         interpolate=False).get('OPENAI_API_KEY') or '')
    if not key.strip():
        raise ValueError('OPENAI_API_KEY is missing from the credential file')
    client = OpenAI(api_key=key.strip(), timeout=300, max_retries=0)
    record = dict(
        created=datetime.now(timezone.utc).isoformat(), model=MODEL,
        samples_per_task=SAMPLES, schema=SCHEMA, bound_usd=round(bound(), 4),
        tasks={})
    for task in TASKS:
        text = prompt(task)
        rows = []
        for i in range(SAMPLES):
            t0 = time.perf_counter()
            row = dict(sample=i)
            try:
                response = client.responses.create(**body(task))
                usage = response.usage
                row.update(
                    status=response.status, service_tier=response.service_tier,
                    response_model=response.model,
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    usd=round(dollars(usage.input_tokens,
                                      usage.output_tokens), 6),
                    raw=response.output_text)
                data = json.loads(response.output_text)
                row['plan'] = data['plan']
                row['rationale'] = data['rationale']
                row['matches_reference'] = (as_tuples(data['plan'])
                                            == REFERENCE[task])
            except Exception as error:          # recorded, never retried
                row.update(error=f'{type(error).__name__}: {error}'[:300])
            row['seconds'] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            print(f"{task} sample {i}: "
                  f"{row.get('matches_reference', row.get('error'))}")
        record['tasks'][task] = dict(
            prompt=text,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            reference=[list(s) for s in REFERENCE[task]], samples=rows)
    record['total_usd'] = round(sum(r.get('usd', 0) for t in record[
        'tasks'].values() for r in t['samples']), 6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1), encoding='utf-8')
    return record


def summarize(record):
    """Per task: how many samples reproduce the teacher's plan."""
    out = {}
    for task, entry in record['tasks'].items():
        plans = [tuple(as_tuples(r['plan'])) for r in entry['samples']
                 if 'plan' in r]
        modal, count = (Counter(plans).most_common(1)[0]
                        if plans else (None, 0))
        out[task] = dict(
            valid=len(plans), samples=len(entry['samples']),
            matches=sum(r.get('matches_reference', False)
                        for r in entry['samples']),
            modal_plan=[list(s) for s in modal] if modal else None,
            modal_count=count,
            modal_matches_reference=(list(modal) == REFERENCE[task]
                                     if modal else False))
    return out


FROZEN = ROOT / 'research/stage_plans/llm_plans_20260929.json'
INTERPRETATION = {
    'selection': 'For each task, the most frequent plan over the samples; '
                 'a tie goes to the plan sampled first. Declared before '
                 'freezing, never chosen by learning outcomes.',
    'repeat': 'each_room_before_the_goal_room expands, as one block with '
              'adjacent repeated steps, once per room before the goal room '
              'in MultiRoom only. DoorKey and KeyCorridor have no room chain '
              'toward the goal, so there it means once. On KeyCorridor the '
              "LLM marked its key step this way ('search each side room "
              "until you pick up the key'); the potential targets the key's "
              'true cell, which is privileged information the plan did not '
              'assume.',
    'door_unlocked': 'For a door locked at episode start, unlocked; for a '
                     "door never locked (MultiRoom), open: the LLM's "
                     "rationale is 'open and pass through each closed door'.",
    'semantics': 'teachers/subgoal_potential.PlanStages',
}


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(',', ':')).encode()
                          ).hexdigest()


def freeze(record):
    """The modal plan per task, with digests and the interpretation."""
    from teachers.subgoal_potential import HUMAN_PLAN, plan_sha256
    tasks = {}
    for task, entry in record['tasks'].items():
        plans = [tuple(as_tuples(r['plan'])) for r in entry['samples']
                 if 'plan' in r]
        counts = Counter(plans)
        best = max(counts.values())
        modal = next(p for p in plans if counts[p] == best)
        tasks[task] = dict(
            plan=[list(s) for s in modal], sha256=plan_sha256(modal),
            modal_count=best, valid_samples=len(plans),
            distinct_plans=len(counts),
            equals_hand_coded=list(modal) == [tuple(s) for s in
                                              HUMAN_PLAN[task]])
    frozen = dict(
        model=record['model'], source=OUT.relative_to(ROOT).as_posix(),
        source_sha256=canonical_sha256(record),
        created=datetime.now(timezone.utc).isoformat(),
        interpretation=INTERPRETATION, tasks=tasks)
    FROZEN.write_text(json.dumps(frozen, indent=1) + '\n', encoding='utf-8')
    return frozen


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--dry-run', action='store_true')
    cli.add_argument('--collect', action='store_true')
    cli.add_argument('--check', action='store_true')
    cli.add_argument('--freeze', action='store_true',
                     help='write the modal plan per task to ' +
                          FROZEN.relative_to(ROOT).as_posix())
    cli.add_argument('--credential-file', type=Path)
    args = cli.parse_args()
    if args.dry_run:
        for task in TASKS:
            print(f'===== {task}\n{prompt(task)}\n')
        print(f'{len(TASKS) * SAMPLES} calls; worst-case bound '
              f'${bound():.4f}')
        return 0
    if args.collect:
        if not args.credential_file:
            cli.error('--collect needs --credential-file')
        record = collect(args.credential_file)
        print(f"total ${record['total_usd']}")
    else:
        record = json.loads(OUT.read_text(encoding='utf-8'))
    for task, s in summarize(record).items():
        print(f"{task:18s} {s['matches']}/{s['samples']} samples equal the "
              f"teacher's plan; modal plan ({s['modal_count']}x) "
              f"{'EQUALS' if s['modal_matches_reference'] else 'DIFFERS FROM'}"
              f" it: {s['modal_plan']}")
    if args.freeze:
        frozen = freeze(record)
        for task, entry in frozen['tasks'].items():
            print(f"froze {task}: {entry['modal_count']}/"
                  f"{entry['valid_samples']} samples, sha256 "
                  f"{entry['sha256'][:12]}, equals hand-coded: "
                  f"{entry['equals_hand_coded']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
