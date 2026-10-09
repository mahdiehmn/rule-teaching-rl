"""Teacher-view ablation: the same rules, checked on the full map.

Protocol:
research/fix_wave_protocol_2026-09-29.md, addendum 6 (frozen before any
run). No API call anywhere.

The paper restricts a privileged teacher's rules to predicates the student
evaluates from its own view, where an unseen object is unknown and a rule
abstains. This builds the ablation: each confirmed bank, byte-for-byte the
same rules, whose `observer` reads the same predicate names from the full
map instead (teachers/minigrid/teacher_view.py). Training with these banks
is fix-wave suites dk_tv, mr_tv and kc_tv.

    python -m scripts.teacher_view_20261001 banks    # write the four banks
    python -m scripts.teacher_view_20261001 check    # rules identical?
    python -m scripts.teacher_view_20261001 labels   # offline comparison

`labels` compares the two views on sampled states against each task's
exact oracle: coverage, precision, the labels only the full map gives, and
how often one student view receives different full-map labels (advice the
student cannot tell apart). States are sampled as in
scripts/larger_map_labels_20261001.py, from layout seeds 38,000,000+,
which no other study uses.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from teachers.minigrid.teacher_view import SUFFIX

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = 'research/rule_banks/teacher_view_20261001'
# The banks behind the confirmed fix-wave results (addenda 2 and 3), and
# their teacher-view copies.
BANKS = {
    'research/rule_banks/v3_20260928/blind_strict.json':
        f'{OUT_DIR}/doorkey_blind_strict.json',
    'research/rule_banks/multiroom_20260928/scoped.json':
        f'{OUT_DIR}/multiroom_scoped.json',
    'research/rule_banks/multiroom_20260928/blind_strict.json':
        f'{OUT_DIR}/multiroom_blind_strict.json',
    'research/rule_banks/keycorridor_mem_20260929/'
    'self_checked_pooled_valid.json':
        f'{OUT_DIR}/keycorridor_mem_pooled_valid.json',
}
SEED0 = 38_000_000
WALK = (0, 1, 2, 3, 4, 5)
PREFER = (2, 5, 3, 4, 0, 1)        # forward, toggle, pickup, drop, turns


def sha256(path):
    """Of the content with LF line ends, so a Windows checkout (autocrlf)
    and the cluster's agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(
        b'\r\n', b'\n')).hexdigest()


def derived(source, root=ROOT):
    bank = json.loads((Path(root) / source).read_text(encoding='utf-8'))
    bank['observer'] = bank.get('observer', 'doorkey_v3') + SUFFIX
    bank['teacher_view_of'] = dict(bank=source,
                                   sha256=sha256(Path(root) / source))
    return bank


def write_banks(root=ROOT):
    for source, target in BANKS.items():
        path = Path(root) / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derived(source, root), indent=1) + '\n',
                        encoding='utf-8')
        print('wrote', target)


def check_banks(root=ROOT):
    """Each copy: the source's exact rules and mode; only the observer."""
    for source, target in BANKS.items():
        src = json.loads((Path(root) / source).read_text(encoding='utf-8'))
        tv = json.loads((Path(root) / target).read_text(encoding='utf-8'))
        if tv != derived(source, root):
            raise ValueError(f'{target} is not the teacher-view copy of '
                             f'{source}')
        if tv['rules'] != src['rules'] or tv['mode'] != src['mode']:
            raise ValueError(f'{target}: rules differ from {source}')
    print(f'PASS {len(BANKS)} teacher-view banks: rules identical to their '
          'sources; only the observer differs')
    return True


# ------------------------------------------------------------------ labels

CASES = (
    ('DoorKey-8x8', 'MiniGrid-DoorKey-8x8-v0', 'doorkey',
     ('research/rule_banks/v3_20260928/blind_strict.json',)),
    ('MultiRoom-N6', 'MiniGrid-MultiRoom-N6-v0', 'multiroom',
     ('research/rule_banks/multiroom_20260928/scoped.json',
      'research/rule_banks/multiroom_20260928/blind_strict.json')),
    ('KeyCorridor-S3R3', 'BabyAI-KeyCorridorS3R3-v0', 'keycorridor',
     ('research/rule_banks/keycorridor_mem_20260929/'
      'self_checked_pooled_valid.json',)),
)


def oracle(kind, env_id):
    if kind == 'doorkey':
        from scripts.larger_map_labels_20261001 import oracle as dk
        return dk('doorkey', env_id)
    if kind == 'multiroom':
        from scripts import conditional_rules_multiroom as mr
        return mr.optimal
    from scripts import conditional_rules_keycorridor as kc
    return kc.optimal


def sampled_states(env_id, optimal, layouts, per_layout, rng):
    """Like larger_map_labels.states: a point on the oracle route, half the
    time followed by a short random walk; unsolvable states skipped."""
    import gymnasium as gym
    import minigrid  # noqa: F401

    def fresh(seed):
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    for i in range(layouts):
        seed = SEED0 + i
        env, actions = fresh(seed), []
        done, reward = False, 0
        for _ in range(env.unwrapped.max_steps):
            opt = optimal(env.unwrapped)
            if not opt:
                break
            a = next(a for a in PREFER if a in opt)
            actions.append(a)
            _, reward, done, trunc, _ = env.step(a)
            if done or trunc:
                break
        solved = bool(actions) and done and reward > 0
        env.close()
        if not solved:
            continue
        for _ in range(per_layout):
            env, done = fresh(seed), False
            for a in actions[:int(rng.integers(len(actions)))]:
                env.step(a)
            if rng.random() < .5:
                for _ in range(int(rng.integers(1, 7))):
                    _, _, done, trunc, _ = env.step(int(rng.choice(WALK)))
                    if done or trunc:
                        break
            opt = () if done else optimal(env.unwrapped)
            if opt:
                yield env, opt
            env.close()


def labels(layouts=60, per_layout=8, seed=0, root=ROOT):
    import numpy as np
    from scripts import conditional_rules_v3 as v3
    from teachers.minigrid.rule_bank import RuleBankTeacher

    rng = np.random.default_rng(seed)
    result = {}
    for label, env_id, kind, sources in CASES:
        optimal = oracle(kind, env_id)
        teachers = {s: (RuleBankTeacher(Path(root) / s),
                        RuleBankTeacher(Path(root) / BANKS[s]))
                    for s in sources}
        rows = {s: [] for s in sources}
        for env, opt in sampled_states(env_id, optimal, layouts,
                                       per_layout, rng):
            u = env.unwrapped
            image = u.gen_obs()['image']
            for s, (sv, tv) in teachers.items():
                a_sv, _ = sv._advise(image, u)
                a_tv, _ = tv._advise(image, u)
                # What the student itself can tell apart: its view, plus
                # its memory where the vocabulary has one.
                pred = sv.observe(image, u) if sv.stateful else \
                    sv.observe(image)
                view = hashlib.sha256(image.tobytes() + str(
                    pred.get('door_unlocked')).encode()).hexdigest()
                rows[s].append((view, a_sv, a_tv, tuple(opt)))
        result[label] = {s: summarize(r) for s, r in rows.items()}
        for s, m in result[label].items():
            print(f"{label:17s} {Path(s).parent.name}/{Path(s).stem}: "
                  f"{m['states']} states", flush=True)
            for view in ('student_view', 'teacher_view'):
                v = m[view]
                print(f"   {view:13s} coverage {v['coverage']:.3f}  "
                      f"precision {fmt(v['precision'])}  "
                      f"ambiguous-label share {fmt(v['ambiguous_share'])}")
            d = m['teacher_view_vs_student_view']
            print(f"   full map only: {d['only_teacher']} labels "
                  f"(precision {fmt(d['only_teacher_precision'])}); "
                  f"changed {d['changed']}; same {d['same']}; "
                  f"student-view only {d['only_student']}")
    return result


def fmt(x):
    return 'n/a' if x is None else f'{x:.3f}'


def summarize(rows):
    """Per view: coverage, precision against the oracle, and the share of
    labels given on a student view that receives more than one action."""
    out = dict(states=len(rows))
    for name, k in (('student_view', 1), ('teacher_view', 2)):
        given = [r for r in rows if r[k] is not None]
        actions = defaultdict(set)
        for r in given:
            actions[r[0]].add(r[k])
        ambiguous = [r for r in given if len(actions[r[0]]) > 1]
        out[name] = dict(
            labelled=len(given),
            coverage=len(given) / max(len(rows), 1),
            precision=(sum(r[k] in r[3] for r in given) / len(given)
                       if given else None),
            ambiguous_share=(len(ambiguous) / len(given) if given
                             else None))
    only_t = [r for r in rows if r[2] is not None and r[1] is None]
    out['teacher_view_vs_student_view'] = dict(
        only_teacher=len(only_t),
        only_teacher_precision=(sum(r[2] in r[3] for r in only_t)
                                / len(only_t) if only_t else None),
        only_student=sum(r[1] is not None and r[2] is None for r in rows),
        changed=sum(None not in (r[1], r[2]) and r[1] != r[2]
                    for r in rows),
        same=sum(r[1] is not None and r[1] == r[2] for r in rows))
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('banks', 'check', 'labels'))
    cli.add_argument('--layouts', type=int, default=60)
    cli.add_argument('--per-layout', type=int, default=8)
    cli.add_argument('--out', type=Path, default=Path(
        'docs/assets/teacher_view_2026-10-01/labels.json'))
    args = cli.parse_args()
    if args.action == 'banks':
        write_banks()
        check_banks()
    elif args.action == 'check':
        check_banks()
    else:
        result = labels(args.layouts, args.per_layout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
