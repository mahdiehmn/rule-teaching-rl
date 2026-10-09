"""
Derive simulator-corrected categorical plan targets without teacher calls.

Only the existing plan categories change. The
original full-state bank, order, split, observations, actions, other formats
and donor mapping remain fixed; no case is filtered on quality or visibility.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from scripts import explanation_formats as formats
from scripts.explanation_screen_panel import file_hash, restore

VERSION = 'plan_repair_targets_20260927_v1'
EXPECTED_COUNTS = (182, 58)
FIELDS = tuple(name for name, _ in formats.PLAN_FIELDS)


def read_json(path):
    """
    Read a UTF-8 JSON artifact, accepting an optional Windows BOM.
    """

    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def native_plan_truth(case):
    """
    Independently classify restored native objects using view coordinates.
    """

    env = restore(case['state'])
    try:
        doors = [obj for obj in env.grid.grid
                 if obj is not None and obj.type == 'door']
        if len(doors) != 1:
            raise ValueError('Expected one native DoorKey door')
        if not doors[0].is_locked:
            phase, target = 'reach_target', 'none'
        elif env.carrying is not None and env.carrying.type == 'key':
            phase, target = 'unlock_door', 'goal'
        else:
            phase, target = 'seek_key', 'door'
        if case['phase'] != phase:
            raise ValueError('Saved phase disagrees with native state')
        result = dict(next_target=target, next_direction='none',
                      next_distance='none', next_hidden=False)
        if target == 'none':
            return result

        # Locate native objects without indexing the serialized grid.
        positions = [(x, y) for x in range(env.width)
                     for y in range(env.height)
                     if (obj := env.grid.get(x, y)) is not None
                     and obj.type == target]
        if len(positions) != 1:
            raise ValueError(f'Expected one native {target}')
        tx, ty = positions[0]
        vx, vy = env.get_view_coords(tx, ty)
        forward = env.agent_view_size - 1 - vy
        sideways = vx - env.agent_view_size // 2
        if abs(forward) >= abs(sideways):
            direction = 'ahead' if forward > 0 else 'behind'
        else:
            direction = 'right' if sideways > 0 else 'left'
        distance = sum(abs(a - b) for a, b in
                       zip(env.agent_pos, positions[0]))
        observed_grid, visibility = env.gen_obs_grid()
        visible = any(visibility[x, y]
                      and (obj := observed_grid.get(x, y)) is not None
                      and obj.type == target
                      for x in range(env.agent_view_size)
                      for y in range(env.agent_view_size))
        result.update(next_direction=direction,
                      next_distance=('near' if distance <= 3 else
                                     'medium' if distance <= 7 else 'far'),
                      next_hidden=not visible)
        return result
    finally:
        env.close()


def validate_sources(raw, panel, expected_counts=EXPECTED_COUNTS):
    """
    Join every immutable row to its native state and verify stored truth.
    """

    if raw.get('information_access') != 'full_state':
        raise ValueError('Plan repair requires the full_state bank')
    if 'plan_repair' in raw:
        raise ValueError('Raw source must not already be a repaired bank')
    counts = tuple(len(raw[s]) for s in ('train', 'audit'))
    if expected_counts is not None and counts != tuple(expected_counts):
        raise ValueError(f'Case counts changed: {counts}')
    cases = {c['case_id']: c for c in panel['cases']}
    if len(cases) != len(panel['cases']):
        raise ValueError('Duplicate panel case identity')
    rows = raw['train'] + raw['audit']
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate bank case identity')
    episodes = [{formats.canonical(r['episode']) for r in raw[s]}
                for s in ('train', 'audit')]
    if episodes[0] & episodes[1]:
        raise ValueError('Training and audit episodes overlap')
    truth = {}
    for split in ('train', 'audit'):
        expected_order = [c['case_id'] for c in panel['cases']
                          if c['case_id'] in {r['case_id']
                                             for r in raw[split]}]
        if expected_order != [r['case_id'] for r in raw[split]]:
            raise ValueError('Bank case order differs from source panel')
        for row in raw[split]:
            case = cases.get(row['case_id'])
            if case is None:
                raise ValueError('Bank case absent from source panel')
            identity = dict(split=case['split'],
                            episode=case['episode_group'],
                            phase=case['phase'],
                            observation=case['state']['local_obs'],
                            positive_action=case['positive_action'],
                            foil_action=case['foil_action'])
            if row['split'] != split or any(row[k] != v
                                           for k, v in identity.items()):
                raise ValueError(f'Case identity changed: {row["case_id"]}')
            for name, values in formats.PLAN_FIELDS:
                if row['targets']['plan'][name] not in values:
                    raise ValueError('Raw plan category is outside its enum')
            native = native_plan_truth(case)
            if native != formats.plan_truth(case):
                raise ValueError('Plan truth disagrees with native simulator')
            categorical = {name: native[name] for name in FIELDS}
            if row['checks']['plan_truth'] != categorical:
                raise ValueError('Stored plan truth is inconsistent')
            correct = {name: row['targets']['plan'][name] == categorical[name]
                       for name in FIELDS}
            if (row['checks']['plan'] != correct
                    or row['checks']['plan_next_hidden']
                    != native['next_hidden']):
                raise ValueError('Stored plan checks are inconsistent')
            truth[row['case_id']] = native

    # Reuse the old target-blind permutation exactly in every arm.
    donors = formats.bundle_permutation(raw['train'])
    for i, row in enumerate(raw['train']):
        donor = raw['train'][donors[i]]
        if row['permuted'] != dict(donor_id=donor['case_id'],
                                   targets=donor['targets']):
            raise ValueError('Frozen donor mapping or bundle changed')
    return truth


def build_corrected_bank(raw, panel, raw_sha, panel_sha,
                         expected_counts=EXPECTED_COUNTS):
    """
    Copy the bank and replace only aligned/permuted categorical plans.
    """

    truth = validate_sources(raw, panel, expected_counts)
    corrected = deepcopy(raw)
    for split in ('train', 'audit'):
        for row in corrected[split]:
            row['targets']['plan'] = {
                name: truth[row['case_id']][name] for name in FIELDS}
    lookup = {r['case_id']: r for r in corrected['train']}
    for row in corrected['train']:
        donor = lookup[row['permuted']['donor_id']]
        row['permuted']['targets']['plan'] = deepcopy(donor['targets']['plan'])
    corrected['plan_repair'] = dict(
        version=VERSION, source_bank_sha256=raw_sha,
        source_panel_sha256=panel_sha,
        changed_fields=['targets.plan', 'permuted.targets.plan'],
        retained_checks='Historical checks score the unchanged raw replies.',
        inherited_metadata=(
            'All inherited metadata, claims, coverage and changed fractions '
            'describe the historical raw bank, not corrected plan quality.'),
        categorical_target_source='Simulator-derived next-subgoal categories',
        corrected_permuted_changed_fraction=sum(
            r['targets']['plan'] != r['permuted']['targets']['plan']
            for r in corrected['train']) / len(corrected['train']),
        native_crosscheck='Native object/phase and MiniGrid view coordinates',
        filtering='none; every source case retained in its original order',
        calibration='Original raw-bank plan scale reused without refitting',
        other_formats='Unchanged original teacher outputs; not certified')
    return corrected


def validate_derived_bank(corrected, raw, panel, raw_sha, panel_sha,
                          expected_counts=EXPECTED_COUNTS):
    """
    Reject any derived-bank change beyond the declared plan replacement.
    """

    expected = build_corrected_bank(raw, panel, raw_sha, panel_sha,
                                    expected_counts)
    if corrected != expected:
        raise ValueError('Derived bank differs from exact plan-only repair')
    return target_report(raw, corrected)


def _bundle(target):
    return '|'.join(target[name] for name in FIELDS)


def _marginals(targets):
    result = {name: {value: sum(t[name] == value for t in targets)
                     for value in values}
              for name, values in formats.PLAN_FIELDS}
    result['bundles'] = dict(sorted(Counter(map(_bundle, targets)).items()))
    return result


def _changed(left, right):
    return dict(cases=len(left), changed_cases=sum(a != b
                for a, b in zip(left, right)),
                changed_fraction=sum(a != b for a, b in zip(left, right))
                / len(left), fields={name: sum(a[name] != b[name]
                    for a, b in zip(left, right)) / len(left)
                    for name in FIELDS})


def _ambiguity(rows, targets):
    groups = defaultdict(list)
    for row, target in zip(rows, targets):
        key = formats.canonical([row['observation'], row['positive_action'],
                                 row['foil_action']])
        groups[key].append(_bundle(target))
    conflicts = [v for v in groups.values() if len(set(v)) > 1]
    correct = sum(Counter(v).most_common(1)[0][1] for v in groups.values())
    return dict(cases=len(rows), distinct_inputs=len(groups),
                conflicting_inputs=len(conflicts),
                cases_in_conflicting_inputs=sum(map(len, conflicts)),
                empirical_exact_input_majority_accuracy=correct / len(rows),
                interpretation='Observed collisions are a lower bound on '
                'ambiguity; unique sampled inputs do not prove observability')


def _conditional(rows, variants, key):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[key(row)].append(i)
    result = {}
    for group, ids in sorted(groups.items()):
        marginals = {name: _marginals([targets[i] for i in ids])
                     for name, targets in variants.items()}
        differences = {}
        for left, right in (('raw', 'corrected'),
                            ('corrected', 'permuted_corrected')):
            if right not in marginals:
                continue
            differences[f'{left}_vs_{right}'] = {
                field: sum(abs(marginals[left][field].get(value, 0)
                               - marginals[right][field].get(value, 0))
                           for value in (set(marginals[left][field])
                                         | set(marginals[right][field])))
                / (2 * len(ids)) for field in (*FIELDS, 'bundles')}
        result[group] = dict(cases=len(ids), targets=marginals,
                             total_variation=differences)
    return result


def target_report(raw, corrected):
    """
    Describe corrections, permutation coverage and conditional structure.
    """

    report = dict(version=VERSION, splits={})
    for split in ('train', 'audit'):
        rows = raw[split]
        variants = dict(raw=[r['targets']['plan'] for r in rows],
                        corrected=[r['targets']['plan']
                                   for r in corrected[split]])
        if split == 'train':
            variants['permuted_raw'] = [r['permuted']['targets']['plan']
                                        for r in rows]
            variants['permuted_corrected'] = [
                r['permuted']['targets']['plan'] for r in corrected[split]]
        entry = dict(
            cases=len(rows), case_ids=[r['case_id'] for r in rows],
            hidden_next_target_cases=sum(
                r['checks']['plan_next_hidden'] for r in rows),
            correction=_changed(variants['raw'], variants['corrected']),
            marginals={k: _marginals(v) for k, v in variants.items()},
            input_ambiguity={k: _ambiguity(rows, v)
                             for k, v in variants.items()},
            by_phase=_conditional(rows, variants, lambda r: r['phase']),
            by_action_pair=_conditional(rows, variants, lambda r:
                f'{r["positive_action"]}:{r["foil_action"]}'),
            by_phase_action_pair=_conditional(rows, variants, lambda r:
                f'{r["phase"]}:{r["positive_action"]}:{r["foil_action"]}'))
        if split == 'train':
            entry['permutation'] = _changed(
                variants['corrected'], variants['permuted_corrected'])
            entry['donors'] = {r['case_id']: r['permuted']['donor_id']
                               for r in rows}
            if (entry['marginals']['corrected']
                    != entry['marginals']['permuted_corrected']):
                raise ValueError('Corrected bundle marginals changed')
        report['splits'][split] = entry

    # Fit shortcuts only on training episodes; evaluate once on audit.
    shortcuts = {}
    for label, bank in (('raw', raw), ('corrected', corrected)):
        grouped = defaultdict(Counter)
        total = Counter()
        for row in bank['train']:
            bundle = _bundle(row['targets']['plan'])
            grouped[(row['positive_action'], row['foil_action'])][bundle] += 1
            total[bundle] += 1
        def majority(counts):
            return sorted(counts, key=lambda x: (-counts[x], x))[0]
        predictions = [majority(grouped.get((r['positive_action'],
                       r['foil_action']), total)) for r in raw['audit']]
        shortcuts[label] = dict(
            audit_cases=len(predictions),
            fitted_split='train only; lexicographic tie break',
            audit_raw_accuracy=sum(p == _bundle(r['targets']['plan'])
                for p, r in zip(predictions, raw['audit'])) / len(predictions),
            audit_corrected_accuracy=sum(p == _bundle(r['targets']['plan'])
                for p, r in zip(predictions, corrected['audit']))
                / len(predictions))
    report['action_pair_majority_baselines'] = shortcuts
    return report


def calibration_scale(calibration, raw_sha, raw):
    """
    Require the frozen training-only raw-bank calibration without refitting.
    """

    if calibration.get('bank_sha256') != raw_sha:
        raise ValueError('Calibration is not tied to the original raw bank')
    if calibration.get('frozen_before_rl') is not True:
        raise ValueError('Calibration was not frozen before RL')
    ids = calibration.get('calibration_cases', [])
    train_ids = {r['case_id'] for r in raw['train']}
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= train_ids:
        raise ValueError('Calibration cases must be unique training cases')
    scale = calibration['primary_scale']['plan']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Plan calibration scale must be finite and positive')
    return scale


def derive(source_batch, output_dir):
    """
    Validate original seals and write a new versioned repair artifact set.
    """

    source, output = Path(source_batch), Path(output_dir)
    paths = dict(raw_bank=source / 'banks/full_state/format_bank.json',
                 panel=source / 'source/panel.json',
                 calibration=source / 'calibration.json')
    hashes = {name: file_hash(path) for name, path in paths.items()}
    frozen = read_json(source / 'banks/full_state/FROZEN.json')
    manifest = read_json(source / 'manifest.json')
    if (hashes['raw_bank'] != frozen['bank_sha256']
            or hashes['calibration'] != frozen['calibration_sha256']
            or hashes['panel'] != manifest['artifact_hashes'][
                'source/panel.json']):
        raise ValueError('Original source seal differs')
    raw, panel, calibration = (read_json(paths[k])
                               for k in ('raw_bank', 'panel', 'calibration'))
    scale = calibration_scale(calibration, hashes['raw_bank'], raw)
    corrected = build_corrected_bank(raw, panel, hashes['raw_bank'],
                                    hashes['panel'])
    report = target_report(raw, corrected)
    if output.exists():
        raise FileExistsError('Repair output exists; choose a fresh directory')
    output.mkdir(parents=True)
    for name, value in (('corrected_bank', corrected),
                        ('target_report', report)):
        path = output / f'{name}.json'
        path.write_text(formats.canonical(value) + '\n', encoding='utf-8')
        paths[name], hashes[name] = path, file_hash(path)
    result = dict(version=VERSION, lesson_scale=scale,
                  **{k: str(p.resolve()) for k, p in paths.items()},
                  **{f'{k}_sha256': value for k, value in hashes.items()})
    (output / 'repair_manifest.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-batch', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.source_batch, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
