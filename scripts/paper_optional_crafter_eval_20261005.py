"""O2: all fifty historical Crafter policies on fifty fresh shared worlds.

No training, teacher, API call or submission.
The parent optional launcher supplies a committed frozen source archive.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

import numpy as np
from scipy import stats
import torch

from scripts import audit_crafter_learning_20261001 as audit
from scripts import crafter_fresh_worlds_20261001 as old
from scripts import paper_bulk_crafter_20261005 as bulk
from scripts import run_crafter_vulcan_20261002 as historical
from scripts.reproduce_crafter_endpoints_20261001 import state_hash


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_optional_crafter_eval_20261005_v1'
PROTOCOL = 'research/paper_optional_crafter_eval_2026-10-05.md'
TIME = '02:00:00'
MAX_SECONDS = 7000
LAYOUTS = tuple(range(45_500_000, 45_500_050))
CAP = 3000
FILES = ('args.json', 'run_summary.json', 'evaluations.jsonl',
         'final_model.pt')
PRIMARY = (('rules_v3b_weak', 'none'),
           ('count_rules_v3b', 'count_none'),
           ('rules_v3b_weak', 'rules_noprog_weak'))
SECONDARY = (('rules_noprog_weak', 'none'), ('count_none', 'none'),
             ('count_rules_v3b', 'rules_v3b_weak'))


def write(path, value):
    """
    Preserve every manifest, attempt and result as an exclusive write.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def runtime():
    """
    Admit the two known Crafter builds and retain exact worker identity.
    """

    torch.set_num_threads(1)
    packages = {name: version(name) for name in (
        'torch', 'numpy', 'scipy', 'crafter', 'opensimplex', 'gymnasium',
        'Pillow')}
    if packages['crafter'] not in ('1.8.3', '1.8.3+computecanada'):
        raise ValueError(f'Unsupported Crafter build: {packages["crafter"]}')
    return dict(python=sys.version, packages=packages, device='cpu',
                torch_threads=torch.get_num_threads(),
                torch_git_version=torch.version.git_version)


def source_hashes(root=ROOT):
    """
    Bind executable source and the protocol, including imported helpers.
    """

    root = Path(root).resolve()
    files = {Path(PROTOCOL)}
    for directory in ('algos', 'envs', 'scripts', 'teachers', 'advising',
                      'monitoring'):
        files.update(path.relative_to(root)
                     for path in (root / directory).rglob('*.py'))
    files.update(Path(name) for name in (bulk.BANK, bulk.NOPROG))
    return {path.as_posix(): hashlib.sha256(
        (root / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for path in sorted(files)}


def cells():
    """
    Keep every original arm and seed, without checkpoint selection.
    """

    return [dict(index=cell['index'], arm=cell['arm'], seed=cell['seed'],
                 source_name=f'{cell["arm"]}_s{cell["seed"]}')
            for cell in historical.cells()]


def load_model(path):
    """
    Require finite exact tensors, then freeze the unchanged policy.
    """

    from algos.ppo_crafter import Net

    tensors = torch.load(path, map_location='cpu', weights_only=True)
    schema = bulk._checkpoint_schema()
    if not isinstance(tensors, dict) or set(tensors) != set(schema):
        raise ValueError('Wrong Crafter checkpoint keys')
    for name, value in tensors.items():
        if (not isinstance(value, torch.Tensor)
                or (tuple(value.shape), value.dtype) != schema[name]
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid Crafter checkpoint tensor: {name}')
    with torch.random.fork_rng(devices=[]):
        net = Net()
    net.load_state_dict(tensors, strict=True)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def validate_source_run(folder, cell):
    """
    Reuse strict original curve checks with the exact five-arm recipe.
    """

    folder = Path(folder)
    arm, seed = cell['arm'], cell['seed']
    cfg = historical.ARMS[arm]
    bonus = 'count' if cfg.get('count_coef', 0.) else 'none'
    expected = bulk.frozen_args(seed, bonus, cfg['rule_bank'])
    saved = audit.read(folder / 'args.json')
    # Preserve the historical output spelling after results are copied.
    # All scientific fields remain exact; original paths are provenance.
    original = PurePosixPath(saved['out'].replace('\\', '/'))
    if (not original.is_absolute()
            or original.name != cell['source_name']
            or original.parent.name != historical.STUDY):
        raise ValueError('Historical output path has the wrong identity')
    expected['out'] = saved['out']
    checker_arm = 'rules_weak' if cfg['rule_bank'] else 'none'
    row = bulk._checker(expected).validate_run(folder, checker_arm, seed)
    summary = audit.read(folder / 'run_summary.json')
    for key in ('auc', 'wall_time_sec', 'sps'):
        if not math.isfinite(summary[key]):
            raise ValueError('Nonfinite historical run metric')
    count = summary['count_bonus']
    if (count['coef'] != expected['count_coef']
            or not math.isfinite(count['total'])
            or count['total'] < 0
            or (count['total'] > 0) != (bonus == 'count')
            or count['total'] > expected['count_coef'] * 999424):
        raise ValueError('Invalid historical count-bonus record')
    if any(type(summary['teacher'][key]) is not int
           for key in ('asked', 'labelled', 'llm_calls')):
        raise ValueError('Historical teacher counts must be integers')
    net = load_model(folder / 'final_model.pt')
    return dict(arm=arm, seed=seed, args=saved,
                artifact_sha256=row['artifact_sha256'],
                policy_sha256=state_hash(net), historical_auc=row['auc'])


def check(root=ROOT):
    """
    Check software admission without requiring historical model files.
    """

    identity = runtime()
    if len(cells()) != 50 or len(LAYOUTS) != 50:
        raise ValueError('Wrong frozen evaluation dimensions')
    # Reserved prior construction/check panels span 30M and 43M;
    # known old and current training pools span 40M, evaluation 41M,
    # and the previous fresh-world study used 44M.
    old_panels = set(range(44_000_000, 44_000_050))
    old_panels.update(range(41_000_000, 41_000_010))
    for seed in (*range(1, 41), *range(141, 151)):
        old_panels.update(range(40_000_000 + seed * 1000,
                                40_000_200 + seed * 1000))
    if set(LAYOUTS) & old_panels or any(
            30_000_000 <= x < 31_000_000 or 43_000_000 <= x < 44_000_000
            for x in LAYOUTS):
        raise ValueError('Fresh panel overlaps a reserved historical block')
    return dict(study=STUDY, cells=50, episodes=2500,
                layouts=list(LAYOUTS), runtime=identity,
                source_sha256=source_hashes(root), policy_updates=0,
                llm_calls=0)


def preflight(source_batch, root=ROOT):
    """
    Validate all fifty source policies without creating any output paths.
    """

    source_batch = Path(source_batch).resolve()
    admitted = check(root)
    # Inventory every required artifact before loading any model. A
    # metadata-only sync must fail before the parent creates an archive.
    required = [source_batch / 'manifest.json']
    for cell in cells():
        required.append(source_batch / 'exits' / f'{cell["index"]}.json')
        required.extend(source_batch / cell['source_name'] / name
                        for name in FILES)
    missing = [path for path in required if not path.is_file()]
    if missing:
        examples = ', '.join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f'{len(missing)} missing historical artifacts: {examples}')
    source_manifest = audit.read(source_batch / 'manifest.json')
    if (source_manifest['study'] != historical.STUDY
            or source_manifest['protocol'] != historical.PROTOCOL
            or source_manifest['cells'] != historical.cells()
            or source_manifest['llm_calls'] != 0):
        raise ValueError('Historical manifest does not describe all 50 runs')
    if {p.name for p in source_batch.glob('*_s*') if p.is_dir()} != {
            c['source_name'] for c in cells()}:
        raise ValueError('Missing or extra historical run directories')
    records = []
    for cell in cells():
        exit_path = source_batch / 'exits' / f'{cell["index"]}.json'
        receipt = audit.read(exit_path)
        if (receipt['cell'] != historical.cells()[cell['index']]
                or receipt['returncode'] != 0
                or receipt['completed'] is not True):
            raise ValueError('Historical worker did not complete this cell')
        validated = validate_source_run(source_batch / cell['source_name'],
                                        cell)
        records.append(dict(**cell, validated=validated,
                            exit_sha256=audit.digest(exit_path)))
    return dict(
        **admitted, source_batch=str(source_batch),
        source_manifest=source_manifest,
        source_manifest_sha256=audit.digest(source_batch / 'manifest.json'),
        records=records)


def prepare(source_batch, out, root=ROOT):
    """
    Validate all inputs and freeze private model copies before evaluation.
    """

    source_batch, out = Path(source_batch).resolve(), Path(out).resolve()
    if (out / 'manifest.json').exists():
        manifest = validate_manifest(out, root)
        if manifest['source_batch'] != str(source_batch):
            raise ValueError('Prepared batch has another historical source')
        return manifest
    if out.exists():
        raise FileExistsError('Preserve incomplete preparation for inspection')
    admitted = preflight(source_batch, root)
    # Only validated complete inputs are copied. Copies bind execution
    # even if a historical results tree is later moved or changed.
    out.mkdir(parents=True)
    for cell in admitted['records']:
        target = out / 'inputs' / cell['source_name']
        target.mkdir(parents=True)
        for name, expected in cell['validated']['artifact_sha256'].items():
            shutil.copyfile(source_batch / cell['source_name'] / name,
                            target / name)
            if audit.digest(target / name) != expected:
                raise ValueError('Historical artifact changed during copy')
    from crafter.constants import achievements

    manifest = dict(
        **admitted, created_utc=datetime.now(timezone.utc).isoformat(),
        achievement_names=list(achievements),
        episode_cap=CAP, max_seconds=MAX_SECONDS, max_attempts=1,
        action_mode='sampled', advisor_on=False,
        generator='training_seed_once_per_model_carried_across_worlds')
    write(out / 'manifest.json', manifest)
    return manifest


def validate_manifest(out, root=ROOT, worker=False):
    """
    Bind protocol, source, panel, records and exact worker runtime.
    """

    out = Path(out)
    manifest = audit.read(out / 'manifest.json')
    fixed = dict(study=STUDY, cells=50, episodes=2500,
                 layouts=list(LAYOUTS), episode_cap=CAP,
                 max_seconds=MAX_SECONDS, max_attempts=1,
                 action_mode='sampled', advisor_on=False,
                 policy_updates=0, llm_calls=0,
                 generator=(
                     'training_seed_once_per_model_carried_across_worlds'))
    if any(manifest.get(k) != value for k, value in fixed.items()):
        raise ValueError('Frozen Crafter evaluation contract changed')
    if manifest['source_sha256'] != source_hashes(root):
        raise ValueError('Frozen evaluation source changed')
    if worker and not Path(__file__).resolve().is_relative_to(
            Path(root).resolve()):
        raise ValueError('Worker was not imported from its frozen archive')
    if worker and manifest['runtime'] != runtime():
        raise ValueError('Worker runtime differs from prepared runtime')
    records = manifest['records']
    if [{k: r[k] for k in c} for r, c in zip(records, cells())] != cells():
        raise ValueError('Frozen evaluation cell identities changed')
    if len(records) != 50:
        raise ValueError('Wrong number of historical policies')
    original = manifest['source_manifest']
    if (original['study'] != historical.STUDY
            or original['protocol'] != historical.PROTOCOL
            or original['cells'] != historical.cells()
            or original['llm_calls'] != 0):
        raise ValueError('Frozen historical manifest changed')
    for record in records:
        validated = record['validated']
        cfg = historical.ARMS[record['arm']]
        bonus = 'count' if cfg.get('count_coef', 0.) else 'none'
        expected = bulk.frozen_args(record['seed'], bonus, cfg['rule_bank'])
        expected['out'] = validated['args']['out']
        if (validated['arm'] != record['arm']
                or validated['seed'] != record['seed']
                or validated['args'] != expected
                or set(validated['artifact_sha256']) != set(FILES)):
            raise ValueError('Frozen historical input recipe changed')
        for value in (validated['policy_sha256'], record['exit_sha256'],
                      *validated['artifact_sha256'].values()):
            if len(value) != 64:
                raise ValueError('Invalid input hash')
            int(value, 16)
    from crafter.constants import achievements

    if manifest['achievement_names'] != list(achievements):
        raise ValueError('Achievement vocabulary changed')
    return manifest


def _identity(out, record):
    """
    Bind each episode and receipt to its immutable manifest and model.
    """

    return dict(manifest_sha256=audit.digest(Path(out) / 'manifest.json'),
                index=record['index'], arm=record['arm'], seed=record['seed'],
                checkpoint_sha256=record['validated']['artifact_sha256'][
                    'final_model.pt'])


def _aggregate(episodes, names):
    """
    Reuse reviewed episode checks with the new fixed panel in isolation.
    """

    spec = importlib.util.spec_from_file_location(
        '_optional_crafter_episode_checker', old.__file__)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.LAYOUTS = LAYOUTS
    return checker.aggregate(episodes, names)


def run_cell(index, out, root=ROOT):
    """
    Execute one policy once, with retained journals and failure receipts.
    """

    out = Path(out).resolve()
    manifest = validate_manifest(out, root, worker=True)
    if type(index) is not int or not 0 <= index < 50:
        raise ValueError('Evaluation index must be between 0 and 49')
    cell = manifest['records'][index]
    identity = _identity(out, cell)
    directory = out / 'cells' / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write(directory / 'start.json', dict(
        **identity, started_unix=started, runtime=runtime(),
        slurm_job_id=os.getenv('SLURM_JOB_ID')))
    try:
        frozen = out / 'inputs' / cell['source_name']
        for name, expected in cell['validated']['artifact_sha256'].items():
            if audit.digest(frozen / name) != expected:
                raise ValueError('Frozen historical input changed')
        net = load_model(frozen / 'final_model.pt')
        before = state_hash(net)
        if before != cell['validated']['policy_sha256']:
            raise ValueError('Loaded policy differs from frozen policy')
        from envs import crafter_symbolic as cs

        deadline = started + MAX_SECONDS
        rng = torch.Generator().manual_seed(cell['seed'])
        rows = []
        with (directory / 'episodes.jsonl').open(
                'x', encoding='utf-8') as journal:
            for layout in LAYOUTS:
                old.check_time(deadline)
                world = cs.world_pool([layout])[0]
                old.check_time(deadline)
                row = dict(**identity, layout=layout, **old.episode(
                    net, cs.CrafterSymbolic([world], seed=0), rng, deadline))
                journal.write(json.dumps(row, allow_nan=False) + '\n')
                journal.flush()
                rows.append(row)
        old.check_time(deadline)
        after = state_hash(net)
        if before != after:
            raise ValueError('Evaluation changed policy parameters')
        write(directory / 'result.json', dict(
            **identity, policy_before=before, policy_after=after,
            episodes=50, policy_updates=0, advisor_on=False, llm_calls=0,
            journal_sha256=audit.digest(directory / 'episodes.jsonl'),
            summary=_aggregate(rows, manifest['achievement_names'])))
        write(directory / 'exit.json', dict(
            **identity, status='completed', ended_unix=time.time()))
    except Exception as exc:
        write(directory / 'exit.json', dict(
            **identity, status='failed', ended_unix=time.time(),
            error_type=type(exc).__name__, error=str(exc)))
        raise
    return 0


def contrasts(rows):
    """
    Use model means as observations, paired over ten reused training seeds.
    """

    grid = {(r['arm'], r['seed']): r for r in rows}
    if len(rows) != 50 or set(grid) != {
            (c['arm'], c['seed']) for c in cells()}:
        raise ValueError('Require all fifty policy endpoints')
    results = []
    for a, b in (*PRIMARY, *SECONDARY):
        differences = np.array([
            grid[a, seed]['achievement_total']
            - grid[b, seed]['achievement_total']
            for seed in historical.SEEDS], dtype=float) / 50
        estimate = audit.interval(differences)
        p = None if estimate['zero_empirical_variance'] else float(
            stats.ttest_1samp(differences, 0).pvalue)
        results.append(dict(
            contrast=f'{a} - {b}', n=10, differences=differences.tolist(),
            role='primary_Holm3' if (a, b) in PRIMARY else 'descriptive',
            p_two_sided=p, **estimate))
    order = sorted(range(3), key=lambda i: (
        1. if results[i]['p_two_sided'] is None
        else results[i]['p_two_sided']))
    running = 0.
    for rank, i in enumerate(order):
        p = results[i]['p_two_sided']
        running = max(running, min(1., (3 - rank) * (1. if p is None else p)))
        results[i]['p_holm3'] = None if p is None else running
        results[i]['positive_effect_supported'] = bool(
            p is not None and results[i]['mean'] > 0 and running < .05)
    return results


def report(out, root=ROOT):
    """
    Recompute all 2500 episodes before any endpoint inference.
    """

    out = Path(out).resolve()
    manifest = validate_manifest(out, root)
    if {p.name for p in (out / 'cells').iterdir() if p.is_dir()} != {
            str(i) for i in range(50)}:
        raise ValueError('Require exactly fifty evaluation directories')
    rows = []
    for cell in manifest['records']:
        directory = out / 'cells' / str(cell['index'])
        identity = _identity(out, cell)
        start, end, saved = (audit.read(directory / name) for name in (
            'start.json', 'exit.json', 'result.json'))
        if any(any(item.get(k) != v for k, v in identity.items())
               for item in (start, end, saved)):
            raise ValueError('Evaluation receipt identity differs')
        elapsed = end['ended_unix'] - start['started_unix']
        if (end['status'] != 'completed' or not 0 <= elapsed <= MAX_SECONDS
                or start['runtime'] != manifest['runtime']):
            raise ValueError('Incomplete or invalid evaluation attempt')
        expected_policy = cell['validated']['policy_sha256']
        if (saved['policy_before'] != expected_policy
                or saved['policy_after'] != expected_policy
                or saved['episodes'] != 50 or saved['policy_updates'] != 0
                or saved['advisor_on'] is not False
                or saved['llm_calls'] != 0):
            raise ValueError('Evaluation changed the teacher-off contract')
        frozen = out / 'inputs' / cell['source_name']
        for name, expected in cell['validated']['artifact_sha256'].items():
            if audit.digest(frozen / name) != expected:
                raise ValueError('Frozen historical input changed')
        journal = directory / 'episodes.jsonl'
        if saved['journal_sha256'] != audit.digest(journal):
            raise ValueError('Episode journal changed')
        episodes = [json.loads(line) for line in journal.read_text(
            encoding='utf-8').splitlines()]
        if any(any(e.get(k) != v for k, v in identity.items())
               for e in episodes):
            raise ValueError('Episode identity differs')
        actual = _aggregate(episodes, manifest['achievement_names'])
        if actual != saved['summary']:
            raise ValueError('Saved endpoint differs from raw episodes')
        rows.append(dict(arm=cell['arm'], seed=cell['seed'], **actual))
    result = dict(study=STUDY, complete=50, episodes=2500, rows=rows,
                  contrasts=contrasts(rows),
                  manifest_sha256=audit.digest(out / 'manifest.json'),
                  independent_result_review='pending',
                  scope='Endpoint generalization on a fresh shared panel; '
                  'ten reused training seeds, not new learning evidence')
    target = out / 'report.json'
    if target.exists():
        if audit.read(target) != result:
            raise ValueError('Existing report differs; preserve both inputs')
    else:
        write(target, result)
    return result


def main():
    """
    Expose source admission, preparation, one-policy workers and reporting.
    """

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('check', 'preflight', 'prepare',
                                        'run-cell', 'report'))
    cli.add_argument('--source-batch', type=Path)
    cli.add_argument('--out', type=Path)
    cli.add_argument('--root', type=Path, default=ROOT)
    cli.add_argument('--index', type=int)
    args = cli.parse_args()
    if args.action == 'check':
        result = check(args.root)
    elif args.action == 'preflight':
        result = preflight(args.source_batch, args.root)
    elif args.action == 'prepare':
        result = prepare(args.source_batch, args.out, args.root)
    elif args.action == 'run-cell':
        return run_cell(args.index, args.out, args.root)
    else:
        result = report(args.out, args.root)
    print(json.dumps({k: result[k] for k in (
        'study', 'cells', 'episodes', 'complete') if k in result}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
