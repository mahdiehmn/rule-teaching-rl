"""Four-arm, free KeyCorridor progress/exposure follow-up.

This separate runner never registers suites
in the existing fix-wave runner. Only the user submits cluster jobs.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from scripts import run_fix_wave_20260929 as fw

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'rule_timing_20261005_v1'
PROTOCOL = 'research/rule_timing_protocol_2026-10-05.md'
CALIBRATION = 'research/rule_timing_calibration_2026-10-05.json'
ARMS = ('none', 'full', 'no_progress', 'full_thinned')
REPS = range(90, 100)
BASE_ARMS = dict(none='none', full='rules_mem_weak',
                 no_progress='rules_mem_weak_noprog',
                 full_thinned='rules_mem_weak')
REQUIRED = (PROTOCOL, CALIBRATION,
            'scripts/run_rule_timing_20261005.py',
            'scripts/submit_rule_timing.sh', 'monitoring/rule_timing.py',
            'algos/ppo_distill.py', 'teachers/evidence.py',
            'teachers/minigrid/rule_bank.py', fw.MEM_BANK, fw.NOPROG_BANK)


def calibrate(source, out):
    """
    Freeze one pooled fraction from twenty existing development runs.
    """

    source = Path(source).resolve()
    records = []
    totals = {}
    for suite, arm in (('kc_mem_confirm', 'rules_mem_weak'),
                       ('kc_prog', 'rules_mem_weak_noprog')):
        batch = Path(source) / suite
        manifest = fw.read(batch / 'manifest.json')
        cells = [c for c in manifest['cells']
                 if c['bonus'] == 'none' and c['arm'] == arm]
        if sorted(c['replicate'] for c in cells) != list(range(30, 40)):
            raise ValueError('Calibration needs all ten development pairs')
        labels = 0
        for cell in cells:
            exit_path = batch / 'cells' / str(cell['index']) / 'exit.json'
            receipt = fw.read(exit_path)
            if receipt['artifact_status'] != 'terminal_contract_validated':
                raise ValueError('Incomplete calibration run')
            run = fw.inside(batch, receipt['runs'][0])
            summary_path = run / 'run_summary.json'
            summary = fw.read(summary_path)
            metrics = fw.validate(run, cell)
            latest = summary['latest']
            count = latest['advisor_control']['teacher_labels']
            if (count != latest['advising']['num_delivered']
                    or count <= 0 or metrics['api_dollars'] != 0
                    or summary['args']['advice_keep_fraction'] != 1.0
                    or summary['args']['total_timesteps'] != 5_000_000):
                raise ValueError('Calibration exposure contract differs')
            labels += count
            records.append(dict(
                suite=suite, arm=arm, seed=cell['seed'], labels=count,
                bank=summary['args']['rule_bank'],
                bank_sha256=summary['args']['rule_bank_sha256'],
                summary_path=summary_path.relative_to(source).as_posix(),
                summary_sha256=fw.digest(summary_path),
                manifest_sha256=fw.digest(batch / 'manifest.json'),
                exit_sha256=fw.digest(exit_path)))
        totals[arm] = labels
    fraction = totals['rules_mem_weak_noprog'] / totals['rules_mem_weak']
    if not 0 < fraction < 1:
        raise ValueError('Full-bank thinning cannot match these totals')
    payload = dict(
        author='mahdiehmn', date='2026-10-05', study=STUDY,
        status='development-only pooled total-label calibration',
        thinned_arm='full', keep_fraction=fraction, totals=totals,
        records=records,
        limitation='Approximate exposure; no per-interval or gradient match')
    out = Path(out)
    if out.exists() and fw.read(out) != payload:
        raise FileExistsError('Preserve the existing frozen calibration')
    fw.write(out, payload)
    return payload


def calibration(root=ROOT):
    """
    Check the frozen direction and pooled arithmetic before cell creation.
    """

    data = fw.read(Path(root) / CALIBRATION)
    records = data['records']
    totals = {arm: sum(r['labels'] for r in records if r['arm'] == arm)
              for arm in ('rules_mem_weak', 'rules_mem_weak_noprog')}
    for arm in totals:
        seeds = sorted(r['seed'] for r in records if r['arm'] == arm)
        if seeds != list(range(14703000, 14704000, 100)):
            raise ValueError('Calibration seed identities differ')
    fraction = totals['rules_mem_weak_noprog'] / totals['rules_mem_weak']
    if (data['study'] != STUDY or data['thinned_arm'] != 'full'
            or data['totals'] != totals or not 0 < fraction < 1
            or fraction != data['keep_fraction']):
        raise ValueError('Frozen calibration identity or arithmetic differs')
    return data


def cells(root=ROOT):
    """
    Pair four arms on reserved fresh seeds, without modifying old suites.
    """

    fraction = calibration(root)['keep_fraction']
    result = []
    for replicate in REPS:
        for arm in ARMS:
            args = fw.arm_args('kc_fresh', BASE_ARMS[arm], replicate,
                               'none', root)
            args = replace(
                args, experiment_id=f'{STUDY}_{arm}',
                rule_timing_diagnostics=arm != 'none',
                advice_keep_fraction=fraction if arm == 'full_thinned'
                else 1.0)
            fw.cell(result, STUDY, arm, args, replicate)
    return result


def validate(run, record):
    """
    Check complete teacher-off curves and all retained-label accounting.
    """

    from scripts.run_rule_bank_pilot_20260928 import read_jsonl
    from scripts.run_explanation_formats_20260925 import derived_of
    from monitoring.evaluation_diagnostics import diagnostic_milestones
    metrics = fw.validate(run, record)
    summary = fw.read(Path(run) / 'run_summary.json')
    latest = summary['latest']
    args = fw.args_of(record)
    geometry = derived_of(args)
    points = diagnostic_milestones(
        args.diagnostic_eval_frames, geometry['batch_size'],
        geometry['num_iterations'])
    early = read_jsonl(Path(run) / 'diagnostic_evaluations.jsonl')
    regular = read_jsonl(Path(run) / 'evaluations.jsonl')
    for row in regular + early:
        keys = ['success_rate']
        if args.eval_sampled:
            keys.append('sampled_success_rate')
        if any(not isinstance(row.get(k), (int, float))
               or not math.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in keys):
            raise ValueError('Nonfinite or out-of-range success rate')
    if ([r['iteration'] for r in early] != sorted(points)
            or any(r['global_step'] != r['iteration'] *
                   geometry['batch_size']
                   or r['requested_frames'] != points[r['iteration']]
                   or r['teacher_on'] or r['episodes'] != args.eval_episodes
                   or r['seed_base'] != args.seed + 50_000
                   or r['policy_sha256_before'] != r['policy_sha256_after']
                   or not (r['rng_isolated'] or
                           r['reused_regular_evaluation'])
                   for r in early)
            or (0 in points and (
                early[0]['training_episodes'] != 0
                or early[0]['policy_sha256_before'] !=
                metrics['initial_sha256']))):
        raise ValueError('Dense early evaluation contract differs')
    labels = latest['advisor_control']['teacher_labels']
    if latest['teacher_cost_dollars'] != 0:
        raise ValueError('Timing study must have zero API cost')
    metrics['retained_labels'] = labels
    if record['arm'] == 'none':
        if labels or latest['teacher_total_queries']:
            raise ValueError('No-advice control received advice')
        return metrics
    rows = read_jsonl(Path(run) / 'rule_timing.jsonl')
    updates = read_jsonl(Path(run) / 'rule_timing_updates.jsonl')
    n = geometry['num_iterations']
    if ([r['rollout'] for r in rows] != list(range(n))
            or [r['rollout'] for r in updates] != list(range(n))
            or [r['global_step'] for r in rows] != [
                (i + 1) * geometry['batch_size'] for i in range(n)]):
        raise ValueError('Incomplete timing diagnostics')
    emitted = retained = 0
    for row in rows:
        emitted += row['emitted']
        retained += row['retained']
        phases = row['phases'].values()
        if (not 0 <= row['retained'] <= row['emitted']
                or row['coefficient'] != fw.ppo.distill_coef(
                    row['rollout'] + 1, n, args)
                or row['total_emitted'] != emitted
                or row['total_retained'] != retained
                or sum(p['advised'] for p in phases) != row['emitted']
                or sum(p['retained'] for p in phases) != row['retained']
                or (row['coefficient'] == 0 and row['retained'])):
            raise ValueError('Timing exposure accounting differs')
        for phase in phases:
            if (phase['queried'] != sum(phase[k] for k in (
                    'advised', 'no_rule', 'conflict'))
                    or sum(phase['emitted_actions'].values()) !=
                    phase['advised']
                    or sum(phase['retained_actions'].values()) !=
                    phase['retained']
                    or any(v < 0 for v in phase['emitted_actions'].values())
                    or any(v < 0 or v > phase['emitted_actions'].get(k, 0)
                           for k, v in phase['retained_actions'].items())):
                raise ValueError('Timing phase accounting differs')
    for update, row in zip(updates, rows):
        if (not 0 <= update['labeled_minibatches'] <=
                update['minibatches'] <= args.update_epochs *
                args.num_minibatches
                or not math.isfinite(update['mean_auxiliary_loss'])
                or update['mean_auxiliary_loss'] < 0
                or (not row['retained'] and (
                    update['labeled_minibatches'] or
                    update['mean_auxiliary_loss'] != 0))):
            raise ValueError('Timing optimization accounting differs')
    if (retained != labels
            or emitted != latest['advising']['num_delivered']):
        raise ValueError('Timing counters disagree with trainer')
    fraction = record['args']['advice_keep_fraction']
    if fraction < 1:
        gates = read_jsonl(Path(run) / 'evidence_gate.jsonl')
        if len(gates) != n or any(
                g['rollout'] != r['rollout']
                or g['global_step'] != r['global_step']
                or g['usable'] != r['emitted']
                or g['accepted'] != r['retained']
                or g['used'] != r['retained']
                or g['rejected'] != r['emitted'] - r['retained']
                or g['cap'] != round(fraction * r['emitted'])
                or r['retained'] != round(fraction * r['emitted'])
                for g, r in zip(gates, rows)):
            raise ValueError('Random-retention contract differs')
    elif retained != emitted:
        raise ValueError('Unfiltered arm unexpectedly dropped labels')
    metrics['emitted_labels'] = emitted
    metrics['labeled_minibatches'] = sum(
        r['labeled_minibatches'] for r in updates)
    return metrics


def verify(batch, rebuild=True):
    """
    Refuse modified manifests or archived source before any worker runs.
    """

    batch = Path(batch).resolve()
    data = fw.read(batch / 'manifest.json')
    if (data['study'] != STUDY
            or fw.digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or (batch / 'READY').read_text().strip() != data['commit']
            or not set(REQUIRED) <= set(data['source_hashes'])):
        raise ValueError('Frozen timing preparation identity differs')
    for name, digest in data['source_hashes'].items():
        if fw.digest(fw.inside(batch / 'code', name)) != digest:
            raise ValueError(f'Archived source changed: {name}')
    if rebuild and data['cells'] != cells(batch / 'code'):
        raise ValueError('Frozen timing cells changed')
    return data


def prepare(root=ROOT):
    """
    Archive committed source into a new batch with an immutable manifest.
    """

    root = Path(root).resolve()
    batch = root / 'results/rule_timing' / STUDY
    if batch.exists():
        verify(batch)
        return batch
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete timing packet first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    manifest = dict(
        author='mahdiehmn', study=STUDY, protocol=PROTOCOL, commit=commit,
        api_calls=0, runtime=fw.runtime_identity(),
        cells=cells(batch / 'code'),
        calibration_sha256=fw.digest(batch / 'code' / CALIBRATION),
        source_hashes={p.relative_to(batch / 'code').as_posix():
                       fw.digest(p) for p in (batch / 'code').rglob('*')
                       if p.is_file()})
    fw.write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(fw.digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    return batch


def run_cell(batch, index):
    """
    Execute exactly one archived cell and preserve any failed attempt.
    """

    batch = Path(batch).resolve()
    manifest = verify(batch)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute archived timing source')
    if fw.runtime_identity() != manifest['runtime']:
        raise ValueError('Worker environment differs from preparation')
    record = manifest['cells'][index]
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    fw.write(directory / 'dispatch.json', dict(
        cell=record, commit=manifest['commit'],
        manifest_sha256=fw.digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID')))
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    args = record['args']
    pattern = f"*{args['experiment_id']}__{args['seed']}__*/run_summary.json"
    try:
        if list((ROOT / 'results/runs').glob(pattern)):
            raise FileExistsError('An earlier attempt exists for this cell')
        process = subprocess.run([
            sys.executable, '-u', '-m', 'algos.ppo_distill',
            *fw.trainer_argv(args)], cwd=ROOT)
        outcome['returncode'] = process.returncode
        runs = list((ROOT / 'results/runs').glob(pattern))
        outcome['runs'] = [p.parent.relative_to(batch).as_posix()
                           for p in runs]
        if process.returncode == 0 and len(runs) == 1:
            outcome['metrics'] = validate(runs[0].parent, record)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    fw.write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit(batch):
    """
    User-only entry point: one unthrottled Slurm array of forty cells.
    """

    batch = Path(batch).resolve()
    manifest = verify(batch)
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as handle:
        handle.write('Single attempt per planned cell; no auto retries.\n')
    command = [
        'sbatch', '--parsable', '--job-name=rule_timing',
        f"--array=0-{len(manifest['cells']) - 1}", '--time=1-06:00:00',
        f'--output={batch}/slurm/%x_%A_%a.out',
        str(batch / 'code/scripts/submit_rule_timing.sh'), str(batch)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    fw.write(batch / 'submission.json', dict(command=command, job_id=job))
    print(f'Timing study: job {job}, 40 cells')


def report(batch):
    """
    Revalidate complete artifacts; withhold inference on partial cohorts.
    """

    batch = Path(batch).resolve()
    manifest = verify(batch, rebuild=False)
    groups = {arm: {} for arm in ARMS}
    statuses = dict(validated=[], failed=[], dispatched_no_exit=[],
                    unattempted=[])
    for cell in manifest['cells']:
        path = batch / 'cells' / str(cell['index']) / 'exit.json'
        if not path.exists():
            state = ('dispatched_no_exit' if
                     (path.parent / 'dispatch.json').exists()
                     else 'unattempted')
            statuses[state].append(cell['index'])
            continue
        receipt = fw.read(path)
        if receipt['artifact_status'] == 'terminal_contract_validated':
            metrics = validate(fw.inside(batch, receipt['runs'][0]), cell)
            groups[cell['arm']][cell['seed']] = metrics
            statuses['validated'].append(cell['index'])
        else:
            statuses['failed'].append(cell['index'])
    result = dict(study=STUDY, completed={
        arm: len(rows) for arm, rows in groups.items()}, expected=40,
        cell_status=statuses)
    if any(len(rows) != 10 for rows in groups.values()):
        result['inference'] = 'Deferred until all forty cells validate'
        return result
    contrasts = [('full_thinned', 'no_progress'), ('full', 'none'),
                 ('full', 'no_progress')]
    tests = [fw.paired(groups[a], groups[b], f'{a}-{b}')
             for a, b in contrasts]
    if any('error' in test for test in tests):
        raise ValueError('Paired initial policies differ')
    for test, adjusted in zip(tests, fw.holm([r['p'] for r in tests])):
        test['holm_p'] = adjusted
    full = groups['full_thinned']
    ablated = groups['no_progress']
    denominator = sum(r['retained_labels'] for r in ablated.values())
    ratio = (sum(r['retained_labels'] for r in full.values()) / denominator
             if denominator else None)
    result.update(contrasts=tests, exposure_ratio=ratio,
                  aggregate_exposure_comparable=(
                      ratio is not None and 0.9 <= ratio <= 1.1),
                  exposure_by_seed={str(seed): {
                      arm: rows[seed]['retained_labels']
                      for arm, rows in groups.items()}
                      for seed in full})
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=(
        'calibrate', 'check', 'prepare', 'launch', 'run-cell', 'report'))
    cli.add_argument('--root', type=Path, default=ROOT)
    cli.add_argument('--source', type=Path)
    cli.add_argument('--out', type=Path)
    cli.add_argument('--batch', type=Path)
    cli.add_argument('--index', type=int)
    args = cli.parse_args()
    if args.action == 'calibrate':
        if args.source is None or args.out is None:
            cli.error('calibrate requires --source and --out')
        print(json.dumps(calibrate(args.source, args.out)['totals']))
    elif args.action == 'check':
        values = cells(args.root)
        print(f'PASS {STUDY}: {len(values)} cells, 10 paired seeds, '
              f"keep fraction {calibration(args.root)['keep_fraction']}")
    elif args.action == 'run-cell':
        return run_cell(args.batch, args.index)
    elif args.action == 'report':
        result = report(args.batch)
        if args.out:
            fw.write(args.out, result)
        print(json.dumps(result, indent=2))
    else:
        batch = prepare(args.root)
        if args.action == 'launch':
            submit(batch)
        else:
            print(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
