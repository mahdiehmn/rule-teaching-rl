"""Prepare, run and report the free, frozen plan-correction diagnostic."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from scipy.stats import t
import torch
import tyro

from algos.ppo_plan_repair import Args
from algos.ppo_progress import policy_hash
from scripts import plan_repair_targets as targets
from scripts.plan_explanation_formats import ISOLATION_SHARED
from scripts.run_explanation_formats_20260925 import (
    derived_of, encoded, eval_steps, lesson_schedule, read_jsonl,
    runtime_identity, trainer_argv,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'plan_repair_20260927_v1'
SOURCE_STUDY = 'explanation_formats_20260925_v1'
SEEDS = tuple(12_700_000 + 100 * i for i in range(5))
ARMS = ('ppo', 'raw', 'corrected', 'permuted_corrected')
SCALE = 0.22896900710982449
PINS = {
    'banks/full_state/format_bank.json':
        '4c12a20ed0783d67c2d1aa83007f06a0834db7b9702877736ba5d3a217d22ce9',
    'source/panel.json':
        '71825bc9254feac2ffb6f0952259f2865984ff369c63ea0c5c2fca2574a9225a',
    'calibration.json':
        '8c6a539c8de72aab773e8b1bc1d51dd890fd30636c0ca708460755ef149837ce',
}
PATHS = {
    'raw_bank': 'corpora/raw_bank.json',
    'repair_panel': 'corpora/panel.json',
    'repair_calibration': 'corpora/calibration.json',
}
REQUIRED = (
    'scripts/run_plan_repair_20260927.py',
    'scripts/plan_repair_targets.py', 'algos/ppo_plan_repair.py',
    'algos/ppo_distill.py', 'algos/advice_control.py',
    'advising/uniform_queries.py', 'scripts/submit_plan_repair.sh',
    'scripts/launch_plan_repair.sh',
    'research/plan_repair_protocol_2026-09-27.md',
    'tests/test_plan_repair_launch.py',
    'tests/test_plan_repair_targets.py',
    'tests/test_advice_rng_isolation.py',
)


def read(path):
    """Read evidence without accessing credentials or a budget ledger."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """Identify the exact artifact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Refuse to overwrite an earlier manifest, dispatch or result."""
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def inside(batch, relative):
    """Bind portable manifest paths without accepting directory escapes."""
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Expected a relative POSIX artifact path')
    result = (Path(batch) / relative).resolve()
    result.relative_to(Path(batch).resolve())
    return result


def check_source(source):
    """Pin the already collected cases and old calibration, not a new bank."""
    source = Path(source)
    for name, expected in PINS.items():
        if digest(source / name) != expected:
            raise ValueError(f'Original input fingerprint differs: {name}')
    calibration = read(source / 'calibration.json')
    if calibration['primary_scale']['plan'] != SCALE:
        raise ValueError('The original plan scale changed')


def menu(hashes):
    """Four conditions differ in target assignment, with five fresh seeds."""
    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            bank = ('corpora/raw_bank.json' if arm in ('ppo', 'raw')
                    else 'corpora/corrected_bank.json')
            values = dict(
                **ISOLATION_SHARED, seed=seed, bonus='count',
                experiment_id=f'{STUDY}_{arm}',
                advisor_rng_isolation=True, advisor_sham=False,
                format_bank=bank, format_bank_sha256=hashes[bank],
                format_access='full_state', repair_arm=arm,
                raw_bank=PATHS['raw_bank'],
                raw_bank_sha256=hashes[PATHS['raw_bank']],
                repair_panel=PATHS['repair_panel'],
                repair_panel_sha256=hashes[PATHS['repair_panel']],
                repair_calibration=PATHS['repair_calibration'],
                repair_calibration_sha256=hashes[PATHS['repair_calibration']],
                lesson_format='plan', lesson_every=4, lesson_offset=0,
                lesson_exposures=1830, lesson_action_coef=0.0,
                lesson_mode='ppo' if arm == 'ppo' else 'explanation',
                lesson_targets=('permuted' if arm == 'permuted_corrected'
                                else 'aligned'),
                lesson_scale=0.0 if arm == 'ppo' else SCALE,
            )
            args = Args(**values)
            if asdict(tyro.cli(Args, args=trainer_argv(asdict(args)),
                              console_outputs=False)) != asdict(args):
                raise ValueError('Trainer command changed resolved settings')
            cells.append(dict(index=len(cells), seed=seed, arm=arm,
                              args=asdict(args)))
    return cells


def prepare(root, source):
    """Archive committed code and derive new immutable target artifacts."""
    root, source = Path(root).resolve(), Path(source).resolve()
    batch = root / 'results/explanation_lessons' / STUDY
    if batch.exists():
        verify(batch)
        return batch
    check_source(source)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', commit],
        cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete reviewed packet first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    # The builder retains original cases, donor mapping and all splits.
    targets.derive(source, batch / 'corpora')
    copies = {
        'banks/full_state/format_bank.json': 'corpora/raw_bank.json',
        'source/panel.json': 'corpora/panel.json',
        'calibration.json': 'corpora/calibration.json',
    }
    for old, new in copies.items():
        destination = batch / new
        if destination.exists():
            if digest(destination) != digest(source / old):
                raise ValueError('Derived input copy changed source bytes')
        else:
            shutil.copyfile(source / old, destination)
    artifacts = {p.relative_to(batch).as_posix(): digest(p)
                 for p in (batch / 'corpora').iterdir() if p.is_file()}
    manifest = dict(
        study=STUDY, commit=commit, runtime=runtime_identity(),
        source_study=SOURCE_STUDY, input_pins=PINS,
        api_calls=0, new_cost_usd=0.0, cells=menu(artifacts),
        source_hashes={p.relative_to(batch / 'code').as_posix(): digest(p)
                       for p in (batch / 'code').rglob('*') if p.is_file()},
        artifact_hashes=artifacts,
        protocol='research/plan_repair_protocol_2026-09-27.md',
    )
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    return batch


def verify(batch):
    """Reject missing, modified or differently configured preparation."""
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    if (digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or manifest['study'] != STUDY
            or (batch / 'READY').read_text().strip() != manifest['commit']
            or manifest['input_pins'] != PINS
            or manifest['source_study'] != SOURCE_STUDY
            or manifest['api_calls'] != 0 or manifest['new_cost_usd'] != 0
            or not set(REQUIRED) <= set(manifest['source_hashes'])
            or manifest['cells'] != menu(manifest['artifact_hashes'])):
        raise ValueError('Frozen preparation identity differs')
    for name, expected in manifest['source_hashes'].items():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    for name, expected in manifest['artifact_hashes'].items():
        if digest(inside(batch, name)) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return manifest


def resolved(batch, cell):
    """Resolve only declared input paths; every other setting stays frozen."""
    values = dict(cell['args'])
    for field in (*PATHS, 'format_bank'):
        values[field] = str(inside(batch, values[field]))
    return Args(**values)


def portable_settings(values):
    """Permit transferred root paths while preserving input filenames."""
    values = dict(values)
    for field in (*PATHS, 'format_bank'):
        value = values[field].replace('\\', '/')
        if '/corpora/' not in value and not value.startswith('corpora/'):
            raise ValueError('Unexpected artifact input path')
        values[field] = 'corpora/' + value.split('corpora/')[-1]
    return values


def validate_run(run, args):
    """Validate portable settings, exposure, evaluation and policy evidence."""
    run = Path(run)
    summary, contract, finish = (read(run / name) for name in (
        'run_summary.json', 'lesson_contract.json', 'lesson_finished.json'))
    derived = derived_of(args)
    endpoint = derived['num_iterations'] * derived['batch_size']
    settings = portable_settings({**asdict(args), **derived})
    if (summary.get('status') != 'completed'
            or summary['global_step'] != endpoint
            or finish['global_step'] != endpoint
            or portable_settings(summary['args']) != settings
            or portable_settings(contract['args']) != settings):
        raise ValueError('Incomplete endpoint or different settings')
    latest = summary.get('latest', {})
    zero_fields = (
        'teacher_total_queries', 'teacher_total_abstains',
        'teacher_total_declined', 'teacher_total_aliased',
        'teacher_total_gated', 'teacher_cost_dollars', 'teacher_wall_time_s',
        'teacher_compute_units', 'reference_calls', 'reference_labels',
        'reference_wall_time_s', 'reference_compute_units', 'distill_loss',
        'distill_coef',
    )
    control = latest.get('advisor_control', {})
    if (any(latest.get(key) != 0 for key in zero_fields)
            or latest.get('reference_teacher') is not None
            or control != dict(
                rng_mode='isolated_numpy_query_stream_v1',
                query_seed=args.seed + 90_117, sham=False,
                teacher_instances=0, query_order_draws=0, teacher_labels=0,
                first_label_global_step=None)):
        raise ValueError('Teacher-free summary evidence differs')
    native = (run / 'initial_policy.sha256').read_text().strip()
    initial = finish['initial_policy_sha256']
    if (not re.fullmatch('[0-9a-f]{64}', native)
            or not re.fullmatch('[0-9a-f]{64}', initial)
            or initial != contract['initial_policy_sha256']):
        raise ValueError('Missing or inconsistent initial policy')
    repair_identity = dict(
        version=targets.VERSION, arm=args.repair_arm,
        raw_bank_sha256=args.raw_bank_sha256,
        panel_sha256=args.repair_panel_sha256,
        calibration_sha256=args.repair_calibration_sha256)
    if (finish.get('plan_repair') != repair_identity or any(
            contract.get('plan_repair', {}).get(key) != value
            for key, value in repair_identity.items())):
        raise ValueError('Repair provenance differs')
    for name in ('consultations.jsonl', 'teacher_requests.jsonl'):
        path = run / name
        if path.exists() and path.read_text().strip():
            raise ValueError('Unexpected online teacher activity')
    schedule = lesson_schedule(derived['num_iterations'], args.lesson_offset,
                               args.lesson_every, args.lesson_exposures)
    active = args.lesson_mode != 'ppo'
    audit = finish.get('audit_plan_metrics', {})
    if (set(audit) != {'raw', 'native_corrected'}
            or finish.get('audit_plan_head_trained') is not active
            or any(not isinstance(scores, dict)
                   or set(scores) != {'next_target', 'next_direction',
                                      'next_distance', 'exact_bundle'}
                   or any(not isinstance(v, (int, float))
                          or not np.isfinite(v) or not 0 <= v <= 1
                          for v in scores.values())
                   for scores in audit.values())):
        raise ValueError('Common native-truth audit evidence is missing')
    integral = len(schedule) * args.lesson_scale if active else 0.0
    if (contract['schedule'] != schedule
            or finish['format_bank_sha256'] != args.format_bank_sha256
            or contract['format_bank_sha256'] != args.format_bank_sha256
            or finish['updates'] != len(schedule)
            or finish['replay_exposures'] != len(schedule) * args.lesson_batch
            or finish['effective_action_exposures'] != 0
            or finish['action_integral'] != 0
            or not np.isclose(finish['explanation_integral'], integral)):
        raise ValueError('Bank, schedule or dose mismatch')
    bank = read(args.format_bank)
    rng = np.random.default_rng(args.seed + 51_007)
    updates = read_jsonl(run / 'lesson_updates.jsonl')
    if [r['iteration'] for r in updates] != schedule:
        raise ValueError('Missing or duplicated replay update')
    for row in updates:
        draw = rng.choice(len(bank['train']), args.lesson_batch,
                          replace=len(bank['train']) < args.lesson_batch)
        ids = [bank['train'][i]['case_id'] for i in draw]
        gradient = row.get('shared_grad_norm')
        if row['ids'] != ids or (active and (
                gradient is None or not np.isfinite(gradient)
                or gradient <= 0)):
            raise ValueError('Replay stream or active gradient mismatch')
        if ('action_loss' in row or (not active and any(
                key in row for key in ('format_loss', 'shared_grad_norm')))
                or (active and (not np.isfinite(row.get('format_loss',
                                                         np.nan))
                                or row['format_loss'] < 0))):
            raise ValueError('Unexpected or missing auxiliary loss evidence')
    curve = read_jsonl(run / 'evaluations.jsonl')
    steps = eval_steps(args, derived)
    if len(steps) < 2 or [r['global_step'] for r in curve] != steps or any(
            r['teacher_on'] or r['episodes'] != args.eval_episodes
            or r['seed_base'] != args.seed + 50_000
            or r['iteration'] * derived['batch_size'] != r['global_step']
            for r in curve):
        raise ValueError('Teacher-off evaluation contract differs')
    y = np.array([r['success_rate'] for r in curve], dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Invalid success values')
    final = policy_hash(torch.load(run / 'agent.pt', map_location='cpu',
                                   weights_only=True))
    if final != finish['final_policy_sha256']:
        raise ValueError('Final policy hash differs')
    heads = torch.load(run / 'lesson_heads.pt', map_location='cpu',
                       weights_only=True)
    if not heads or any(not torch.isfinite(v).all() for v in heads.values()):
        raise ValueError('Missing or nonfinite saved explanation heads')
    auc = float(np.sum(np.diff(steps) * (y[:-1] + y[1:]) / 2)
                / (steps[-1] - steps[0]))
    return dict(auc=auc, final=float(y[-1]), initial_sha256=initial,
                final_sha256=final, wall_seconds=summary['wall_time_sec'],
                evaluations=len(curve), audit_plan_metrics=audit)


def run_cell(batch, index):
    """Execute one user-submitted cell once, preserving all failures."""
    batch = Path(batch).resolve()
    manifest = verify(batch)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute the archived source')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker environment differs from preparation')
    if not 0 <= index < len(manifest['cells']):
        raise ValueError('Invalid cell index')
    cell = manifest['cells'][index]
    args = resolved(batch, cell)
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    write(directory / 'dispatch.json', dict(
        cell=cell, args=asdict(args), commit=manifest['commit'],
        manifest_sha256=digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    pattern = f'*{args.experiment_id}__{args.seed}__*/run_summary.json'
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        if list((ROOT / 'results/runs').glob(pattern)):
            raise FileExistsError('An earlier attempt exists for this cell')
        process = subprocess.run([
            sys.executable, '-u', '-m', 'algos.ppo_plan_repair',
            *trainer_argv(asdict(args))], cwd=ROOT)
        outcome['returncode'] = process.returncode
        runs = list((ROOT / 'results/runs').glob(pattern))
        outcome['runs'] = [p.parent.relative_to(batch).as_posix()
                           for p in runs]
        if process.returncode == 0 and len(runs) == 1:
            outcome['metrics'] = validate_run(runs[0].parent, args)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit(batch):
    """One unthrottled array; an ambiguous attempt never triggers a retry."""
    batch = Path(batch).resolve()
    verify(batch)
    submitted = batch / 'SUBMITTED_JOB'
    if submitted.exists():
        print(f'{STUDY} already submitted: {submitted.read_text().strip()}')
        return
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write('One unthrottled array, 20 new free cells.\n')
    command = [
        'sbatch', '--parsable', '--job-name=plan_repair', '--array=0-19',
        f'--output={batch}/slurm/%x_%A_%a.out',
        str(batch / 'code/scripts/submit_plan_repair.sh'), str(batch),
    ]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f'{STUDY}: {job}')


def report(batch):
    """Report only validated complete pairs; do not turn missing runs to zero."""
    batch = Path(batch).resolve()
    manifest = verify(batch)
    rows, values = [], {}
    for cell in manifest['cells']:
        exit_path = batch / 'cells' / str(cell['index']) / 'exit.json'
        row = dict(index=cell['index'], seed=cell['seed'], arm=cell['arm'],
                   status='unattempted_or_partial')
        if exit_path.exists():
            saved = read(exit_path)
            row['status'] = saved['artifact_status']
            if saved['artifact_status'] == 'terminal_contract_validated':
                dispatch = read(exit_path.with_name('dispatch.json'))
                if (dispatch.get('cell') != cell
                        or dispatch.get('commit') != manifest['commit']
                        or dispatch.get('manifest_sha256') !=
                        digest(batch / 'manifest.json')
                        or portable_settings(dispatch['args']) !=
                        portable_settings(cell['args'])):
                    raise ValueError('Dispatch is not linked to this cell')
                if saved['returncode'] != 0 or len(saved['runs']) != 1:
                    raise ValueError('Exit record contradicts completion')
                metrics = validate_run(inside(batch, saved['runs'][0]),
                                       resolved(batch, cell))
                if metrics != saved['metrics']:
                    raise ValueError('Recomputed metrics differ from exit')
                row.update(metrics)
                values[cell['seed'], cell['arm']] = metrics
        rows.append(row)
    contrasts = []
    for other, threshold in (('raw', .03), ('ppo', .05),
                              ('permuted_corrected', .03)):
        pairs = [(values[s, 'corrected'], values[s, other]) for s in SEEDS
                 if (s, 'corrected') in values and (s, other) in values]
        if any(a['initial_sha256'] != b['initial_sha256'] for a, b in pairs):
            raise ValueError('Paired initialization differs')
        delta = np.array([a['auc'] - b['auc'] for a, b in pairs])
        mean = float(delta.mean()) if len(delta) else None
        half = (float(t.ppf(.975, len(delta) - 1)
                      * delta.std(ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1 else None)
        contrasts.append(dict(
            contrast=f'corrected - {other}', n=len(delta), mean=mean,
            ci95=[mean - half, mean + half] if half is not None else None,
            positive=int(sum(delta > 0)), threshold=threshold,
            nomination_component=(len(delta) == 5 and mean >= threshold
                                  and int(sum(delta > 0)) >= 4)))
    result = dict(study=STUDY, completed=len(values), planned=20, runs=rows,
                  contrasts=contrasts, new_api_calls=0,
                  correction_nomination=(len(values) == 20 and
                      contrasts[0]['nomination_component']),
                  useful_semantic_candidate=(len(values) == 20 and all(
                      c['nomination_component'] for c in contrasts)),
                  inference='development; independent review required')
    print(json.dumps(result, indent=2))
    return result


def main():
    """User-only launch; checks and reports never call a provider."""
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--launch', action='store_true')
    actions.add_argument('--report', action='store_true')
    actions.add_argument('--run-cell', type=int)
    cli.add_argument('--source', type=Path, default=(
        ROOT / 'results/explanation_lessons' / SOURCE_STUDY))
    cli.add_argument('--batch', type=Path, default=(
        ROOT / 'results/explanation_lessons' / STUDY))
    args = cli.parse_args()
    if args.report:
        report(args.batch)
    elif args.run_cell is not None:
        return run_cell(args.batch, args.run_cell)
    elif args.check:
        check_source(args.source)
        names = (*PATHS.values(), 'corpora/corrected_bank.json')
        menu(dict.fromkeys(names, '0' * 64))
        print('PASS: original inputs and 20 resolved trainer commands; '
              'five fresh paired seeds; 9,999,360 transitions; $0 API.')
    else:
        batch = prepare(ROOT, args.source)
        print(f'Prepared {batch}: 20 runs, $0 API; no old runs repeated.')
        if args.launch:
            submit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
