"""Frozen finite-budget rule references; only the user launches jobs."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import tyro

from algos.ppo_distill import Args
from scripts.run_explanation_grid import trainer_argv
from scripts.run_explanation_formats_20260925 import runtime_identity
from scripts.run_guidance_bonus_s3r3 import config as base_config


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'rule_reference_20260927_v1'
PROTOCOL = 'research/rule_reference_protocol_2026-09-27.md'
TASKS = {
    'doorkey_8x8': ('oracle', 12_800_000),
    'multiroom_n6': ('door_bfs', 12_900_000),
    'keycorridor_s3r3': ('bot', 13_000_000),
}
CORE = ('none', 'entropy480', 'probability480', 'random480')
REQUIRED = (
    'scripts/run_rule_reference_20260927.py',
    'scripts/rule_reference_validation.py',
    'scripts/report_rule_reference_20260927.py',
    'scripts/submit_rule_reference.sh', 'scripts/launch_rule_reference.sh',
    'algos/ppo_distill.py', 'teachers/controlled_advice.py', PROTOCOL,
    'tests/test_rule_reference_launch.py',
    'tests/test_rule_reference_validation.py',
    'tests/test_rule_reference_reporting.py',
)


def read(path):
    """Read research artifacts without credentials or live ledgers."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """Fingerprint persisted bytes, not an in-memory configuration."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Exclusive creation preserves earlier attempts and their evidence."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def inside(root, relative):
    """Resolve portable artifact names without directory escapes."""
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Expected a relative POSIX path')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def arms(task, bonus):
    """Add only lower doses already used in the paid reference studies."""
    extra = []
    if bonus == 'count' or task == 'keycorridor_s3r3':
        extra.append('random120')
    if task == 'multiroom_n6' and bonus == 'count':
        extra.extend(('random15', 'random30', 'random60'))
    return (*CORE, *extra)


def menu():
    """Twenty fresh independent seed blocks, paired within each task."""
    cells = []
    for replicate in range(20):
        for task, (teacher, start) in TASKS.items():
            seed = start + 100 * replicate
            for bonus in ('none', 'count'):
                for arm in arms(task, bonus):
                    guided = arm != 'none'
                    random = arm.startswith('random')
                    cap = int(arm[6:]) if random else 480 if guided else 0
                    selective = guided and not random
                    args = replace(
                        base_config(0), task=task, seed=seed, bonus=bonus,
                        experiment_id=f'{STUDY}_{task}_{bonus}_{arm}',
                        teacher=teacher, teacher_model='',
                        teacher_reasoning_effort='', guidance=guided,
                        teacher_stream=guided and teacher == 'bot',
                        advisor=('mistake' if arm == 'probability480'
                                 else 'importance' if arm == 'entropy480'
                                 else 'unlimited'),
                        importance_source='entropy' if selective else 'none',
                        importance_threshold=0.0, advice_rate=0.0,
                        mistake_threshold=.2 if arm == 'probability480'
                        else 0.0,
                        query_budget=cap, advice_budget=cap,
                        uniform_queries=random, query_interval=1,
                        query_windows=0, distill_coef_start=1.0,
                        distill_coef_min=.01, distill_normalization='labeled',
                        advisor_rng_isolation=True, advisor_sham=False,
                        advisor_no_teacher_peek=True,
                        record_initial_policy=True, audit_explanations=True,
                        offline_summary_only=False, explanation='none',
                        action_reference='', advice_replay=False,
                        eval_frame_milestones='2000000,5000000',
                        max_cost_dollars=0.0, budget_ledger='', track=False,
                    )
                    cells.append(dict(index=len(cells), task=task,
                                      replicate=replicate, seed=seed,
                                      bonus=bonus, arm=arm, args=asdict(args)))
    return cells


def check_commands(cells):
    """Check each distinct scientific setting through the real parser."""
    for cell in cells:
        if cell['replicate'] != 0:
            continue
        parsed = tyro.cli(Args, args=trainer_argv(cell['args']),
                          console_outputs=False)
        if asdict(parsed) != cell['args']:
            raise ValueError('Trainer CLI changed a resolved setting')
    if len(cells) != 620 or len({
            (c['task'], c['bonus'], c['arm'], c['seed']) for c in cells
    }) != len(cells):
        raise ValueError('Frozen menu is incomplete or duplicated')


def verify(batch):
    """Validate source and frozen identities before dispatch or reporting."""
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    if (digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or manifest['study'] != STUDY
            or manifest['cells'] != menu()
            or manifest.get('protocol') != PROTOCOL
            or manifest.get('api_calls') != 0
            or manifest.get('new_cost_usd') != 0
            or (batch / 'READY').read_text().strip() != manifest['commit']
            or not set(REQUIRED) <= set(manifest['source_hashes'])):
        raise ValueError('Frozen batch identity differs')
    for name, expected in manifest['source_hashes'].items():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    return manifest


def prepare(root):
    """Archive committed source once; old experiments are never repeated."""
    root = Path(root).resolve()
    batch = root / 'results/efficiency' / STUDY
    if batch.exists():
        verify(batch)
        return batch
    check_commands(menu())
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         commit],
        cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete reviewed packet first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for directory in ('cells', 'slurm'):
        (batch / directory).mkdir()
    manifest = dict(
        study=STUDY, protocol=PROTOCOL, commit=commit,
        runtime=runtime_identity(), api_calls=0, new_cost_usd=0.0,
        cells=menu(), source_hashes={
            p.relative_to(batch / 'code').as_posix(): digest(p)
            for p in (batch / 'code').rglob('*') if p.is_file()},
    )
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    return batch


def run_cell(batch, index):
    """Run one cell once; persist nonzero exits and artifact failures."""
    from scripts.rule_reference_validation import validate_run

    batch = Path(batch).resolve()
    manifest = verify(batch)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute archived source')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker packages differ from preparation')
    if type(index) is not int or index not in range(len(manifest['cells'])):
        raise ValueError('Invalid cell index')
    cell = manifest['cells'][index]
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    write(directory / 'dispatch.json', dict(
        cell=cell, commit=manifest['commit'],
        manifest_sha256=digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    args = Args(**cell['args'])
    pattern = f'*{args.experiment_id}__{args.seed}__*'
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        if list((ROOT / 'results/runs').glob(pattern)):
            raise FileExistsError('An earlier attempt exists for this cell')
        process = subprocess.run([
            sys.executable, '-u', '-m', 'algos.ppo_distill',
            *trainer_argv(cell['args'])], cwd=ROOT)
        outcome['returncode'] = process.returncode
        runs = [p for p in (ROOT / 'results/runs').glob(pattern) if p.is_dir()]
        outcome['runs'] = [p.relative_to(batch).as_posix() for p in runs]
        if process.returncode == 0 and len(runs) == 1:
            outcome['metrics'] = validate_run(runs[0], cell['args'])
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'exit.json', outcome)
    if outcome['artifact_status'] != 'terminal_contract_validated':
        print(json.dumps(outcome), flush=True)
        return 1
    return 0


def submission_command(batch):
    """All cells may run concurrently, subject only to cluster policy."""
    return ['sbatch', '--parsable', '--job-name=rule_reference',
            '--array=0-619', f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch / 'code/scripts/submit_rule_reference.sh'), str(batch)]


def submit(batch):
    """User-only submission; ambiguous receipts never trigger a retry."""
    batch = Path(batch).resolve()
    verify(batch)
    submitted = batch / 'SUBMITTED_JOB'
    if submitted.exists():
        print(f'{STUDY} already submitted: {submitted.read_text().strip()}')
        return
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write('One array; 620 cells; $0 API.\n')
    command = submission_command(batch)
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized receipt; inspect submission attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f'{STUDY}: {job}')


def report(batch, output):
    """Revalidate native artifacts before seed-level reporting and plots."""
    from scripts.rule_reference_validation import validate_run
    from scripts.report_rule_reference_20260927 import build_report

    batch = Path(batch).resolve()
    manifest = verify(batch)
    rows = []
    for cell in manifest['cells']:
        row = {key: cell[key] for key in
               ('index', 'task', 'bonus', 'arm', 'seed')}
        directory = batch / 'cells' / str(cell['index'])
        row['status'] = ('started_without_exit' if directory.exists()
                         else 'unattempted_or_untransferred')
        if (directory / 'exit.json').exists():
            saved = read(directory / 'exit.json')
            row['status'] = saved.get('artifact_status', 'failed')
            if row['status'] == 'terminal_contract_validated':
                dispatch = read(directory / 'dispatch.json')
                if (dispatch.get('cell') != cell
                        or dispatch.get('commit') != manifest['commit']
                        or dispatch.get('manifest_sha256') !=
                        digest(batch / 'manifest.json')
                        or saved.get('returncode') != 0
                        or len(saved.get('runs', [])) != 1):
                    raise ValueError('Dispatch/completion identity differs')
                metrics = validate_run(inside(batch, saved['runs'][0]),
                                       cell['args'])
                if metrics != saved['metrics']:
                    raise ValueError('Recomputed terminal metrics differ')
                row.update(metrics)
                row['status'] = 'complete'
        rows.append(row)
    result = build_report(rows, output_dir=output)
    print(f'Saved rule reference report and separate task figures: {output}')
    return result


def main():
    """Preview/check are read-only; launch is explicitly user controlled."""
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    for mode in ('check', 'prepare', 'launch', 'report'):
        modes.add_argument('--' + mode, action='store_true')
    modes.add_argument('--run-cell', type=int)
    cli.add_argument('--batch', type=Path,
                     default=ROOT / 'results/efficiency' / STUDY)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args()
    if args.run_cell is not None:
        return run_cell(args.batch, args.run_cell)
    if args.report:
        report(args.batch, args.output or args.batch / 'analysis')
        return 0
    print('620 new runs: 480 core + 140 low-dose; 20 paired seeds/task; '
          '9,999,360 transitions/run; $0 API. No array throttle.')
    print('KC tracks every active state with its free planner; selected '
          'consultations are capped; planner calls are reported separately.')
    if args.check:
        check_commands(menu())
        print('PASS: 31 distinct trainer configurations; all frozen cells.')
    elif args.prepare or args.launch:
        batch = prepare(ROOT)
        print(f'Prepared: {batch}')
        if args.launch:
            submit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
