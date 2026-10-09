"""Frozen optional studies; only the user invokes launch on Vulcan.

Reserve every selected paid group before any
submission. Existing core controls and all attempts remain untouched.
"""

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from scripts import paper_optional_crafter_eval_20261005 as crafter
from scripts import paper_optional_online_20261005 as online
from scripts import paper_optional_regeneration_20261005 as regen
from scripts import run_advising_strength_grid as funding
from scripts import run_fix_wave_20260929 as fw
from scripts import run_paper_bulk_20261005 as bulk
from teachers.budget import BudgetError, CostLedger

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_optionals_20261005_v1'
PROTOCOL = 'research/paper_optionals_protocol_2026-10-05.md'
GROUPS = ('crafter_eval', 'regeneration', 'online')
STAGES = dict(crafter_eval=('evaluate',),
              regeneration=('generate', 'train'), online=('train',))
AMOUNTS = dict(crafter_eval=0., regeneration=7.40, online=15.10)
REQUIRED = (
    PROTOCOL, online.PRICES, crafter.PROTOCOL, *regen.REQUIRED,
    'scripts/run_paper_optionals_20261005.py',
    'scripts/paper_optional_online_20261005.py',
    'scripts/paper_optional_crafter_eval_20261005.py',
    'scripts/launch_paper_optionals.sh',
    'scripts/submit_paper_optionals.sh',
)


def batch_dir(root, group):
    return Path(root).resolve() / 'results/paper_optionals' / STUDY / group


def crafter_source_path(root, explicit=None):
    """
    Find the known historical batch without selecting among outcomes.
    """
    if explicit:
        return Path(explicit).resolve()
    root = Path(root).resolve()
    suffix = 'results/crafter_vulcan/crafter_vulcan_20261002'
    candidates = (root / suffix,
                  root.parent / '.vlm-crafter-1a1335a4b6ec' / suffix)
    for path in candidates:
        if (path / 'manifest.json').is_file():
            return path.resolve()
    raise FileNotFoundError(
        'Historical Crafter batch missing. Set VLM_CRAFTER_SOURCE or '
        '--crafter-source to the complete original batch with final_model.pt '
        f'for all50 policies. Checked: {list(map(str, candidates))}')


def check(root=ROOT, groups=GROUPS):
    """
    Check the finite menu without credentials, calls or result selection.
    """
    result = dict(study=STUDY, groups={})
    for group in groups:
        if group == 'crafter_eval':
            crafter.check(root)
            row = dict(training_runs=0, evaluation_policies=50, episodes=2500,
                       api_calls_max=0)
        elif group == 'regeneration':
            regen.consultation_requests(regen.corpus(root))
            if len(regen.training_plan()) != 70:
                raise ValueError('Wrong regeneration training count')
            row = dict(training_runs=70, generation_tasks=5, api_calls_max=360)
        else:
            records = online.cells(root)
            if (len(records) != 40 or round(sum(r['allocation_usd']
                                               for r in records), 2) != 15.10):
                raise ValueError('Wrong online count/allocation')
            row = dict(training_runs=40, api_calls_max=sum(
                r['args']['query_budget'] for r in records))
        result['groups'][group] = dict(row, reservation_usd=AMOUNTS[group])
    result['reservation_usd'] = round(sum(AMOUNTS[g] for g in groups), 2)
    print(json.dumps(result, indent=2), flush=True)
    return result


def source_controls(root):
    """
    Freeze only configurations/provenance; do not read learning outcomes.
    """
    result = {}
    for suite in online.SOURCES:
        batch = fw.batch_dir(Path(root), suite).resolve()
        fw.verify(batch, rebuild=False)
        bulk.compatible_existing(batch, suite, root)
        result[suite] = dict(batch=str(batch), manifest_sha256=fw.digest(
            batch / 'manifest.json'))
    return result


def parent_hold(manifest):
    """
    Workers may spend only their funded allocation under an intact hold.
    """
    if not manifest['allocation_usd']:
        return
    status = funding.check_pool(manifest['ledger'], 0.)
    holds = [r for r in status['open_reservations']
             if r['run_id'] == manifest['parent_run_id']]
    if len(holds) != 1 or holds[0]['reserved'] != manifest['allocation_usd']:
        raise BudgetError('Optional group parent reservation is missing')


def pristine_child(path, amount):
    state = CostLedger(str(path)).status()
    if (state['allowance_usd'] != amount or state['settled_usd'] != 0
            or state['reserved_usd'] != 0 or state['open_reservations']
            or state['overspent_runs']):
        raise BudgetError('Child ledger is not its pristine funded allocation')


def verify(batch, rebuild=True):
    """
    Bind immutable inputs and recipes to the committed source archive.
    """
    batch = Path(batch).resolve()
    m = fw.read(batch / 'manifest.json')
    if (m['study'] != STUDY or m['group'] not in GROUPS
            or fw.digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or (batch / 'READY').read_text().strip() != m['commit']
            or not set(REQUIRED) <= set(m['source_hashes'])
            or m['allocation_usd'] != AMOUNTS[m['group']]):
        raise ValueError('Optional manifest identity differs')
    for name, expected in m['source_hashes'].items():
        if fw.digest(fw.inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    for name, expected in m['input_hashes'].items():
        if fw.digest(fw.inside(batch, name)) != expected:
            raise ValueError(f'Frozen optional input changed: {name}')
    group = m['group']
    if group == 'online':
        if rebuild and m['cells'] != online.cells(
                batch / 'code', batch / 'cells/train'):
            raise ValueError('Online recipe changed')
        for control in m['controls'].values():
            if fw.digest(Path(control['batch']) / 'manifest.json') != control[
                    'manifest_sha256']:
                raise ValueError('Paired source-control manifest changed')
    elif group == 'regeneration':
        regen.verify_generation_inputs(batch / 'generation')
        if m['cells'] != regen.training_plan():
            raise ValueError('Crossed training plan changed')
    else:
        crafter.validate_manifest(batch / 'evaluation', batch / 'code')
    return m


def prepare_group(root, group, ledger, credential, crafter_source):
    """
    Freeze one group and fund children without touching existing holds.
    """
    root = Path(root).resolve()
    batch = batch_dir(root, group)
    if batch.exists():
        m = verify(batch)
        parent_hold(m)
        return batch
    controls = source_controls(root) if group == 'online' else {}
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                     cwd=root, text=True).strip()
    archive = subprocess.check_output([
        'git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', commit],
        cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete optional package first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    (batch / 'slurm').mkdir()
    inputs = {}
    if group == 'crafter_eval':
        crafter.prepare(crafter_source, batch / 'evaluation', batch / 'code')
        inputs['evaluation/manifest.json'] = fw.digest(
            batch / 'evaluation/manifest.json')
        records = crafter.cells()
    elif group == 'regeneration':
        regen.prepare_generation(batch / 'generation', batch / 'code')
        inputs['generation/generation_manifest.json'] = fw.digest(
            batch / 'generation/generation_manifest.json')
        records = regen.training_plan()
    else:
        records = online.cells(batch / 'code', batch / 'cells/train')
    manifest = dict(
        author='mahdiehmn', study=STUDY, group=group, commit=commit,
        protocol=PROTOCOL, runtime=fw.runtime_identity(), cells=records,
        controls=controls, input_hashes=inputs, allocation_usd=AMOUNTS[group],
        ledger=str(Path(ledger).resolve()),
        credential=str(Path(credential).resolve()),
        parent_run_id=f'{STUDY}:{group}',
        source_hashes={p.relative_to(batch / 'code').as_posix(): fw.digest(p)
                       for p in (batch / 'code').rglob('*') if p.is_file()})
    for stage in STAGES[group]:
        count = 5 if stage == 'generate' else len(records)
        for index in range(count):
            directory = batch / 'cells' / stage / str(index)
            directory.mkdir(parents=True)
            if stage == 'generate' or group == 'online':
                amount = 1.48 if stage == 'generate' else records[index][
                    'allocation_usd']
                CostLedger.initialize(
                    str(directory / 'budget.json'), amount,
                    note=f"Child of {manifest['parent_run_id']}")
    fw.write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(fw.digest(batch / 'manifest.json'))
    if AMOUNTS[group]:
        funding.check_pool(ledger, AMOUNTS[group])
        CostLedger(str(ledger)).reserve(
            manifest['parent_run_id'], AMOUNTS[group],
            note='Full optional group; keep until audit')
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    parent_hold(manifest)
    return batch


def prepare_groups(root, groups, ledger, credential, crafter_source):
    """
    Complete all selected admission and funding before the first sbatch.
    """
    check(root, groups)
    required = round(sum(AMOUNTS[g] for g in groups
                         if not batch_dir(root, g).exists()), 2)
    if any(AMOUNTS[g] for g in groups):
        # Validate the credential without printing its contents.
        funding.backend_environment('openai', credential)
        funding.check_pool(ledger, required)
    if 'online' in groups:
        source_controls(root)
    if ('crafter_eval' in groups
            and not batch_dir(root, 'crafter_eval').exists()):
        # Missing old policies must fail before archives or holds exist.
        crafter.preflight(crafter_source, root)
    batches = {}
    for group in groups:
        batch = prepare_group(root, group, ledger, credential, crafter_source)
        batches[group] = batch
        print(f'PREPARED {group}: {batch}', flush=True)
    return batches


def resolved_records(batch, m, stage):
    if stage == 'generate':
        return [dict(index=i, bank=regen.BANKS[i]) for i in range(5)]
    if m['group'] == 'regeneration':
        records = regen.cells(batch / 'code', batch / 'generation')
        if [{k: r[k] for k in plan} for r, plan in zip(records, m['cells'])
                ] != m['cells']:
            raise ValueError('Resolved banks changed the predeclared plan')
        return records
    return m['cells']


def worker(batch, stage, index):
    """
    One immutable attempt per cell, with durable failed/partial evidence.
    """
    batch = Path(batch).resolve()
    m = verify(batch)
    if (ROOT.resolve() != (batch / 'code').resolve()
            or fw.runtime_identity() != m['runtime']):
        raise ValueError('Worker source/runtime differs from preparation')
    if stage not in STAGES[m['group']]:
        raise ValueError('Invalid worker stage')
    records = resolved_records(batch, m, stage)
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError('Index outside frozen plan')
    record = records[index]
    directory = batch / 'cells' / stage / str(index)
    parent_hold(m)
    if stage == 'generate' or m['group'] == 'online':
        pristine_child(directory / 'budget.json', 1.48 if stage == 'generate'
                       else record['allocation_usd'])
    with (directory / 'CLAIMED').open('x') as handle:
        handle.write('One attempt; no automatic scientific or paid retry.\n')
    fw.write(directory / 'dispatch.json', dict(
        cell=record, stage=stage, commit=m['commit'],
        manifest_sha256=fw.digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        if stage == 'generate':
            outcome['generation'] = regen.generation_worker(
                batch / 'generation', index, m['credential'],
                directory / 'budget.json')
            outcome['returncode'] = 0
        elif stage == 'evaluate':
            outcome['returncode'] = crafter.run_cell(
                index, batch / 'evaluation', ROOT)
        else:
            args = record['args']
            pattern = (f"*{args['experiment_id']}__{args['seed']}__*/"
                       'run_summary.json')
            # Detect partial directories as well as completed summaries.
            directories = pattern.split('/')[0]
            if list((ROOT / 'results/runs').glob(directories)):
                raise FileExistsError('A previous training attempt exists')
            env = (funding.backend_environment('openai', m['credential'])
                   if m['group'] == 'online' else dict(os.environ))
            process = subprocess.run([
                sys.executable, '-u', '-m', record['trainer'],
                *fw.trainer_argv(args)], cwd=ROOT, env=env)
            outcome['returncode'] = process.returncode
            runs = list((ROOT / 'results/runs').glob(pattern))
            outcome['runs'] = [r.parent.relative_to(batch).as_posix()
                               for r in runs]
            if process.returncode != 0 or len(runs) != 1:
                raise ValueError('Training did not produce one completed run')
            outcome['metrics'] = validate_training(runs[0].parent, record,
                                                    m['group'])
        if outcome['returncode'] != 0:
            raise ValueError('Worker did not finish its contract')
        outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        # API exceptions can contain request data; never copy their text here.
        outcome.update(error_type=type(error).__name__)
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    fw.write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def validate_training(run, record, group):
    metrics = (online.validate(run, record) if group == 'online'
               else online.validate_learning(run, record))
    if group == 'regeneration' and metrics['api_dollars'] != 0:
        raise ValueError('Frozen-bank training incurred unexpected API cost')
    return metrics


def submission_state(batch, stage):
    directory = Path(batch) / 'submissions' / stage
    receipt = directory / 'submission.json'
    if receipt.exists():
        job = fw.read(receipt).get('job_id')
        if not isinstance(job, str) or not re.fullmatch('[1-9][0-9]*', job):
            raise ValueError('Invalid scheduler receipt')
        return job
    if (directory / 'ATTEMPTED').exists():
        raise ValueError(f'Ambiguous submission; inspect {directory}')
    return None


def submit(batch, stage, dependency=None):
    batch = Path(batch).resolve()
    m = verify(batch)
    parent_hold(m)
    previous = submission_state(batch, stage)
    if previous:
        print(f"ALREADY SUBMITTED {m['group']}/{stage}: {previous}",
              flush=True)
        return previous
    count = 5 if stage == 'generate' else len(m['cells'])
    if stage not in STAGES[m['group']]:
        raise ValueError('Invalid submission stage')
    if m['group'] == 'regeneration' and stage == 'train':
        if dependency != submission_state(batch, 'generate') or not dependency:
            raise ValueError('Regeneration training needs its generation job')
    time = (crafter.TIME if stage == 'evaluate' else
            '02:00:00' if stage == 'generate' else
            online.TIME if m['group'] == 'online' else regen.TIME)
    command = ['sbatch', '--parsable', f"--job-name=po_{m['group']}_{stage}",
               f'--array=0-{count - 1}', f'--time={time}',
               f'--output={batch}/slurm/%x_%A_%a.out']
    if stage == 'evaluate':
        command += ['--cpus-per-task=1', '--mem=8G']
    if dependency:
        command += [f'--dependency=afterok:{dependency}']
    command += [str(batch / 'code/scripts/submit_paper_optionals.sh'),
                str(batch), stage]
    directory = batch / 'submissions' / stage
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'ATTEMPTED').open('x') as handle:
        handle.write('User-owned submission; inspect ambiguous receipts.\n')
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; preserve attempt')
    fw.write(directory / 'submission.json', dict(command=command, job_id=job))
    print(f"SUBMITTED {m['group']}/{stage}: {job}", flush=True)
    return job


def launch(root, groups, ledger, credential, crafter_source):
    batches = prepare_groups(root, groups, ledger, credential, crafter_source)
    for group, batch in batches.items():
        for stage in STAGES[group]:
            submission_state(batch, stage)
    for group, batch in batches.items():
        dependency = None
        for stage in STAGES[group]:
            # Resume with each group's own frozen submission recipe.
            source = (
                'from scripts.run_paper_optionals_20261005 import submit; '
                'import sys; submit(sys.argv[1], sys.argv[2], '
                'sys.argv[3] or None)')
            subprocess.run([sys.executable, '-c', source, str(batch), stage,
                            dependency or ''], cwd=batch / 'code', check=True)
            dependency = submission_state(batch, stage)


def rows_and_status(batch, m, stage):
    statuses = dict(validated=[], worker_completed_pending_artifact_check=[],
                    failed=[], dispatched_no_exit=[], unattempted=[])
    rows = []
    # Bank generation may be pending; inventory still accounts for all70 slots.
    try:
        records = resolved_records(batch, m, stage)
    except (FileNotFoundError, ValueError):
        if m['group'] != 'regeneration' or stage != 'train':
            raise
        records = m['cells']
    for record in records:
        directory = batch / 'cells' / stage / str(record['index'])
        state = 'unattempted'
        if (directory / 'CLAIMED').exists():
            state = 'dispatched_no_exit'
        if (directory / 'exit.json').exists():
            state = 'failed'
            receipt = fw.read(directory / 'exit.json')
            if receipt['artifact_status'] == 'terminal_contract_validated':
                dispatch = fw.read(directory / 'dispatch.json')
                if (dispatch['cell'] != record or dispatch['stage'] != stage
                        or dispatch['commit'] != m['commit']
                        or dispatch['manifest_sha256'] != fw.digest(
                            batch / 'manifest.json')
                        or receipt['returncode'] != 0):
                    raise ValueError('Successful dispatch provenance differs')
                if stage == 'train':
                    if len(receipt['runs']) != 1:
                        raise ValueError('Successful cell has wrong run count')
                    metrics = validate_training(
                        fw.inside(batch, receipt['runs'][0]),
                        record, m['group'])
                    if receipt['metrics'] != metrics:
                        raise ValueError('Saved metrics differ from raw run')
                    rows.append(dict(record=record, metrics=metrics))
                state = ('validated' if stage == 'train' else
                         'worker_completed_pending_artifact_check')
        statuses[state].append(record['index'])
    return statuses, rows


def control_rows(m):
    """
    Revalidate paired completed source runs; missing outcomes stay missing.
    """
    rows = []
    for reference in m['controls'].values():
        batch = Path(reference['batch'])
        old = fw.verify(batch, rebuild=False)
        for record in old['cells']:
            if record['arm'] not in ('none', 'rules_weak'):
                continue
            directory = batch / 'cells' / str(record['index'])
            if not (directory / 'exit.json').exists():
                continue
            end = fw.read(directory / 'exit.json')
            if end.get('artifact_status') != 'terminal_contract_validated':
                continue
            start = fw.read(directory / 'dispatch.json')
            if (start['cell'] != record or start['commit'] != old['commit']
                    or start['manifest_sha256'] != reference['manifest_sha256']
                    or end['returncode'] != 0 or len(end['runs']) != 1):
                raise ValueError('Source control dispatch differs')
            metrics = online.validate_learning(
                fw.inside(batch, end['runs'][0]), record)
            rows.append(dict(record=record, metrics=metrics))
    return rows


def report(root=ROOT):
    root = Path(root).resolve()
    result = dict(study=STUDY, groups={}, independent_result_review='pending')
    for group in GROUPS:
        batch = batch_dir(root, group)
        if not (batch / 'READY').exists():
            result['groups'][group] = dict(status='not_ready_or_not_prepared')
            continue
        m = verify(batch)
        entry = dict(stages={}, allocation_usd=m['allocation_usd'])
        rows = []
        for stage in STAGES[group]:
            states, stage_rows = rows_and_status(batch, m, stage)
            entry['stages'][stage] = states
            rows += stage_rows
        if group == 'crafter_eval':
            done = entry['stages']['evaluate'][
                'worker_completed_pending_artifact_check']
            entry['results'] = (
                crafter.report(batch / 'evaluation', batch / 'code')
                if len(done) == 50 else dict(status='PROVISIONAL'))
            if entry['results'].get('complete') == 50:
                entry['stages']['evaluate']['validated'] = list(range(50))
                entry['stages']['evaluate'][
                    'worker_completed_pending_artifact_check'] = []
        elif group == 'regeneration':
            if len(entry['stages']['generate'][
                    'worker_completed_pending_artifact_check']) == 5:
                regen.freeze_generation(batch / 'generation')
                entry['stages']['generate']['validated'] = list(range(5))
                entry['stages']['generate'][
                    'worker_completed_pending_artifact_check'] = []
            entry['results'] = regen.report([
                dict(arm=r['record']['arm'], seed=r['record']['seed'],
                     auc=r['metrics']['auc'],
                     initial_sha256=r['metrics']['initial_sha256'])
                for r in rows])
        else:
            controls = control_rows(m)
            entry['online_cost_and_dose'] = rows
            admitted = len(rows) == 40 and all(
                r['metrics']['dose_admissible'] for r in rows)
            entry['primary'] = bulk.compare_family(
                controls + (rows if admitted else []),
                online.primary_contrasts())
            entry['secondary_online_minus_none'] = bulk.compare_family(
                controls + (rows if admitted else []), [
                    (task, bonus, 'online_main', 'none')
                    for task in ('doorkey_8x8', 'multiroom_n6')
                    for bonus in ('none', 'count')])
            entry['dose_admissible'] = admitted
        result['groups'][group] = entry
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/paper_optionals' / STUDY / 'reports' / stamp
    output.mkdir(parents=True)
    fw.write(output / 'report.json', result)
    print(f'Report: {output / "report.json"}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'launch',
                                           'run-cell', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--groups', nargs='+', choices=GROUPS)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--crafter-source', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--stage', choices=('generate', 'train', 'evaluate'))
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    groups = args.groups or GROUPS
    if len(groups) != len(set(groups)):
        parser.error('Duplicate groups are not permitted')
    if args.action == 'run-cell':
        return worker(args.batch, args.stage, args.index)
    if args.action == 'report':
        report(args.root)
    elif args.action == 'check':
        check(args.root, groups)
    else:
        (launch if args.action == 'launch' else prepare_groups)(
            args.root, groups,
            args.ledger or args.root / 'results/budget_ledger.json',
            args.credential_file or args.root / '.env',
            crafter_source_path(args.root, args.crafter_source)
            if 'crafter_eval' in groups else Path('.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
