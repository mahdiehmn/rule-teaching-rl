"""Prepare and submit the complete, outcome-independent paper batch.

Submission is a user-only operation. Existing
core and timing archives retain their own workers and result locations.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import io
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from scripts import paper_bulk_crafter_20261005 as crafter
from scripts import paper_bulk_sensitivity_20261005 as sensitivity
from scripts import paper_bulk_view_20261005 as view
from scripts import run_fix_wave_20260929 as fw
from scripts import run_rule_timing_20261005 as timing

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_bulk_20261005_v1'
PROTOCOL = 'research/paper_bulk_protocol_2026-10-05.md'
CORE = (*fw.FRESH_CORE, *fw.FRESH_OPTIONAL)
EXTRAS = ('view', 'crafter', *sensitivity.SPECS)
GROUPS = (*CORE, 'timing', *EXTRAS)
COUNTS = dict(zip(GROUPS, (40, 40, 60, 20, 20, 20, 20,
                          40, 120, 60, 100, 100)))
TIMES = dict(view=view.TIME, crafter=crafter.TIME,
             **{name: sensitivity.TIME for name in sensitivity.SPECS})
CRAFTER_VERSIONS = ('1.8.3', '1.8.3+computecanada')
REQUIRED = (
    PROTOCOL, 'scripts/run_paper_bulk_20261005.py',
    'scripts/launch_paper_bulk.sh', 'scripts/submit_paper_bulk.sh',
    'scripts/paper_bulk_crafter_20261005.py', crafter.PROTOCOL,
    'scripts/audit_crafter_learning_20261001.py',
    'algos/ppo_crafter.py', crafter.BANK, crafter.NOPROG,
    *timing.REQUIRED, *view.REQUIRED, *sensitivity.REQUIRED,
)


def group_cells(group, root=ROOT):
    """
    Resolve a single declared group with its existing configuration code.
    """

    if group in CORE:
        return fw.suite_cells(group, root)
    if group == 'timing':
        return timing.cells(root)
    if group == 'view':
        return view.cells(root)
    if group == 'crafter':
        return crafter.cells(root)
    return sensitivity.cells(root, suite=group)


def check(root=ROOT, groups=GROUPS):
    """
    Admit all selected cells before any submission, with explicit costs.
    """

    identities = set()
    summary = {}
    if 'crafter' in groups:
        crafter.check(root)
        extra_runtime('crafter')
    for group in groups:
        records = group_cells(group, root)
        if len(records) != COUNTS[group]:
            raise ValueError(f'Wrong cell count: {group}')
        transitions = 0
        for index, record in enumerate(records):
            if record['index'] != index:
                raise ValueError('Cell indices must be contiguous')
            args = record['args']
            name = args.get('experiment_id', record.get('experiment_id'))
            identity = (record['trainer'], name, record['seed'])
            if identity in identities:
                raise ValueError('A run was duplicated across groups')
            identities.add(identity)
            if (args.get('guidance', False)
                    and args.get('teacher') != 'rule_bank'):
                raise ValueError('Only frozen-bank/free controls admitted')
            transitions += args.get('total_timesteps', args.get('total_steps'))
        summary[group] = dict(cells=len(records),
                              nominal_transitions=transitions)
    result = dict(study=STUDY, groups=summary,
                  cells=sum(g['cells'] for g in summary.values()),
                  nominal_transitions=sum(g['nominal_transitions']
                                          for g in summary.values()),
                  paid_model_calls=0)
    print(json.dumps(result, indent=2))
    return result


def extra_runtime(group):
    """
    Freeze the Crafter dependency as well as the common trainer runtime.
    """

    identity = fw.runtime_identity()
    if group == 'crafter':
        version = importlib.metadata.version('crafter')
        # Vulcan adds a local build tag to the upstream release.
        # Keep that tag: preparation and workers must match exactly.
        if version not in CRAFTER_VERSIONS:
            raise ValueError(
                f'Crafter requires one of {CRAFTER_VERSIONS}; '
                f'found {version!r}')
        identity['packages']['crafter'] = version
    return identity


def extra_batch(root, group):
    return Path(root).resolve() / 'results/paper_bulk' / STUDY / group


def verify_extra(batch, rebuild=True):
    """
    Check immutable identity and every archived source file.
    """

    batch = Path(batch).resolve()
    manifest = fw.read(batch / 'manifest.json')
    if (manifest['study'] != STUDY or manifest['group'] not in EXTRAS
            or fw.digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or (batch / 'READY').read_text().strip() != manifest['commit']
            or not set(REQUIRED) <= set(manifest['source_hashes'])):
        raise ValueError('Frozen bulk identity differs')
    for name, expected in manifest['source_hashes'].items():
        if fw.digest(fw.inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    if rebuild and manifest['cells'] != group_cells(
            manifest['group'], batch / 'code'):
        raise ValueError('Frozen bulk cell configuration differs')
    return manifest


def prepare_extra(root, group):
    """
    Snapshot committed source for one additional, independently queued group.
    """

    root = Path(root).resolve()
    batch = extra_batch(root, group)
    if batch.exists():
        verify_extra(batch)
        return batch
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output([
        'git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
        commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete bulk packet first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    manifest = dict(
        author='mahdiehmn', study=STUDY, group=group, commit=commit,
        protocol=PROTOCOL, api_calls=0, runtime=extra_runtime(group),
        cells=group_cells(group, batch / 'code'),
        source_hashes={p.relative_to(batch / 'code').as_posix():
                       fw.digest(p) for p in (batch / 'code').rglob('*')
                       if p.is_file()})
    fw.write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(fw.digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    verify_extra(batch)
    return batch


def compatible_existing(batch, group, root):
    """
    Accept an already prepared equivalent core without silently rerunning it.
    """

    saved = fw.read(Path(batch) / 'manifest.json')['cells']
    expected = group_cells(group, root)
    if len(saved) != len(expected):
        raise ValueError('Existing batch has a different planned count')
    for old, new in zip(saved, expected):
        a, b = dict(old), dict(new)
        # The new read-only option defaults off in older core archives.
        # Compare full resolved settings, not only names or seeds.
        missing = set(b['args']) - set(a['args'])
        if missing - {'rule_timing_diagnostics'}:
            raise ValueError('Existing core is missing required arguments')
        a['args'] = asdict(fw.ppo.Args(**a['args']))
        b['args'] = asdict(fw.ppo.Args(**b['args']))
        if a != b:
            raise ValueError(f'Existing {group} configuration differs')


def prepare_groups(root, groups):
    """
    Finish all admission/preparation checks before the first sbatch call.
    """

    root = Path(root).resolve()
    check(root, groups)
    batches = {}
    for group in groups:
        if group in CORE:
            batch = fw.prepare(root, group)
            compatible_existing(batch, group, root)
        elif group == 'timing':
            batch = timing.prepare(root)
        else:
            batch = prepare_extra(root, group)
        batches[group] = str(batch)
        print(f'PREPARED {group}: {batch}', flush=True)
    return batches


def submission_state(batch, group):
    """
    Reject ambiguous prior attempts before submitting any other array.
    """

    batch = Path(batch)
    receipt = batch / 'submission.json'
    marker = batch / 'SUBMITTED_JOB'
    if receipt.exists():
        job = fw.read(receipt).get('job_id')
        if not isinstance(job, str) or not re.fullmatch('[1-9][0-9]*', job):
            raise ValueError(f'Invalid submission receipt: {batch}')
        if group in CORE and (
                not marker.exists() or marker.read_text().strip() != job):
            raise ValueError(f'Core submission markers disagree: {batch}')
        return job
    if (batch / 'SUBMISSION_ATTEMPTED').exists() or marker.exists():
        raise ValueError(f'Ambiguous prior submission; inspect: {batch}')
    return None


def validate_extra(run, record, group):
    if group == 'crafter':
        return crafter.validate(run, record)
    # All new MiniGrid guided arms carry the timing diagnostics. The
    # checker includes dense/regular teacher-off evaluation contracts.
    return timing.validate(run, record)


def worker(batch, index):
    """
    Execute one immutable cell; preserve failures without automatic retry.
    """

    batch = Path(batch).resolve()
    manifest = verify_extra(batch, rebuild=False)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute archived bulk source')
    if manifest['runtime'] != extra_runtime(manifest['group']):
        raise ValueError('Worker runtime differs from preparation')
    if index is None or not 0 <= index < len(manifest['cells']):
        raise ValueError('Worker index is outside the frozen manifest')
    record = manifest['cells'][index]
    group = manifest['group']
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    fw.write(directory / 'dispatch.json', dict(
        cell=record, commit=manifest['commit'],
        manifest_sha256=fw.digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    args = dict(record['args'])
    try:
        if group == 'crafter':
            run = (directory / 'run').resolve()
            args['out'] = str(run)
        else:
            pattern = (f"*{args['experiment_id']}__{args['seed']}__*/"
                       'run_summary.json')
            if list((ROOT / 'results/runs').glob(pattern)):
                raise FileExistsError('A previous cell attempt exists')
        process = subprocess.run([
            sys.executable, '-u', '-m', record['trainer'],
            *fw.trainer_argv(args)], cwd=ROOT)
        outcome['returncode'] = process.returncode
        if group != 'crafter':
            matches = list((ROOT / 'results/runs').glob(pattern))
            if len(matches) != 1:
                raise ValueError('Expected exactly one produced run')
            run = matches[0].parent
        outcome['runs'] = [run.relative_to(batch).as_posix()]
        if process.returncode == 0:
            outcome['metrics'] = validate_extra(run, record, group)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    fw.write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit_extra(batch):
    """
    Submit one group with a durable receipt and no concurrency throttle.
    """

    batch = Path(batch).resolve()
    manifest = verify_extra(batch)
    receipt = batch / 'submission.json'
    job = submission_state(batch, manifest['group'])
    if job:
        print(f"ALREADY SUBMITTED {manifest['group']}: "
              f'{job}')
        return
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as handle:
        handle.write('One user-owned array; no automatic retries.\n')
    command = [
        'sbatch', '--parsable', f"--job-name=pb_{manifest['group']}",
        f"--array=0-{len(manifest['cells']) - 1}",
        f"--time={TIMES[manifest['group']]}",
        f'--output={batch}/slurm/%x_%A_%a.out',
        str(batch / 'code/scripts/submit_paper_bulk.sh'), str(batch)]
    if manifest['group'] == 'crafter':
        command[2:2] = ['--cpus-per-task=1', '--mem=8G']
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    fw.write(receipt, dict(command=command, job_id=job))
    print(f"SUBMITTED {manifest['group']}: {job}", flush=True)


def launch_groups(root, groups):
    """
    Queue every selected group without scientific-result dependencies.
    """

    batches = prepare_groups(root, groups)
    submitted = {group: submission_state(path, group)
                 for group, path in batches.items()}
    for group, path in batches.items():
        batch = Path(path)
        if submitted[group]:
            print(f'ALREADY SUBMITTED {group}: {submitted[group]}')
            continue
        if group in CORE:
            module, function = 'scripts.run_fix_wave_20260929', 'submit'
        elif group == 'timing':
            module, function = 'scripts.run_rule_timing_20261005', 'submit'
        else:
            module = 'scripts.run_paper_bulk_20261005'
            function = 'submit_extra'
        # Import from each batch's frozen source, including an older
        # already prepared core. No shell interpolation of batch paths.
        source = (f'from {module} import {function}; import sys; '
                  f'{function}(sys.argv[1])')
        subprocess.run([sys.executable, '-c', source, str(batch)],
                       cwd=batch / 'code', check=True)


def status_rows(batch, group):
    """
    Revalidate saved runs and account for every planned cell explicitly.
    """

    manifest = verify_extra(batch, rebuild=False)
    if manifest['group'] != group:
        raise ValueError('Report group differs from manifest')
    statuses = dict(validated=[], failed=[], dispatched_no_exit=[],
                    unattempted=[])
    rows = []
    for record in manifest['cells']:
        folder = Path(batch) / 'cells' / str(record['index'])
        if not (folder / 'exit.json').exists():
            dispatched = (folder / 'dispatch.json').exists()
            state = ('dispatched_no_exit' if dispatched
                     else 'unattempted')
        else:
            receipt = fw.read(folder / 'exit.json')
            state = 'failed'
            if receipt['artifact_status'] == 'terminal_contract_validated':
                dispatch = fw.read(folder / 'dispatch.json')
                if (dispatch['cell'] != record
                        or dispatch['commit'] != manifest['commit']
                        or dispatch['manifest_sha256'] !=
                        fw.digest(Path(batch) / 'manifest.json')
                        or receipt['returncode'] != 0
                        or len(receipt['runs']) != 1):
                    raise ValueError('Completed dispatch provenance differs')
                metrics = validate_extra(
                    fw.inside(batch, receipt['runs'][0]), record, group)
                rows.append(dict(record=record, metrics=metrics))
                state = 'validated'
        statuses[state].append(record['index'])
    return statuses, rows


def compare_family(rows, contrasts):
    """
    Withhold incomplete/degenerate inference and retain its Holm slot.
    """

    groups = {}
    for row in rows:
        rec, metrics = row['record'], row['metrics']
        key = (rec['task'], rec['bonus'], rec['arm'])
        groups.setdefault(key, {})[rec['seed']] = metrics
    tests = []
    for task, bonus, a, b in contrasts:
        aa = groups.get((task, bonus, a), {})
        bb = groups.get((task, bonus, b), {})
        label = f'{task}/{bonus}/{a}-{b}'
        if len(aa) != 10 or set(aa) != set(bb):
            tests.append(dict(contrast=label, status='incomplete', p=None))
            continue
        test = fw.paired(aa, bb, label)
        if 'error' in test:
            raise ValueError(test['error'])
        # A constant paired difference does not provide an estimated
        # sampling variance; do not promote p=0 from that degeneracy.
        deltas = [aa[k]['auc'] - bb[k]['auc'] for k in sorted(aa)]
        if len(set(deltas)) == 1:
            test.update(p=None, status='undefined_zero_variance')
        else:
            test['status'] = 'complete'
        tests.append(test)
    if any(t['status'] == 'incomplete' for t in tests):
        for test in tests:
            test.update(p=None, ci95=None,
                        inference='withheld_until_family_complete')
        return dict(status='PROVISIONAL', tests=tests)
    adjusted = fw.holm([t['p'] if t['p'] is not None else 1.0 for t in tests])
    for test, p in zip(tests, adjusted):
        test['holm_p'] = p if test['p'] is not None else None
    return dict(status='complete', tests=tests)


def report(root=ROOT):
    """
    Write an inventory plus new-family results alongside legacy reports.
    """

    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/paper_bulk' / STUDY / 'reports' / stamp
    output.mkdir(parents=True, exist_ok=False)
    summaries, collected = {}, {}
    for group in EXTRAS:
        batch = extra_batch(root, group)
        if not (batch / 'manifest.json').exists():
            summaries[group] = dict(status='not_prepared',
                                    expected=COUNTS[group])
            collected[group] = []
            continue
        statuses, rows = status_rows(batch, group)
        summaries[group] = dict(cell_status=statuses, expected=COUNTS[group])
        collected[group] = rows
    families = dict(view=compare_family(
        collected['view'], view.PRIMARY_CONTRASTS))
    sensitivity_rows = [r for name in sensitivity.SPECS
                        for r in collected[name]]
    for name, contrasts in sensitivity.families().items():
        families[name] = compare_family(sensitivity_rows, contrasts)
    crafter_rows = [r['metrics'] for r in collected['crafter']]
    if len(crafter_rows) == 60:
        families['crafter'] = crafter.report(crafter_rows)
    else:
        families['crafter'] = dict(status='PROVISIONAL',
                                   complete=len(crafter_rows))
    # Preserve every validation status in the machine-readable output.
    result = dict(study=STUDY, groups=summaries, families=families)
    fw.write(output / 'bulk_extra_report.json', result)
    fw.report(root, list(CORE), out=output / 'core_report.json')
    timing_batch = root / 'results/rule_timing' / timing.STUDY
    if (timing_batch / 'manifest.json').exists():
        fw.write(output / 'timing_report.json', timing.report(timing_batch))
    print(f'Reports: {output}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'launch',
                                           'run-cell', 'report'))
    parser.add_argument('--groups', nargs='+', choices=GROUPS)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    groups = args.groups or GROUPS
    if len(groups) != len(set(groups)):
        parser.error('Do not repeat a group in the launch request')
    if args.action == 'run-cell':
        return worker(args.batch, args.index)
    if args.action == 'report':
        report(args.root)
    elif args.action == 'check':
        check(args.root, groups)
    elif args.action == 'prepare':
        prepare_groups(args.root, groups)
    else:
        launch_groups(args.root, groups)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
