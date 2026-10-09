"""Audit completed native GPT blocks and prepare the two missing seeds.

Preview is read-only. Preparation retains a conservative hold for completed
work, including every unknown-cost attempt, and freezes the original code.
It neither settles provider bills nor submits jobs or makes API requests.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
import zipfile

from scripts import run_advising_strength_grid as paid
from scripts.analyze_phase1_native_20260915 import (
    audit_journal, audit_settings, json_lines, read_json)
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError, CostLedger, PriceTable, _lock


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'phase1_native_20260914_v1'
COMMIT = '1ab5fc0d9c9d22b7147d58ba92245bcc42c0b008'
JOBS = ('927025', '927026', '927027')
# These are the original deployed manifests, independently inventoried
# in the September 15 transfer. Do not silently accept revised cohorts.
MANIFEST_HASHES = (
    '0b747f51a2e34e32999347372786db89e6a0e387712b24efa8ba5539e6e494a2',
    'a88837289d02c8c3ce24f40fe05eb761ffc0fac3b1fd22182f9c4339eb824622',
    '7fe0a9ca00c3575d46447841bf86a8cc1b81eb6a3de5f6df4bc4f833830136e0',
)


def accounting_rows(text, jobs=JOBS):
    """Require every original cell to be complete, not just array parents."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')),
                               delimiter='|'))
    result = {}
    for job in jobs:
        for index in range(8):
            identity = f'{job}_{index}'
            matches = [r for r in rows if r['JobID'] == identity]
            if (len(matches) != 1 or matches[0]['State'] != 'COMPLETED'
                    or matches[0]['ExitCode'] != '0:0'):
                raise BudgetError(f'Completion unverified: {identity}')
            result[identity] = matches[0]
    return result


def remaining_manifest(template, replicate):
    """Change identities and seed only; preserve all scientific settings."""
    if replicate not in (3, 4):
        raise ValueError('Only the missing planned replicates 3 and 4')
    manifest = copy.deepcopy(template)
    identity = f'{STUDY}_r{replicate}'
    manifest.update(experiment_id=identity, replicate=replicate,
                    parent_run_id='grid:' + identity)
    for cell in manifest['cells']:
        cell.update(seed=7_600_000 + 100 * replicate, replicate=replicate)
        cell['args'].update(
            seed=cell['seed'],
            experiment_id=f'{identity}_{cell["bonus"]}_{cell["strategy"]}')
    return manifest


def bounded_cost(rows, args, prices):
    """Price known usage and retain a full request bound for unknown usage."""
    bound = prices.call_bound('gpt-5-mini', 16384, 3500)
    rate_in, rate_out = prices.chat_rates('gpt-5-mini')
    known, unknown = 0.0, 0
    if (args['max_attempts'] != 1 or args['max_input_tokens'] != 16384
            or args['max_output_tokens'] != 3500
            or args['teacher_model'] != 'gpt-5-mini'
            or args['explanation'] != 'none'):
        raise BudgetError('This audit only covers the frozen native recipe')
    if len(rows) != 480:
        raise BudgetError('Expected all 480 consultation records')
    if [r['consultation'] for r in rows] != list(range(1, 481)):
        raise BudgetError('Missing or duplicate consultation record')
    for row in rows:
        meta = row['metadata']
        attempts = meta.get('attempt_count')
        cost = row.get('known_response_dollars')
        if row['teacher_model'] != 'gpt-5-mini':
            raise BudgetError('Unexpected teacher model')
        if meta.get('cache_hit'):
            if attempts != 0 or cost != 0:
                raise BudgetError('Cache hit has ambiguous billing')
            continue
        if type(attempts) is not int or attempts != 1:
            raise BudgetError('Attempt count exceeds frozen bound')
        if cost is None:
            unknown += 1
            continue
        ti, to = row['tokens_in'], row['tokens_out']
        if (type(ti) is not int or type(to) is not int
                or not 0 <= ti <= 16384 or not 0 <= to <= 3500
                or meta.get('service_tier') != 'default'
                or meta.get('cost_known') is not True
                or meta.get('usage_known') is not True):
            raise BudgetError('Known usage or pricing tier unverified')
        estimated = (ti * rate_in + to * rate_out) / 1e6
        if (type(cost) not in (int, float) or not math.isfinite(cost)
                or not math.isclose(cost, estimated, abs_tol=1e-9)
                or not 0 <= cost <= bound):
            raise BudgetError('Recorded cost disagrees with frozen prices')
        known += estimated
    return dict(known_standard_rate_usd=known, unknown_attempts=unknown,
                unknown_maximum_usd=unknown * bound,
                retained_bound_usd=known + unknown * bound)


def audit_blocks(data, accounting, *, transferred=False, jobs=JOBS,
                 manifest_hashes=MANIFEST_HASHES, batch_names=None):
    """Join terminal workers to complete summaries, journals and ledgers."""
    reports, evidence, template = [], {}, None
    if len(jobs) != len(manifest_hashes):
        raise ValueError('Each completed block needs a pinned manifest hash')
    if batch_names is not None and len(batch_names) != len(jobs):
        raise ValueError('Each explicit batch needs its own job and hash')
    for replicate, job in enumerate(jobs):
        name = (batch_names[replicate] if batch_names is not None
                else f'{STUDY}_r{replicate}')
        batch = data / 'advising_strength' / name
        mp = batch / 'manifest.json'
        if (digest(mp) != manifest_hashes[replicate]
                or (batch / 'manifest.sha256').read_text().strip()
                != manifest_hashes[replicate]):
            raise BudgetError(f'Original manifest changed: {batch}')
        manifest = read_json(mp)
        if manifest['commit'] != COMMIT:
            raise BudgetError('Original source commit changed')
        if not transferred and (batch / 'READY').read_text().strip() != COMMIT:
            raise BudgetError('Missing original READY marker')
        evidence[str(mp)] = digest(mp)
        for name, checksum in manifest['snapshot_hashes'].items():
            path = batch / 'code' / name
            # The transfer intentionally omits some documentation/assets.
            # Live preparation requires every original source file.
            if transferred and not path.is_file():
                if path.suffix == '.py':
                    raise BudgetError(f'Missing transferred code: {name}')
                continue
            if digest(path) != checksum:
                raise BudgetError(f'Frozen source changed: {path}')
        prices = PriceTable.load(str(batch / 'code' / paid.PRICES))
        runs = [p for p in (batch / 'code/results/runs').iterdir()
                if p.is_dir()]
        if len(runs) != len(manifest['cells']):
            raise BudgetError('Unexpected extra/missing run directories')
        summaries = [(p, read_json(p / 'run_summary.json')) for p in runs]
        costs = []
        for cell in manifest['cells']:
            identity = f'{job}_{cell["index"]}'
            if identity not in accounting:
                raise BudgetError(f'Missing scheduler evidence: {identity}')
            directory = batch / 'cells' / str(cell['index'])
            dispatch = read_json(directory / 'dispatch.json')
            if (read_json(directory / 'exit.json') != {'returncode': 0}
                    or dispatch['commit'] != COMMIT):
                raise BudgetError(f'Worker did not finish: {identity}')
            matches = [(p, s) for p, s in summaries
                       if s['args']['experiment_id']
                       == cell['args']['experiment_id']
                       and s['args']['seed'] == cell['args']['seed']]
            if len(matches) != 1:
                raise BudgetError('Nonunique run identity')
            run, summary = matches[0]
            if (summary['status'] != 'completed'
                    or summary['global_step'] != 9_999_360):
                raise BudgetError('Training did not finish the frozen horizon')
            audit_settings(cell, summary)
            audit_journal(run, summary)
            for path in (directory / 'dispatch.json', directory / 'exit.json',
                         run / 'run_summary.json'):
                evidence[str(path)] = digest(path)
            if not cell['args']['guidance']:
                continue
            journal = run / 'consultations.jsonl'
            rows = json_lines(journal)
            costs.append(bounded_cost(rows, cell['args'], prices))
            child_path = directory / 'budget.json'
            child = read_json(child_path)
            entries = child['entries']
            if (child['allowance_usd'] != cell['allocation_usd']
                    or len(entries) != 1
                    or entries[0]['run_id'] != run.name
                    or entries[0]['state'] != 'open'
                    or entries[0]['actual'] is not None
                    or entries[0]['reserved'] != cell['worst_case_usd']):
                raise BudgetError('Child accounting differs from frozen hold')
            evidence[str(journal)] = digest(journal)
            evidence[str(child_path)] = digest(child_path)
        report = {key: sum(c[key] for c in costs) for key in costs[0]}
        report.update(run_id=manifest['parent_run_id'],
                      old_hold_usd=manifest['grid_reservation_usd'],
                      retained_usd=math.ceil(
                          report['retained_bound_usd'] * 100) / 100)
        reports.append(report)
        if replicate == 0:
            template = manifest
    return template, reports, evidence


def original_archive(root, template):
    """Obtain original code from Git, independent of the current checkout."""
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf',
         'archive', '--format=zip', COMMIT], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        hashes = {name: hashlib.sha256(handle.read(name)).hexdigest()
                  for name in handle.namelist() if not name.endswith('/')}
    if hashes != template['snapshot_hashes']:
        raise BudgetError('Git archive differs from the deployed source')
    return archive


def reduce_holds(ledger, reports, audit_path):
    """Retain conservative liabilities as OPEN holds, never paid invoices."""
    book = CostLedger(str(ledger))
    with _lock(str(ledger)):
        payload = book._read()
        original = copy.deepcopy(payload)
        for report in reports:
            if (not math.isfinite(report['retained_usd'])
                    or not 0 <= report['retained_usd']
                    <= report['old_hold_usd']):
                raise BudgetError('Invalid reduced reservation')
            matches = [e for e in payload['entries']
                       if e['run_id'] == report['run_id']]
            if (len(matches) != 1 or matches[0]['state'] != 'open'
                    or matches[0]['actual'] is not None
                    or matches[0].get('overspent')
                    or matches[0]['reserved'] not in (
                        report['old_hold_usd'], report['retained_usd'])):
                raise BudgetError(
                    'Parent hold differs; inspect before editing')
            entry = matches[0]
            if entry['reserved'] == report['retained_usd']:
                continue
            entry.setdefault('reservation_adjustments', []).append({
                'previous_usd': entry['reserved'],
                'retained_usd': report['retained_usd'],
                'audit_path': str(audit_path),
                'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'reason': 'Completed; known usage plus unknown-attempt bound',
            })
            entry['reserved'] = report['retained_usd']
        # Preserve the exact previous ledger beside the audit before writing.
        write_json(audit_path.with_suffix('.ledger-before.json'), original)
        book._write(payload)


def prepare_block(root, ledger, manifest, archive):
    """Build a new funded block using the byte-identical original snapshot."""
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    batch.mkdir(parents=True, exist_ok=False)
    snapshot = batch / 'code'
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        handle.extractall(snapshot)
    manifest.update(parent_ledger=str(ledger),
                    credential_file=str(root / '.env'))
    write_json(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(
        digest(batch / 'manifest.json') + '\n', encoding='utf-8')
    (batch / 'slurm').mkdir()
    CostLedger(str(ledger)).reserve(
        manifest['parent_run_id'], manifest['grid_reservation_usd'],
        note=f'{batch}; parent allocation, do not settle while jobs run')
    for cell in manifest['cells']:
        directory = batch / 'cells' / str(cell['index'])
        directory.mkdir(parents=True)
        if cell['allocation_usd']:
            CostLedger.initialize(str(directory / 'budget.json'),
                                  cell['allocation_usd'],
                                  note='Funded by '
                                  + manifest['parent_run_id'])
    (batch / 'READY').write_text(COMMIT + '\n', encoding='utf-8')
    command = shlex.join([
        'sbatch', '--parsable', '--job-name=p1_native', '--array=0-7',
        f'--output={batch}/slurm/%x_%A_%a.out',
        str(snapshot / paid.WORKER), str(batch)])
    (batch / 'submit.sh').write_text(
        '#!/bin/bash\nset -euo pipefail\n' + command + '\n',
        encoding='utf-8', newline='\n')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'results')
    parser.add_argument('--accounting', type=Path,
                        help='Transfer receipt, read-only audit only')
    parser.add_argument('--write-commands', type=Path,
                        default=ROOT / 'results/submit_phase1_final_seeds.sh')
    args = parser.parse_args()
    if args.prepare and (args.accounting or args.data_root.resolve()
                         != ROOT / 'results'):
        parser.error('Preparation requires live local artifacts/accounting')
    if args.accounting:
        text = args.accounting.read_text(encoding='utf-8-sig')
    else:
        text = subprocess.check_output([
            'sacct', '-j', ','.join(JOBS), '--starttime=2026-09-15',
            '--allocations', '--parsable2',
            '--format=JobID,State,ExitCode'], text=True)
    accounting = accounting_rows(text)
    template, reports, evidence = audit_blocks(
        args.data_root, accounting, transferred=bool(args.accounting))
    archive = original_archive(ROOT, template)
    manifests = [remaining_manifest(template, r) for r in (3, 4)]
    amount = round(sum(m['grid_reservation_usd'] for m in manifests), 2)
    print(json.dumps(reports, indent=2))
    print(f'Missing: 16 runs, seeds 7600300/7600400; ${amount:.2f}.')
    print('Original source/settings; 9,999,360 transitions; 480 calls/run; '
          'no query windows or array throttle.')
    if not args.accounting:
        status = paid.check_pool(args.ledger, 0)
        holds = {h['run_id']: h['reserved']
                 for h in status['open_reservations']}
        released = 0.0
        for report in reports:
            current = holds.get(report['run_id'])
            if current not in (report['old_hold_usd'], report['retained_usd']):
                raise BudgetError('Completed parent hold is missing/changed')
            released += current - report['retained_usd']
        projected = status['available_usd'] + released
        print(f'Free after retaining completed costs: ${projected:.2f}; '
              f'after funding missing seeds: ${projected - amount:.2f}.')
        if projected < amount:
            raise BudgetError('Insufficient funds; no holds changed')
    if args.prepare:
        if args.write_commands.exists():
            raise FileExistsError(
                'Submission file exists; inspect, do not repeat')
        for manifest in manifests:
            batch = (ROOT / 'results/advising_strength'
                     / manifest['experiment_id'])
            if batch.exists():
                raise FileExistsError(f'Already prepared: {batch}')
        paid.backend_environment('openai', ROOT / '.env')
        # Store the evidence before the sole shared-ledger mutation.
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        audit = (ROOT / 'results/diagnostics'
                 / f'phase1_completion_{stamp}.json')
        write_json(audit, dict(reports=reports, evidence_sha256=evidence,
                               accounting=accounting, original_commit=COMMIT,
                               note='Bounds retained; actual bills unsettled'))
        reduce_holds(args.ledger, reports, audit)
        paid.check_pool(args.ledger, amount)
        commands = [prepare_block(ROOT, args.ledger.resolve(), m, archive)
                    for m in manifests]
        args.write_commands.parent.mkdir(parents=True, exist_ok=True)
        with args.write_commands.open(
                'x', encoding='utf-8', newline='\n') as f:
            f.write('#!/bin/bash\nset -euo pipefail\n')
            f.write('\n'.join(commands) + '\n')
        script = shlex.quote(str(args.write_commands))
        print(f'Prepared. Submit: bash {script}')
    else:
        print('Audit only. No ledger, batch or submission file changed.')
    print('No API calls, training or Slurm submissions made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
