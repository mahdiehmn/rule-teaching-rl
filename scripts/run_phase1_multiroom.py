"""Freeze the 20-run MultiRoom replication; only the user submits Slurm."""

import argparse
import copy
import io
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace
import zipfile

from scripts import finish_phase1_native as finish
from scripts import run_advising_strength_grid as paid
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import (
    BudgetError, CostLedger, PriceTable, reservation_for)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/phase1_multiroom_native_20260916_v1.json'
PROTOCOL = 'research/phase1_multiroom_protocol_2026-09-16.md'
JOBS = finish.JOBS + ('938980', '938981')
HASHES = finish.MANIFEST_HASHES + (
    'c7aea8bf2f255ee36546923e043eb3c94a1cf0e421a3f27c2c6ed8db26269e3f',
    'a176e4cfabd72f6df92dc23327d271a5d5b4b86185230cf1f3c101c6628be0e8',
)


def make_manifest(root=ROOT):
    """Change only task, seed and identity from deployed DoorKey Args."""
    config = json.loads((root / CONFIG).read_text())
    if config['source_commit'] != finish.COMMIT:
        raise ValueError('Expected the completed native DoorKey trainer')
    prices = PriceTable(json.loads(subprocess.check_output(
        ['git', 'show', f'{finish.COMMIT}:{paid.PRICES}'], cwd=root)))
    cells = []
    for replicate, seed in enumerate(config['seeds']):
        for template in config['templates']:
            values = copy.deepcopy(template['args'])
            bonus, strategy = template['bonus'], template['strategy']
            values.update(task=config['task'], seed=seed, experiment_id=(
                f'{config["study_id"]}_{bonus}_{strategy}'))
            bound = (reservation_for(SimpleNamespace(**values), None,
                                     prices)[0] if values['guidance'] else 0)
            cells.append(dict(index=len(cells), seed=seed,
                              replicate=replicate, bonus=bonus,
                              strategy=strategy, provider='openai',
                              args=values, worst_case_usd=bound,
                              allocation_usd=math.ceil(bound * 100) / 100))
    return dict(
        author=config['author'], date=config['date'],
        experiment_id=config['study_id'], study_id=config['study_id'],
        commit=config['source_commit'], provider='openai',
        teacher_model='gpt-5-mini', teacher_reasoning_effort='',
        config_path=CONFIG, config_sha256=digest(root / CONFIG),
        protocol=PROTOCOL, protocol_sha256=digest(root / PROTOCOL),
        prices_sha256=config['source_hashes'][paid.PRICES],
        scope='Exploratory transfer of native entropy advice to MultiRoom',
        timing='Native entropy and delivery pacing; no fixed query windows',
        array_concurrency=0, cells=cells, consultation_cap=4800,
        query_budget_per_guided_run=480, advice_budget_per_guided_run=480,
        grid_reservation_usd=round(sum(c['allocation_usd'] for c in cells), 2),
    )


def original_archive(root=ROOT):
    """Check every original source byte before freezing the old trainer."""
    config = json.loads((root / CONFIG).read_text())
    return finish.original_archive(root, {
        'snapshot_hashes': config['source_hashes']})


def funding(root, ledger, amount, reconcile):
    """Audit completed work before proposing any release of unused holds."""
    status = paid.check_pool(ledger, 0)
    reports, evidence, accounting = [], {}, {}
    if reconcile:
        text = subprocess.check_output([
            'sacct', '-j', ','.join(JOBS), '--starttime=2026-09-15',
            '--allocations', '--parsable2', '--format=JobID,State,ExitCode'],
            text=True)
        accounting = finish.accounting_rows(text, jobs=JOBS)
        _, reports, evidence = finish.audit_blocks(
            root / 'results', accounting, jobs=JOBS,
            manifest_hashes=HASHES)
    holds = {r['run_id']: r['reserved'] for r in status['open_reservations']}
    released = 0.0
    for report in reports:
        current = holds.get(report['run_id'])
        if current not in (report['old_hold_usd'], report['retained_usd']):
            raise BudgetError('Completed hold missing or changed; inspect it')
        released += current - report['retained_usd']
    available = status['available_usd'] + released
    print(f'Pool free now ${status["available_usd"]:.2f}; '
          f'after audited hold reductions ${available:.2f}.')
    if available + 1e-9 < amount:
        raise BudgetError(f'Need ${amount:.2f}; only ${available:.2f} '
                          'available. No holds changed.')
    return reports, evidence, accounting


def submission_command(batch):
    """The existing snapshot worker accepts all twenty cell indices."""
    return ['sbatch', '--parsable', '--job-name=p1_multiroom',
            '--array=0-19', f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch / 'code' / paid.WORKER), str(batch)]


def prepare(root, ledger, manifest, archive, config_path=CONFIG,
            protocol_path=PROTOCOL):
    """Reserve one whole batch; retain the original executable source."""
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    snapshot = batch / 'code'
    snapshot.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        handle.extractall(snapshot)
    # Add provenance documents, without modifying original executable code.
    for name in (config_path, protocol_path):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    manifest.update(
        launcher_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        snapshot_hashes={p.relative_to(snapshot).as_posix(): digest(p)
                         for p in snapshot.rglob('*') if p.is_file()},
        parent_ledger=str(ledger.resolve()),
        parent_run_id='grid:' + manifest['experiment_id'],
        credential_file=str(root / '.env'))
    write_json(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'slurm').mkdir()
    CostLedger(str(ledger)).reserve(
        manifest['parent_run_id'], manifest['grid_reservation_usd'],
        note=f'{batch}; retain until completed usage is audited')
    for cell in manifest['cells']:
        directory = batch / 'cells' / str(cell['index'])
        directory.mkdir(parents=True)
        if cell['allocation_usd']:
            CostLedger.initialize(
                str(directory / 'budget.json'), cell['allocation_usd'],
                note='Funded by ' + manifest['parent_run_id'])
    (batch / 'READY').write_text(manifest['commit'] + '\n')
    return batch


def main():
    """Preview, verify or prepare; never make requests or submit jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    parser.add_argument('--reconcile-completed', action='store_true')
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    parser.add_argument('--write-commands', type=Path,
                        default=ROOT / 'results/submit_phase1_multiroom.sh')
    args = parser.parse_args()
    manifest = make_manifest(ROOT)
    amount = manifest['grid_reservation_usd']
    print(f'{manifest["experiment_id"]}: 20 runs, five paired seeds, '
          f'9,999,360 transitions/run; reserve ${amount:.2f}.')
    print('PPO and Count-PPO x no teacher/entropy GPT-5-mini; '
          '480 calls/guided run, 4,800 total. No windows or array throttle.')
    if not (args.check or args.prepare):
        print('Preview only. No changes, API calls or submissions.')
        return 0
    if args.prepare:
        raise RuntimeError(
            'This v1 batch already ran as array 959691; do not duplicate it. '
            'The subgoal replacement was withdrawn. For new matched '
            'advisor comparisons use scripts.run_multiroom_advisors.')
    batch = ROOT / 'results/advising_strength' / manifest['experiment_id']
    if batch.exists() or args.write_commands.exists():
        raise FileExistsError('Batch or submission file already exists; '
                              'inspect it, do not submit twice')
    # Preflight every condition before changing the shared ledger.
    archive = original_archive(ROOT)
    paid.backend_environment('openai', ROOT / '.env')
    reports, evidence, accounting = funding(
        ROOT, args.ledger, amount, args.reconcile_completed)
    print('PASS: original source, credentials and full-batch funding.')
    if args.prepare:
        if reports:
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            audit = (ROOT / 'results/diagnostics'
                     / f'multiroom_funding_{stamp}.json')
            write_json(audit, dict(
                reports=reports, evidence_sha256=evidence,
                accounting=accounting,
                note='Open bounds retained; no invoice settlement'))
            finish.reduce_holds(args.ledger, reports, audit)
        paid.check_pool(args.ledger, amount)
        batch = prepare(ROOT, args.ledger, manifest, archive)
        content = ('#!/bin/bash\nset -euo pipefail\n'
                   + shlex.join(submission_command(batch)) + '\n')
        args.write_commands.parent.mkdir(parents=True, exist_ok=True)
        with args.write_commands.open(
                'x', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print('Prepared. Submit once: bash ' + shlex.quote(
            str(args.write_commands)))
    else:
        print('Check only; no reservations changed.')
    print('No API calls, training or Slurm submissions made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
