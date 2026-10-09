"""Extend completed MultiRoom controls with two advisor arms; never submit.

Original learner plus an opt-in random query gate.
"""

import argparse
import copy
import csv
import io
import json
import shlex
import subprocess
import time
import zipfile
from pathlib import Path

from scripts import run_phase1_multiroom as prior
from scripts import finish_phase1_native as finish
from scripts import run_advising_strength_grid as paid
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_multiroom_advisors.py'
HELPER = 'advising/uniform_queries.py'
PROTOCOL = 'research/multiroom_advisors_extension_protocol_2026-09-17.md'
STUDY = 'phase1_multiroom_advisors_extension_20260917_v2'
SUPERSEDED_STUDY = 'phase1_multiroom_advisors_20260917_v1'
PRIOR = 'phase1_multiroom_native_20260916_v1'
PRIOR_HASH = 'be748859a9236d6446a07ae984ce43c7fbcf59a6b4a732efa107ea65619f1e7a'
STRATEGIES = ('probability20', 'random')


def make_manifest(replicate, root=ROOT):
    """Freeze missing arms against the completed seed-matched cohort."""
    if type(replicate) is not int or replicate not in range(5):
        raise ValueError('Replicate must be an integer from 0 through 4')
    original = prior.make_manifest(root)
    identity = f'{STUDY}_r{replicate}'
    seed = 8_100_000 + 100 * replicate
    cells = []
    for bonus in ('none', 'count'):
        for strategy in STRATEGIES:
            template = next(c for c in original['cells']
                            if c['replicate'] == replicate
                            and c['bonus'] == bonus
                            and c['strategy'] == 'entropy')
            cell = copy.deepcopy(template)
            values = cell['args']
            values.update(seed=seed, experiment_id=(
                f'{identity}_{bonus}_{strategy}'), uniform_queries=False)
            if strategy == 'probability20':
                values.update(advisor='mistake', mistake_threshold=.2)
            elif strategy == 'random':
                values.update(advisor='unlimited', importance_source='none',
                              uniform_queries=True)
            cell.update(index=len(cells), seed=seed, replicate=replicate,
                        strategy=strategy)
            cells.append(cell)
    manifest = copy.deepcopy(original)
    manifest.update(
        author='mahdiehmn', date='2026-09-17',
        experiment_id=identity, study_id=STUDY, replicate=replicate,
        planned_replicates=list(range(5)), cells=cells,
        config_path=SOURCE, config_sha256=digest(root / SOURCE),
        protocol=PROTOCOL, protocol_sha256=digest(root / PROTOCOL),
        base_config_path=prior.CONFIG,
        base_config_sha256=digest(root / prior.CONFIG),
        uniform_helper_sha256=digest(root / HELPER),
        budget_source_sha256=digest(root / 'teachers/budget.py'),
        authorized_project_ceiling_usd=203.13,
        source_note='Pinned original learner with opt-in uniform query gate',
        scope='Matched extension of completed MultiRoom cohort; '
              'existing none/entropy reused and exact excluded',
        timing='Native selective advisors versus uniform clock sampling',
        prior_batch=PRIOR, prior_job='959691',
        prior_seed=seed, reused_strategies=['none', 'entropy'],
        consultation_cap=1920, grid_reservation_usd=21.32,
    )
    return manifest


def patch_trainer(source):
    """Apply explicit checked edits to the pinned historical trainer.

    Existing advisor paths default to off. Both old and patched sources
    are saved in the archive for byte-level inspection.
    """
    edits = [
        ('    query_windows: int = 0\n',
         '    query_windows: int = 0\n'
         '    uniform_queries: bool = False\n'),
        ("    if args.distill_normalization not in ('labeled', 'batch'):\n",
         "    if args.uniform_queries and (not args.guidance\n"
         "            or args.query_budget <= 0 or args.query_windows\n"
         "            or args.query_interval != 1 or args.teacher_stream\n"
         "            or args.action_reference or args.advisor != 'unlimited'\n"
         "            or args.importance_source != 'none' or args.advice_rate):\n"
         "        raise ValueError('Uniform query configuration is invalid')\n"
         "    if args.distill_normalization not in ('labeled', 'batch'):\n"),
        ('    windows = None\n',
         '    windows = None\n'
         '    uniform_queries = None\n'
         '    if args.uniform_queries:\n'
         '        from advising.uniform_queries import UniformQueries\n'
         '        active = [r for r in range(args.num_iterations)\n'
         '                  if distill_coef(r + 1, args.num_iterations, args) > 0]\n'
         '        uniform_queries = UniformQueries(\n'
         '            active, args.num_steps, args.num_envs,\n'
         '            args.query_budget, args.seed)\n'),
        ('                for i in order:\n                    i = int(i)\n',
         '                for i in order:\n                    i = int(i)\n'
         '                    if (uniform_queries is not None and not\n'
         '                            uniform_queries.allows(iteration-1, step, i)):\n'
         '                        continue\n'),
        ('                    if consulted:\n'
         '                        fresh_episode[i] = False\n',
         '                    if consulted:\n'
         '                        if uniform_queries is not None:\n'
         '                            uniform_queries.record(iteration-1, step, i)\n'
         '                        fresh_episode[i] = False\n'),
        ('        if windows is not None:\n'
         "            extra['query_schedule'] = windows.manifest()\n",
         '        if uniform_queries is not None:\n'
         "            extra['uniform_query_schedule'] = uniform_queries.manifest()\n"
         '        if windows is not None:\n'
         "            extra['query_schedule'] = windows.manifest()\n"),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError('Historical trainer patch anchor changed')
        source = source.replace(old, new)
    compile(source, 'algos/ppo_distill.py', 'exec')
    return source


def training_archive(root=ROOT):
    """Validate the original archive, then record the two training changes."""
    archive = prior.original_archive(root)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as old, zipfile.ZipFile(
            out, 'w', compression=zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            data = old.read(name)
            if name == 'algos/ppo_distill.py':
                new.writestr('provenance/ppo_distill.original.py.txt', data)
                data = patch_trainer(data.decode('utf-8')).encode('utf-8')
            elif name == paid.RUNNER:
                source = data.decode('utf-8')
                old_limit = 'AUTHORIZED_ALLOWANCE_USD = 150.0'
                if source.count(old_limit) != 1:
                    raise ValueError('Historical worker ceiling changed')
                data = source.replace(old_limit,
                    'AUTHORIZED_ALLOWANCE_USD = 203.13').encode('utf-8')
            elif name == 'teachers/budget.py':
                # Preserve historical billing behavior; repair only exact
                # subtraction so the fifth equal-cent reservation can fit.
                current = (root / name).read_text(encoding='utf-8')
                start = current.index('    @staticmethod\n    def _available(')
                end = current.index('    def status(self):', start)
                source = data.decode('utf-8')
                changes = [
                    ('import json\n', 'import json\nfrom decimal import Decimal\n'),
                    ('    def status(self):', current[start:end]
                     + '    def status(self):'),
                    ("'available_usd': payload['allowance_usd'] - settled - held,",
                     "'available_usd': self._available(payload),"),
                    ("available = payload['allowance_usd'] - settled - held",
                     'available = self._available(payload)'),
                ]
                for before, after in changes:
                    if source.count(before) != 1:
                        raise ValueError('Historical ledger anchor changed')
                    source = source.replace(before, after)
                compile(source, name, 'exec')
                data = source.encode('utf-8')
            new.writestr(name, data)
        new.writestr(HELPER, (root / HELPER).read_bytes())
    return out.getvalue()


def completed_accounting(text):
    """Require every allocation from the old 20-cell batch to be terminal."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')), delimiter='|'))
    result = {}
    for i in range(20):
        identity = f'959691_{i}'
        hits = [r for r in rows if r['JobID'] == identity]
        if (len(hits) != 1 or hits[0]['State'] != 'COMPLETED'
                or hits[0]['ExitCode'] != '0:0'):
            raise BudgetError(f'Completion unverified: {identity}')
        result[identity] = hits[0]
    return result


def completed_audit(data, accounting, transferred=False):
    return finish.audit_blocks(
        data, accounting, transferred=transferred, jobs=('959691',),
        manifest_hashes=(PRIOR_HASH,), batch_names=(PRIOR,))


def funding(root, ledger, amount, reconcile):
    """Read-only preflight; known usage and unknown costs remain held."""
    status = paid.check_pool(ledger, 0)
    reports, evidence, accounting = [], {}, {}
    if reconcile:
        text = subprocess.check_output([
            'sacct', '-j', '959691', '--starttime=2026-09-16',
            '--allocations', '--parsable2', '--format=JobID,State,ExitCode'],
            text=True)
        accounting = completed_accounting(text)
        _, reports, evidence = completed_audit(root / 'results', accounting)
    holds = {r['run_id']: r['reserved'] for r in status['open_reservations']}
    available = status['available_usd']
    for r in reports:
        current = holds.get(r['run_id'])
        if current not in (r['old_hold_usd'], r['retained_usd']):
            raise BudgetError('Old MultiRoom hold changed; inspect before editing')
        available += current - r['retained_usd']
    print(f'Free now ${status["available_usd"]:.2f}; '
          f'after verified completed hold reduction ${available:.2f}.')
    if available + 1e-9 < amount:
        raise BudgetError(f'Need ${amount:.2f}; only ${available:.2f} free. '
                          'No holds changed. Select fewer complete seed blocks.')
    return reports, evidence, accounting


def parse_replicates(value):
    try:
        values = [int(v) for v in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integers 0..4') from exc
    if not values or len(set(values)) != len(values) or set(values) - set(range(5)):
        raise argparse.ArgumentTypeError('Use distinct replicate indices 0..4')
    return sorted(values)


def check_cohort(root, manifest):
    """Later funding stages must preserve the first block's contract."""
    fields = ('commit', 'config_sha256', 'protocol_sha256',
              'base_config_sha256', 'uniform_helper_sha256',
              'budget_source_sha256', 'authorized_project_ceiling_usd')
    for batch in (root/'results/advising_strength').glob(STUDY+'_r*'):
        path = batch/'manifest.json'
        if not path.exists():
            raise ValueError(f'Incomplete previous preparation: {batch}')
        old = json.loads(path.read_text())
        if any(old.get(k) != manifest[k] for k in fields):
            raise ValueError('Existing seed block has different source/settings; '
                             'do not mix study versions')

    # The superseded fresh-seed design must not coexist with this extension.
    for batch in (root/'results/advising_strength').glob(
            SUPERSEDED_STUDY+'_r*'):
        raise ValueError(
            f'Superseded fresh-seed batch exists: {batch}. Inspect it before '
            'preparing the matched extension.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    parser.add_argument('--replicates', type=parse_replicates, default=list(range(5)))
    parser.add_argument('--reconcile-completed', action='store_true')
    parser.add_argument('--ledger', type=Path, default=ROOT/'results/budget_ledger.json')
    parser.add_argument('--write-commands', type=Path)
    args = parser.parse_args()
    manifests = [make_manifest(r) for r in args.replicates]
    amount = round(sum(m['grid_reservation_usd'] for m in manifests), 2)
    for m in manifests:
        print(f'{m["experiment_id"]}: 4 runs, seed={m["cells"][0]["seed"]}, '
              f'reservation ${m["grid_reservation_usd"]:.2f}')
    print(f'Selected: {len(manifests)*4} new runs; ${amount:.2f}; '
          f'{len(manifests)*1920} maximum GPT consultations.')
    print('Full extension: 20 new runs on the five completed seeds; $106.60 '
          'worst-case holds. Existing none/entropy runs are reused, not rerun. '
          'No exact-action arm, query windows or array throttle.')
    if not (args.check or args.prepare):
        print('Preview only; no credentials, API calls, ledger changes or jobs.')
        return 0
    if args.prepare and args.write_commands is None:
        parser.error('--prepare requires a new --write-commands filename')
    if args.write_commands and args.write_commands.exists():
        raise FileExistsError('Submission file exists; do not submit twice')
    check_cohort(ROOT, manifests[0])
    for m in manifests:
        if (ROOT/'results/advising_strength'/m['experiment_id']).exists():
            raise FileExistsError(f'Batch already exists: {m["experiment_id"]}')
    archive = training_archive()
    paid.backend_environment('openai', ROOT/'.env')
    reports, evidence, accounting = funding(
        ROOT, args.ledger, amount, args.reconcile_completed)
    print('PASS: source, credentials, funding; check made no requests or changes.')
    if not args.prepare:
        return 0
    if reports:
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        audit = ROOT/'results/diagnostics'/f'multiroom_advisors_funding_{stamp}.json'
        write_json(audit, dict(reports=reports, evidence_sha256=evidence,
                               accounting=accounting))
        finish.reduce_holds(args.ledger, reports, audit)
    paid.check_pool(args.ledger, amount)
    commands = ['#!/bin/bash', 'set -euo pipefail']
    for m in manifests:
        batch = prior.prepare(ROOT, args.ledger, m, archive,
                              config_path=SOURCE, protocol_path=PROTOCOL)
        commands.append(shlex.join([
            'sbatch', '--parsable', '--job-name=mr_adv_ext', '--array=0-3',
            f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch/'code'/paid.WORKER), str(batch)]))
    args.write_commands.parent.mkdir(parents=True, exist_ok=True)
    with args.write_commands.open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(commands)+'\n')
    print('Prepared; user submits once: bash '+shlex.quote(str(args.write_commands)))
    print('No API calls or jobs submitted by this preparation command.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
