"""Complete the Phase 1 advisor grid at 120 GPT calls on three tasks.

Preparation only; the user submits jobs.
Every new cell copies a completed, hash-pinned Phase 1 cell and changes
only the declared advisor fields, cap and identity.
"""

import argparse
import copy
import hashlib
import io
import math
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts import finish_phase1_native as native
from scripts import run_advising_strength_grid as paid
from scripts import run_budget_transfer as transfer
from scripts import run_multiroom_advisors as inherited
from scripts import run_phase1_multiroom as prior
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError, CostLedger, PriceTable

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_phase1_lowcap120.py'
PROTOCOL = 'research/phase1_lowcap120_protocol_2026-10-08.md'
PREFIX = 'phase1_lowcap120_20261008_v1'
STUDIES = {s: f'{PREFIX}_{s}' for s in ('doorkey', 'multiroom', 'keycorridor')}
CAP = 120
CALL_BOUND = 0.011096  # gpt-5-mini, 16384 input / 3500 output tokens
# Completed low-budget batches; the KeyCorridor one is also the source
# reference whose training snapshot the new batches must reproduce.
KC_BATCH = 'keycorridor_gpt_budget_20260924_v1'
KC_HASH = 'c9456c4c00bf3d09c51042da62d2bdf3978047153b8cf1f7f89cc4b5bd9256c2'
DK120_BATCH = 'doorkey_gpt120_20260924_v1'
DK120_HASH = '2df8b6235896bc28a3a4cd080992ed5bca5eba63a1127d71e63402bec951ffb4'
MR120_BATCH = transfer.mr.STUDY
MR120_HASH = transfer.PINNED[transfer.mr.STUDY]
# New arms per student. Count-PPO random120 already exists on DoorKey and
# MultiRoom; KeyCorridor already has random120 for both students.
ARMS = {
    'doorkey': {'none': ('entropy120', 'probability120', 'random120'),
                'count': ('entropy120', 'probability120')},
    'multiroom': {'none': ('entropy120', 'probability120', 'random120'),
                  'count': ('entropy120', 'probability120')},
    'keycorridor': {'none': ('entropy120', 'probability120'),
                    'count': ('entropy120', 'probability120')},
}
TASKS = {'doorkey': 'doorkey_8x8', 'multiroom': 'multiroom_n6',
         'keycorridor': 'keycorridor_s3r3'}
SEEDS = {'doorkey': 7_600_000, 'multiroom': 8_100_000,
         'keycorridor': 11_600_000}


def convert(args, arm, identity):
    """Apply one Phase 1 advisor recipe at the 120-call cap."""
    values = copy.deepcopy(args)
    values.update(
        query_budget=CAP, advice_budget=CAP, experiment_id=identity,
        guidance=True, teacher='llm_general', uniform_queries=False,
        advisor='importance', importance_source='entropy',
        mistake_threshold=0.0,
    )
    if arm == 'probability120':
        values.update(advisor='mistake', mistake_threshold=.2)
    elif arm == 'random120':
        values.update(advisor='unlimited', importance_source='none',
                      uniform_queries=True)
    elif arm != 'entropy120':
        raise ValueError(f'Unknown arm {arm}')
    return values


def one(cells, **match):
    """Exactly one manifest cell must match."""
    hits = [c for c in cells if all(c.get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise ValueError(f'Expected one cell for {match}, found {len(hits)}')
    return hits[0]


def sources(suite, rep, data):
    """Return (template per student, reused cells) for one seed block."""
    root = data / 'advising_strength'
    if suite == 'doorkey':
        name = f'{native.STUDY}_r{rep}'
        old = transfer.mr.pinned(root / name, prior.HASHES[rep])
        if old['commit'] != native.COMMIT:
            raise ValueError('DoorKey Phase 1 commit differs')
        templates = {b: one(old['cells'], bonus=b, strategy='entropy')
                     for b in ('none', 'count')}
        reused = [(name, prior.HASHES[rep], one(old['cells'], bonus=b,
                                                strategy='none'), 'none')
                  for b in ('none', 'count')]
        dk = transfer.mr.pinned(root / DK120_BATCH, DK120_HASH)
        reused.append((DK120_BATCH, DK120_HASH,
                       one(dk['cells'], replicate=rep, bonus='count'),
                       'random120'))
    elif suite == 'multiroom':
        old = transfer.mr.pinned(root / inherited.PRIOR, inherited.PRIOR_HASH)
        templates = {b: one(old['cells'], replicate=rep, bonus=b,
                            strategy='entropy') for b in ('none', 'count')}
        reused = [(inherited.PRIOR, inherited.PRIOR_HASH,
                   one(old['cells'], replicate=rep, bonus=b, strategy='none'),
                   'none') for b in ('none', 'count')]
        mr = transfer.mr.pinned(root / MR120_BATCH, MR120_HASH)
        reused.append((MR120_BATCH, MR120_HASH,
                       one(mr['cells'], replicate=rep, bonus='count'),
                       'random120'))
    else:
        old = transfer.mr.pinned(root / KC_BATCH, KC_HASH)
        templates = {b: one(old['cells'], replicate=rep, bonus=b,
                            strategy='random120') for b in ('none', 'count')}
        reused = [(KC_BATCH, KC_HASH,
                   one(old['cells'], replicate=rep, bonus=b, strategy=s),
                   'none' if s == 'random0' else s)
                  for b in ('none', 'count') for s in ('random0', 'random120')]
    return templates, reused


def make_manifest(suite, root=ROOT, data=None):
    """Freeze one task's new cells against its completed paired cohort."""
    data = Path(data) if data is not None else root / 'results'
    identity = STUDIES[suite]
    prices = PriceTable.load(str(root / paid.PRICES))
    bound = prices.call_bound('gpt-5-mini', 16384, 3500)
    if not math.isclose(bound, CALL_BOUND, abs_tol=1e-12):
        raise ValueError('Reviewed token/price bound changed')
    reference = transfer.mr.pinned(data / 'advising_strength' / KC_BATCH,
                                   KC_HASH)
    if reference['commit'] != native.COMMIT:
        raise ValueError('Reference batch is not the Phase 1 trainer')
    cells, reused = [], []
    for rep in range(5):
        seed = SEEDS[suite] + 100 * rep
        templates, old = sources(suite, rep, data)
        initial = {}
        for batch, checksum, cell, arm in old:
            evidence = transfer.mr.completed(
                data / 'advising_strength' / batch, cell)
            previous = initial.setdefault(cell['bonus'],
                                          evidence['initial_policy_sha256'])
            if evidence['initial_policy_sha256'] != previous:
                raise ValueError('Reused paired initial policies differ')
            reused.append(dict(replicate=rep, bonus=cell['bonus'], arm=arm,
                               batch=batch, index=cell['index'],
                               manifest_sha256=checksum, **evidence))
        for bonus, arms in ARMS[suite].items():
            template = templates[bonus]
            a = template['args']
            if (a['seed'] != seed or template['seed'] != seed
                    or a['task'] != TASKS[suite] or a['bonus'] != bonus
                    or not a['guidance'] or a['teacher'] != 'llm_general'
                    or a['teacher_model'] != 'gpt-5-mini'
                    or a['max_input_tokens'] != 16384
                    or a['max_output_tokens'] != 3500
                    or a['max_attempts'] != 1
                    or a['total_timesteps'] != 10_000_000):
                raise ValueError(f'Template settings differ: {suite} r{rep}')
            for arm in arms:
                cell = copy.deepcopy(template)
                cell['args'] = convert(
                    a, arm, f'{identity}_{bonus}_{arm}')
                cell.update(index=len(cells), seed=seed, replicate=rep,
                            bonus=bonus, strategy=arm, provider='openai',
                            worst_case_usd=bound * CAP,
                            allocation_usd=math.ceil(bound * CAP * 100) / 100)
                cells.append(cell)
    return dict(
        author='mahdiehmn', date='2026-10-08',
        experiment_id=identity, study_id=identity, suite=suite,
        provider='openai', teacher_model='gpt-5-mini',
        teacher_reasoning_effort='', commit=reference['commit'],
        config_path=SOURCE, config_sha256=digest(root / SOURCE),
        protocol=PROTOCOL, protocol_sha256=digest(root / PROTOCOL),
        prices_sha256=reference['prices_sha256'],
        source_reference=dict(batch=KC_BATCH, manifest_sha256=KC_HASH),
        reused_cohorts=reused, cells=cells, array_concurrency=0,
        authorized_project_ceiling_usd=transfer.CEILING,
        grid_reservation_usd=round(sum(c['allocation_usd'] for c in cells), 2),
        consultation_cap=sum(c['args']['query_budget'] for c in cells),
        query_budget_per_guided_run=CAP, advice_budget_per_guided_run=CAP,
        scope='Exploratory same-seed Phase 1 extension; not confirmation',
        timing='Native entropy pacing, probability .20 mistake test or '
               'uniform clock slots; cap 120 versus completed 480 cohorts',
    )


def training_archive(root, data, manifests):
    """Reviewed archive builder, then byte identity with the KC120 batch."""
    archive = transfer.training_archive(root, data, manifests)
    reference = transfer.mr.pinned(data / 'advising_strength' / KC_BATCH,
                                   KC_HASH)['snapshot_hashes']
    provenance = {transfer.SOURCE, transfer.PROTOCOL}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = {n for n in z.namelist() if not n.endswith('/')}
        expected = set(reference) - provenance
        if names != expected:
            raise ValueError('Archive file set differs from the KC120 batch: '
                             f'{sorted(names ^ expected)[:5]}')
        for name in names:
            if hashlib.sha256(z.read(name)).hexdigest() != reference[name]:
                raise ValueError(f'Archive differs from KC120 batch: {name}')
    return archive


def prepare(root, ledger, manifest, archive):
    """Publish one immutable batch only after its complete hold is taken."""
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    if batch.exists():
        raise FileExistsError(batch)
    with tempfile.TemporaryDirectory(prefix='.p1-lowcap-', dir=root) as temp:
        stage = Path(temp) / 'batch'
        snapshot = stage / 'code'
        snapshot.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            z.extractall(snapshot)
        for name in (SOURCE, PROTOCOL):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        saved = copy.deepcopy(manifest)
        saved.update(
            launcher_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            snapshot_hashes={p.relative_to(snapshot).as_posix(): digest(p)
                             for p in snapshot.rglob('*') if p.is_file()},
            parent_ledger=str(ledger.resolve()),
            parent_run_id='grid:' + manifest['experiment_id'],
            credential_file=str((root / '.env').resolve()),
        )
        write_json(stage / 'manifest.json', saved)
        (stage / 'manifest.sha256').write_text(digest(stage / 'manifest.json'))
        (stage / 'slurm').mkdir()
        for cell in saved['cells']:
            folder = stage / 'cells' / str(cell['index'])
            folder.mkdir(parents=True)
            CostLedger.initialize(str(folder / 'budget.json'),
                                  cell['allocation_usd'],
                                  note='Funded by ' + saved['parent_run_id'])
        (stage / 'READY').write_text(saved['commit'] + '\n')
        paid.check_pool(ledger, saved['grid_reservation_usd'])
        CostLedger(str(ledger)).reserve(
            saved['parent_run_id'], saved['grid_reservation_usd'],
            note=f'{batch}; full bound retained pending completed audit')
        # Never roll back a successful reservation on an I/O error.
        batch.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', *STUDIES), default='all')
    parser.add_argument('--data', type=Path, default=ROOT / 'results')
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    args = parser.parse_args(argv)
    suites = list(STUDIES) if args.suite == 'all' else [args.suite]
    try:
        manifests = [make_manifest(s, ROOT, args.data) for s in suites]
        for m in manifests:
            print(f'{m["experiment_id"]}: {len(m["cells"])} runs; '
                  f'{m["consultation_cap"]} maximum calls; '
                  f'${m["grid_reservation_usd"]:.2f} hold')
        if not (args.check or args.prepare):
            print('Preview only; no reservation, API call or submission.')
            return 0
        archive = training_archive(ROOT, args.data, manifests)
        paid.backend_environment('openai', ROOT / '.env')
        existing = [transfer.existing_batch(ROOT, args.ledger, m)
                    for m in manifests]
        amount = round(sum(m['grid_reservation_usd']
                           for m, b in zip(manifests, existing) if b is None), 2)
        paid.check_pool(args.ledger, amount)
        print(f'PASS: pinned cohorts, KC120-identical source, credentials '
              f'and ${amount:.2f} new funding.')
        if args.prepare:
            for m, batch in zip(manifests, existing):
                print('Prepared:', batch or prepare(ROOT, args.ledger, m,
                                                    archive))
        print('No API calls, training or Slurm submissions made.')
        return 0
    except BudgetError as exc:
        print(f'Admission pending: {exc}. No API requests or jobs submitted.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
