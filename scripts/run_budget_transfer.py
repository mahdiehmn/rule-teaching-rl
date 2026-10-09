"""Prepare lower-dose extensions and a matched KeyCorridor transfer study."""

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts import run_advising_strength_grid as paid
from scripts import run_doorkey_random_extension as door
from scripts import run_multiroom_budget120 as mr
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError, CostLedger, PriceTable

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_budget_transfer.py'
PROTOCOL = 'research/budget_transfer_protocol_2026-09-24.md'
CEILING = 453.13
STUDIES = {
    'multiroom': 'multiroom_gpt30_60_20260924_v1',
    'doorkey': 'doorkey_gpt120_20260924_v1',
    'keycorridor': 'keycorridor_gpt_budget_20260924_v1',
}
PINNED = {
    mr.STUDY: '54b7a22a5a7912658e44b04f592e02db520b2fc3a159a49bdd9bbf8199c3ec50',
    door.STUDY: 'd12dcf0684917868662d20c6c578ad5008a41a6db019c548410cf1f031b7d58d',
}


def make_manifest(suite, root=ROOT, data=None):
    """Use actual archived arguments instead of today's trainer defaults."""
    data = Path(data) if data is not None else root / 'results'
    if suite == 'keycorridor':
        check_fresh_seeds(data)
    identity = STUDIES[suite]
    origin = door.STUDY if suite == 'doorkey' else mr.STUDY
    batch = data / 'advising_strength' / origin
    original = mr.pinned(batch, PINNED[origin])
    prices = PriceTable.load(str(root / paid.PRICES))
    call_bound = prices.call_bound('gpt-5-mini', 16384, 3500)
    if not math.isclose(call_bound, 0.011096, abs_tol=1e-12):
        raise ValueError('Reviewed token/price bound changed')
    cells, reused = [], []
    for rep in range(5):
        source = next(
            c
            for c in original['cells']
            if c['replicate'] == rep and c['bonus'] == 'count'
        )
        if suite != 'keycorridor':
            evidence = dict(
                batch=origin,
                index=source['index'],
                manifest_sha256=PINNED[origin],
                **mr.completed(batch, source),
            )
            reused.append(evidence)
        doses = (30, 60) if suite == 'multiroom' else (120,)
        backgrounds = (
            ('none', 'count') if suite == 'keycorridor' else ('count',)
        )
        if suite == 'keycorridor':
            doses = (0, 120, 480)
        for bonus in backgrounds:
            for dose in doses:
                cell = copy.deepcopy(source)
                values = cell['args']
                seed = (
                    11_600_000 + 100 * rep
                    if suite == 'keycorridor'
                    else values['seed']
                )
                values.update(
                    seed=seed,
                    bonus=bonus,
                    query_budget=dose,
                    advice_budget=dose,
                    experiment_id=(f'{identity}_{bonus}_random{dose}'),
                )
                if suite == 'keycorridor':
                    values.update(task='keycorridor_s3r3')
                if dose == 0:
                    values.update(
                        guidance=False, uniform_queries=False, teacher='oracle'
                    )
                bound = call_bound * dose
                cell.update(
                    index=len(cells),
                    seed=seed,
                    replicate=rep,
                    bonus=bonus,
                    strategy=f'random{dose}',
                    worst_case_usd=bound,
                    allocation_usd=math.ceil(bound * 100) / 100,
                )
                cells.append(cell)
    # These identities bind the reused outcomes to the immutable inputs.
    # KeyCorridor borrows a recipe, not another task's outcome/control.
    if suite == 'multiroom':
        reference = mr.make_manifest(root, data)
        for entry in reference['reused_cohorts']:
            rep = entry['replicate']
            if (
                reused[rep]['initial_policy_sha256']
                != entry['control']['initial_policy_sha256']
            ):
                raise ValueError('MultiRoom120 and reused controls differ')
        reused.extend(reference['reused_cohorts'])
    elif suite == 'doorkey':
        for rep, expected in enumerate(door.prior.HASHES):
            name = f'{door.native.STUDY}_r{rep}'
            old_batch = data / 'advising_strength' / name
            old = mr.pinned(old_batch, expected)
            control = next(
                c
                for c in old['cells']
                if c['bonus'] == 'count' and c['strategy'] == 'none'
            )
            evidence = mr.completed(old_batch, control)
            if (
                evidence['initial_policy_sha256']
                != reused[rep]['initial_policy_sha256']
            ):
                raise ValueError(
                    'DoorKey random and control initializations differ'
                )
            reused.append(
                dict(
                    batch=name,
                    index=control['index'],
                    manifest_sha256=expected,
                    **evidence,
                )
            )
    return dict(
        experiment_id=identity,
        study_id=identity,
        date='2026-09-24',
        suite=suite,
        provider='openai',
        teacher_model='gpt-5-mini',
        teacher_reasoning_effort='',
        commit=original['commit'],
        config_path=SOURCE,
        config_sha256=digest(root / SOURCE),
        protocol=PROTOCOL,
        protocol_sha256=digest(root / PROTOCOL),
        prices_sha256=original['prices_sha256'],
        source_reference=dict(batch=origin, manifest_sha256=PINNED[origin]),
        reused_cohorts=reused,
        cells=cells,
        array_concurrency=0,
        authorized_project_ceiling_usd=CEILING,
        grid_reservation_usd=round(sum(c['allocation_usd'] for c in cells), 2),
        consultation_cap=sum(c['args']['query_budget'] for c in cells),
        scope='Exploratory dose/transfer study; not independent confirmation',
        timing='Uniform clock slots without replacement during original first75%; not nested across budgets',
    )


def check_fresh_seeds(data):
    """Reject collisions with another prepared cohort, including eval seeds."""
    proposed = [
        (
            11_600_000 + 100 * i + offset,
            11_600_000 + 100 * i + offset + count - 1,
        )
        for i in range(5)
        for offset, count in ((0, 8), (50_000, 50))
    ]
    for family in (
        'advising_strength',
        'efficiency',
        'explanation_grid',
        'explanation_lessons',
    ):
        for path in (data / family).glob('*/manifest.json'):
            old = mr.read(path)
            if old.get('experiment_id') == STUDIES['keycorridor']:
                continue
            for cell in old.get('cells', []):
                values = cell.get('args', {})
                seed = values.get('seed', cell.get('seed'))
                if type(seed) is not int:
                    continue
                blocks = [
                    (seed + offset, seed + offset + count - 1)
                    for offset, count in (
                        (0, values.get('num_envs', 8)),
                        (50_000, values.get('eval_episodes', 50)),
                    )
                ]
                if any(
                    a <= d and c <= b for a, b in blocks for c, d in proposed
                ):
                    raise ValueError(
                        f'Fresh train/eval seed collision: {path.parent.name}'
                    )


def training_archive(root, data, manifests):
    """Require scientific source identity; change only worker admission cap."""
    archive = mr.inherited.training_archive(root)
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(archive)) as src,
        zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as dst,
    ):
        for name in src.namelist():
            value = src.read(name)
            if name == mr.inherited.HELPER:
                value = value.replace(b'\r\n', b'\n')
            if name == paid.RUNNER:
                anchor = b'AUTHORIZED_ALLOWANCE_USD = 203.13'
                if value.count(anchor) != 1:
                    raise ValueError(
                        'Historical worker admission anchor differs'
                    )
                value = value.replace(
                    anchor, b'AUTHORIZED_ALLOWANCE_USD = 453.13'
                )
            dst.writestr(name, value)
    result = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(result)) as z:
        for manifest in manifests:
            ref = manifest['source_reference']
            old = mr.pinned(
                data / 'advising_strength' / ref['batch'],
                ref['manifest_sha256'],
            )
            for name, expected in old['snapshot_hashes'].items():
                if (
                    name.startswith(
                        ('algos/', 'advising/', 'envs/', 'teachers/')
                    )
                    and name != 'teachers/budget.py'
                ):
                    if hashlib.sha256(z.read(name)).hexdigest() != expected:
                        raise ValueError(f'Scientific source differs: {name}')
            if (
                hashlib.sha256(z.read(paid.PRICES)).hexdigest()
                != manifest['prices_sha256']
            ):
                raise ValueError('Frozen prices differ')
    return result


def existing_batch(root, ledger, manifest):
    """Verify a prior preparation without reserving or submitting again."""
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    if not batch.exists():
        if any(
            e['run_id'] == 'grid:' + manifest['experiment_id']
            for e in CostLedger(str(ledger))._read()['entries']
        ):
            raise BudgetError('Reservation exists without a batch; inspect it')
        return None
    saved = mr.read(batch / 'manifest.json')
    if (
        saved.get('parent_run_id') != 'grid:' + manifest['experiment_id']
        or Path(saved.get('parent_ledger', '')).resolve() != ledger.resolve()
    ):
        raise BudgetError(
            'Prepared batch has a different parent ledger or hold'
        )
    if (
        not (batch / 'READY').is_file()
        or digest(batch / 'manifest.json')
        != (batch / 'manifest.sha256').read_text().strip()
    ):
        raise ValueError('Existing batch is incomplete or changed')
    for key, value in manifest.items():
        if saved.get(key) != value:
            raise ValueError(f'Existing batch differs: {key}')
    for name, checksum in saved['snapshot_hashes'].items():
        if digest(batch / 'code' / name) != checksum:
            raise ValueError(f'Prepared source differs: {name}')
    status = paid.check_pool(ledger, 0)
    holds = [
        h
        for h in status['open_reservations']
        if h['run_id'] == saved['parent_run_id']
    ]
    if len(holds) != 1 or holds[0]['reserved'] < saved['grid_reservation_usd']:
        raise BudgetError('Prepared batch hold is not intact')
    return batch


def prepare(root, ledger, manifest, archive):
    """Publish one immutable batch only after its complete hold is taken."""
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    if batch.exists():
        raise FileExistsError(batch)
    with tempfile.TemporaryDirectory(
        prefix='.budget-transfer-', dir=root
    ) as temp:
        stage = Path(temp) / 'batch'
        snapshot = stage / 'code'
        snapshot.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            z.extractall(snapshot)
        for name in (SOURCE, PROTOCOL):
            (snapshot / name).write_bytes((root / name).read_bytes())
        saved = copy.deepcopy(manifest)
        saved.update(
            launcher_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True
            ).strip(),
            snapshot_hashes={
                p.relative_to(snapshot).as_posix(): digest(p)
                for p in snapshot.rglob('*')
                if p.is_file()
            },
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
            CostLedger.initialize(
                str(folder / 'budget.json'), cell['allocation_usd']
            )
        (stage / 'READY').write_text(saved['commit'] + '\n')
        paid.check_pool(ledger, saved['grid_reservation_usd'])
        CostLedger(str(ledger)).reserve(
            saved['parent_run_id'],
            saved['grid_reservation_usd'],
            note=f'{batch}; full bound retained pending completed audit',
        )
        batch.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', *STUDIES), default='all')
    parser.add_argument('--data', type=Path, default=ROOT / 'results')
    parser.add_argument(
        '--ledger', type=Path, default=ROOT / 'results/budget_ledger.json'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    parser.add_argument('--apply-authorized-funding', action='store_true')
    args = parser.parse_args(argv)
    if args.apply_authorized_funding and not args.prepare:
        parser.error('Funding update requires explicit preparation')
    suites = list(STUDIES) if args.suite == 'all' else [args.suite]
    manifests = [make_manifest(s, ROOT, args.data) for s in suites]
    for m in manifests:
        print(
            f'{m["experiment_id"]}: {len(m["cells"])} runs; '
            f'{m["consultation_cap"]} maximum calls; ${m["grid_reservation_usd"]:.2f}'
        )
    if not (args.prepare or args.check):
        print('Preview only; no funding change, API call or submission.')
        return 0
    archive = training_archive(ROOT, args.data, manifests)
    paid.backend_environment('openai', ROOT / '.env')
    if args.apply_authorized_funding:
        # This is an absolute, idempotent ceiling, never repeated +250.
        paid.check_pool(args.ledger, 0)
        CostLedger(str(args.ledger)).increase_allowance(
            CEILING,
            note='User authorized additional250USD on2026-09-24; preserve all previous spending and holds',
        )
    existing = [existing_batch(ROOT, args.ledger, m) for m in manifests]
    amount = round(
        sum(
            m['grid_reservation_usd']
            for m, b in zip(manifests, existing)
            if b is None
        ),
        2,
    )
    paid.check_pool(args.ledger, amount)
    print(
        f'PASS: archived source, completed reuse, credentials and ${amount:.2f} new funding.'
    )
    if args.prepare:
        for m, batch in zip(manifests, existing):
            batch = batch or prepare(ROOT, args.ledger, m, archive)
            print('Prepared:', batch)
    print('No API calls, training or Slurm submissions made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
