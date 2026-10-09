"""Add five GPT120 cells to completed MultiRoom control/GPT480 pairs.

Preparation only; no API or scheduler calls.
"""

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
from scripts import run_multiroom_advisors as inherited
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError, CostLedger, PriceTable

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'multiroom_gpt120_extension_20260923_v1'
SOURCE = 'scripts/run_multiroom_budget120.py'
PROTOCOL = 'research/multiroom_budget120_protocol_2026-09-23.md'
HASHES = (
    '27156ab4bacf2dbf363ffa467c2d18648068a60fe69d8de3353e8301294df66a',
    '310981152f5b4959302fa0875d7528f5b8dbc95276587827e59f3a7bf2d7cd3d',
    '32b0ff1135f34cb76ca4734768d6ef1dd25935d005a67fd6eed8c47b14d24ef1',
    'cfa72a47df17a55d886a46217407917553f727bc35a628610e76a774ad1c02cc',
    '8b796a9f58b4a1d100c6249f0961f901e427e0029424cd569e48d37b2a0690ea',
)


def read(path):
    """Read an artifact, never credentials."""
    return json.loads(path.read_text(encoding='utf-8'))


def completed(batch, cell):
    """Reject missing, ambiguous or incomplete reused runs."""
    directory = batch / 'cells' / str(cell['index'])
    if read(directory / 'exit.json').get('returncode') != 0:
        raise ValueError('Reused cell did not complete')
    args = cell['args']
    paths = list((batch / 'code/results/runs').glob(
        f'*{args["experiment_id"]}__{args["seed"]}__*/run_summary.json'
    ))
    if len(paths) != 1:
        raise ValueError('Reused run is missing or ambiguous')
    summary = read(paths[0])
    if (summary.get('status') != 'completed'
            or summary.get('global_step') != 9_999_360):
        raise ValueError('Reused run did not reach the planned horizon')
    initial = (paths[0].parent / 'initial_policy.sha256').read_text().strip()
    if len(initial) != 64 or any(c not in '0123456789abcdef' for c in initial):
        raise ValueError('Reused initial-policy hash is malformed')
    return dict(summary_sha256=digest(paths[0]), run=paths[0].parent.name,
                initial_policy_sha256=initial)


def pinned(batch, expected):
    """Require the original unmodified manifest and its digest marker."""
    path = batch / 'manifest.json'
    if (digest(path) != expected
            or (batch / 'manifest.sha256').read_text().strip() != expected):
        raise ValueError(f'Reused manifest changed: {batch.name}')
    return read(path)


def make_manifest(root=ROOT, data=None):
    """Copy actual matched480 settings; change only cap and identity."""
    data = root / 'results' if data is None else Path(data)
    prices = PriceTable.load(str(root / paid.PRICES))
    controls = data / 'advising_strength' / inherited.PRIOR
    baseline = pinned(controls, inherited.PRIOR_HASH)
    cells, reused = [], []
    for replicate, expected in enumerate(HASHES):
        name = f'{inherited.STUDY}_r{replicate}'
        batch = data / 'advising_strength' / name
        old = pinned(batch, expected)
        source = next(c for c in old['cells']
                      if c['bonus'] == 'count' and c['strategy'] == 'random')
        control = next(c for c in baseline['cells']
                       if c['replicate'] == replicate
                       and c['bonus'] == 'count' and c['strategy'] == 'none')
        args = source['args']
        if (args['seed'] != 8_100_000 + 100 * replicate
                or args['task'] != 'multiroom_n6'
                or not args['uniform_queries'] or args['query_budget'] != 480
                or args['teacher_model'] != 'gpt-5-mini'
                or old['commit'] != inherited.finish.COMMIT
                or old['prices_sha256'] != hashlib.sha256(
                    (root / paid.PRICES).read_bytes().replace(
                        b'\r\n', b'\n')).hexdigest()):
            raise ValueError('Original treatment identity differs')
        reused.append(dict(
            replicate=replicate, batch=name, manifest_sha256=expected,
            random480=completed(batch, source),
            control=completed(controls, control),
        ))
        if reused[-1]['random480']['initial_policy_sha256'] != reused[-1][
                'control']['initial_policy_sha256']:
            raise ValueError('Reused paired initial policies differ')
        cell = copy.deepcopy(source)
        cell['args'].update(
            query_budget=120, advice_budget=120,
            experiment_id=STUDY + '_count_random120',
        )
        bound = prices.call_bound('gpt-5-mini', 16384, 3500) * 120
        if args['max_attempts'] != 1 or not math.isclose(bound, 1.33152):
            raise ValueError('Reviewed full-attempt bound differs')
        cell.update(index=replicate, strategy='random120',
                    worst_case_usd=bound,
                    allocation_usd=math.ceil(bound * 100) / 100)
        cells.append(cell)
    manifest = copy.deepcopy(old)
    # Discard stale accounting and snapshot fields from the reused batch.
    for key in ('snapshot_hashes', 'parent_ledger', 'parent_run_id',
                'credential_file', 'launcher_commit', 'replicate',
                'prior_seed'):
        manifest.pop(key, None)
    manifest.update(
        author='mahdiehmn', date='2026-09-23', experiment_id=STUDY,
        study_id=STUDY, cells=cells, reused_cohorts=reused,
        config_path=SOURCE, config_sha256=digest(root / SOURCE),
        protocol=PROTOCOL, protocol_sha256=digest(root / PROTOCOL),
        grid_reservation_usd=6.70, consultation_cap=600,
        query_budget_per_guided_run=120, advice_budget_per_guided_run=120,
        scope='Exploratory same-seed budget extension; not confirmation',
        array_concurrency=0,
        replicates=list(range(5)), reused_strategies=['none', 'random480'],
        timing='Uniform clock sampling; cap120 versus original480',
    )
    return manifest


def preflight(root, data, ledger, manifest):
    """Check funding and exact training-source compatibility first."""
    batch = root / 'results/advising_strength' / STUDY
    if batch.exists():
        raise FileExistsError('Batch already prepared; do not duplicate')
    paid.check_pool(ledger, manifest['grid_reservation_usd'])
    if any(e['run_id'] == 'grid:' + STUDY
           for e in CostLedger(str(ledger))._read()['entries']):
        raise BudgetError('Existing accounting entry requires inspection')
    paid.backend_environment('openai', root / '.env')
    archive = inherited.training_archive(root)
    # Git's Windows checkout can give this overlaid helper CRLF bytes.
    # Canonicalize only line endings; the original scientific hash below
    # still rejects any actual source change.
    normalized = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as old, zipfile.ZipFile(
            normalized, 'w', compression=zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            value = old.read(name)
            if name == inherited.HELPER:
                value = value.replace(b'\r\n', b'\n')
            new.writestr(name, value)
    archive = normalized.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        if hashlib.sha256(handle.read(paid.PRICES)).hexdigest() != manifest[
                'prices_sha256']:
            raise ValueError('Frozen training price table differs')
        for reused in manifest['reused_cohorts']:
            old = read(data / 'advising_strength' / reused['batch']
                       / 'manifest.json')
            for name, checksum in old['snapshot_hashes'].items():
                if name.startswith(('algos/', 'advising/', 'envs/',
                                    'teachers/')) and name != 'teachers/budget.py':
                    if hashlib.sha256(handle.read(name)).hexdigest() != checksum:
                        raise ValueError(f'Training source differs: {name}')
    return archive


def prepare(root, data, ledger, manifest):
    """Stage before reservation; never release a liability on I/O failure."""
    archive = preflight(root, data, ledger, manifest)
    batch = root / 'results/advising_strength' / STUDY
    with tempfile.TemporaryDirectory(prefix='.mr120-', dir=root) as tmp:
        frozen = Path(tmp) / 'batch'
        snapshot = frozen / 'code'
        snapshot.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            handle.extractall(snapshot)
        for name in (SOURCE, PROTOCOL):
            (snapshot / name).write_bytes((root / name).read_bytes())
        saved = copy.deepcopy(manifest)
        saved.update(
            launcher_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            snapshot_hashes={p.relative_to(snapshot).as_posix(): digest(p)
                             for p in snapshot.rglob('*') if p.is_file()},
            parent_ledger=str(ledger.resolve()), parent_run_id='grid:' + STUDY,
            credential_file=str((root / '.env').resolve()),
        )
        write_json(frozen / 'manifest.json', saved)
        (frozen / 'manifest.sha256').write_text(digest(frozen / 'manifest.json'))
        (frozen / 'slurm').mkdir()
        for cell in cells_from(saved):
            directory = frozen / 'cells' / str(cell['index'])
            directory.mkdir(parents=True)
            CostLedger.initialize(str(directory / 'budget.json'),
                                  cell['allocation_usd'])
        (frozen / 'READY').write_text(saved['commit'] + '\n')
        CostLedger(str(ledger)).reserve(saved['parent_run_id'], 6.70,
                                      note=f'{batch}; completed costs retained')
        batch.parent.mkdir(parents=True, exist_ok=True)
        frozen.rename(batch)
    return batch


def cells_from(manifest):
    """Assert the funded scope before creating child pools."""
    cells = manifest['cells']
    if len(cells) != 5 or any(c['allocation_usd'] != 1.34 for c in cells):
        raise ValueError('Expected five fully funded120-call cells')
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT / 'results')
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    args = parser.parse_args()
    manifest = make_manifest(ROOT, args.data)
    print('MultiRoom: five new Count-PPO GPT120 runs; reuse five480 and five'
          ' no-advice controls. 9,999,360 transitions/run; reserve $6.70.')
    if args.check:
        preflight(ROOT, args.data, args.ledger, manifest)
        print('PASS: paired artifacts, source, credentials and funding.')
    elif args.prepare:
        print(prepare(ROOT, args.data, args.ledger, manifest))
    else:
        print('Preview only; no reservation, API call or submission.')


if __name__ == '__main__':
    main()
