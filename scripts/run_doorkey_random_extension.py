"""Prepare only the ten missing native DoorKey random-advice cells.

No calls, reconciliation, or job submission.
"""

import argparse
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import zipfile

from scripts import finish_phase1_native as native
from scripts import run_advising_strength_grid as paid
from scripts import run_multiroom_advisors as random_gate
from scripts import run_phase1_multiroom as prior
from scripts.run_explanation_grid import digest, write_json
from teachers.budget import BudgetError, CostLedger


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_doorkey_random_extension.py'
PROTOCOL = 'research/doorkey_random_extension_2026-09-18.md'
STUDY = 'doorkey_random_extension_20260918_v1'


def make_manifest(root=ROOT, data=None):
    """Copy pinned entropy cells; keep all learner/teacher settings intact."""
    data = root / 'results' if data is None else Path(data)
    cells, reused = [], []
    for replicate, checksum in enumerate(prior.HASHES):
        name = f'{native.STUDY}_r{replicate}'
        batch = data / 'advising_strength' / name
        path = batch / 'manifest.json'
        if (
            digest(path) != checksum
            or (batch / 'manifest.sha256').read_text().strip() != checksum
        ):
            raise BudgetError(f'Original DoorKey manifest changed: {name}')
        old = json.loads(path.read_text(encoding='utf-8'))
        seed = 7_600_000 + 100 * replicate
        if old['commit'] != native.COMMIT or len(old['cells']) != 8:
            raise BudgetError('Original DoorKey cohort identity differs')
        reused.append(
            dict(
                batch=name, job=prior.JOBS[replicate], manifest_sha256=checksum
            )
        )
        for bonus in ('none', 'count'):
            matches = [
                c
                for c in old['cells']
                if c['bonus'] == bonus and c['strategy'] == 'entropy'
            ]
            if len(matches) != 1:
                raise BudgetError('Missing or duplicate original entropy cell')
            cell = copy.deepcopy(matches[0])
            args = cell['args']
            if (
                cell['seed'] != seed
                or args['seed'] != seed
                or args['task'] != 'doorkey_8x8'
                or args['bonus'] != bonus
                or args['advisor'] != 'importance'
                or not args['guidance']
                or args['query_budget'] != 480
                or cell['allocation_usd'] != 5.33
            ):
                raise BudgetError('Original entropy settings differ')
            args.update(
                experiment_id=f'{STUDY}_{bonus}_random',
                advisor='unlimited',
                importance_source='none',
                uniform_queries=True,
            )
            cell.update(
                index=len(cells), replicate=replicate, strategy='random'
            )
            cells.append(cell)
    return dict(
        author='mahdiehmn',
        date='2026-09-18',
        experiment_id=STUDY,
        study_id=STUDY,
        commit=native.COMMIT,
        config_path=SOURCE,
        config_sha256=digest(root / SOURCE),
        protocol=PROTOCOL,
        protocol_sha256=digest(root / PROTOCOL),
        provider='openai',
        teacher_model='gpt-5-mini',
        teacher_reasoning_effort='',
        prices_sha256=old['prices_sha256'],
        cells=cells,
        array_concurrency=0,
        consultation_cap=4800,
        query_budget_per_guided_run=480,
        advice_budget_per_guided_run=480,
        grid_reservation_usd=round(sum(c['allocation_usd'] for c in cells), 2),
        authorized_project_ceiling_usd=203.13,
        reused_cohorts=reused,
        scope='Exploratory extension: reuse forty original DoorKey cells',
        timing='UniformQueries without replacement in the teacher window; '
        'reset slots are not replaced and may yield fewer than 480 calls',
        uniform_helper_sha256=digest(root / random_gate.HELPER),
        archive_builder_sha256=digest(root / random_gate.SOURCE),
        budget_source_sha256=digest(root / 'teachers/budget.py'),
    )


def preflight(root, ledger, manifest):
    """Ordinary admission refusals happen before writing batch artifacts."""
    batch = root / 'results/advising_strength' / STUDY
    if batch.exists():
        raise FileExistsError(
            f'Already prepared: {batch}; do not submit again'
        )
    paid.check_pool(ledger, manifest['grid_reservation_usd'])
    if any(
        e['run_id'] == 'grid:' + STUDY
        for e in CostLedger(str(ledger))._read()['entries']
    ):
        raise BudgetError(
            'This experiment already has an accounting entry; inspect it'
        )
    paid.backend_environment('openai', root / '.env')
    return random_gate.training_archive(root)


def prepare(root, ledger, manifest):
    """Stage source, reserve atomically, then publish one immutable batch."""
    root, ledger = Path(root).resolve(), Path(ledger).resolve()
    archive = preflight(root, ledger, manifest)
    batch = root / 'results/advising_strength' / STUDY
    # A competing reservation may win after preflight. In that case the
    # temporary snapshot is removed and no new batch or parent hold remains.
    with tempfile.TemporaryDirectory(
        prefix='.doorkey-random-', dir=root
    ) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != root:
            raise ValueError(
                'Staging directory must remain within the workspace'
            )
        frozen = stage / 'batch'
        snapshot = frozen / 'code'
        snapshot.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            handle.extractall(snapshot)
        for name in (SOURCE, PROTOCOL):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
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
            parent_ledger=str(ledger),
            parent_run_id='grid:' + STUDY,
            credential_file=str(root / '.env'),
        )
        write_json(frozen / 'manifest.json', saved)
        (frozen / 'manifest.sha256').write_text(
            digest(frozen / 'manifest.json')
        )
        (frozen / 'slurm').mkdir()
        for cell in saved['cells']:
            directory = frozen / 'cells' / str(cell['index'])
            directory.mkdir(parents=True)
            CostLedger.initialize(
                str(directory / 'budget.json'),
                cell['allocation_usd'],
                note='Funded by ' + saved['parent_run_id'],
            )
        (frozen / 'READY').write_text(saved['commit'] + '\n')
        paid.check_pool(ledger, saved['grid_reservation_usd'])
        CostLedger(str(ledger)).reserve(
            saved['parent_run_id'],
            saved['grid_reservation_usd'],
            note=f'{batch}; retain until completed usage is audited',
        )
        # Never roll back a successful reservation automatically on an I/O
        # error. Retain the liability and inspect before manual recovery.
        batch.parent.mkdir(parents=True, exist_ok=True)
        frozen.rename(batch)
    return batch


def submission_command(batch):
    return shlex.join(
        [
            'sbatch',
            '--parsable',
            '--job-name=door_random',
            '--array=0-9',
            f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch / 'code' / paid.WORKER),
            str(batch),
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    parser.add_argument(
        '--data',
        type=Path,
        default=ROOT / 'results',
        help='Original pinned cohort artifacts; read-only',
    )
    parser.add_argument(
        '--ledger', type=Path, default=ROOT / 'results/budget_ledger.json'
    )
    parser.add_argument('--write-commands', type=Path)
    args = parser.parse_args(argv)
    if args.prepare and args.write_commands is None:
        parser.error('--prepare requires --write-commands')
    try:
        manifest = make_manifest(ROOT, args.data)
        print(
            'DoorKey random: 10 runs; five existing seeds; PPO/Count-PPO; '
            '480 query opportunities each. Reuses forty completed controls.'
        )
        print(
            'Reservation $53.30 within the unchanged $203.13 ceiling. '
            'Current jobs and holds remain unchanged.'
        )
        if not (args.check or args.prepare):
            print(
                'Preview only; no reservation, API request or job submission.'
            )
            return 0
        if args.write_commands and args.write_commands.exists():
            raise FileExistsError(
                'Submission file already exists; do not overwrite it'
            )
        if args.check:
            preflight(ROOT, args.ledger, manifest)
            print(
                'PASS: cohorts, funding, credentials and archived source.'
            )
            return 0
        batch = prepare(ROOT, args.ledger, manifest)
        args.write_commands.parent.mkdir(parents=True, exist_ok=True)
        with args.write_commands.open(
            'x', encoding='utf-8', newline='\n'
        ) as handle:
            handle.write(
                '#!/bin/bash\nset -euo pipefail\n'
                + submission_command(batch)
                + '\n'
            )
        print(
            'Prepared. User submission: bash '
            + shlex.quote(str(args.write_commands))
        )
        return 0
    except BudgetError as exc:
        print(f'Admission pending: {exc}. No API requests or jobs submitted.')
        return 2
    except (OSError, ValueError) as exc:
        print(
            f'Preparation stopped: {exc}. Inspect artifacts before retrying.'
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
