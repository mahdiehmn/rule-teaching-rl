"""Prepare a fully funded grid; run exactly one reserved manifest cell.

The login-node coordinator reserves the ENTIRE grid in the shared ledger
once. Each cell receives a separate, funded child ledger. Compute nodes
never mutate the shared ledger, so their starts need no cross-node ledger
lock. Keep the parent hold until all children and provider spend have
been reconciled. Child balances are allocations of that hold, not money
added to the account.

Preparation makes no API calls and submits no jobs. The user runs the
printed sbatch command. A claimed cell cannot be retried automatically.
"""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

from advising.schedule import QueryWindows
from scripts.dev_window_env import environment
from teachers.budget import BudgetError, CostLedger, PriceTable
from teachers.budget import reservation_for


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/explanation_s3r3_v1.json'
PRICES = 'configs/explanation_prices_2026-09-10.json'
ARMS = [('R0', False, 'none'), ('R1', True, 'none'),
        ('R2', True, 'correct'), ('R3', True, 'shuffled'),
        ('R4', True, 'detached')]
VARIANTS = ARMS + [('S2', True, 'correct'), ('S4', True, 'detached')]


def digest(path):
    """Hash a file without including its absolute filesystem location."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    """Write a new artifact; refuse to replace earlier experiment state."""
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')


def make_manifest(root=ROOT, config_path=CONFIG):
    """Resolve all trainer defaults and dollar bounds without API access."""
    from algos.ppo_distill import Args

    root = Path(root).resolve()
    config_path = (root / config_path).resolve().relative_to(root).as_posix()
    config = json.loads((root / config_path).read_text(encoding='utf-8'))
    selected = config.get('arms', [arm for arm, _, _ in ARMS])
    if (not selected or len(set(selected)) != len(selected)
            or set(selected) - {arm for arm, _, _ in VARIANTS}):
        raise ValueError('Select distinct arms R0-R4 or structured S2/S4.')
    if (set(selected) & {'S2', 'S4'}
            and not config['shared'].get('structured_explanations')):
        raise ValueError('Structured arms require structured model replies.')
    if config.get('controlled_reference') and (
            not config['shared'].get('action_reference')
            or not config['shared'].get('structured_explanations')):
        raise ValueError('Controlled grids need reference actions and schema.')
    seeds = config['seeds']
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Select distinct nonnegative integer seeds.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['experiment_id']):
        raise ValueError('experiment_id must be a safe directory name.')
    job_name = config.get('job_name')
    if job_name is not None and (not isinstance(job_name, str)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', job_name)):
        raise ValueError('job_name must be a safe Slurm name.')
    prices = PriceTable.load(str(root / PRICES))
    cells = []
    for seed in config['seeds']:
        for arm, guided, explanation in VARIANTS:
            if arm not in selected:
                continue
            args = Args(**config['shared'], seed=seed,
                        experiment_id=config['experiment_id'] + (
                            '_' + arm if config.get('controlled_reference')
                            else ''),
                        guidance=guided, explanation=explanation)
            if arm in ('S2', 'S4'):
                from teachers.controlled_advice import SUBGOALS
                args.explanation_target = 'subgoal'
                args.embed_dim = len(SUBGOALS)
            if config.get('controlled_reference') and arm == 'R1':
                args.rationale_queries = False
                args.query_budget = args.query_windows = 0
            # R0 cannot create a paid teacher even if a future default
            # changes the no-guidance path. Its query cap is zero.
            if not guided:
                args.teacher = 'bot'
                args.query_windows = args.query_budget = 0
                args.advice_budget = 0
                args.action_reference = ''
            bound = (reservation_for(args, None, prices)[0]
                     if guided and args.rationale_queries else 0.0)
            cells.append({
                'index': len(cells), 'arm': arm, 'seed': seed,
                'args': asdict(args), 'worst_case_usd': bound,
                'allocation_usd': math.ceil(bound * 100) / 100,
            })
    shared = config['shared']
    schedule = QueryWindows(
        total_timesteps=shared['total_timesteps'],
        batch_size=shared['num_envs'] * shared['num_steps'],
        num_steps=shared['num_steps'], num_envs=shared['num_envs'],
        num_windows=shared['query_windows'],
        window_steps=shared['query_window_steps'],
        guidance_fraction=shared['query_window_fraction'],
    ).manifest()
    return {
        'experiment_id': config['experiment_id'],
        'job_name': job_name,
        'config_path': config_path,
        'config_sha256': digest(root / config_path),
        'scope': config.get('scope', 'full-grid candidate'),
        'protocol': config.get('protocol'),
        'controlled_reference': bool(config.get('controlled_reference')),
        'recovery_of': config.get('recovery_of'),
        'prices_sha256': digest(root / PRICES),
        'schedule': schedule, 'cells': cells,
        'grid_reservation_usd': round(sum(
            c['allocation_usd'] for c in cells), 2),
        'consultation_cap': sum(c['args']['query_budget'] for c in cells),
    }


def trainer_argv(values):
    """Serialize the frozen dataclass, including false and empty values."""
    result = []
    for key, value in values.items():
        name = key.replace('_', '-')
        if isinstance(value, bool):
            result.append('--' + (name if value else 'no-' + name))
        elif value is not None:
            result.extend(['--' + name, str(value)])
    return result


def check_pool(path, amount):
    """Check affordability without discarding existing pilot holds."""
    status = CostLedger(str(path)).status()
    if (not 0 < status['allowance_usd'] <= 203.13
            or status['overspent_runs']):
        raise BudgetError('The existing pool must be within $203.13 and '
                          'have no unresolved overspend.')
    if not math.isfinite(amount) or status['available_usd'] < amount:
        raise BudgetError(f'Whole grid needs ${amount:.2f}; only '
                          f'${status["available_usd"]:.2f} is free.')
    return status


def prepare(root, ledger, manifest):
    """Freeze committed code and fund all children in one admission."""
    root, ledger = Path(root).resolve(), Path(ledger).resolve()
    subprocess.run(['git', 'diff', '--exit-code', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    amount = manifest['grid_reservation_usd']
    batch = root / 'results' / 'explanation_grid' / manifest['experiment_id']
    if batch.exists():
        raise FileExistsError(f'{batch} already exists; do not prepare twice')
    check_pool(ledger, amount)
    batch.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive directory creation prevents repeated preparation, even
    # after a crash. A failure leaves evidence and never releases money.
    batch.mkdir()
    snapshot = batch / 'code'
    snapshot.mkdir()
    archive = subprocess.check_output(['git', 'archive', '--format=zip',
                                       commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        handle.extractall(snapshot)
    hashes = {str(p.relative_to(snapshot)).replace('\\', '/'): digest(p)
              for p in snapshot.rglob('*') if p.is_file()}
    # Refuse an untracked launcher/config that git archive left behind.
    config_path = manifest.get('config_path', CONFIG)
    for required in (config_path, PRICES, 'scripts/run_explanation_grid.py',
                     'scripts/submit_explanation_grid.sh'):
        if required not in hashes:
            raise RuntimeError(f'{required} is absent from HEAD')
    # Git may normalize CRLF to LF in the archive. Record the bytes
    # actually executed, while the tracked-diff guard checks edits.
    manifest['config_sha256'] = hashes[config_path]
    manifest['prices_sha256'] = hashes[PRICES]
    manifest.update(commit=commit, snapshot_hashes=hashes,
                    parent_ledger=str(ledger),
                    parent_run_id='grid:' + manifest['experiment_id'],
                    env_file=str(root / '.env'))
    write_json(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(
        digest(batch / 'manifest.json') + '\n', encoding='utf-8')
    (batch / 'slurm').mkdir()
    (batch / 'cells').mkdir()
    # This is the ONLY shared-ledger mutation in the entire launch.
    CostLedger(str(ledger)).reserve(
        manifest['parent_run_id'], amount,
        note=f'{batch}; parent allocation, do not settle while jobs run')
    for cell in manifest['cells']:
        directory = batch / 'cells' / str(cell['index'])
        directory.mkdir()
        if cell['allocation_usd']:
            CostLedger.initialize(
                str(directory / 'budget.json'), cell['allocation_usd'],
                note=f'Funded by {manifest["parent_run_id"]}; '
                     'do not add this allowance to the parent total')
    # A partial preparation must never start a worker.
    (batch / 'READY').write_text(commit + '\n', encoding='utf-8')
    return batch


def run_cell(batch, index):
    """Claim and execute one immutable, funded cell, without resumption."""
    batch = Path(batch).resolve()
    if not (batch / 'READY').is_file():
        raise RuntimeError('Batch preparation did not finish; no dispatch.')
    if (batch / 'STOP').exists():
        raise RuntimeError('An earlier cell failed; review before spending.')
    if digest(batch / 'manifest.json') != (
            batch / 'manifest.sha256').read_text().strip():
        raise RuntimeError('Manifest changed after preparation.')
    manifest = json.loads((batch / 'manifest.json').read_text())
    snapshot = batch / 'code'
    if not 0 <= index < len(manifest['cells']):
        raise ValueError('Cell index is outside this manifest.')
    if Path(__file__).resolve().parents[1] != snapshot:
        raise RuntimeError('Run the worker from the frozen code directory.')
    for name, checksum in manifest['snapshot_hashes'].items():
        if digest(snapshot / name) != checksum:
            raise RuntimeError(f'Frozen source changed: {name}')
    status = CostLedger(manifest['parent_ledger']).status()
    holds = [e for e in status['open_reservations']
             if e['run_id'] == manifest['parent_run_id']]
    if (len(holds) != 1 or holds[0]['reserved'] <
            manifest['grid_reservation_usd'] or status['overspent_runs']
            or status['available_usd'] < 0):
        raise BudgetError('The whole-grid parent reservation is not intact.')
    cell = manifest['cells'][index]
    directory = batch / 'cells' / str(index)
    if cell['allocation_usd']:
        child = CostLedger(str(directory / 'budget.json')).status()
        if (child['allowance_usd'] != cell['allocation_usd']
                or child['open_reservations'] or child['settled_usd']):
            raise BudgetError('Child allocation changed or was already used.')
    # mkdir is one atomic operation, not a read-modify-write ledger
    # transaction. Requeue/duplicate sbatch must not buy a second run.
    (directory / 'CLAIMED').mkdir()
    values = dict(cell['args'])
    values['price_table'] = str(snapshot / PRICES)
    values['embed_cache'] = str(directory / 'embeddings.json')
    values['budget_ledger'] = (str(directory / 'budget.json')
                               if cell['allocation_usd'] else '')
    env = environment(manifest['env_file'])
    env['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'
    env['PYTHONPATH'] = str(snapshot)
    # Pin these before either client is constructed, not only after
    # the trainer has reserved. Never inherit another run's retries.
    env['LLM_MAX_ATTEMPTS'] = str(values['max_attempts'])
    env['LLM_MAX_SDK_RETRIES'] = '0'
    command = [sys.executable, '-u', '-m', 'algos.ppo_distill',
               *trainer_argv(values)]
    write_json(directory / 'dispatch.json', {
        'index': index, 'commit': manifest['commit'], 'args': values,
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
    })
    result = subprocess.run(command, cwd=snapshot, env=env)
    write_json(directory / 'exit.json', {'returncode': result.returncode})
    if result.returncode:
        (batch / 'STOP').touch(exist_ok=True)
    return result.returncode


def main():
    """Default to a read-only preview; preparation never submits Slurm."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run-cell', type=int)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--config', default=CONFIG,
                        help='tracked experiment config inside this repo')
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    args = parser.parse_args()
    if args.run_cell is not None:
        if args.batch is None:
            parser.error('--run-cell needs --batch')
        try:
            return run_cell(args.batch, args.run_cell)
        except Exception:
            # New workers stop; already running cells may finish from
            # their existing allocations. No automatic paid retry.
            (args.batch / 'STOP').touch(exist_ok=True)
            raise
    manifest = make_manifest(config_path=args.config)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'cells'},
                     indent=2))
    for cell in manifest['cells']:
        print(f'{cell["index"]:2d}  {cell["arm"]} seed={cell["seed"]} '
              f'allocation=${cell["allocation_usd"]:.2f}')
    if not (args.check or args.prepare):
        return 0
    # Parse every actual CLI without initializing a teacher/client.
    import tyro
    from algos.ppo_distill import Args
    for cell in manifest['cells']:
        tyro.cli(Args, args=trainer_argv(cell['args']))
    environment(ROOT / '.env')
    status = check_pool(args.ledger, manifest['grid_reservation_usd'])
    print(f'Pool free: ${status["available_usd"]:.2f}; credentials '
          f'present; all {len(manifest["cells"])} trainer commands parse. '
          'No API calls.')
    if args.prepare:
        batch = prepare(ROOT, args.ledger, manifest)
        print('Prepared; submit once with:')
        job_option = ([f'--job-name={manifest["job_name"]}']
                      if manifest.get('job_name') else [])
        print(shlex.join([
            'sbatch', *job_option,
            f'--array=0-{len(manifest["cells"]) - 1}',
            f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch / 'code/scripts/submit_explanation_grid.sh'),
            str(batch),
        ]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
