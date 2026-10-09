"""Freeze, fund and dispatch the advising-strategy by count-bonus pilot.

Preparation makes no API call and never submits Slurm. The printed array
command is the only supported launch path. OpenAI grids reserve every allowed
attempt before dispatch; Aleph grids are free and use a separate credential.
"""

import argparse
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import zipfile

from dotenv import dotenv_values

from scripts.run_explanation_grid import digest, trainer_argv, write_json
from teachers.budget import (
    BudgetError, CostLedger, PriceTable, reservation_for)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/advising_strength_gpt5mini_pilot_v1.json'
PRICES = 'configs/explanation_prices_2026-09-10.json'
WORKER = 'scripts/submit_advising_strength_grid.sh'
RUNNER = 'scripts/run_advising_strength_grid.py'
# User-authorized total ceiling, including earlier spending and holds.
# Increasing a ledger's allowance is not a new spending authorization.
# User added250USD on2026-09-24 to the existing203.13USD ceiling.
AUTHORIZED_ALLOWANCE_USD = 453.13
STRATEGIES = {
    'none': dict(guidance=False, advisor='unlimited',
                 importance_source='none', mistake_threshold=0.0),
    'early': dict(guidance=True, advisor='early',
                  importance_source='none', mistake_threshold=0.0),
    'entropy': dict(guidance=True, advisor='importance',
                    importance_source='entropy', mistake_threshold=0.0),
    # This is exact sampled-action disagreement, gated by the same student
    # entropy signal as the importance arm. It is not a free oracle peek.
    'mistake_exact': dict(guidance=True, advisor='mistake',
                          importance_source='entropy',
                          mistake_threshold=0.0),
}


def _load_config(root, config_path):
    root = Path(root).resolve()
    relative = (root / config_path).resolve().relative_to(root).as_posix()
    return relative, json.loads((root / relative).read_text(encoding='utf-8'))


def make_manifest(root=ROOT, config_path=CONFIG):
    """Resolve all Args defaults and worst-case funding without a client."""
    from algos.ppo_distill import Args

    root = Path(root).resolve()
    config_path, config = _load_config(root, config_path)
    provider = config['provider']
    if provider not in ('openai', 'aleph'):
        raise ValueError('provider must be openai or aleph')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['experiment_id']):
        raise ValueError('experiment_id must be a safe directory name')
    seeds = config['seeds']
    bonuses = config['student_bonuses']
    strategies = config['strategies']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError('seeds must be distinct nonnegative integers')
    if bonuses != ['none', 'count']:
        raise ValueError('pilot requires student_bonuses [none, count]')
    if (strategies != list(STRATEGIES)
            or len(set(strategies)) != len(strategies)):
        raise ValueError('pilot requires none/early/entropy/mistake_exact')
    if (type(config['query_budget']) is not int
            or config['query_budget'] <= 0
            or config['advice_budget'] != config['query_budget']):
        raise ValueError('positive matched query/advice caps are required')
    # Zero leaves scheduling to Slurm.
    concurrency = config.get('array_concurrency', 0)
    if type(concurrency) is not int or concurrency < 0:
        raise ValueError('array_concurrency must be nonnegative; 0 is uncapped')

    prices = PriceTable.load(str(root / PRICES))
    cells = []
    for seed in seeds:
        for bonus in bonuses:
            for strategy in strategies:
                options = STRATEGIES[strategy]
                guided = options['guidance']
                args = Args(
                    **config['shared'], seed=seed, bonus=bonus,
                    experiment_id=config['experiment_id'],
                    teacher='llm_general' if guided else 'bot',
                    teacher_model=config['teacher_model'],
                    teacher_reasoning_effort=(
                        config['teacher_reasoning_effort'] if guided else ''),
                    query_budget=config['query_budget'] if guided else 0,
                    advice_budget=config['advice_budget'] if guided else 0,
                    **options,
                )
                if args.query_windows != 0 or args.explanation != 'none':
                    raise ValueError('This pilot uses native advisor timing '
                                     'and no head')
                if args.audit_explanations is not True:
                    raise ValueError('Pilot requires initial-policy recording '
                                     'via audit_explanations')
                bound = (reservation_for(args, None, prices)[0]
                         if provider == 'openai' and guided else 0.0)
                cells.append({
                    'index': len(cells), 'seed': seed, 'bonus': bonus,
                    'strategy': strategy, 'provider': provider,
                    'args': asdict(args), 'worst_case_usd': bound,
                    'allocation_usd': math.ceil(bound * 100) / 100,
                })
    return {
        'experiment_id': config['experiment_id'],
        'author': config['author'], 'date': config['date'],
        'scope': config['scope'], 'provider': provider,
        'teacher_model': config['teacher_model'],
        'teacher_reasoning_effort': config['teacher_reasoning_effort'],
        'config_path': config_path,
        'config_sha256': digest(root / config_path),
        'prices_sha256': digest(root / PRICES),
        'timing': 'native advisor timing; no fixed query windows',
        'query_budget_per_guided_run': config['query_budget'],
        'advice_budget_per_guided_run': config['advice_budget'],
        'array_concurrency': concurrency, 'cells': cells,
        'grid_reservation_usd': round(sum(
            c['allocation_usd'] for c in cells), 2),
        'consultation_cap': sum(c['args']['query_budget'] for c in cells),
    }


def backend_environment(provider, credential_file, inherited=None):
    """Load one credential as dotenv data, never as shell code."""
    result = dict(os.environ if inherited is None else inherited)
    values = {}
    path = Path(credential_file)
    if path.is_file():
        values = dotenv_values(path, encoding='utf-8-sig', interpolate=False)
    if provider == 'openai':
        key = (result.get('OPENAI_API_KEY') or
               values.get('OPENAI_API_KEY') or '').strip()
        if not key:
            raise ValueError('OPENAI_API_KEY is missing')
        result.update(OPENAI_API_KEY=key, LLM_PROVIDER='openai',
                      MODEL='gpt-5-mini',
                      OPENAI_BASE_URL='https://api.openai.com/v1')
    else:
        key = (result.get('TYK_KEY') or result.get('ALEPH_API_KEY') or
               values.get('TYK_KEY') or values.get('ALEPH_API_KEY')
               or '').strip()
        if not key:
            raise ValueError('TYK_KEY/ALEPH_API_KEY is missing')
        result.update(TYK_KEY=key, LLM_PROVIDER='aleph',
                      ALEPH_BASE_URL='https://inference.vulcan.alliancecan.ca/v1')
    result['LLM_MAX_SDK_RETRIES'] = '0'
    return result


def check_pool(path, amount):
    """Read the shared paid pool without changing any earlier hold."""
    status = CostLedger(str(path)).status()
    allowance = status['allowance_usd']
    if (type(allowance) not in (int, float) or not math.isfinite(allowance)
            or not 0 < allowance <= AUTHORIZED_ALLOWANCE_USD):
        raise BudgetError(
            'Pool allowance must be finite, positive and at most '
            f'${AUTHORIZED_ALLOWANCE_USD:.2f} authorized total')
    if (type(amount) not in (int, float) or not math.isfinite(amount)
            or amount < 0):
        raise BudgetError('Admission amount must be finite and nonnegative')
    if any(type(status[k]) not in (int, float)
           or not math.isfinite(status[k]) or status[k] < 0
           for k in ('settled_usd', 'reserved_usd', 'available_usd')):
        raise BudgetError('Pool contains invalid or negative accounting')
    if status['overspent_runs'] or status['available_usd'] < amount:
        raise BudgetError(f'Grid needs ${amount:.2f}; pool has '
                          f'${status["available_usd"]:.2f} free')
    return status


def prepare(root, ledger, manifest):
    """Create an immutable batch and reserve the whole paid grid once."""
    root, ledger = Path(root).resolve(), Path(ledger).resolve()
    subprocess.run(['git', 'diff', '--exit-code', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    amount = manifest['grid_reservation_usd']
    batch = root / 'results/advising_strength' / manifest['experiment_id']
    if batch.exists():
        raise FileExistsError(f'{batch} already exists; do not prepare twice')
    if amount:
        check_pool(ledger, amount)
    batch.parent.mkdir(parents=True, exist_ok=True)
    batch.mkdir()
    snapshot = batch / 'code'
    snapshot.mkdir()
    archive = subprocess.check_output(
        ['git', 'archive', '--format=zip', commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        handle.extractall(snapshot)
    hashes = {p.relative_to(snapshot).as_posix(): digest(p)
              for p in snapshot.rglob('*') if p.is_file()}
    for required in (manifest['config_path'], PRICES, RUNNER, WORKER):
        if required not in hashes:
            raise RuntimeError(f'{required} is absent from HEAD')
    manifest['config_sha256'] = hashes[manifest['config_path']]
    manifest['prices_sha256'] = hashes[PRICES]
    credential = (root / '.env' if manifest['provider'] == 'openai'
                  else Path.home() / '.aleph_tyk.env')
    manifest.update(
        commit=commit, snapshot_hashes=hashes,
        parent_ledger=str(ledger) if amount else None,
        parent_run_id=(('grid:' + manifest['experiment_id'])
                       if amount else None),
        credential_file=str(credential),
    )
    write_json(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(
        digest(batch / 'manifest.json') + '\n', encoding='utf-8')
    (batch / 'slurm').mkdir()
    (batch / 'cells').mkdir()
    if amount:
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
    (batch / 'READY').write_text(commit + '\n', encoding='utf-8')
    return batch


def run_cell(batch, index):
    """Claim one cell and run it from the immutable snapshot."""
    batch = Path(batch).resolve()
    if not (batch / 'READY').is_file() or (batch / 'STOP').exists():
        raise RuntimeError('Batch is incomplete or stopped')
    if digest(batch / 'manifest.json') != (
            batch / 'manifest.sha256').read_text().strip():
        raise RuntimeError('Manifest changed after preparation')
    manifest = json.loads((batch / 'manifest.json').read_text())
    snapshot = batch / 'code'
    if not 0 <= index < len(manifest['cells']):
        raise ValueError('Cell index is outside this manifest')
    if Path(__file__).resolve().parents[1] != snapshot:
        raise RuntimeError('Run the worker from the frozen code directory')
    for name, checksum in manifest['snapshot_hashes'].items():
        if digest(snapshot / name) != checksum:
            raise RuntimeError(f'Frozen source changed: {name}')
    if manifest['grid_reservation_usd']:
        # The grid is already held, so recheck the ceiling without asking
        # for a second allocation. This also catches a changed pool.
        status = check_pool(manifest['parent_ledger'], 0.0)
        holds = [h for h in status['open_reservations']
                 if h['run_id'] == manifest['parent_run_id']]
        if (len(holds) != 1
                or holds[0]['reserved'] < manifest['grid_reservation_usd']
                or status['overspent_runs'] or status['available_usd'] < 0):
            raise BudgetError('Whole-grid parent reservation is not intact')
    cell = manifest['cells'][index]
    directory = batch / 'cells' / str(index)
    if cell['allocation_usd']:
        child = CostLedger(str(directory / 'budget.json')).status()
        if (child['allowance_usd'] != cell['allocation_usd']
                or child['open_reservations'] or child['settled_usd']):
            raise BudgetError('Child allocation changed or was already used')
    (directory / 'CLAIMED').mkdir()
    values = dict(cell['args'])
    values['price_table'] = str(snapshot / PRICES)
    values['embed_cache'] = str(directory / 'unused_embeddings.json')
    values['budget_ledger'] = (str(directory / 'budget.json')
                               if cell['allocation_usd'] else '')
    env = backend_environment(
        manifest['provider'], manifest['credential_file'])
    env['PYTHONPATH'] = str(snapshot)
    env['LLM_MAX_ATTEMPTS'] = str(values['max_attempts'])
    env['LLM_MAX_INPUT_TOKENS'] = str(values['max_input_tokens'])
    env['LLM_MAX_OUTPUT_TOKENS'] = str(values['max_output_tokens'])
    if manifest['provider'] == 'aleph':
        env['ALEPH_REASONING_EFFORT'] = manifest['teacher_reasoning_effort']
        env['ALEPH_MAX_OUTPUT_TOKENS'] = str(values['max_output_tokens'])
    command = [sys.executable, '-u', '-m', 'algos.ppo_distill',
               *trainer_argv(values)]
    write_json(directory / 'dispatch.json', {
        'index': index, 'commit': manifest['commit'],
        'provider': cell['provider'],
        'bonus': cell['bonus'], 'strategy': cell['strategy'], 'args': values,
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
    })
    result = subprocess.run(command, cwd=snapshot, env=env)
    write_json(directory / 'exit.json', {'returncode': result.returncode})
    if result.returncode:
        (batch / 'STOP').touch(exist_ok=True)
    return result.returncode


def main():
    """Preview/check/prepare a grid, or execute one frozen cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run-cell', type=int)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    args = parser.parse_args()
    if args.run_cell is not None:
        if args.batch is None:
            parser.error('--run-cell needs --batch')
        try:
            return run_cell(args.batch, args.run_cell)
        except Exception:
            (args.batch / 'STOP').touch(exist_ok=True)
            raise
    manifest = make_manifest(config_path=args.config)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'cells'},
                     indent=2))
    for cell in manifest['cells']:
        print(f'{cell["index"]:2d} {cell["provider"]} seed={cell["seed"]} '
              f'bonus={cell["bonus"]} strategy={cell["strategy"]} '
              f'allocation=${cell["allocation_usd"]:.2f}')
    if not (args.check or args.prepare):
        return 0
    import tyro
    from algos.ppo_distill import Args
    for cell in manifest['cells']:
        parsed = tyro.cli(Args, args=trainer_argv(cell['args']))
        if asdict(parsed) != cell['args']:
            raise ValueError('Trainer CLI did not preserve resolved Args')
    credential = (ROOT / '.env' if manifest['provider'] == 'openai'
                  else Path.home() / '.aleph_tyk.env')
    backend_environment(manifest['provider'], credential)
    if manifest['grid_reservation_usd']:
        status = check_pool(args.ledger, manifest['grid_reservation_usd'])
        funding = f'pool free ${status["available_usd"]:.2f}'
    else:
        funding = 'free Aleph backend; shared dollar ledger unchanged'
    print(f'Credentials present; {funding}; all {len(manifest["cells"])} '
          'commands parse. No API calls.')
    if args.prepare:
        batch = prepare(ROOT, args.ledger, manifest)
        last = len(manifest['cells']) - 1
        concurrent = manifest['array_concurrency']
        array = f'0-{last}' + (f'%{concurrent}' if concurrent else '')
        print('Prepared; submit once with:')
        print(shlex.join([
            'sbatch', f'--array={array}',
            f'--output={batch}/slurm/%x_%A_%a.out',
            str(batch / 'code' / WORKER), str(batch),
        ]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
