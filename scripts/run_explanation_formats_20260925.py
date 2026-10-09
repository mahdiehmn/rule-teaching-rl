"""Freeze, gate and run the explanation-format and timing studies.

Protocol, frozen before any collection
or learning outcome: research/explanation_formats_protocol_2026-09-25.md.

Two batches, each archived from one committed revision and executed only
from that snapshot. Submission and every paid request are user-owned.

    timing   free, 30 cells: 5 seeds x 2 backgrounds x {PPO, action-only
             early, action-only late}. The cached categorical lesson bank
             supplies images and action pairs only.
    formats  paid (about $4.9 worst case), one collection per explanation
             writer, then 170 learning cells: 120 primary (full-state
             writer; PPO, action-only and five formats x aligned/permuted)
             and 50 access-comparison cells (local-only writer, aligned).

One user command submits the whole formats chain; Slurm orders it:

    writer full_state: collect -> freeze -> gate -> calibrate (one job)
        afterok -> formats array 0-119
        afterok -> writer local_only: collect -> freeze -> gate (one job)
                       afterok -> access array 0-49

The two writer jobs run one after the other, so their ledger admissions
never coincide (teachers/budget.py). Each collection reserves its
writer's full chat bound before its first request, makes one attempt per
case with zero SDK retries, and settles known costs plus the full bound
of any unknown-cost attempt; each freeze reserves and settles its
embedding bound. A failed gate cancels only its dependents. Banks, gate
records and the calibration are written once and re-verified by every
learning cell; nothing is replaced, resubmitted or retried automatically.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

import numpy as np
import torch
import tyro

from algos.ppo_lesson_formats import Args, lesson_schedule
from algos.ppo_progress import policy_hash
from scripts import collect_explanation_formats as collection
from scripts import plan_explanation_formats as plan
from scripts.calibrate_explanation_formats import calibrate
from scripts.explanation_formats import ACCESS, FORMATS, request_identity
from scripts.run_explanation_grid import trainer_argv
from teachers.budget import BudgetError, PriceTable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'research/explanation_formats_protocol_2026-09-25.md'
RUNNER = 'scripts/run_explanation_formats_20260925.py'
WORKER = 'scripts/submit_explanation_formats.sh'
REQUIRED_SOURCES = (
    PROTOCOL, RUNNER, WORKER, 'scripts/launch_explanation_formats.sh',
    'algos/ppo_lesson_formats.py', 'algos/ppo_distill.py',
    'scripts/explanation_formats.py', 'scripts/collect_explanation_formats.py',
    'scripts/calibrate_explanation_formats.py',
    'scripts/plan_explanation_formats.py', collection.PRICES,
    'tests/test_explanation_formats.py',
    'tests/test_explanation_formats_launch.py',
)
TIMING = 'explanation_timing_20260925_v1'
FORMATS_BATCH = 'explanation_formats_20260925_v1'
SOURCE_STUDY = 'contrastive_lessons_20260924_v1'
PANEL_SHA256 = (
    '71825bc9254feac2ffb6f0952259f2865984ff369c63ea0c5c2fca2574a9225a')
LESSONS_SHA256 = (
    '594983dc4e9b91d193d6ddcb549c3f89b5ab7cbbcfaa68f2796150af638765f0')
DERIVED = dict(batch_size=1024, minibatch_size=256, num_iterations=9765)
# Technical floors frozen before collection. A bank with fewer cases known
# in every format needs inspection before any learning run. The primary
# floor also guarantees the 64 calibration cases (120 + 149 - 192 >= 64).
GATES = dict(full_state=dict(train=120, audit=40),
             local_only=dict(train=48, audit=16))
EMBEDDING_DIMENSION = 1536
CALIBRATION_SEEDS = plan.SEEDS
CALIBRATION_SIZE = 64
SUITES = ('timing', 'formats', 'access')
WRITER_ORDER = ('full_state', 'local_only')
SUBMISSION = (
    # name, stage, target, array, dependency
    ('writer_full_state', 'writer', 'full_state', None, None),
    ('formats', 'cell', 'formats', '0-119', 'writer_full_state'),
    ('writer_local_only', 'writer', 'local_only', None, 'writer_full_state'),
    ('access', 'cell', 'access', '0-49', 'writer_local_only'),
)


# ----------------------------------------------------------------- helpers

def digest(path):
    """Identify exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    """Serialize records identically on every host."""
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def write(path, value, exclusive=False):
    """Write a JSON record; `exclusive` refuses to replace an existing one."""
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write(encoded(value))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalized(value):
    """Compare contracts as they survive a JSON round trip."""
    return json.loads(json.dumps(value))


def read_jsonl(path):
    """Refuse a torn terminal journal rather than dropping its tail."""
    payload = Path(path).read_bytes()
    if not payload or not payload.endswith(b'\n'):
        raise ValueError(f'Empty or unterminated terminal journal: {path}')
    return [json.loads(line) for line in payload.decode('utf-8').splitlines()]


def within(base, relative):
    """Resolve portable artifact names while rejecting directory escapes."""
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Artifact names must use relative POSIX paths')
    path = (Path(base) / relative).resolve()
    path.relative_to(Path(base).resolve())
    return path


def runtime_identity():
    """Workers must run the preparation interpreter and packages."""
    return dict(python=platform.python_version(), packages={
        name: version(name) for name in
        ('numpy', 'torch', 'gymnasium', 'minigrid', 'tyro', 'openai',
         'httpx')})


def pinned_prices(prices):
    if (prices.chat_rates(collection.MODEL) != (0.25, 2.0)
            or prices.embedding_rate(collection.EMBED_MODEL) != 0.02):
        raise BudgetError('Reviewed GPT-5-mini or embedding prices changed')


def check_commands(cells):
    """Every listed cell must survive the real trainer command line."""
    for cell in cells:
        parsed = tyro.cli(Args, args=trainer_argv(cell['args']),
                          console_outputs=False)
        if asdict(parsed) != cell['args']:
            raise ValueError('Command-line parsing changed frozen arguments')


def placeholder_cells():
    """One seed's distinct commands with placeholder banks and scales."""
    scales = {f: 1.0 for f in FORMATS}
    return (plan.format_menu('banks/full_state/format_bank.json', '0' * 64,
                             scales)[:24]
            + plan.access_menu('banks/local_only/format_bank.json',
                               '1' * 64, scales)[:10])


# --------------------------------------------------------------- contracts

def timing_contract():
    return dict(
        study=TIMING, suite='timing', protocol=PROTOCOL,
        source_study=SOURCE_STUDY, lessons_sha256=LESSONS_SHA256,
        cells=plan.timing_menu('corpora/lessons.json', LESSONS_SHA256),
        training_seeds=plan.SEEDS, backgrounds=list(plan.BACKGROUNDS),
        schedules=plan.schedule_summary(),
        exposures=plan.TIMING_EXPOSURES, derived_args=DERIVED, api_calls=0,
        primary='teacher_off_greedy_auc_over_all_observed_evaluations')


def formats_contract(panel, prices):
    identities = {a: request_identity(
        panel, a, model=collection.MODEL, effort=collection.EFFORT,
        max_output_tokens=collection.MAX_OUTPUT_TOKENS) for a in ACCESS}
    bounds = {a: collection.price(panel, prices, a) for a in ACCESS}
    return dict(
        study=FORMATS_BATCH, suite='formats', protocol=PROTOCOL,
        source_study=SOURCE_STUDY, panel_sha256=PANEL_SHA256,
        lessons_sha256=LESSONS_SHA256, writers=list(WRITER_ORDER),
        primary_access='full_state', model=collection.MODEL,
        effort=collection.EFFORT,
        max_output_tokens=collection.MAX_OUTPUT_TOKENS,
        embedding_model=collection.EMBED_MODEL, attempts_per_case=1,
        sdk_retries=0, identities=identities, bounds=bounds,
        total_bound_usd=sum(b['total_bound_usd'] for b in bounds.values()),
        gates=GATES, embedding_dimension=EMBEDDING_DIMENSION,
        pairing='none: each writer is frozen on its own all-format '
                'intersection',
        calibration=dict(bank='full_state', seeds=CALIBRATION_SEEDS,
                         size=CALIBRATION_SIZE,
                         applies_to=['formats', 'access']),
        suites=dict(formats=120, access=50),
        submission=[list(s) for s in SUBMISSION],
        training_seeds=plan.SEEDS, backgrounds=list(plan.BACKGROUNDS),
        schedule=dict(offset=0, every=4, exposures=1830),
        derived_args=DERIVED,
        primary='teacher_off_greedy_auc_over_all_observed_evaluations')


# ------------------------------------------------------------- preparation

def archive(root, batch):
    """Extract the committed revision; uncommitted source is refused."""
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    payload = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(payload)) as handle:
        if not set(REQUIRED_SOURCES) <= set(handle.namelist()):
            raise ValueError('Commit the complete formats packet first')
        # Readiness is written last. A partial preparation stays visible
        # and cannot be mistaken for an admitted batch.
        batch.mkdir(parents=True, exist_ok=False)
        handle.extractall(batch / 'code')
    return commit


def seal(batch, manifest, commit, artifacts):
    manifest.update(
        commit=commit, runtime=runtime_identity(),
        source_hashes={p.relative_to(batch / 'code').as_posix(): digest(p)
                       for p in (batch / 'code').rglob('*') if p.is_file()},
        artifact_hashes={name: digest(batch / name) for name in artifacts})
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(commit + '\n')
    return verify(batch)


def prepare_timing(root, source):
    """Free batch: archived code plus the cached categorical lessons."""
    root = Path(root).resolve()
    batch = root / 'results/explanation_lessons' / TIMING
    if batch.exists():
        verify(batch)
        return batch
    lessons = Path(source) / 'lessons.json'
    if digest(lessons) != LESSONS_SHA256:
        raise ValueError('Cached lesson bank checksum differs')
    contract = timing_contract()
    check_commands(contract['cells'][:6])
    commit = archive(root, batch)
    for name in ('corpora', 'cells', 'slurm'):
        (batch / name).mkdir()
    (batch / 'corpora/lessons.json').write_bytes(lessons.read_bytes())
    seal(batch, contract, commit, ('corpora/lessons.json',))
    return batch


def prepare_formats(root, source, ledger, credential):
    """Paid batch: price both writers and check the live pool read-only."""
    root = Path(root).resolve()
    batch = root / 'results/explanation_lessons' / FORMATS_BATCH
    if batch.exists():
        verify(batch)
        return batch
    source = Path(source)
    for name, checksum in (('panel.json', PANEL_SHA256),
                           ('lessons.json', LESSONS_SHA256)):
        if digest(source / name) != checksum:
            raise ValueError(f'{name} checksum differs from the frozen source')
    panel = read(source / 'panel.json')
    prices = PriceTable.load(str(root / collection.PRICES))
    pinned_prices(prices)
    contract = formats_contract(panel, prices)
    funds = collection.check_ledger(ledger, contract['total_bound_usd'])
    if not funds['sufficient']:
        raise BudgetError(
            f"Both writers need ${contract['total_bound_usd']:.2f} worst "
            f"case; only ${funds['available_usd']:.2f} is free")
    from scripts.run_advising_strength_grid import backend_environment
    backend_environment('openai', credential)   # presence only; not stored
    check_commands(placeholder_cells())
    commit = archive(root, batch)
    for name in ('source', 'collect', 'banks', 'cells', 'slurm'):
        (batch / name).mkdir()
    for name in ('panel.json', 'lessons.json'):
        (batch / 'source' / name).write_bytes((source / name).read_bytes())
    contract.update(ledger=str(Path(ledger).resolve()),
                    credential_file=str(Path(credential).resolve()))
    seal(batch, contract, commit, ('source/panel.json', 'source/lessons.json'))
    return batch


def verify(batch):
    """Rebuild the frozen contract before every stage and every cell."""
    batch = Path(batch).resolve()
    if (not (batch / 'READY').is_file()
            or digest(batch / 'manifest.json')
            != (batch / 'manifest.sha256').read_text().strip()):
        raise ValueError('Incomplete or modified manifest')
    manifest = read(batch / 'manifest.json')
    if ((batch / 'READY').read_text().strip() != manifest['commit']
            or not set(REQUIRED_SOURCES) <= set(manifest['source_hashes'])):
        raise ValueError('Incomplete frozen source inventory')
    for hashes, base in ((manifest['source_hashes'], batch / 'code'),
                         (manifest['artifact_hashes'], batch)):
        for name, checksum in hashes.items():
            if digest(within(base, name)) != checksum:
                raise ValueError(f'Frozen artifact changed: {name}')
    if manifest.get('suite') == 'timing':
        expected = timing_contract()
        artifacts = {'corpora/lessons.json': LESSONS_SHA256}
    elif manifest.get('suite') == 'formats':
        prices = PriceTable.load(str(batch / 'code' / collection.PRICES))
        pinned_prices(prices)
        expected = formats_contract(read(batch / 'source/panel.json'), prices)
        artifacts = {'source/panel.json': PANEL_SHA256,
                     'source/lessons.json': LESSONS_SHA256}
    else:
        raise ValueError('Unknown batch kind')
    if (manifest['artifact_hashes'] != artifacts
            or any(manifest.get(k) != v
                   for k, v in normalized(expected).items())):
        raise ValueError('Frozen cells, identities or contract differ')
    return manifest


def check_worker(batch, manifest):
    if ROOT.resolve() != (Path(batch) / 'code').resolve():
        raise ValueError('Run workers from the archived code directory')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker runtime differs from preparation')


# ------------------------------------------------------ writers and gates

def gate_bank(bank, access):
    """Technical floors only; semantic quality is the independent audit."""
    floor = GATES[access]
    checks = dict(
        train=bank['train_cases'] >= floor['train'],
        audit=bank['audit_cases'] >= floor['audit'],
        embedding_dimension=(bank['embedding_dimension']
                             == EMBEDDING_DIMENSION),
        information_access=bank['information_access'] == access,
        unpaired=bank['paired_restriction'] is False,
    )
    if access == 'full_state':
        # The permuted control must actually move targets in every format.
        checks['permutation_changes_every_format'] = all(
            bank['permuted_changed_fraction'][f] > 0 for f in FORMATS)
    return dict(floor=floor, checks=checks, passed=all(checks.values()))


def frozen(batch, access, commit):
    """A writer's sealed bank, gate and calibration, re-verified."""
    batch = Path(batch)
    path = batch / 'banks' / access / 'FROZEN.json'
    if not path.is_file():
        raise ValueError(f'No frozen {access} bank; its writer job has not '
                         'passed')
    record = read(path)
    if (digest(path) != path.with_suffix('.sha256').read_text().strip()
            or record['access'] != access or record['commit'] != commit
            or not record['gate']['passed']
            or digest(within(batch, record['bank'])) != record['bank_sha256']
            or digest(within(batch, record['calibration']))
            != record['calibration_sha256']
            or read(within(batch, record['calibration']))['primary_scale']
            != record['primary_scale']):
        raise ValueError(f'The frozen {access} record is modified')
    if (read(within(batch, record['bank'])).get('information_access')
            != access):
        raise ValueError(f'The frozen {access} bank has another access')
    return record


def writer_stage(batch, access, client=None):
    """Collect, freeze and gate one writer; the primary also calibrates."""
    batch = Path(batch).resolve()
    manifest = verify(batch)
    if manifest['suite'] != 'formats' or access not in ACCESS:
        raise ValueError('Writers belong to the formats batch')
    check_worker(batch, manifest)
    if access == 'local_only':
        # The comparison reuses the primary scales: never without them.
        frozen(batch, 'full_state', manifest['commit'])
    record_path = batch / 'banks' / access / 'FROZEN.json'
    if record_path.exists():
        return frozen(batch, access, manifest['commit'])
    panel = read(batch / 'source/panel.json')
    prices = PriceTable.load(str(batch / 'code' / collection.PRICES))
    pinned_prices(prices)
    collected = batch / 'collect' / access
    out = batch / 'banks' / access
    owned = client is None
    if owned:
        client = collection.openai_client(manifest['credential_file'])
    try:
        if not (collected / 'collection_summary.json').exists():
            if collected.exists():
                raise FileExistsError(
                    f'Partial {access} collection exists; inspect it, '
                    'never retry automatically')
            collection.collect(panel, client, prices, manifest['ledger'],
                               collected, workers=8, access=access)
        summary = read(collected / 'collection_summary.json')
        replies = collection.replies_by_case(collected / 'raw_replies.jsonl')
        identity = manifest['identities'][access]
        if (summary['request_sha256'] != identity['request_sha256']
                or summary['information_access'] != access
                or summary['requests'] != len(panel['cases'])
                or set(replies) != {c['case_id'] for c in panel['cases']}):
            raise ValueError('Collection differs from its frozen requests')
        # An earlier partial freeze blocks: inspect rather than re-embed.
        out.mkdir(parents=True, exist_ok=False)
        bank, cost = collection.embed_and_freeze(
            panel, replies, client, prices, manifest['ledger'], out, access)
    finally:
        if owned:
            client.close()
    record = dict(
        access=access, commit=manifest['commit'],
        bank=f'banks/{access}/format_bank.json',
        bank_sha256=digest(out / 'format_bank.json'),
        raw_replies_sha256=digest(collected / 'raw_replies.jsonl'),
        collection=summary, embeddings=cost,
        train_cases=bank['train_cases'], audit_cases=bank['audit_cases'],
        coverage=bank['coverage'],
        train_plan_next_hidden_fraction=bank[
            'train_plan_next_hidden_fraction'],
        permuted_changed_fraction=bank['permuted_changed_fraction'],
        gate=gate_bank(bank, access),
        frozen_at=datetime.now(timezone.utc).isoformat())
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ('coverage', 'collection')}, indent=2),
          flush=True)
    if not record['gate']['passed']:
        write(out / 'GATE_FAILED.json', record, exclusive=True)
        raise ValueError(f'The {access} bank failed its frozen gate; its '
                         'learning cells are withheld')
    if access == 'full_state':
        result = calibrate(out / 'format_bank.json',
                           batch / 'source/lessons.json',
                           CALIBRATION_SEEDS, CALIBRATION_SIZE)
        scales = result['primary_scale']
        if (set(scales) != set(FORMATS) or not all(
                math.isfinite(v) and v > 0 for v in scales.values())):
            raise ValueError('Calibration produced an invalid scale')
        write(batch / 'calibration.json', result, exclusive=True)
        print(json.dumps(dict(primary_scale=scales,
                              ratio_spread=result['ratio_spread']),
                         indent=2), flush=True)
    record.update(calibration='calibration.json',
                  calibration_sha256=digest(batch / 'calibration.json'),
                  primary_scale=read(batch / 'calibration.json')[
                      'primary_scale'])
    write(record_path, record, exclusive=True)
    record_path.with_suffix('.sha256').write_text(digest(record_path))
    return frozen(batch, access, manifest['commit'])


# ------------------------------------------------------------ learning cells

def cells_for(batch, manifest, suite):
    """Frozen cells; paid suites bind their bank and scales at dispatch."""
    if suite == 'timing':
        if manifest['suite'] != 'timing':
            raise ValueError('Timing cells belong to the timing batch')
        return manifest['cells']
    if manifest['suite'] != 'formats' or suite not in ('formats', 'access'):
        raise ValueError('Unknown suite for this batch')
    full = frozen(batch, 'full_state', manifest['commit'])
    if suite == 'formats':
        return plan.format_menu(full['bank'], full['bank_sha256'],
                                full['primary_scale'])
    local = frozen(batch, 'local_only', manifest['commit'])
    if local['calibration_sha256'] != full['calibration_sha256']:
        raise ValueError('The access comparison must reuse primary scales')
    return plan.access_menu(local['bank'], local['bank_sha256'],
                            full['primary_scale'])


def resolved_args(batch, cell):
    """Bind the relative bank path only when dispatching."""
    values = dict(cell['args'])
    values['format_bank'] = str(within(batch, values['format_bank']))
    return Args(**values)


def derived_of(args):
    size = args.num_envs * args.num_steps
    return dict(batch_size=size, minibatch_size=size // args.num_minibatches,
                num_iterations=args.total_timesteps // size)


def eval_steps(args, derived):
    """The trainer's teacher-off evaluation points (ppo_distill.train)."""
    n, size = derived['num_iterations'], derived['batch_size']
    if args.eval_interval <= 0:
        return []
    milestones = {int(x) // size for x in args.eval_frame_milestones.split(',')
                  if x.strip()}
    return [i * size for i in range(1, n + 1)
            if i % args.eval_interval == 0 or i in milestones or i == n]


def validate_completed(run_dir, args):
    """Admit a terminal run only with its curve, dose and replay evidence."""
    run_dir = Path(run_dir)
    summary = read(run_dir / 'run_summary.json')
    finish = read(run_dir / 'lesson_finished.json')
    contract = read(run_dir / 'lesson_contract.json')
    native = (run_dir / 'initial_policy.sha256').read_text().strip()
    # The native trainer and the adapter hash parameters differently;
    # both identities are kept, never compared across encodings.
    if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (
            native, finish.get('initial_policy_sha256', ''),
            contract.get('initial_policy_sha256', ''))):
        raise ValueError('Missing or malformed initial policy identity')
    derived = derived_of(args)
    horizon = derived['num_iterations'] * derived['batch_size']
    expected_args = normalized({**asdict(args), **derived})
    if (summary.get('status') != 'completed'
            or summary.get('global_step') != horizon
            or finish.get('global_step') != horizon
            or summary.get('args') != expected_args
            or contract.get('args') != expected_args):
        raise ValueError('Terminal settings or endpoint differ')
    schedule = lesson_schedule(derived['num_iterations'], args.lesson_offset,
                               args.lesson_every, args.lesson_exposures)
    mode, n, width = args.lesson_mode, len(schedule), args.lesson_batch
    explains = mode in ('explanation', 'combined', 'detached')
    acts = mode in ('actions', 'combined')
    if (finish['format_bank_sha256'] != args.format_bank_sha256
            or contract['format_bank_sha256'] != args.format_bank_sha256
            or finish['initial_policy_sha256']
            != contract['initial_policy_sha256']
            or contract['schedule'] != schedule
            or finish['updates'] != n or finish['replay_exposures'] != n * width
            or finish['effective_action_exposures'] != (n * width if acts
                                                        else 0)
            or [finish['schedule_first'], finish['schedule_last']]
            != [schedule[0], schedule[-1]]
            or not math.isclose(finish['explanation_integral'],
                                args.lesson_scale * n if explains else 0.0,
                                rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(finish['action_integral'],
                                args.lesson_action_coef * n if acts else 0.0,
                                rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError('Lesson identity, schedule or dose differs')
    # Reconstruct the replay stream instead of trusting a count.
    ids = [r['case_id'] for r in read(args.format_bank)['train']]
    rng = np.random.default_rng(args.seed + 51_007)
    replay = hashlib.sha256()
    updates = read_jsonl(run_dir / 'lesson_updates.jsonl')
    if [u['iteration'] for u in updates] != schedule:
        raise ValueError('Replay schedule differs')
    for update in updates:
        drawn = rng.choice(len(ids), width, replace=len(ids) < width)
        expected = [ids[i] for i in drawn]
        if update['ids'] != expected:
            raise ValueError('Replay IDs differ from the seeded stream')
        replay.update(encoded(expected))
        grad = update.get('shared_grad_norm')
        if mode in ('explanation', 'combined'):
            if grad is None or not math.isfinite(grad) or grad <= 0:
                raise ValueError('Explanation update without a shared '
                                 'gradient')
        elif grad is not None:
            raise ValueError('Unexpected explanation gradient')
    curve = read_jsonl(run_dir / 'evaluations.jsonl')
    steps = eval_steps(args, derived)
    if (not steps or [r['global_step'] for r in curve] != steps
            or any(r['teacher_on'] or r['episodes'] != args.eval_episodes
                   or r['seed_base'] != args.seed + 50_000
                   or r['iteration'] * derived['batch_size']
                   != r['global_step'] for r in curve)):
        raise ValueError('Teacher-off evaluation grid or identity differs')
    success = np.array([r['success_rate'] for r in curve], dtype=float)
    if not np.isfinite(success).all() or np.any((success < 0) | (success > 1)):
        raise ValueError('Invalid teacher-off success values')
    checkpoint = run_dir / 'agent.pt'
    final = policy_hash(torch.load(checkpoint, map_location='cpu',
                                   weights_only=True))
    if final != finish['final_policy_sha256']:
        raise ValueError('Checkpoint contradicts the final policy identity')
    x = np.asarray(steps, dtype=float)
    auc = (float((np.diff(x) * (success[1:] + success[:-1]) / 2).sum()
                 / (x[-1] - x[0])) if len(x) > 1 else float(success[0]))
    return dict(
        teacher_off_auc=auc, final_success=float(success[-1]),
        auc_interval=[steps[0], steps[-1]], curve=success.tolist(),
        replay_ids_sha256=replay.hexdigest(), lesson_updates=n,
        initial_policy_sha256=finish['initial_policy_sha256'],
        initial_policy_native_sha256=native, final_policy_sha256=final,
        checkpoint_sha256=digest(checkpoint),
        explanation_integral=finish['explanation_integral'],
        action_integral=finish['action_integral'],
        audit_head_metrics=finish['audit_head_metrics'])


def run_cell(batch, suite, index):
    """Dispatch once from the snapshot; failures are kept, never retried."""
    batch = Path(batch).resolve()
    manifest = verify(batch)
    check_worker(batch, manifest)
    cells = cells_for(batch, manifest, suite)
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError(f'Cell index outside the frozen {suite} range')
    cell = cells[index]
    args = resolved_args(batch, cell)
    pattern = f'*{args.experiment_id}__{args.seed}__*/run_summary.json'
    if list((ROOT / 'results/runs').glob(pattern)):
        raise FileExistsError('Existing run for this cell; do not resubmit')
    directory = batch / 'cells' / suite / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / 'dispatch.json', dict(
        **cell, suite=suite, commit=manifest['commit'],
        manifest_sha256=digest(batch / 'manifest.json'),
        runtime=runtime_identity(), resolved_args=asdict(args),
        started_at=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    outcome = dict(returncode=None, runs=[], artifact_status='not_validated')
    try:
        process = subprocess.run(
            [sys.executable, '-u', '-m', cell['trainer'],
             *trainer_argv(asdict(args))], cwd=ROOT)
        outcome['returncode'] = process.returncode
        runs = sorted((ROOT / 'results/runs').glob(pattern))
        outcome['runs'] = [p.parent.relative_to(batch).as_posix()
                           for p in runs]
        if process.returncode:
            outcome['artifact_status'] = 'process_failed'
        elif len(runs) != 1:
            raise ValueError('Expected exactly one completed run artifact')
        else:
            outcome['metrics'] = validate_completed(runs[0].parent, args)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(artifact_status='failed',
                       error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    outcome['worker_returncode'] = int(
        outcome['artifact_status'] != 'terminal_contract_validated')
    write(directory / 'exit.json', outcome)
    return outcome['worker_returncode']


# -------------------------------------------------------------- submission

def claim(batch, text):
    """One submission attempt per batch; a rerun reports, never repeats."""
    with (Path(batch) / 'SUBMISSION_ATTEMPTED').open('x') as handle:
        handle.write(text + '\n')


def sbatch(batch, name, stage, target, array=None, dependency=None,
           resources=()):
    command = ['sbatch', '--parsable', f'--job-name=xf_{name}',
               f'--output={batch}/slurm/%x_%A_%a.out', *resources]
    if array:
        command.append('--array=' + array)
    if dependency:
        command += [f'--dependency=afterok:{dependency}',
                    '--kill-on-invalid-dep=yes']
    command += [str(batch / 'code' / WORKER), str(batch), stage, target]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    with (batch / 'submitted_jobs.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(name=name, job=job,
                                     command=command)) + '\n')
    print(name, job, flush=True)
    return job


def submit_timing(batch):
    batch = Path(batch).resolve()
    verify(batch)
    claim(batch, 'timing array 0-29')
    job = sbatch(batch, 'timing', 'cell', 'timing', array='0-29')
    (batch / 'SUBMITTED_JOB').write_text(json.dumps(dict(timing=job)) + '\n')
    return dict(timing=job)


def submit_formats(batch):
    batch = Path(batch).resolve()
    verify(batch)
    claim(batch, 'writer chain and two learning arrays')
    jobs = {}
    for name, stage, target, array, after in SUBMISSION:
        resources = (('--mem=8G', '--time=0-12:00:00')
                     if stage == 'writer' else ())
        jobs[name] = sbatch(batch, name, stage, target, array,
                            jobs[after] if after else None, resources)
    (batch / 'SUBMITTED_JOB').write_text(json.dumps(jobs) + '\n')
    return jobs


def launch(root, suites, source, ledger, credential):
    """User-owned: prepare (idempotent) and submit what is not submitted."""
    steps = dict(
        timing=(lambda: prepare_timing(root, source), submit_timing),
        formats=(lambda: prepare_formats(root, source, ledger, credential),
                 submit_formats))
    for suite in suites:
        prepare, submit = steps[suite]
        batch = prepare()
        if (batch / 'SUBMITTED_JOB').exists():
            print(f'{batch.name} already submitted: '
                  f'{(batch / "SUBMITTED_JOB").read_text().strip()}')
        elif (batch / 'SUBMISSION_ATTEMPTED').exists():
            raise FileExistsError(f'{batch} has an unfinished submission '
                                  'attempt; inspect it before retrying')
        else:
            submit(batch)


def check(source, ledger):
    """Offline resolution of both batches; nothing prepared or reserved."""
    panel = read(Path(source) / 'panel.json')
    prices = PriceTable.load(str(ROOT / collection.PRICES))
    pinned_prices(prices)
    contract = formats_contract(panel, prices)
    check_commands(timing_contract()['cells'][:6] + placeholder_cells())
    result = dict(
        timing_cells=len(timing_contract()['cells']),
        schedules=plan.schedule_summary(),
        formats_cells=contract['suites'],
        bounds_usd={a: round(b['total_bound_usd'], 4)
                    for a, b in contract['bounds'].items()},
        total_bound_usd=round(contract['total_bound_usd'], 4),
        request_sha256={a: i['request_sha256']
                        for a, i in contract['identities'].items()},
        panel_matches=digest(Path(source) / 'panel.json') == PANEL_SHA256,
        lessons_match=digest(Path(source) / 'lessons.json') == LESSONS_SHA256)
    if Path(ledger).is_file():
        result['ledger'] = collection.check_ledger(
            ledger, contract['total_bound_usd'])
    print(json.dumps(result, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true',
                        help='resolve both batches offline')
    action.add_argument('--launch', action='store_true',
                        help='prepare and submit (user-owned)')
    action.add_argument('--writer', choices=ACCESS,
                        help='worker: one writer job')
    action.add_argument('--run-cell', type=int, metavar='INDEX',
                        help='worker: one learning cell')
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--suites', nargs='+', choices=('timing', 'formats'),
                        default=['timing', 'formats'])
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--source', type=Path, default=(
        ROOT / 'results/explanation_lessons' / SOURCE_STUDY))
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / 'results/budget_ledger.json')
    parser.add_argument('--credential-file', type=Path,
                        default=ROOT / '.env')
    cli = parser.parse_args()
    if cli.check:
        return check(cli.source, cli.ledger)
    if cli.launch:
        launch(ROOT, cli.suites, cli.source, cli.ledger, cli.credential_file)
        return 0
    if cli.batch is None:
        parser.error('workers need --batch')
    if cli.writer:
        writer_stage(cli.batch, cli.writer)
        return 0
    if cli.suite is None:
        parser.error('--run-cell needs --suite')
    return run_cell(cli.batch, cli.suite, cli.run_cell)


if __name__ == '__main__':
    raise SystemExit(main())
