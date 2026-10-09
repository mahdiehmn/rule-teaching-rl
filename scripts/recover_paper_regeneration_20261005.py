"""Bounded, explicitly amended recovery of one failed bank consultation.

Original attempts and holds remain immutable.
Only user-submitted jobs call the provider or train policies.
"""

import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
import zipfile

from scripts import paper_optional_regeneration_20261005 as regen
from scripts import run_paper_optionals_20261005 as old
from scripts import run_fix_wave_20260929 as fw
from teachers.budget import BudgetError, CostLedger

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'regeneration_recovery_20261005_v1'
PROTOCOL = 'research/paper_regeneration_recovery_2026-10-05.md'
MODULE = 'scripts/recover_paper_regeneration_20261005.py'
WRAPPER = 'scripts/submit_paper_regeneration_recovery.sh'
ORIGINAL_COMMIT = '2c419fa1edbdff1365ec20309c5e9b4c556dbca0'
FAILED_CASE = 'consult_018'
ALLOCATION = .76
REQUIRED = (*old.REQUIRED, PROTOCOL, MODULE, WRAPPER)
read = regen.read
write_new = regen.write_new


def jsonl(path):
    """Read every nonempty durable journal record."""

    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]


def hashes(directory):
    """Freeze a tree, including raw evidence and attempt markers."""

    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): fw.digest(p)
            for p in directory.rglob('*') if p.is_file()
            and not p.name.endswith('.lock')}


def validate_cost(end):
    """Retain a full bound when provider usage was unavailable."""

    if end.get('dollars') is None:
        if end.get('dollars_bound') != regen.PER_CALL_BOUND:
            raise ValueError('Unknown attempt lacks its full cost bound')
        return regen.PER_CALL_BOUND
    incoming, outgoing = end.get('tokens_in'), end.get('tokens_out')
    if (type(incoming) is not int or not 0 <= incoming <= 16384
            or type(outgoing) is not int or not 0 <= outgoing <= 8192):
        raise ValueError('Invalid original attempt token accounting')
    cost = (incoming * .25 + outgoing * 2) / 1e6
    if end['dollars'] != cost or not math.isfinite(cost):
        raise ValueError('Original attempt cost differs from token usage')
    return cost


def finite_nonnegative(value):
    """NaN and booleans must never pass numeric accounting comparisons."""

    return (type(value) in (int, float) and math.isfinite(value)
            and value >= 0)


def cost_breakdown(*paths):
    """Separate provider usage from conservative unknown-attempt bounds."""

    known, unknown, known_n, unknown_n = 0., 0., 0, 0
    for path in paths:
        for row in jsonl(path):
            if row.get('event') != 'END':
                continue
            cost = validate_cost(row)
            if row.get('dollars') is None:
                unknown += cost
                unknown_n += 1
            else:
                known += cost
                known_n += 1
    return dict(usage_proven_usd=known, unknown_attempt_bound_usd=unknown,
                usage_proven_attempts=known_n, unknown_attempts=unknown_n)


def audit_failed_bank(directory):
    """Admit only the observed single capped consultation failure."""

    import jsonschema
    directory = Path(directory)
    requests = regen.collect.load_frozen(
        directory / 'consult_requests.json',
        directory / 'consult_requests_manifest.json', 'all')
    if requests != regen.consultation_requests(regen.corpus(
            panel_source=directory / 'panels.json')):
        raise ValueError('Original consultation payloads changed')
    forbidden = ('blind_requests.json', 'blind_requests_manifest.json',
                 'blind_replies.raw.jsonl', 'blind_replies.jsonl',
                 'refine_plan.json', 'bank.json', 'generation_receipt.json')
    if any((directory / name).exists() for name in forbidden):
        raise ValueError('Failed bank has unexpected later-stage artifacts')
    failure = read(directory / 'generation_failure.json')
    if failure != dict(study=regen.STUDY, generation_index=2,
                       error_type='ValueError', status='generation_failed',
                       reservation_retained=True):
        raise ValueError('Original failure receipt differs')
    if not (directory / 'GENERATION_ATTEMPTED').is_file():
        raise ValueError('Original attempt marker missing')
    raw = jsonl(directory / 'consult_replies.raw.jsonl')
    replies = jsonl(directory / 'consult_replies.jsonl')
    if len(raw) != 72 or len(replies) != 36:
        raise ValueError('Expected exactly36 original consultation attempts')
    good, costs, mapping = [], {}, []
    for request in requests:
        regen.validate_request(request)
        identity = {k: request[k] for k in (
            'case_id', 'condition', 'request_sha256')}
        events = [(i, r) for i, r in enumerate(raw)
                  if all(r.get(k) == v for k, v in identity.items())]
        answers = [(i, r) for i, r in enumerate(replies)
                   if all(r.get(k) == v for k, v in identity.items())]
        if ([r['event'] for _, r in events] != ['START', 'END']
                or len(answers) != 1):
            raise ValueError('Original attempt identities differ')
        start, end = (r for _, r in events)
        reply = answers[0][1]
        if (start.get('model') != regen.MODEL
                or start.get('bound_usd') != regen.PER_CALL_BOUND
                or end.get('served_model') != regen.MODEL
                or reply.get('served_model') != regen.MODEL):
            raise ValueError('Original model or bound differs')
        case = request['case_id']
        costs[case] = validate_cost(end)
        if case == FAILED_CASE:
            if (end.get('response_status') != 'failed'
                    or end.get('error_type') != 'ValueError'
                    or end.get('http_status') is not None
                    or end.get('tokens_out') != 8192
                    or reply.get('response_status') != 'failed'
                    or reply.get('answer') is not None):
                raise ValueError('Failure is outside this bounded amendment')
        else:
            answer = json.loads(end['output_text'])
            if (end.get('response_status') != 'completed'
                    or reply.get('response_status') != 'completed'
                    or reply['answer'] != answer):
                raise ValueError('Another original consultation failed')
            jsonschema.validate(answer, request['request']['text'][
                'format']['schema'])
            regen.v3.parse(answer)
            good.append(case)
        mapping.append(dict(case_id=case, raw_rows=[i for i, _ in events],
                            reply_row=answers[0][0]))
    return dict(cost_usd=sum(costs.values()),
                failed_cost_usd=costs[FAILED_CASE], good_case_ids=good,
                mapping=mapping, cost_breakdown=cost_breakdown(
                    directory / 'consult_replies.raw.jsonl'))


def audit_complete_bank(directory, index):
    """Recompute completed banks and their receipt costs from raw data."""

    directory = Path(directory)
    receipt = read(directory / 'generation_receipt.json')
    if (receipt.get('study') != regen.STUDY
            or type(receipt.get('generation_index')) is not int
            or receipt.get('generation_index') != index
            or receipt.get('status') != 'technical_generation_complete'
            or type(receipt.get('attempts')) is not int
            or type(receipt.get('rules')) is not int
            or not finite_nonnegative(receipt.get('settled_liability_usd'))
            or set(receipt['hashes']) != set(regen.RECEIPT_FILES)):
        raise ValueError('Completed bank receipt identity differs')
    for name, digest in receipt['hashes'].items():
        if fw.digest(fw.inside(directory, name)) != digest:
            raise ValueError('Completed bank output hash changed')
    bank = read(directory / 'bank.json')
    if bank != regen.build_bank(directory, index):
        raise ValueError('Completed bank differs from the strict filter')
    count, cost = regen.audit_stage(directory, 'consult')
    blind_count, blind_cost = regen.audit_stage(directory, 'blind')
    plan, requests = regen.expected_blind(directory)
    if (count != 36 or blind_count > 36
            or read(directory / 'refine_plan.json') != plan
            or read(directory / 'blind_requests.json') != requests
            or receipt['attempts'] != count + blind_count
            or receipt['rules'] != len(bank['rules'])
            or abs(receipt['settled_liability_usd'] - cost -
                   blind_cost) > 1e-12):
        raise ValueError('Completed bank accounting or recipe differs')
    return dict(attempts=count + blind_count, cost_usd=cost + blind_cost,
                cost_breakdown=cost_breakdown(*[
                    directory / f'{stage}_replies.raw.jsonl'
                    for stage in ('consult', 'blind')]))


def no_original_training(original, manifest):
    """A pending original array must not have claimed any training slot."""

    for cell in manifest['cells']:
        directory = original / 'cells/train' / str(cell['index'])
        if any(directory.glob('*')):
            raise ValueError('Original training already has an attempt')
    runs = original / 'code/results/runs'
    if list(runs.glob(f'*{regen.STUDY}*')):
        raise ValueError('Original training run directories already exist')


def audit_original(original):
    """Verify original frozen source, runtime, attempts and untouched slots."""

    original = Path(original).resolve()
    m = old.verify(original)
    if (m['group'] != 'regeneration' or m['commit'] != ORIGINAL_COMMIT
            or m['runtime'] != fw.runtime_identity()):
        raise ValueError('Original generation source/runtime differs')
    old.parent_hold(m)
    no_original_training(original, m)
    regen.verify_generation_inputs(original / 'generation')
    audit = dict(completed={}, failed=None)
    for index in range(5):
        cell = original / 'cells/generate' / str(index)
        end = read(cell / 'exit.json')
        start = read(cell / 'dispatch.json')
        if (not (cell / 'CLAIMED').is_file()
                or start['cell'] != dict(index=index,
                                         bank=regen.BANKS[index])
                or start['stage'] != 'generate'
                or start['commit'] != ORIGINAL_COMMIT
                or start['manifest_sha256'] != fw.digest(
                    original / 'manifest.json')):
            raise ValueError('Original generation dispatch differs')
        bank = original / 'generation/banks' / regen.BANKS[index]
        if index == 2:
            if (end.get('artifact_status') != 'failed'
                    or end.get('error_type') != 'ValueError'
                    or end.get('returncode') is not None
                    or end.get('runs') != []):
                raise ValueError('Failed generation cell differs')
            audit['failed'] = audit_failed_bank(bank)
            state = CostLedger(str(cell / 'budget.json')).status()
            holds = state['open_reservations']
            if (state['allowance_usd'] != 1.48
                    or state['settled_usd'] != 0 or len(holds) != 1
                    or holds[0]['reserved'] != 1.48
                    or holds[0]['run_id'] != f'{regen.STUDY}_bank_2'
                    or state['overspent_runs']):
                raise BudgetError('Original failed bank hold changed')
        else:
            if (end.get('artifact_status') != 'terminal_contract_validated'
                    or end.get('returncode') != 0):
                raise ValueError('Another original bank did not complete')
            audit['completed'][str(index)] = audit_complete_bank(bank, index)
            if end.get('generation') != read(bank / 'generation_receipt.json'):
                raise ValueError('Original worker generation receipt differs')
            state = CostLedger(str(cell / 'budget.json')).status()
            cost = audit['completed'][str(index)]['cost_usd']
            if (state['allowance_usd'] != 1.48
                    or not finite_nonnegative(state['settled_usd'])
                    or abs(state['settled_usd'] - cost) > 1e-12
                    or state['reserved_usd'] != 0
                    or state['open_reservations'] or state['overspent_runs']):
                raise BudgetError('Completed original child ledger differs')
    return m, audit


def original_files(original):
    """List frozen inputs and original attempt evidence, excluding locks."""

    original = Path(original)
    paths = [original / name for name in (
        'manifest.json', 'manifest.sha256', 'READY')]
    for name in ('generation', 'cells/generate', 'submissions'):
        paths.extend(p for p in (original / name).rglob('*')
                     if p.is_file() and not p.name.endswith('.lock'))
    return {p.relative_to(original).as_posix(): fw.digest(p) for p in paths}


def same_scientific_source(original_manifest, code):
    """Require every preexisting executable/configuration byte unchanged."""

    for name, expected in original_manifest['source_hashes'].items():
        if Path(name).suffix in ('.py', '.json', '.toml', '.yaml', '.yml'):
            if fw.digest(fw.inside(code, name)) != expected:
                raise ValueError(f'Original scientific source changed: {name}')


def batch_dir(root):
    return Path(root).resolve() / 'results/paper_optionals' / old.STUDY / STUDY


def parent_hold(m):
    """Keep the new amendment fully funded without touching old holds."""

    state = old.funding.check_pool(m['ledger'], 0.)
    holds = [r for r in state['open_reservations']
             if r['run_id'] == m['parent_run_id']]
    if len(holds) != 1 or holds[0]['reserved'] != ALLOCATION:
        raise BudgetError('Recovery requires its separate $0.76 parent hold')


def prepare(original, root=ROOT):
    """Archive committed recovery software and original evidence once."""

    original, root = Path(original).resolve(), Path(root).resolve()
    batch = batch_dir(root)
    if batch.exists():
        m = verify(batch)
        if Path(m['original']).resolve() != original:
            raise ValueError('Existing recovery belongs to another source')
        parent_hold(m)
        return batch
    m, audit = audit_original(original)
    old.funding.backend_environment('openai', Path(m['credential']))
    old.funding.check_pool(m['ledger'], ALLOCATION)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                     cwd=root, text=True).strip()
    archive = subprocess.check_output([
        'git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', commit],
        cwd=root)
    evidence = original_files(original)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the entire recovery package first')
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    same_scientific_source(m, batch / 'code')
    for name, digest in evidence.items():
        target = fw.inside(batch / 'original_snapshot', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fw.inside(original, name), target)
        if fw.digest(target) != digest:
            raise ValueError('Original evidence changed during snapshot')
    for index in range(70):
        (batch / 'cells/train' / str(index)).mkdir(parents=True)
    (batch / 'cells/recover/0').mkdir(parents=True)
    (batch / 'slurm').mkdir()
    CostLedger.initialize(str(batch / 'cells/recover/0/budget.json'),
                          ALLOCATION, note='Separate bounded recovery')
    manifest = dict(
        study=STUDY, protocol=PROTOCOL, commit=commit,
        original=str(original), original_commit=ORIGINAL_COMMIT,
        original_hashes=evidence, original_audit=audit,
        cells=regen.training_plan(), runtime=fw.runtime_identity(),
        allocation_usd=ALLOCATION, max_new_attempts=37,
        ledger=m['ledger'], credential=m['credential'],
        parent_run_id=f'{old.STUDY}:{STUDY}',
        source_hashes=hashes(batch / 'code'),
        original_single_attempt_state='4/5 complete; bank_2 failed',
        independent_result_review='pending')
    write_new(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(fw.digest(batch / 'manifest.json'))
    CostLedger(m['ledger']).reserve(
        manifest['parent_run_id'], ALLOCATION,
        note='ONE consult018 second attempt + at most36 new blind requests')
    (batch / 'READY').write_text(commit + '\n')
    verify(batch)
    print(f'PREPARED recovery: {batch}', flush=True)
    return batch


def verify(batch):
    """Check immutable archives without treating derived data as original."""

    batch = Path(batch).resolve()
    m = read(batch / 'manifest.json')
    if (m['study'] != STUDY or m['protocol'] != PROTOCOL
            or m['original_commit'] != ORIGINAL_COMMIT
            or m['allocation_usd'] != ALLOCATION
            or m['max_new_attempts'] != 37
            or m['cells'] != regen.training_plan()
            or (batch / 'READY').read_text().strip() != m['commit']
            or fw.digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or not set(REQUIRED) <= set(m['source_hashes'])):
        raise ValueError('Recovery manifest identity differs')
    # Runtime-generated results are deliberately outside source hashes.
    for name, digest in m['source_hashes'].items():
        if fw.digest(fw.inside(batch / 'code', name)) != digest:
            raise ValueError(f'Recovery source changed: {name}')
    original = Path(m['original'])
    original_m, audit = audit_original(original)
    if (original_files(original) != m['original_hashes']
            or hashes(batch / 'original_snapshot') != m['original_hashes']
            or audit != m['original_audit']
            or m['runtime'] != original_m['runtime']):
        raise ValueError('Original evidence or snapshot changed')
    same_scientific_source(original_m, batch / 'code')
    return m


def selected_consults(batch):
    """Explicitly map 35 original successes and one amended response."""

    original = Path(batch) / 'original_snapshot/generation/banks/bank_2'
    original_raw = jsonl(original / 'consult_replies.raw.jsonl')
    original_answers = jsonl(original / 'consult_replies.jsonl')
    retry = Path(batch) / 'retry'
    regen.audit_stage(retry, 'consult')
    retry_raw = jsonl(retry / 'consult_replies.raw.jsonl')
    retry_answers = jsonl(retry / 'consult_replies.jsonl')
    requests = read(original / 'consult_requests.json')
    expected = [r for r in requests if r['case_id'] == FAILED_CASE]
    if read(retry / 'consult_requests.json') != expected:
        raise ValueError('Second attempt is not the byte-identical request')
    raw, answers, mapping = [], [], []
    for request in requests:
        case = request['case_id']
        amended = case == FAILED_CASE
        source_raw = retry_raw if amended else original_raw
        source_answers = retry_answers if amended else original_answers
        event_rows = [i for i, r in enumerate(source_raw)
                      if r['case_id'] == case]
        reply_rows = [i for i, r in enumerate(source_answers)
                      if r['case_id'] == case]
        if len(event_rows) != 2 or len(reply_rows) != 1:
            raise ValueError('Selection lineage is not one complete attempt')
        raw.extend(source_raw[i] for i in event_rows)
        answers.append(source_answers[reply_rows[0]])
        mapping.append(dict(case_id=case,
                            source='retry' if amended else 'original',
                            raw_rows=event_rows, reply_row=reply_rows[0]))
    return raw, answers, mapping


def write_jsonl(path, rows):
    """Create derived records once, with their lineage recorded separately."""

    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')


def make_effective_generation(batch):
    """Preserve four banks verbatim and derive only the failed bank stream."""

    batch = Path(batch)
    source = batch / 'original_snapshot/generation'
    effective = batch / 'effective_generation'
    effective.mkdir(exist_ok=False)
    for name in ('generation_manifest.json', 'generation_manifest.sha256'):
        shutil.copyfile(source / name, effective / name)
    (effective / 'banks').mkdir()
    for index in (0, 1, 3, 4):
        shutil.copytree(source / 'banks' / f'bank_{index}',
                        effective / 'banks' / f'bank_{index}')
    destination = effective / 'banks/bank_2'
    destination.mkdir()
    for name in ('panels.json', 'consult_requests.json',
                 'consult_requests_manifest.json'):
        shutil.copyfile(source / 'banks/bank_2' / name, destination / name)
    raw, replies, mapping = selected_consults(batch)
    write_jsonl(destination / 'consult_replies.raw.jsonl', raw)
    write_jsonl(destination / 'consult_replies.jsonl', replies)
    return destination, mapping


def worker_context(batch):
    """Workers execute the archived code under the prepared runtime."""

    batch = Path(batch).resolve()
    m = verify(batch)
    if (ROOT.resolve() != (batch / 'code').resolve()
            or fw.runtime_identity() != m['runtime']):
        raise ValueError('Recovery worker source/runtime differs')
    parent_hold(m)
    return batch, m


def dispatch(batch, m, stage, index, record):
    """An exclusive claim prevents partial attempts from being resent."""

    directory = batch / 'cells' / stage / str(index)
    with (directory / 'CLAIMED').open('x') as handle:
        handle.write('One bounded amended attempt; never auto-retry.\n')
    write_new(directory / 'dispatch.json', dict(
        cell=record, stage=stage, commit=m['commit'],
        manifest_sha256=fw.digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    return directory


def recovery_costs(batch, m):
    """Keep all attempts, selected replies and paid reservations distinct."""

    _, retry_cost = regen.audit_stage(Path(batch) / 'retry', 'consult')
    _, blind_cost = regen.audit_stage(
        Path(batch) / 'effective_generation/banks/bank_2', 'blind')
    original_cost = m['original_audit']['failed']['cost_usd'] + sum(
        row['cost_usd'] for row in m['original_audit']['completed'].values())
    return dict(original_attempts_liability_usd=original_cost,
                original_failed_attempt_usd=m['original_audit']['failed'][
                    'failed_cost_usd'],
                new_attempts_liability_usd=retry_cost + blind_cost,
                combined_attempts_liability_usd=original_cost + retry_cost +
                blind_cost, original_parent_hold_usd=7.40,
                additional_parent_hold_usd=ALLOCATION,
                old_holds_released=False,
                new_attempts_breakdown=cost_breakdown(
                    Path(batch) / 'retry/consult_replies.raw.jsonl',
                    Path(batch) / 'effective_generation/banks/bank_2' /
                    'blind_replies.raw.jsonl'))


class RecordingClient:
    """Preserve provider status separately from the collector's validation."""

    def __init__(self, client, path):
        self.client = client
        self.path = Path(path)
        self.path.touch(exist_ok=False)
        self.lock = threading.Lock()
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **body):
        # The existing collector sends this exact body once. This extra
        # log records diagnosis only and never changes or retries it.
        response = self.client.responses.create(**body)
        details = getattr(response, 'incomplete_details', None)
        if details is not None and hasattr(details, 'model_dump'):
            details = details.model_dump(mode='json')
        if details is not None and not isinstance(details, dict):
            details = dict(reason=getattr(details, 'reason', None))
        row = dict(request_sha256=regen.v3.digest(body),
                   provider_status=response.status,
                   incomplete_details=details, served_model=response.model)
        with self.lock:
            regen.append_event(self.path, row)
        return response

    def close(self):
        self.client.close()


def recover(batch):
    """Send at most one repeated consult and 36 unattempted blind checks."""

    batch, m = worker_context(batch)
    directory = batch / 'cells/recover/0'
    old.pristine_child(directory / 'budget.json', ALLOCATION)
    directory = dispatch(batch, m, 'recover', 0,
                         dict(index=0, case_id=FAILED_CASE))
    ledger = CostLedger(str(directory / 'budget.json'))
    run_id = f'{STUDY}:bank_2'
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    client = None
    phase = 'reserve_additional_liability'
    try:
        ledger.reserve(run_id, ALLOCATION,
                       note='ONE second consult + at most36 new blind checks')

        def check_hold():
            parent_hold(m)
            state = ledger.status()
            holds = state['open_reservations']
            if (len(holds) != 1 or holds[0]['run_id'] != run_id
                    or holds[0]['reserved'] != ALLOCATION
                    or state['overspent_runs']):
                raise BudgetError('Recovery child reservation changed')

        retry = batch / 'retry'
        retry.mkdir(exist_ok=False)
        requests = read(batch / 'original_snapshot/generation/banks/bank_2'
                        / 'consult_requests.json')
        selected = [r for r in requests if r['case_id'] == FAILED_CASE]
        regen.export_requests(retry, 'consult', selected)
        phase = 'retry_consultation'
        client = RecordingClient(regen.openai_client(m['credential']),
                                 batch / 'provider_receipts.jsonl')
        result = regen.collect_stage(retry, 'consult', client, check_hold)
        if len(result) != 1 or result[0]['response_status'] != 'completed':
            raise ValueError('Amended consultation failed; stop permanently')
        phase = 'derive_effective_consultations'
        effective, mapping = make_effective_generation(batch)
        phase = 'blind_checks'
        regen.blind_requests(effective)
        results = regen.collect_stage(effective, 'blind', client, check_hold)
        if any(r['response_status'] != 'completed' for r in results):
            raise ValueError('Amended blind check failed; stop permanently')
        phase = 'build_and_validate_bank'
        bank = regen.build_bank(effective, 2)
        write_new(effective / 'bank.json', bank)
        consult_n, consult_cost = regen.audit_stage(effective, 'consult')
        blind_n, blind_cost = regen.audit_stage(effective, 'blind')
        write_new(effective / 'generation_receipt.json', dict(
            study=regen.STUDY, generation_index=2,
            status='technical_generation_complete',
            attempts=consult_n + blind_n, rules=len(bank['rules']),
            settled_liability_usd=consult_cost + blind_cost,
            hashes={name: fw.digest(effective / name)
                    for name in regen.RECEIPT_FILES},
            provenance='DERIVED selected attempts; see recovery_receipt.json',
            original_single_attempt_status='failed'))
        regen.freeze_generation(batch / 'effective_generation')
        costs = recovery_costs(batch, m)
        receipt = dict(
            study=STUDY, status='amended_generation_complete',
            original_single_attempt_state='4/5 complete; bank_2 failed',
            new_attempts=1 + blind_n, selected_consult_lineage=mapping,
            effective_hashes=hashes(batch / 'effective_generation'),
            retry_hashes=hashes(retry), costs=costs,
            provider_receipts_sha256=fw.digest(
                batch / 'provider_receipts.jsonl'),
            independent_result_review='pending')
        write_new(batch / 'recovery_receipt.json', receipt)
        verify_lineage(batch)
        phase = 'settle_additional_liability'
        ledger.settle(run_id, costs['new_attempts_liability_usd'])
        outcome.update(returncode=0,
                       artifact_status='terminal_contract_validated')
    except Exception as error:
        outcome['error_type'] = type(error).__name__
        outcome['failure_stage'] = phase
    finally:
        if client is not None:
            client.close()
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_new(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def verify_lineage(batch):
    """Reconstruct the derived view and account for the excluded failure."""

    batch = Path(batch).resolve()
    m = verify(batch)
    receipt = read(batch / 'recovery_receipt.json')
    if (receipt['study'] != STUDY
            or receipt['status'] != 'amended_generation_complete'
            or receipt['original_single_attempt_state'] !=
            '4/5 complete; bank_2 failed'
            or receipt['effective_hashes'] != hashes(
                batch / 'effective_generation')
            or receipt['retry_hashes'] != hashes(batch / 'retry')):
        raise ValueError('Recovery receipt or derived evidence changed')
    if fw.digest(batch / 'provider_receipts.jsonl') != receipt[
            'provider_receipts_sha256']:
        raise ValueError('Provider diagnostic receipts changed')
    for index in (0, 1, 3, 4):
        effective_bank = batch / f'effective_generation/banks/bank_{index}'
        original_bank = (batch / 'original_snapshot/generation/banks'
                         / f'bank_{index}')
        if hashes(effective_bank) != hashes(original_bank):
            raise ValueError('An originally complete bank was changed')
    raw, replies, mapping = selected_consults(batch)
    effective = batch / 'effective_generation/banks/bank_2'
    if (raw != jsonl(effective / 'consult_replies.raw.jsonl')
            or replies != jsonl(effective / 'consult_replies.jsonl')
            or mapping != receipt['selected_consult_lineage']):
        raise ValueError('Derived consultation lineage differs')
    regen.freeze_generation(batch / 'effective_generation')
    for index in range(5):
        audit_complete_bank(
            batch / 'effective_generation/banks' / f'bank_{index}', index)
    count, _ = regen.audit_stage(effective, 'blind')
    if (type(receipt['new_attempts']) is not int
            or receipt['new_attempts'] != 1 + count
            or not 1 <= receipt['new_attempts'] <= 37
            or receipt['costs'] != recovery_costs(batch, m)):
        raise ValueError('Recovery accounting differs from all attempts')
    return receipt


def successful_recovery(batch, m):
    """A derived receipt alone does not prove a completed paid worker."""

    directory = Path(batch) / 'cells/recover/0'
    end = read(directory / 'exit.json')
    start = read(directory / 'dispatch.json')
    if (not (directory / 'CLAIMED').is_file()
            or end['artifact_status'] != 'terminal_contract_validated'
            or end['returncode'] != 0
            or start['cell'] != dict(index=0, case_id=FAILED_CASE)
            or start['stage'] != 'recover' or start['commit'] != m['commit']
            or start['manifest_sha256'] != fw.digest(
                Path(batch) / 'manifest.json')):
        raise ValueError('Recovery worker dispatch/completion differs')
    receipt = verify_lineage(batch)
    state = CostLedger(str(directory / 'budget.json')).status()
    if (state['allowance_usd'] != ALLOCATION
            or not finite_nonnegative(state['settled_usd'])
            or state['reserved_usd'] != 0 or state['open_reservations']
            or state['overspent_runs']
            or abs(state['settled_usd'] - receipt['costs'][
                'new_attempts_liability_usd']) > 1e-12):
        raise BudgetError('Recovery child settlement differs')
    return receipt


def records(batch, m):
    """Resolve exactly the original70 arms/seeds using the derived banks."""

    verify_lineage(batch)
    result = regen.cells(batch / 'code', batch / 'effective_generation')
    if [{k: r[k] for k in plan} for r, plan in zip(result, m['cells'])
        ] != m['cells'] or len(result) != 70:
        raise ValueError('Amendment changed the original training plan')
    return result


def train(batch, index):
    """Run one original scientific cell once in an isolated new directory."""

    batch, m = worker_context(batch)
    if type(index) is not int or index not in range(70):
        raise ValueError('Training index outside original plan')
    successful_recovery(batch, m)
    record = records(batch, m)[index]
    directory = dispatch(batch, m, 'train', index, record)
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        args = record['args']
        pattern = f"*{args['experiment_id']}__{args['seed']}__*"
        runs_root = batch / 'code/results/runs'
        if list(runs_root.glob(pattern)):
            raise FileExistsError('Previous recovery training attempt exists')
        process = subprocess.run([
            sys.executable, '-u', '-m', record['trainer'],
            *fw.trainer_argv(args)], cwd=batch / 'code', env=dict(os.environ))
        outcome['returncode'] = process.returncode
        runs = list(runs_root.glob(pattern + '/run_summary.json'))
        outcome['runs'] = [p.parent.relative_to(batch).as_posix()
                           for p in runs]
        if process.returncode != 0 or len(runs) != 1:
            raise ValueError('Training did not produce one completed run')
        outcome['metrics'] = old.validate_training(
            runs[0].parent, record, 'regeneration')
        outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome['error_type'] = type(error).__name__
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_new(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submission_state(batch, stage):
    return old.submission_state(batch, stage)


def submit(batch, stage, dependency=None):
    """Idempotently submit one recovery or the70 unchanged training slots."""

    batch = Path(batch).resolve()
    m = verify(batch)
    parent_hold(m)
    if stage not in ('recover', 'train'):
        raise ValueError('Unknown recovery stage')
    previous = submission_state(batch, stage)
    if previous:
        print(f'ALREADY SUBMITTED recovery/{stage}: {previous}', flush=True)
        return previous
    if stage == 'train' and (not dependency or dependency !=
                            submission_state(batch, 'recover')):
        raise ValueError('Training must depend on this recovery job')
    count = 1 if stage == 'recover' else 70
    time_limit = '02:00:00' if stage == 'recover' else regen.TIME
    command = [
        'sbatch', '--parsable', f'--job-name=po_recovery_{stage}',
        f'--array=0-{count - 1}', f'--time={time_limit}',
        f'--output={batch}/slurm/%x_%A_%a.out']
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
    command += [str(batch / 'code' / WRAPPER), str(batch), stage]
    directory = batch / 'submissions' / stage
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'ATTEMPTED').open('x') as handle:
        handle.write('Preserve ambiguous scheduler submissions.\n')
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Ambiguous scheduler receipt; inspect saved attempt')
    write_new(directory / 'submission.json', dict(command=command, job_id=job))
    print(f'SUBMITTED recovery/{stage}: {job}', flush=True)
    return job


def launch(original, root=ROOT):
    """Prepare first, then submit the new jobs from their frozen source."""

    batch = prepare(original, root)
    for stage in ('recover', 'train'):
        submission_state(batch, stage)
    dependency = None
    for stage in ('recover', 'train'):
        source = (
            'from scripts.recover_paper_regeneration_20261005 import submit; '
            'import sys; submit(sys.argv[1],sys.argv[2],sys.argv[3] or None)')
        subprocess.run([sys.executable, '-c', source, str(batch), stage,
                        dependency or ''], cwd=batch / 'code', check=True)
        dependency = submission_state(batch, stage)
    return batch


def report(batch):
    """Report original failures and amended progress without erasing either."""

    batch = Path(batch).resolve()
    m = verify(batch)
    result = dict(study=STUDY,
                  original_single_attempt_state='4/5 complete; bank_2 failed',
                  original_attempt_audit=m['original_audit'],
                  independent_result_review='pending',
                  recovery_status='unattempted', training={})
    result['new_attempt_inventory'] = partial_attempt_accounting(batch)
    recovery_cell = batch / 'cells/recover/0'
    if (recovery_cell / 'CLAIMED').exists():
        result['recovery_status'] = 'dispatched_no_exit'
    if (recovery_cell / 'exit.json').exists():
        end = read(recovery_cell / 'exit.json')
        result['recovery_status'] = end['artifact_status']
        result['recovery_exit'] = end
    plan = m['cells']
    if result['recovery_status'] == 'terminal_contract_validated':
        result['recovery_receipt'] = successful_recovery(batch, m)
        plan = records(batch, m)
    statuses = {state: [] for state in (
        'validated', 'failed', 'dispatched_no_exit', 'unattempted')}
    rows = []
    for record in plan:
        index = record['index']
        cell = batch / 'cells/train' / str(index)
        state = 'dispatched_no_exit' if (cell / 'CLAIMED').exists() else (
            'unattempted')
        if (cell / 'exit.json').exists():
            state = 'failed'
            end = read(cell / 'exit.json')
            if end['artifact_status'] == 'terminal_contract_validated':
                if 'args' not in record:
                    raise ValueError('Training completed without recovery')
                start = read(cell / 'dispatch.json')
                if (start['cell'] != record or start['stage'] != 'train'
                        or start['commit'] != m['commit']
                        or start['manifest_sha256'] != fw.digest(
                            batch / 'manifest.json')
                        or end['returncode'] != 0 or len(end['runs']) != 1):
                    raise ValueError('Amended training dispatch differs')
                metrics = old.validate_training(
                    fw.inside(batch, end['runs'][0]), record, 'regeneration')
                if metrics != end['metrics']:
                    raise ValueError('Amended metrics differ from raw data')
                state = 'validated'
                rows.append(dict(arm=record['arm'], seed=record['seed'],
                                 auc=metrics['auc'], initial_sha256=metrics[
                                     'initial_sha256']))
        statuses[state].append(index)
    result['training'] = statuses
    result['results'] = regen.report(rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = batch / 'reports' / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'report.json', result)
    print(f'Report: {output / "report.json"}')
    return result


def partial_attempt_accounting(batch):
    """Keep failed and interrupted paid attempts visible in partial reports."""

    batch = Path(batch)
    result = dict(started=0, ended=0, usage_proven_usd=0.,
                  unknown_attempt_bound_usd=0.,
                  full_reservation_usd=ALLOCATION)
    for relative in ('retry/consult_replies.raw.jsonl',
                     'effective_generation/banks/bank_2/'
                     'blind_replies.raw.jsonl'):
        path = batch / relative
        if not path.exists():
            continue
        rows = jsonl(path)
        starts = [r for r in rows if r.get('event') == 'START']
        ends = [r for r in rows if r.get('event') == 'END']
        if len({r['case_id'] for r in starts}) != len(starts):
            raise ValueError('Duplicate paid recovery attempt')
        for start in starts:
            matches = [r for r in ends if all(r.get(k) == start[k]
                       for k in ('case_id', 'condition', 'request_sha256'))]
            if len(matches) > 1:
                raise ValueError('Duplicate paid recovery outcome')
            cost = (validate_cost(matches[0]) if matches
                    else regen.PER_CALL_BOUND)
            key = ('usage_proven_usd' if matches and
                   matches[0].get('dollars') is not None else
                   'unknown_attempt_bound_usd')
            result[key] += cost
        if len(ends) > len(starts):
            raise ValueError('Recovery outcome without a start record')
        result['started'] += len(starts)
        result['ended'] += len(ends)
    if result['started'] > 37:
        raise ValueError('Recovery exceeded its maximum attempt count')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=(
        'prepare', 'launch', 'recover', 'train', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.action in ('prepare', 'launch'):
        if args.original is None:
            parser.error('--original is required')
        (prepare if args.action == 'prepare' else launch)(
            args.original, args.root)
    else:
        batch = args.batch or batch_dir(args.root)
        if args.action == 'recover':
            return recover(batch)
        if args.action == 'train':
            return train(batch, args.index)
        report(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
