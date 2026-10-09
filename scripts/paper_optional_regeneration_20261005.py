"""Five unselected DoorKey bank repetitions and their learning cells.

Paid requests run only inside user-submitted
generation jobs with pre-funded child ledgers. No calls occur on import.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from scripts import collect_prompt_reliability_20260927 as collect
from scripts import conditional_rules_v3 as v3
from scripts import run_fix_wave_20260929 as fw
from teachers.budget import BudgetError, CostLedger

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_optional_regeneration_20261005_v1'
PROTOCOL = 'research/paper_optional_regeneration_2026-10-05.md'
FIXTURE = 'research/fixtures/paper_optional_dk_corpus_20261005.json'
CORPUS_SHA256 = (
    'd1105d126a8137cbc491a923102a4982323c9070165772d3299473f801de5a70')
BANKS = tuple(f'bank_{i}' for i in range(5))
ARMS = ('none', 'historical_selected', *BANKS)
REPS = range(130, 140)
TIME = '1-06:00:00'
MODEL = 'gpt-5-mini-2025-08-07'
CONTRACT = dict(
    study=STUDY, model=MODEL, banks=5, consult_per_bank=36,
    max_blind_per_bank=36, max_attempts_per_bank=72, attempts_per_case=1,
    sdk_retries=0, workers_per_bank=6, timeout_seconds=300,
    max_input_tokens=16384, wire_byte_limit=15360,
    max_output_tokens=8192, reasoning_effort='low',
    input_per_million=.25, output_per_million=2.,
    reservation_per_bank_usd=1.48, full_reservation_usd=7.40,
    endpoint='https://api.openai.com/v1/responses',
)
PER_CALL_BOUND = (.25 * 16384 + 2 * 8192) / 1e6
RECEIPT_FILES = (
    'bank.json', 'consult_replies.jsonl', 'consult_replies.raw.jsonl',
    'blind_requests.json', 'blind_requests_manifest.json',
    'blind_replies.jsonl', 'blind_replies.raw.jsonl', 'refine_plan.json',
)
REQUIRED = (
    PROTOCOL, FIXTURE, 'scripts/paper_optional_regeneration_20261005.py',
    'scripts/conditional_rules_v3.py',
    'scripts/collect_prompt_reliability_20260927.py',
    'scripts/run_rule_bank_pilot_20260928.py', 'teachers/budget.py',
    'research/rule_banks/v3_20260928/blind_strict.json',
)


def read(path):
    """
    Read JSON without relying on the platform's default encoding.
    """

    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    """
    Preserve any prior generation attempt or frozen artifact.
    """

    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=1, allow_nan=False)
        handle.write('\n')


def corpus(root=ROOT, panel_source=None):
    """
    Verify the exact historical consultation and experience pool.
    """

    data = read(panel_source or Path(root) / FIXTURE)
    data = data.get('corpus', data)
    panels = {name: data[name] for name in ('consult', 'pool')}
    if (len(panels['consult']) != 36 or len(panels['pool']) != 343
            or v3.digest(panels) != CORPUS_SHA256):
        raise ValueError('Historical DoorKey corpus identity differs')
    return panels


def validate_request(row):
    """
    Freeze the historical request recipe and a conservative input bound.
    """

    body = row['request']
    if (row['model'] != MODEL or body['model'] != MODEL
            or row['request_sha256'] != v3.digest(body)
            or body['max_output_tokens'] != 8192
            or body['reasoning'] != {'effort': 'low'}
            or body['service_tier'] != 'default' or body['store'] is not False
            or len(json.dumps(body).encode()) > CONTRACT['wire_byte_limit']):
        raise ValueError('Generation request recipe or byte bound differs')


def consultation_requests(panels):
    """
    Repeat the same 36 original prompts without adding a variant hint.
    """

    rows = []
    for index, state in enumerate(panels['consult']):
        body = v3.body(v3.consult_prompt(state), v3.consult_schema(),
                       'scoped_rule_v3')
        row = dict(case_id=f'consult_{index:03d}', condition='v3_consult',
                   split='consult', model=MODEL, request=body,
                   request_sha256=v3.digest(body))
        validate_request(row)
        rows.append(row)
    return rows


def export_requests(directory, stage, rows):
    """
    Use the existing collector's frozen-request manifest convention.
    """

    write_new(directory / f'{stage}_requests.json', rows)
    write_new(directory / f'{stage}_requests_manifest.json', dict(
        study=STUDY, stage=stage, model=MODEL, cases=len(rows),
        requests_sha256=v3.digest(rows)))


def prepare_generation(batch, root=ROOT, panel_source=None):
    """
    Freeze API inputs without creating a ledger or sending a request.
    """

    batch = Path(batch).resolve()
    if (batch / 'generation_manifest.json').exists():
        verify_generation_inputs(batch)
        return batch
    panels = corpus(root, panel_source)
    requests = consultation_requests(panels)
    batch.mkdir(parents=True, exist_ok=True)
    (batch / 'banks').mkdir(exist_ok=False)
    hashes = {}
    for bank in BANKS:
        directory = batch / 'banks' / bank
        directory.mkdir()
        write_new(directory / 'panels.json', panels)
        export_requests(directory, 'consult', requests)
        for name in ('panels.json', 'consult_requests.json',
                     'consult_requests_manifest.json'):
            path = directory / name
            hashes[path.relative_to(batch).as_posix()] = fw.digest(path)
    manifest = dict(contract=CONTRACT, corpus_sha256=CORPUS_SHA256,
                    input_sha256=hashes, training_replicates=list(REPS))
    write_new(batch / 'generation_manifest.json', manifest)
    (batch / 'generation_manifest.sha256').write_text(
        fw.digest(batch / 'generation_manifest.json') + '\n')
    return batch


def verify_generation_inputs(batch):
    """
    Refuse changed prompts, corpus, contract, or generation identities.
    """

    batch = Path(batch).resolve()
    manifest = read(batch / 'generation_manifest.json')
    if (manifest['contract'] != CONTRACT
            or manifest['corpus_sha256'] != CORPUS_SHA256
            or manifest['training_replicates'] != list(REPS)
            or fw.digest(batch / 'generation_manifest.json') !=
            (batch / 'generation_manifest.sha256').read_text().strip()):
        raise ValueError('Frozen generation contract changed')
    expected = set()
    for bank in BANKS:
        directory = batch / 'banks' / bank
        panels = corpus(panel_source=directory / 'panels.json')
        rows = collect.load_frozen(
            directory / 'consult_requests.json',
            directory / 'consult_requests_manifest.json', 'all')
        if rows != consultation_requests(panels):
            raise ValueError('Consultation payload differs from recipe')
        for name in ('panels.json', 'consult_requests.json',
                     'consult_requests_manifest.json'):
            relative = f'banks/{bank}/{name}'
            expected.add(relative)
            if fw.digest(batch / relative) != manifest['input_sha256'][
                    relative]:
                raise ValueError('Generation input changed')
    if set(manifest['input_sha256']) != expected:
        raise ValueError('Generation input inventory differs')
    return manifest


def wire_guard(request):
    """
    Check the actual outgoing body before any network transmission.
    """

    if (request.method != 'POST'
            or str(request.url) != CONTRACT['endpoint']
            or len(request.content) > CONTRACT['wire_byte_limit']):
        raise BudgetError('Unexpected endpoint or oversized API request')
    body = json.loads(request.content)
    validate_request(dict(model=MODEL, request=body,
                          request_sha256=v3.digest(body)))


def openai_client(credential):
    """
    Reuse the project's credential loader with zero SDK retries.
    """

    import httpx
    from openai import OpenAI
    from scripts.run_advising_strength_grid import backend_environment
    environment = backend_environment('openai', Path(credential))
    return OpenAI(
        api_key=environment['OPENAI_API_KEY'],
        base_url='https://api.openai.com/v1', max_retries=0,
        timeout=CONTRACT['timeout_seconds'],
        http_client=httpx.Client(event_hooks={'request': [wire_guard]}))


def append_event(path, value):
    """
    Persist a request attempt before dispatch and its receipt afterward.
    """

    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def collect_stage(directory, stage, client, check_hold):
    """
    Collect at most 36 requests once, retaining every failed receipt.
    """

    import jsonschema
    rows = collect.load_frozen(
        directory / f'{stage}_requests.json',
        directory / f'{stage}_requests_manifest.json', 'all')
    if len(rows) > 36:
        raise ValueError('Generation stage exceeds its request limit')
    for row in rows:
        validate_request(row)
    raw = directory / f'{stage}_replies.raw.jsonl'
    replies = directory / f'{stage}_replies.jsonl'
    raw.touch(exist_ok=False)
    replies.touch(exist_ok=False)
    lock = threading.Lock()

    def one(row):
        identity = {k: row[k] for k in (
            'case_id', 'condition', 'request_sha256')}
        check_hold()
        with lock:
            append_event(raw, dict(event='START', **identity,
                                   bound_usd=PER_CALL_BOUND,
                                   model=MODEL, time=time.time()))
        end = dict(event='END', **identity, response_status='failed',
                   served_model=None, dollars=None,
                   dollars_bound=PER_CALL_BOUND)
        answer = None
        started = time.monotonic()
        try:
            response = client.responses.create(**row['request'])
            end.update(served_model=response.model,
                       response_status=response.status,
                       output_text=response.output_text)
            usage = response.usage
            if usage is not None:
                tokens_in, tokens_out = usage.input_tokens, usage.output_tokens
                if (type(tokens_in) is not int or type(tokens_out) is not int
                        or tokens_in < 0 or tokens_out < 0):
                    raise BudgetError('Invalid provider token accounting')
                end.update(tokens_in=tokens_in, tokens_out=tokens_out,
                           dollars=(tokens_in * .25 + tokens_out * 2) / 1e6)
                if tokens_in > 16384 or tokens_out > 8192:
                    raise BudgetError('Provider usage exceeds paid contract')
            if response.model != MODEL or response.status != 'completed':
                raise ValueError('Wrong model or incomplete generation')
            answer = json.loads(response.output_text)
            jsonschema.validate(answer, row['request']['text']['format'][
                'schema'])
            if stage == 'consult':
                v3.parse(answer)
            end['response_status'] = 'completed'
        except Exception as error:
            end.update(response_status='failed',
                       error_type=type(error).__name__,
                       http_status=getattr(error, 'status_code', None))
            answer = None
        end['seconds'] = time.monotonic() - started
        with lock:
            append_event(raw, end)
            append_event(replies, dict(
                **identity, served_model=end['served_model'],
                response_status=end['response_status'], answer=answer))
        return end

    with ThreadPoolExecutor(CONTRACT['workers_per_bank']) as pool:
        return list(pool.map(one, rows))


def blind_requests(directory):
    """
    Reuse the original pool selection and strict checker prompt recipe.
    """

    # This exports candidate refinement prompts but never sends them.
    # Its saved plan supplies the original RNG-7 matching situations.
    v3.export_refine(directory)
    plan, rows = expected_blind(directory)
    if read(directory / 'refine_plan.json') != plan:
        raise ValueError('Original blind-check sampling recipe changed')
    export_requests(directory, 'blind', rows)
    return rows


def expected_blind(directory):
    """
    Reconstruct the original plan without mutating frozen generation files.
    """

    panels, parsed = v3.consult_rules(directory)
    rng = np.random.default_rng(7)
    plan = []
    rows = []
    for index, (_state, _now, rule, _status) in enumerate(parsed):
        if rule is None:
            continue
        matches = [state for state in panels['pool']
                   if v3.executable(rule, state['v3'])]
        if not matches:
            plan.append(dict(case=index, situations=0))
            continue
        situations = [matches[i] for i in sorted(rng.choice(
            len(matches), min(v3.K_SITUATIONS, len(matches)),
            replace=False))]
        plan.append(dict(case=index, situations=len(situations),
                         pool_matches=len(matches),
                         natives=[state['native'] for state in situations]))
        body = v3.body(v3.blind_prompt(situations), v3.blind_schema(),
                       v3.BLIND_VERSION)
        row = dict(case_id=f'blind_{index:03d}',
                   condition='v3_blind', split='blind', model=MODEL,
                   request=body, request_sha256=v3.digest(body))
        validate_request(row)
        rows.append(row)
    return plan, rows


def build_bank(directory, index):
    """
    Retain the original strict filter without a learning-quality gate.
    """

    from scripts.run_rule_bank_pilot_20260928 import rule_json
    panels, parsed = v3.consult_rules(directory)
    kept, status = v3.blind_filtered(directory, parsed, panels, strict=True)
    return dict(
        study=STUDY, generation_index=index, model=MODEL,
        observer='doorkey_v3', mode='scoped',
        rules=[rule_json(rule) for rule in kept],
        blind_check_status=status, corpus_sha256=CORPUS_SHA256,
        source_replies_sha256={stage: fw.digest(
            directory / f'{stage}_replies.jsonl')
            for stage in ('consult', 'blind')})


def generation_worker(batch, index, credential, ledger_path):
    """
    Make one bounded generation attempt under an existing child allowance.
    """

    batch = Path(batch).resolve()
    verify_generation_inputs(batch)
    if type(index) is not int or index not in range(5):
        raise ValueError('Generation index must be between zero and four')
    directory = batch / 'banks' / BANKS[index]
    ledger = CostLedger(str(Path(ledger_path).resolve()))
    status = ledger.status()
    if (status['allowance_usd'] != 1.48 or status['settled_usd'] != 0
            or status['reserved_usd'] != 0 or status['overspent_runs']):
        raise BudgetError('Generation needs its pristine $1.48 child ledger')
    with (directory / 'GENERATION_ATTEMPTED').open('x') as handle:
        handle.write('One generation attempt; no implicit retries.\n')
    run_id = f'{STUDY}_{BANKS[index]}'
    ledger.reserve(run_id, 1.48, note='36 consult + at most36 blind attempts')

    def check_hold():
        status = ledger.status()
        holds = [entry for entry in status['open_reservations']
                 if entry['run_id'] == run_id]
        if (len(holds) != 1 or holds[0]['reserved'] != 1.48
                or status['overspent_runs']):
            raise BudgetError('Full bank-generation reservation is missing')

    client = None
    try:
        client = openai_client(credential)
        consult = collect_stage(directory, 'consult', client, check_hold)
        if any(row['response_status'] != 'completed' for row in consult):
            raise ValueError('Consultation attempt incomplete; preserve it')
        blind_requests(directory)
        blind = collect_stage(directory, 'blind', client, check_hold)
        if any(row['response_status'] != 'completed' for row in blind):
            raise ValueError('Blind-check attempt incomplete; preserve it')
        bank = build_bank(directory, index)
        write_new(directory / 'bank.json', bank)
        rows = consult + blind
        liability = sum(row['dollars'] if row['dollars'] is not None
                        else PER_CALL_BOUND for row in rows)
        ledger.settle(run_id, liability)
        receipt = dict(
            study=STUDY, generation_index=index,
            status='technical_generation_complete', attempts=len(rows),
            rules=len(bank['rules']), settled_liability_usd=liability,
            hashes={name: fw.digest(directory / name)
                    for name in RECEIPT_FILES})
        write_new(directory / 'generation_receipt.json', receipt)
        return receipt
    except Exception as error:
        # Early failures retain the hold. A receipt-write failure can
        # follow successful settlement; report the actual ledger state.
        # Reconciliation never requires resending a paid request.
        held = any(entry['run_id'] == run_id for entry in
                   ledger.status()['open_reservations'])
        write_new(directory / 'generation_failure.json', dict(
            study=STUDY, generation_index=index,
            error_type=type(error).__name__, status='generation_failed',
            reservation_retained=held))
        raise
    finally:
        if client is not None:
            client.close()


def audit_stage(directory, stage):
    """
    Check every frozen request against one durable start and end receipt.
    """

    import jsonschema
    requests = collect.load_frozen(
        directory / f'{stage}_requests.json',
        directory / f'{stage}_requests_manifest.json', 'all')
    raw = [json.loads(line) for line in (
        directory / f'{stage}_replies.raw.jsonl').read_text().splitlines()]
    replies = [json.loads(line) for line in (
        directory / f'{stage}_replies.jsonl').read_text().splitlines()]
    if len(raw) != 2 * len(requests) or len(replies) != len(requests):
        raise ValueError('Incomplete or duplicate generation receipts')
    liability = 0.
    for request in requests:
        identity = {k: request[k] for k in (
            'case_id', 'condition', 'request_sha256')}
        events = [r for r in raw if all(r[k] == v
                  for k, v in identity.items())]
        imported = [r for r in replies if all(r[k] == v
                    for k, v in identity.items())]
        if ([r['event'] for r in events] != ['START', 'END']
                or len(imported) != 1):
            raise ValueError('Generation request/receipt identities differ')
        start, end = events
        if (start['bound_usd'] != PER_CALL_BOUND or start['model'] != MODEL
                or end['served_model'] != MODEL
                or end['response_status'] != 'completed'
                or imported[0]['served_model'] != MODEL
                or imported[0]['response_status'] != 'completed'):
            raise ValueError('Generation reply is not technically complete')
        answer = json.loads(end['output_text'])
        if answer != imported[0]['answer']:
            raise ValueError('Raw and imported generation replies differ')
        jsonschema.validate(answer, request['request']['text']['format'][
            'schema'])
        if end['dollars'] is None:
            liability += PER_CALL_BOUND
        else:
            expected = (end['tokens_in'] * .25 + end['tokens_out'] * 2) / 1e6
            if (not 0 <= end['tokens_in'] <= 16384
                    or not 0 <= end['tokens_out'] <= 8192
                    or end['dollars'] != expected):
                raise ValueError('Generation cost accounting differs')
            liability += expected
    return len(requests), liability


def freeze_generation(batch):
    """
    Verify all five repetitions; never select the best or replace a bank.
    """

    batch = Path(batch).resolve()
    verify_generation_inputs(batch)
    paths = {}
    for index, name in enumerate(BANKS):
        directory = batch / 'banks' / name
        receipt = read(directory / 'generation_receipt.json')
        if (receipt['study'] != STUDY
                or receipt['generation_index'] != index
                or receipt['status'] != 'technical_generation_complete'
                or not 36 <= receipt['attempts'] <= 72
                or set(receipt['hashes']) != set(RECEIPT_FILES)):
            raise ValueError('Generation receipt identity differs')
        for artifact, digest in receipt['hashes'].items():
            if fw.digest(fw.inside(directory, artifact)) != digest:
                raise ValueError('Generation output changed after freezing')
        bank = read(directory / 'bank.json')
        if bank != build_bank(directory, index):
            raise ValueError('Frozen bank differs from original filter')
        if receipt['rules'] != len(bank['rules']):
            raise ValueError('Generation receipt rule count differs')
        consult_n, consult_cost = audit_stage(directory, 'consult')
        blind_n, blind_cost = audit_stage(directory, 'blind')
        plan, requests = expected_blind(directory)
        if (read(directory / 'refine_plan.json') != plan
                or read(directory / 'blind_requests.json') != requests):
            raise ValueError('Blind-check recipe differs from frozen corpus')
        if (consult_n != 36 or blind_n > 36
                or receipt['attempts'] != consult_n + blind_n
                or abs(receipt['settled_liability_usd'] -
                       consult_cost - blind_cost) > 1e-12):
            raise ValueError('Generation receipt totals differ')
        paths[name] = directory / 'bank.json'
    return paths


def training_plan():
    """
    Freeze all training slots before the generation outcomes are known.
    """

    return [dict(index=7 * offset + index, replicate=replicate,
                 seed=14_500_000 + 100 * replicate, arm=arm)
            for offset, replicate in enumerate(REPS)
            for index, arm in enumerate(ARMS)]


def cells(root=ROOT, banks_dir=None):
    """
    Resolve seven arms only after all five bank receipts are validated.
    """

    if banks_dir is None:
        raise ValueError('A technically complete generation batch is needed')
    banks = freeze_generation(banks_dir)
    result = []
    for replicate in REPS:
        for arm in ARMS:
            args = fw.arm_args('dk_fresh', 'none' if arm == 'none'
                               else 'rules_weak', replicate, 'none', root)
            args = replace(args, experiment_id=f'{STUDY}_{arm}',
                           rule_timing_diagnostics=arm != 'none')
            if arm in banks:
                args = replace(args, rule_bank=str(banks[arm]),
                               rule_bank_sha256=fw.digest(banks[arm]))
            fw.cell(result, STUDY, arm, args, replicate)
    return result


def report(rows):
    """
    Summarize a crossed bank-by-training-seed design without pseudo-n.
    """

    if len(rows) != 70:
        return dict(status='PROVISIONAL', completed=len(rows), expected=70)
    if any(type(row.get('auc')) not in (int, float)
           or not np.isfinite(row['auc']) or not 0 <= row['auc'] <= 1
           for row in rows):
        raise ValueError('Regeneration AUC must be finite and in [0, 1]')
    grouped = {arm: {r['seed']: r for r in rows if r['arm'] == arm}
               for arm in ARMS}
    seeds = [14_500_000 + 100 * replicate for replicate in REPS]
    if any(sorted(grouped[arm]) != seeds for arm in ARMS):
        raise ValueError('Regeneration report needs all70 planned cells')
    if any(len({grouped[arm][seed]['initial_sha256'] for arm in ARMS}) != 1
           for seed in seeds):
        raise ValueError('Regeneration initial policies differ')
    baseline = np.array([grouped['none'][seed]['auc'] for seed in seeds])
    values = np.array([[grouped[bank][seed]['auc'] for seed in seeds]
                       for bank in BANKS])
    if not np.isfinite(values).all() or not np.isfinite(baseline).all():
        raise ValueError('Regeneration report contains nonfinite AUC')
    delta = values - baseline
    # Resample banks and seeds separately, preserving shared controls.
    # Five generation repetitions are not fifty independent banks.
    rng = np.random.default_rng(20261005)
    sampled = []
    for _ in range(10000):
        banks = rng.integers(0, 5, 5)
        chosen = rng.integers(0, 10, 10)
        sampled.append(float(delta[np.ix_(banks, chosen)].mean()))
    historical = fw.paired(grouped['historical_selected'], grouped['none'],
                           'historical selected - none')
    historical_delta = [grouped['historical_selected'][seed]['auc'] -
                        grouped['none'][seed]['auc'] for seed in seeds]
    if np.ptp(historical_delta) < 1e-14:
        historical.update(p=None, test_status='undefined_zero_variance')
    return dict(
        status='complete_exploratory', generation_repetitions=5,
        training_seeds=10, learning_cells=70,
        bank_mean_auc=dict(zip(BANKS, values.mean(axis=1).tolist())),
        bank_mean_delta=dict(zip(BANKS, delta.mean(axis=1).tolist())),
        mean_regenerated_minus_none=float(delta.mean()),
        crossed_bootstrap_ci95=np.quantile(sampled, [.025, .975]).tolist(),
        ci_limitation='Exploratory percentile interval with only five banks',
        historical_selected_minus_none=historical)
