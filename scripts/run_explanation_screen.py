"""Prepare, collect and score an Aleph-only common-state explanation screen."""

import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import jsonschema

from scripts.explanation_screen_panel import (
    FORMATS, PHASES, build_panel, file_hash, make_request, object_hash,
    response_schema,
)


EXPERIMENT = 'aleph_explanation_screen_20260915_v1'
MODEL = 'gpt-oss-120b'
ENDPOINT = 'https://inference.vulcan.alliancecan.ca/v1'
RUNNER = 'scripts/run_explanation_screen.py'
WORKER = 'scripts/submit_explanation_screen.sh'
PANEL_MODULE = 'scripts/explanation_screen_panel.py'
ROOT = Path(__file__).resolve().parents[1]
REQUEST_LIMIT = 108
MAX_ATTEMPTS = 2
TIMEOUT_SECONDS = 180
DEADLINE_SECONDS = 180 * 60


def write_json(path, value):
    """Write a readable artifact; immutable files get hashes separately."""
    Path(path).write_text(json.dumps(value, indent=2) + '\n',
                          encoding='utf-8')


def append_row(path, row):
    """Persist request starts before sending, including crash uncertainty."""
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(path):
    """Ignore only a final torn append; retain its existence as evidence."""
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines(keepends=True) if line.endswith('\n')]


def credentials(path):
    """Read Aleph credentials as BOM-safe data, never execute an env file."""
    from dotenv import dotenv_values
    values = (dotenv_values(path, encoding='utf-8-sig', interpolate=False)
              if Path(path).is_file() else {})
    key = (os.getenv('TYK_KEY') or os.getenv('ALEPH_API_KEY') or
           values.get('TYK_KEY') or values.get('ALEPH_API_KEY') or '').strip()
    if not key:
        raise ValueError('No TYK_KEY/ALEPH_API_KEY in the environment or '
                         f'{path}; no OpenAI fallback is permitted')
    return key


def verify_batch(batch, check_source=True):
    """Verify immutable questions, truth, contract and executable source."""
    batch = Path(batch).resolve()
    if not (batch / 'READY').is_file():
        raise ValueError('Incomplete preparation: READY is absent')
    if file_hash(batch / 'manifest.json') != (
            batch / 'manifest.sha256').read_text().strip():
        raise ValueError('Manifest changed after preparation')
    manifest = json.loads((batch / 'manifest.json').read_text())
    for name, checksum in manifest['artifact_sha256'].items():
        if file_hash(batch / name) != checksum:
            raise ValueError(f'Frozen question/truth changed: {name}')
    if check_source:
        for name, checksum in manifest['source_sha256'].items():
            if file_hash(batch / 'code' / name) != checksum:
                raise ValueError(f'Frozen source changed: {name}')
    if (manifest['provider'] != 'aleph' or manifest['endpoint'] != ENDPOINT
            or manifest['model'] != MODEL
            or manifest['logical_requests'] != REQUEST_LIMIT
            or manifest['max_attempts'] != MAX_ATTEMPTS):
        raise ValueError('Unsupported screen contract')
    return manifest


def prepare(root, corpus, credential_file):
    """Freeze questions plus committed source; never submit a Slurm job."""
    batch = root / 'results/explanation_screens' / EXPERIMENT
    if batch.exists():
        verify_batch(batch)
        return batch
    panel = build_panel(corpus)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'],
                   cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ['git', 'archive', '--format=zip', commit], cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        if not {RUNNER, WORKER, PANEL_MODULE} <= set(handle.namelist()):
            raise ValueError('Commit the screen implementation first')
        batch.mkdir(parents=True)
        handle.extractall(batch / 'code')
    requests = []
    for kind in FORMATS:
        for case in panel['cases']:
            request = make_request(case, kind)
            requests.append({
                'format': kind, 'case_id': case['case_id'],
                'request': request, 'request_sha256': object_hash(request),
            })
    write_json(batch / 'panel.json', panel)
    write_json(batch / 'requests.json', requests)
    packages = {name: version(name) for name in
                ('minigrid', 'gymnasium', 'openai', 'jsonschema')}
    manifest = {
        'experiment_id': EXPERIMENT, 'author': 'mahdiehmn',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'exploratory engineering screen; no learning claim',
        'commit': commit, 'provider': 'aleph', 'endpoint': ENDPOINT,
        'model': MODEL, 'reasoning_effort': 'low',
        'logical_requests': len(requests), 'max_attempts': MAX_ATTEMPTS,
        'sdk_retries': 0, 'max_output_tokens': 32768,
        'timeout_seconds': TIMEOUT_SECONDS,
        'deadline_seconds_per_worker': DEADLINE_SECONDS,
        'formats': list(FORMATS), 'packages': packages,
        'credential_file': str(Path(credential_file).resolve()),
        'artifact_sha256': {name: file_hash(batch / name)
                            for name in ('panel.json', 'requests.json')},
        'source_sha256': {
            p.relative_to(batch / 'code').as_posix(): file_hash(p)
            for p in (batch / 'code').rglob('*') if p.is_file()},
    }
    write_json(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(
        file_hash(batch / 'manifest.json') + '\n')
    (batch / 'slurm').mkdir()
    (batch / 'workers').mkdir()
    (batch / 'READY').write_text(commit + '\n')
    verify_batch(batch)
    return batch


def submission_command(batch):
    """Use three workers and no array throttle; return, never execute."""
    args = ['sbatch', '--parsable', '--job-name=expl_screen', '--array=0-2',
            f'--output={batch.as_posix()}/slurm/%x_%A_%a.out',
            (batch / 'code' / WORKER).as_posix(), batch.as_posix()]
    return shlex.join(args)


def collect(directory, requests, client, manifest, clock=time.monotonic,
            identity_fields=()):
    """Collect each planned request once; retry only bounded transients."""
    from teachers.base import call_with_cold_start_retry
    from teachers.reliability import attempt_context, failure_details
    start = clock()
    deadline = start + manifest['deadline_seconds_per_worker']
    failures = consecutive = finished = 0
    reason = 'all_requests_attempted'
    for item in requests:
        if clock() >= deadline:
            reason = 'start_deadline_reached'
            break
        case_start = clock()
        # New diagnostic versions bind conditions as well as questions.
        identity = {name: item[name] for name in identity_fields}

        def sink(event):
            """Retain safe request accounting even if the process dies."""
            append_row(directory / 'attempts.jsonl', {
                'case_id': item['case_id'], 'format': item['format'],
                'at_utc': datetime.now(timezone.utc).isoformat(),
                **event, **identity})

        def request():
            """Check the start deadline on retries as well as first calls."""
            if clock() >= deadline:
                raise TimeoutError('Screen start deadline reached')
            return client.teacher_response(
                model=manifest['model'], **item['request'],
                reasoning={'effort': manifest['reasoning_effort']},
                max_output_tokens=manifest['max_output_tokens'])

        record = {k: item[k] for k in
                  ('case_id', 'format', 'request_sha256')}
        record.update(identity)
        with attempt_context(sink):
            try:
                response = call_with_cold_start_retry(
                    request, max_wait_s=max(0, deadline - clock()))
                raw = response.output_text
                record['raw_output'] = raw
                usage = getattr(response, 'usage', None)
                record['tokens_in'] = getattr(usage, 'input_tokens', None)
                record['tokens_out'] = getattr(usage, 'output_tokens', None)
                answer = json.loads(raw)
                jsonschema.validate(answer, response_schema(item['format']))
                record.update(status='valid', answer=answer)
                consecutive = 0
            except Exception as exc:
                details = failure_details(exc)
                details.pop('message', None)
                record.update(status='failed', failure=details)
                failures += 1
                consecutive += 1
        record['elapsed_seconds'] = clock() - case_start
        append_row(directory / 'responses.jsonl', record)
        finished += 1
        print(f'{item["format"]} {item["case_id"]}: {record["status"]}; '
              f'{finished}/{len(requests)}, failures {failures}', flush=True)
        if record.get('failure', {}).get('http_status') in (400, 401, 403, 404):
            reason = 'configuration_or_auth_failure'
            break
        if consecutive >= 3:
            reason = 'three_consecutive_failures'
            break
    summary = {'finished': finished, 'failed': failures,
               'unattempted': len(requests) - finished,
               'elapsed_seconds': clock() - start, 'stop_reason': reason}
    write_json(directory / 'summary.json', summary)
    return summary


def run_worker(batch, index):
    """Load only the pinned Aleph client and claim a worker exclusively."""
    from teachers.aleph import AlephClient
    batch = Path(batch).resolve()
    manifest = verify_batch(batch)
    if ROOT != batch / 'code':
        raise ValueError('Use the worker in the frozen code directory')
    if index not in range(len(FORMATS)):
        raise ValueError('Worker must be 0, 1 or 2')
    if version('minigrid') != manifest['packages']['minigrid']:
        raise ValueError('MiniGrid version differs from the frozen panel')
    key = credentials(manifest['credential_file'])
    directory = batch / 'workers' / FORMATS[index]
    directory.mkdir(exist_ok=True)
    if (directory / 'DONE').exists():
        print('Worker already finished; no requests repeated.')
        return 0
    # A second worker cannot repeat uncertain calls after a lost job.
    with (directory / 'STARTED').open('x') as handle:
        handle.write(datetime.now(timezone.utc).isoformat() + '\n')
    os.environ.update(LLM_PROVIDER='aleph',
                      LLM_MAX_ATTEMPTS=str(MAX_ATTEMPTS),
                      LLM_MAX_SDK_RETRIES='0')
    client = AlephClient(base_url=ENDPOINT, api_key=key,
                         timeout=manifest['timeout_seconds'], max_retries=0)
    requests = json.loads((batch / 'requests.json').read_text())
    requests = [r for r in requests if r['format'] == FORMATS[index]]
    try:
        summary = collect(directory, requests, client, manifest)
    finally:
        client.close()
    (directory / 'DONE').write_text(summary['stop_reason'] + '\n')
    return int(summary['failed'] > 0 or summary['unattempted'] > 0)


def report(batch):
    """Score actual response content; keep missing and manual work visible."""
    batch = Path(batch).resolve()
    verify_batch(batch)
    panel = json.loads((batch / 'panel.json').read_text())
    cases = {c['case_id']: c for c in panel['cases']}
    requests = json.loads((batch / 'requests.json').read_text())
    expected = {(r['format'], r['case_id']): r for r in requests}
    metrics, review, answers = {}, [], {}
    for kind in FORMATS:
        directory = batch / 'workers' / kind
        rows = load_rows(directory / 'responses.jsonl')
        seen = set()
        valid = correct = critical_correct = 0
        phase_ok, phase_n = Counter(), Counter()
        for row in rows:
            case_id = row['case_id']
            key = (kind, case_id)
            if (case_id in seen or key not in expected
                    or row['format'] != kind
                    or row['request_sha256'] !=
                    expected[key]['request_sha256']):
                raise ValueError('Duplicate or mismatched response identity')
            seen.add(case_id)
            case = cases[case_id]
            if row['status'] != 'valid':
                continue
            answer = json.loads(row['raw_output'])
            jsonschema.validate(answer, response_schema(kind))
            if answer != row['answer']:
                raise ValueError('Parsed answer differs from persisted text')
            exact = all(answer[name] == case['truth'][name]
                        for name in ('reference', 'alternative'))
            valid += 1
            correct += int(exact)
            phase_ok[case['phase']] += int(exact)
            phase_n[case['phase']] += 1
            critical_correct += int(exact and case['stratum'] != 'other')
            answers[key] = exact
            prose = {k: v for k, v in answer.items()
                     if k not in ('reference', 'alternative')}
            review.append({
                'format': kind, 'case_id': case_id, 'phase': case['phase'],
                'stratum': case['stratum'], 'physics_exact': exact,
                'text': json.dumps(prose), 'reviewer': '',
                'factual_ok': '', 'support_current_local_or_privileged': '',
                'adds_beyond_action_label': '', 'notes': '',
            })
        attempts = load_rows(directory / 'attempts.jsonl')
        by_case = {}
        for event in attempts:
            cid = event['case_id']
            if (kind, cid) not in expected or event.get('format') != kind:
                raise ValueError('Attempt has an unknown request identity')
            by_case.setdefault(cid, []).append(event)
        for cid, events in by_case.items():
            opened = None
            number = 0
            for event in events:
                if event['event'] == 'request_start':
                    number += 1
                    if opened is not None or number > MAX_ATTEMPTS:
                        raise ValueError('Overlapping or excess attempts')
                    opened = number
                elif event['event'] == 'request_end':
                    if opened is None or event['attempt'] != opened:
                        raise ValueError('Unlinked request end')
                    opened = None
                else:
                    raise ValueError('Unexpected attempt event')
                if event['attempt'] != number:
                    raise ValueError('Attempt index mismatch')
        for row in rows:
            events = by_case.get(row['case_id'], [])
            if (not events or events[-1]['event'] != 'request_end'
                    or (row['status'] == 'valid'
                        and events[-1].get('outcome') != 'ok')):
                raise ValueError('Response is not backed by a finished call')
        starts = sum(r['event'] == 'request_start' for r in attempts)
        ends = sum(r['event'] == 'request_end' for r in attempts)
        critical_n = sum(c['stratum'] != 'other' for c in cases.values())
        metrics[kind] = {
            'planned': 36, 'returned': len(rows), 'valid': valid,
            'failed': len(rows) - valid, 'missing': 36 - len(rows),
            'exact_action_pairs': correct, 'exact_over_planned': correct / 36,
            'valid_by_phase': dict(phase_n),
            'exact_by_phase': dict(phase_ok),
            'critical_pairs_correct': critical_correct,
            'critical_pairs_planned': critical_n,
            'candidate_for_manual_review': (
                len(rows) == 36 and valid >= 35 and correct >= 33
                and all(phase_ok[p] >= 10 for p in PHASES)
                and critical_correct == critical_n and starts == ends),
            'request_starts': starts, 'request_ends': ends,
            'unclosed_attempts': starts - ends,
            'known_tokens_in': sum(r.get('tokens_in') or 0 for r in attempts
                                   if r['event'] == 'request_end'),
            'known_tokens_out': sum(r.get('tokens_out') or 0 for r in attempts
                                    if r['event'] == 'request_end'),
            'unknown_usage_attempts': sum(
                r.get('tokens_out') is None for r in attempts
                if r['event'] == 'request_end') + starts - ends,
            'elapsed_seconds': sum(r['elapsed_seconds'] for r in rows),
        }
        if starts > 72 or starts < ends:
            raise ValueError('Request accounting violates the attempt bound')
    paired = [cid for cid in cases if all((k, cid) in answers for k in FORMATS)]
    no_change = sum(all(c['truth'][a] == c['unchanged_baseline']
                       for a in ('reference', 'alternative'))
                    for c in cases.values())
    result = {
        'experiment_id': EXPERIMENT, 'formats': metrics,
        'valid_cases_paired_across_all_formats': len(paired),
        'paired_exact_counts': {
            k: sum(answers[k, cid] for cid in paired) for k in FORMATS},
        'no_change_baseline_exact_pairs': no_change,
        'manual_semantic_review': 'pending; see manual_review_template.csv',
        'learning_benefit': 'not tested',
        'claim': 'Stratified development panel; not rollout error prevalence',
    }
    write_json(batch / 'report.json', result)
    if review and not (batch / 'manual_review_template.csv').exists():
        # Re-reporting never overwrites a completed human review file.
        with (batch / 'manual_review_template.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    print(json.dumps(result, indent=2))
    return result


def main():
    """Default to a free preview; user explicitly prepares and submits."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run-worker', type=int)
    actions.add_argument('--report', action='store_true')
    parser.add_argument('--corpus', type=Path,
                        default=ROOT / 'results/advising_strength')
    parser.add_argument('--batch', type=Path,
                        default=ROOT / 'results/explanation_screens'
                        / EXPERIMENT)
    parser.add_argument('--credential-file', type=Path,
                        default=Path.home() / '.aleph_tyk.env')
    args = parser.parse_args()
    if args.report:
        report(args.batch)
        return 0
    if args.run_worker is not None:
        return run_worker(args.batch, args.run_worker)
    print('DoorKey: 36 fixed states x 3 formats = 108 logical requests; '
          'at most 216 HTTP attempts. Aleph gpt-oss-120b only; no training.')
    print('Three parallel workers; no array throttle. No paid API fallback, '
          'no OpenAI ledger changes.')
    if args.check:
        panel = build_panel(args.corpus)
        credentials(args.credential_file)
        print(f'PASS: {len(panel["cases"])} simulator-validated states; '
              'Aleph credential present. No request made.')
    if args.prepare:
        credentials(args.credential_file)
        batch = prepare(ROOT, args.corpus, args.credential_file)
        command = submission_command(batch)
        target = ROOT / 'results/submit_aleph_explanation_screen_20260915.sh'
        target.write_text('#!/bin/bash\nset -euo pipefail\n' + command + '\n',
                          encoding='utf-8')
        print(f'Prepared: {batch}\nSubmit: bash {target}')
        print('No API request or Slurm submission made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
