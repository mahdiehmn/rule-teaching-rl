"""Collect frozen prompt requests: validated, one attempt each, hard-capped.

Works for both request exporters, which deliberately have no API client:
`run_prompt_reliability --prepare` (v3) and
`prepare_scaffold_development --prepare` (v4 hybrid scaffold).

Guarantees:

- Identity. `--requests` must be the batch's full `requests.json`. Its
  digest must equal the manifest's `requests_sha256`, and every body must
  hash to its own `request_sha256`, before anything is sent. Only then is
  the split selected (`development` unless `--split gate` is given;
  `--split all` sends a whole bank export such as the language pilot's).
- Endpoint. One fixed base URL; an httpx hook refuses anything but
  POST https://api.openai.com/v1/responses. No retries (max_retries=0).
- Hard cap by reservation. Before dispatch, the whole pending batch is
  bounded: input tokens <= UTF-8 bytes of the serialized request (a token
  covers at least one byte) plus 64, output tokens <= max_output_tokens.
  Nothing is sent unless recorded spend + unresolved-attempt bounds + the
  pending bound fit under --cap-usd.
- Durable attempts. A START line is flushed and fsynced before each call
  and an END line after it. On resume, finished requests are skipped and a
  START without END is an UNRESOLVED attempt: never resent, counted at its
  full bound. Failures are never retried.

Outputs, next to each other:
    <out>.raw.jsonl  START/END events (END: served model, status, raw
                     text, token usage, dollars, seconds, error)
    <out>.jsonl      import rows for --report: case_id, condition,
                     request_sha256, served_model, response_status, answer
                     (only served replies; missing rows count as failures)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

BASE_URL = 'https://api.openai.com/v1'
ENDPOINT = BASE_URL + '/responses'
PRICES = {  # USD per 1M tokens (input, output); repo price table
    'gpt-5-mini': (0.25, 2.00), 'gpt-5': (1.25, 10.00), 'gpt-5.5': (5.00, 30.00),
    'gpt-4o-mini': (0.15, 0.60),
}
OVERHEAD_TOKENS = 64


def digest(value):
    """Content digest, identical to run_prompt_reliability.digest."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def rate(model):
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES[prefix]
    raise ValueError(f'No price for {model}')


def bound(row):
    rin, rout = rate(row['model'])
    tokens_in = len(json.dumps(row['request']).encode()) + OVERHEAD_TOKENS
    return (tokens_in * rin
            + row['request']['max_output_tokens'] * rout) / 1e6


def load_frozen(requests_path, manifest_path, split):
    rows = json.loads(Path(requests_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if digest(rows) != manifest['requests_sha256']:
        raise SystemExit('requests.json does not match the manifest')
    for r in rows:
        if digest(r['request']) != r['request_sha256']:
            raise SystemExit(f"Body hash mismatch: {r['case_id']}")
        if r['request']['model'] != r['model']:
            raise SystemExit(f"Model mismatch: {r['case_id']}")
    keys = [(r['case_id'], r['condition']) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit('Duplicate (case, condition) in requests')
    return [r for r in rows if split == 'all' or r['split'] == split]


def read_events(raw_path):
    starts, ends = {}, {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            key = (e['case_id'], e['condition'])
            (starts if e['event'] == 'START' else ends)[key] = e
    return starts, ends


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--requests', required=True,
                   help="the batch's full requests.json")
    p.add_argument('--manifest', required=True,
                   help="the batch's manifest.json")
    p.add_argument('--split', choices=('development', 'gate', 'train',
                                       'audit', 'all'),
                   default='development',
                   help='"all" sends every split in the file (bank export)')
    p.add_argument('--out', required=True, help='output stem')
    p.add_argument('--dotenv',
                   default='.env')
    p.add_argument('--cap-usd', type=float, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--price-only', action='store_true')
    args = p.parse_args()

    rows = load_frozen(args.requests, args.manifest, args.split)
    out = Path(args.out)
    raw_path = out.with_name(out.name + '.raw.jsonl')
    reply_path = out.with_name(out.name + '.jsonl')
    starts, ends = read_events(raw_path)
    by_key = {(r['case_id'], r['condition']): r for r in rows}
    if set(starts) - set(by_key):
        raise SystemExit('Existing receipts belong to another batch/split')
    # errored attempts have unknown billing: count them at their bound
    spent = sum(e['dollars'] if e.get('dollars') is not None
                else e.get('dollars_bound', 0.0) for e in ends.values())
    unresolved = [k for k in starts if k not in ends]
    unresolved_bound = sum(bound(by_key[k]) for k in unresolved)
    todo = [r for k, r in by_key.items() if k not in starts]
    pending = sum(bound(r) for r in todo)
    exposure = spent + unresolved_bound + pending
    print(f'{len(rows)} {args.split} requests ({rows[0]["model"]}), '
          f'batch verified against manifest')
    print(f'finished {len(ends)} (recorded ${spent:.3f}); unresolved '
          f'{len(unresolved)} (bound ${unresolved_bound:.3f}, never resent); '
          f'to send {len(todo)} (bound ${pending:.3f})')
    print(f'worst-case total ${exposure:.3f} vs cap ${args.cap_usd:.2f}')
    if exposure > args.cap_usd:
        raise SystemExit('Refusing: worst case exceeds the cap; nothing sent')
    if args.price_only or not todo:
        return

    import httpx
    from dotenv import dotenv_values
    from openai import OpenAI

    def guard(request):
        if request.method != 'POST' or str(request.url) != ENDPOINT:
            raise RuntimeError(f'Blocked request to {request.url}')

    client = OpenAI(api_key=dotenv_values(args.dotenv)['OPENAI_API_KEY'],
                    base_url=BASE_URL, max_retries=0, timeout=300,
                    http_client=httpx.Client(
                        timeout=300, event_hooks={'request': [guard]}))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    total = [spent]

    def append(path, record):
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def one(r):
        key = dict(case_id=r['case_id'], condition=r['condition'],
                   request_sha256=r['request_sha256'])
        with lock:
            append(raw_path, dict(event='START', **key, model=r['model'],
                                  bound_usd=bound(r), time=time.time()))
        t0 = time.time()
        end = dict(event='END', **key, served_model=None,
                   response_status='failed', dollars=None)
        answer = None
        try:
            resp = client.responses.create(**r['request'])
            usage = resp.usage
            rin, rout = rate(r['model'])
            status = resp.status
            text = resp.output_text
            if status == 'completed':
                try:
                    answer = json.loads(text)
                except json.JSONDecodeError:
                    status = 'unparseable'
            end.update(served_model=resp.model, response_status=status,
                       output_text=text, tokens_in=usage.input_tokens,
                       tokens_out=usage.output_tokens,
                       dollars=(usage.input_tokens * rin
                                + usage.output_tokens * rout) / 1e6,
                       incomplete=(str(resp.incomplete_details)
                                   if resp.incomplete_details else None))
        except Exception as exc:
            # Unknown whether the provider billed it: keep the bound.
            end.update(error=repr(exc)[:500], dollars_bound=bound(r))
        end['seconds'] = round(time.time() - t0, 2)
        with lock:
            append(raw_path, end)
            total[0] += end['dollars'] or 0.0
            if end['served_model'] is not None:
                append(reply_path, dict(
                    **key, served_model=end['served_model'],
                    response_status=end['response_status'], answer=answer))
        return end['response_status']

    with ThreadPoolExecutor(args.workers) as pool:
        statuses = list(pool.map(one, todo))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    print(f'sent {len(statuses)}: {counts}; recorded spend now '
          f'${total[0]:.3f}')
    print(f'receipts: {raw_path}\nreport input: {reply_path}')


if __name__ == '__main__':
    main()
