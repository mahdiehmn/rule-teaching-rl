"""Price, check, collect, embed and freeze an explanation-format bank.

Stages are explicit; nothing is paid
without --collect or --freeze, and both first RESERVE their full
worst-case bound in the named ledger (refusing if the pool cannot cover
it):

    --price          exact per-request bounds for all 256 requests (input
                     bound = serialized request bytes, which cannot be
                     fewer than its tokens; output = the request's cap)
                     plus the embedding bound
    --check-ledger   read-only: is that bound free in the given ledger?
    --collect        reserve, then ONE attempt per case, zero SDK retries;
                     every raw reply and failure is kept; settle known
                     costs plus the full bound of any unknown-cost attempt
    --freeze         reserve the embedding bound, embed all known prose
                     once (cached), settle; bank JSON + sha256 +
                     independent audit sheet

The study launcher (scripts/run_explanation_formats_20260925.py) runs the
same functions from an archived snapshot inside one Slurm job per writer,
so ledger admissions never coincide.

Every stage takes --access: full_state (the primary, privileged
explanation writer) or local_only (an opt-in comparison). Run ids,
request identities and bank metadata carry the access and the request
version, so the two conditions can never be confused or overwritten.
--pair-replies freezes a bank on the cases known under BOTH conditions.

This module never chooses cases or actions: they are the contrastive
lesson panel's 256 cases and fixed action pairs.

The key is read from --credential-file as dotenv data (never sourced or
printed) unless OPENAI_API_KEY is already exported; the client pins the
OpenAI endpoint, zero SDK retries and a transport guard that refuses any
other endpoint or an oversized request body.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.explanation_formats import (
    ACCESS, FORMATS, PRIMARY_ACCESS, audit_sheet, freeze_bank, known_cases,
    parse_reply, request, request_identity)
from teachers.budget import BudgetError, CostLedger, PriceTable

PRICES = 'configs/explanation_prices_2026-09-10.json'
MODEL, EFFORT = 'gpt-5-mini', 'low'
MAX_OUTPUT_TOKENS = 4096
MAX_INPUT_TOKENS = 16_384
EMBED_MODEL = 'text-embedding-3-small'
MAX_TEXT_CHARS = 1_000   # longer prose is truncated before embedding
OPENAI_BASE = 'https://api.openai.com/v1'


def run_id(access):
    return f'explanation_formats_20260925_{access}_collection'


def wire_guard(request):
    """Runs inside httpx before dispatch: two endpoints, bounded bodies."""
    url = str(request.url)
    if (request.method != 'POST' or url not in (
            OPENAI_BASE + '/responses', OPENAI_BASE + '/embeddings')):
        raise BudgetError(f'Unexpected paid request endpoint: {url}')
    if url.endswith('/responses') and len(request.content) > MAX_INPUT_TOKENS:
        raise BudgetError('Serialized request exceeds the priced input bound')


def openai_client(credential_file):
    """The project's key-only dotenv loader; zero retries; guarded wire."""
    import httpx
    from openai import OpenAI
    from scripts.run_advising_strength_grid import backend_environment

    environment = backend_environment('openai', credential_file)
    return OpenAI(
        api_key=environment['OPENAI_API_KEY'], base_url=OPENAI_BASE,
        organization=os.environ.get('OPENAI_ORG_ID'),
        project=os.environ.get('OPENAI_PROJECT_ID'),
        max_retries=0, timeout=180,
        http_client=httpx.Client(event_hooks={'request': [wire_guard]}))


def requests_for(panel, access=PRIMARY_ACCESS):
    out = []
    for case in panel['cases']:
        body = request(case, access, MODEL, EFFORT, MAX_OUTPUT_TOKENS)
        size = len(json.dumps(body).encode('utf-8'))
        if size > MAX_INPUT_TOKENS:
            raise BudgetError(f"{case['case_id']} request exceeds the input "
                              'bound')
        out.append((case, body, size))
    return out


def price(panel, prices, access=PRIMARY_ACCESS):
    requests = requests_for(panel, access)
    chat = sum(prices.call_bound(MODEL, size, MAX_OUTPUT_TOKENS)
               for _case, _body, size in requests)
    # Three prose texts per case at most, each capped at MAX_TEXT_CHARS
    # characters (a token is at least one character).
    embed = prices.embedding_bound(EMBED_MODEL, 3 * len(requests),
                                   MAX_TEXT_CHARS)
    return dict(requests=len(requests), chat_bound_usd=chat,
                embedding_bound_usd=embed,
                total_bound_usd=chat + embed,
                max_request_bytes=max(s for _c, _b, s in requests),
                information_access=access,
                model=MODEL, effort=EFFORT,
                max_output_tokens=MAX_OUTPUT_TOKENS)


def check_ledger(path, amount):
    status = CostLedger(str(path)).status()
    if status['overspent_runs']:
        raise BudgetError('Ledger records overspent runs; resolve first')
    return dict(available_usd=status['available_usd'],
                needed_usd=amount,
                sufficient=status['available_usd'] >= amount)


def collect(panel, client, prices, ledger_path, out, workers=8,
            access=PRIMARY_ACCESS):
    """Reserve the chat bound, then exactly one attempt per case."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bound = price(panel, prices, access)['chat_bound_usd']
    identity = request_identity(panel, access, model=MODEL, effort=EFFORT,
                                max_output_tokens=MAX_OUTPUT_TOKENS)
    CostLedger(str(ledger_path)).reserve(
        run_id(access), bound,
        note=f'explanation formats bank, {access}, '
             f'{len(panel["cases"])} single attempts, '
             f'{identity["request_version"]}')
    rate_in, rate_out = prices.chat_rates(MODEL)

    def one(item):
        case, body, size = item
        row = dict(case_id=case['case_id'], attempts=1,
                   bound_usd=prices.call_bound(MODEL, size,
                                               MAX_OUTPUT_TOKENS))
        started = time.time()
        try:
            response = client.responses.create(**body)
            usage = getattr(response, 'usage', None)
            if usage is not None:
                row['tokens_in'] = int(usage.input_tokens)
                row['tokens_out'] = int(usage.output_tokens)
                row['dollars'] = (row['tokens_in'] * rate_in
                                  + row['tokens_out'] * rate_out) / 1e6
            row['response_status'] = getattr(response, 'status', None)
            row['output_text'] = response.output_text
            answer = json.loads(response.output_text)
            parse_reply(answer)
            row.update(status='valid', answer=answer)
        except Exception as error:  # recorded, never retried
            row.update(status='failed', error_type=type(error).__name__,
                       http_status=getattr(error, 'status_code', None))
        row['seconds'] = time.time() - started
        return row

    rows = []
    # Each reply is flushed as soon as it (and every earlier one) is back,
    # so an interrupted collection keeps its paid evidence.
    with ThreadPoolExecutor(workers) as pool, \
            (out / 'raw_replies.jsonl').open('w', encoding='utf-8') as handle:
        for row in pool.map(one, requests_for(panel, access)):
            rows.append(row)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
    known = sum(r.get('dollars', 0.0) for r in rows)
    unknown = sum(r['bound_usd'] for r in rows if 'dollars' not in r)
    CostLedger(str(ledger_path)).settle(run_id(access), known + unknown)
    summary = dict(**identity, requests=len(rows),
                   valid=sum(r['status'] == 'valid' for r in rows),
                   failed=sum(r['status'] != 'valid' for r in rows),
                   known_usd=known, unknown_attempt_bound_usd=unknown,
                   settled_usd=known + unknown, reserved_usd=bound)
    (out / 'collection_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def replies_by_case(path):
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding='utf-8').splitlines() if line]
    return {r['case_id']: r for r in rows}


def prose_texts(panel, replies):
    texts = set()
    for case in panel['cases']:
        row = replies.get(case['case_id'], {})
        if row.get('status') != 'valid':
            continue
        parsed = parse_reply(row['answer'])
        if parsed['plain']:
            texts.add(parsed['plain']['text'])
        if parsed['contrastive']:
            texts.update(parsed['contrastive'].values())
    return sorted(t[:MAX_TEXT_CHARS] for t in texts)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(panel, replies, provider, out, access=PRIMARY_ACCESS,
           restrict_to=None):
    """Freeze the bank; provider.embed must be cache-backed."""
    out = Path(out)

    def embed(texts):
        return provider.embed([t[:MAX_TEXT_CHARS] for t in texts])

    identity = request_identity(panel, access, model=MODEL, effort=EFFORT,
                                max_output_tokens=MAX_OUTPUT_TOKENS)
    bank = freeze_bank(panel, replies, embed, access, identity, restrict_to)
    path = out / 'format_bank.json'
    path.write_text(json.dumps(bank, allow_nan=False), encoding='utf-8')
    (out / 'format_bank.sha256').write_text(sha256_file(path) + '\n')
    sheet = audit_sheet(bank)
    if sheet:
        with (out / 'audit_sheet.csv').open('w', newline='',
                                            encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
            writer.writeheader()
            writer.writerows(sheet)
    return bank


def embed_and_freeze(panel, replies, client, prices, ledger_path, out,
                     access=PRIMARY_ACCESS, restrict_to=None):
    """Reserve the embedding bound, embed once (cached), freeze, settle.

    The bound covers every known prose text of every valid reply, a
    superset of what the frozen intersection embeds.
    """
    from teachers.embeddings import EmbeddingProvider

    out = Path(out)
    texts = prose_texts(panel, replies)
    bound = prices.embedding_bound(EMBED_MODEL, max(1, len(texts)),
                                   MAX_TEXT_CHARS)
    ledger = CostLedger(str(ledger_path))
    name = run_id(access) + '_embeddings'
    ledger.reserve(name, bound, note=f'{len(texts)} prose texts, {access}, '
                                     'cached once')
    provider = EmbeddingProvider(
        EMBED_MODEL, cache_path=str(out / 'embedding_cache.json'),
        client=client)
    try:
        bank = freeze(panel, replies, provider, out, access, restrict_to)
    finally:
        provider.save()
        tokens = provider.num_tokens
        # An embedding request without reported usage keeps its full bound.
        actual = (tokens * prices.embedding_rate(EMBED_MODEL) / 1e6
                  if tokens is not None
                  else (bound if provider.num_requests else 0.0))
        ledger.settle(name, actual)
    return bank, dict(texts=len(texts), reserved_usd=bound,
                      settled_usd=actual, requests=provider.num_requests,
                      tokens=tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True,
                        help='contrastive_lessons_20260924_v1/panel.json')
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--price', action='store_true')
    stage.add_argument('--check-ledger', type=Path)
    stage.add_argument('--collect', action='store_true')
    stage.add_argument('--freeze', action='store_true')
    parser.add_argument('--access', choices=ACCESS, default=PRIMARY_ACCESS)
    parser.add_argument('--pair-replies', type=Path,
                        help='other access condition raw_replies.jsonl; '
                             'freeze only cases known under both')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--credential-file', type=Path, default=Path('.env'),
                        help='dotenv file holding OPENAI_API_KEY (read as '
                             'data; an exported key takes precedence)')
    args = parser.parse_args()
    panel = json.loads(Path(args.panel).read_text())
    prices = PriceTable.load(PRICES)
    bounds = price(panel, prices, args.access)
    if args.price:
        print(json.dumps(bounds, indent=2))
        return 0
    if args.check_ledger:
        print(json.dumps(check_ledger(args.check_ledger,
                                      bounds['total_bound_usd']), indent=2))
        return 0
    if args.ledger is None or args.out is None:
        parser.error('--collect/--freeze need --ledger and --out')
    client = openai_client(args.credential_file)
    if args.collect:
        print(json.dumps(collect(panel, client, prices, args.ledger,
                                 args.out, access=args.access), indent=2))
        return 0
    replies = replies_by_case(args.out / 'raw_replies.jsonl')
    restrict_to = None
    if args.pair_replies:
        restrict_to = (known_cases(panel, replies)
                       & known_cases(panel, replies_by_case(
                           args.pair_replies)))
    bank, _cost = embed_and_freeze(panel, replies, client, prices,
                                   args.ledger, args.out, args.access,
                                   restrict_to)
    print(json.dumps(dict(information_access=bank['information_access'],
                          request_sha256=bank['request_sha256'],
                          train=bank['train_cases'],
                          audit=bank['audit_cases'],
                          coverage=bank['coverage'],
                          formats=list(FORMATS)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
