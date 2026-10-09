"""
A shared dollar ledger for paid model traffic, reserved before it is spent.

The allowance is **one pool across every paid run**, not a per-run limit,
so a guard that only looks at the current process cannot enforce it. Two
runs each staying under their own cap can together exceed the pool.

The design is reserve-then-settle, and the reservation happens **once at
run start for the run's whole worst case** rather than per call:

    available = allowance - settled_spend - open_reservations

A run computes the most it could possibly cost, takes that out of the
pool atomically, and only then makes its first paid request. If the pool
cannot cover it, the run refuses to start instead of discovering the
limit somewhere in the middle, which would leave a half-finished arm
that is not comparable to anything.

That ordering is what makes concurrency tractable. Per-call reservation
would need a locked transaction per request across every node; per-run
reservation needs one transaction per run, and the lock only has to hold
for the few milliseconds around it.

**Failure is deliberately biased toward over-holding.** A run that
crashes leaves its reservation open, so the pool under-counts what is
available until someone reconciles it (`scripts/budget_status.py
--reconcile`). Releasing automatically on failure would be the wrong
default: a request that timed out may still have been billed, and an
ambiguous failure must not silently return money to the pool.

**What this cannot see.** Spend is measured from the usage each API
response reports. A request that is billed but returns no usable
response -- a retry that fails after generation -- contributes no
measured cost, so the reservation must cover **every attempt the code
is permitted to make**, not the number it is expected to need. An
earlier version reserved an expected 1.25 attempts per call on the
grounds that reserving the cap was implausible and priced the grid out
of the allowance. That reasoning was wrong in kind: an expectation is
not a guarantee, and a guard that holds less than the maximum is not a
limit. An offline test demonstrated nine HTTP attempts reaching an offline
transport for one consultation, because the SDK client's own
`max_retries=2` multiplied the outer cap of 3.

Both layers are therefore pinned for a budgeted run: SDK retries are
disabled (`LLM_MAX_SDK_RETRIES=0`) so the client makes exactly one
attempt per call, the outer cold-start retry is capped
(`LLM_MAX_ATTEMPTS`), and the reservation multiplies by that cap. The
consequence is a smaller affordable grid, which is the honest answer
rather than a larger one that was never covered.
"""

import json
from decimal import Decimal
import math
import os
import time


class BudgetError(RuntimeError):
    """
    Raised when a paid dispatch cannot be shown to be affordable.

    Deliberately also raised for *unknown* bounds, not only for
    exceeded ones: a price or token ceiling this code cannot establish
    is not a reason to proceed and hope.
    """


class PriceTable:
    """
    Dollars per million tokens, with provenance, and strict on misses.

    `estimate_dollars` in `llm_prompts.py` returns 0.0 for a model it
    does not know, which is right for reporting -- a missing price
    should not crash a free run -- and catastrophic for a budget guard,
    because an unknown model would appear to cost nothing and no cap
    would ever trigger. This wrapper raises instead.
    """

    def __init__(self, payload, source='<inline>'):
        self.source = source
        self.recorded_on = payload.get('recorded_on')
        self.provenance = payload.get('provenance')
        self.models = payload.get('models') or {}
        self.embeddings = payload.get('embeddings') or {}
        if not self.recorded_on or not self.provenance:
            raise BudgetError(
                f'price table {source} has no recorded_on/provenance; '
                f'an undated price is not evidence of what anything '
                f'costs today'
            )

    @classmethod
    def load(cls, path):
        """
        Read a price table from disk.
        """

        if not path or not os.path.exists(path):
            raise BudgetError(
                f'price table {path!r} not found; paid dispatch needs '
                f'prices it can point at'
            )
        with open(path, encoding='utf-8') as handle:
            return cls(json.load(handle), source=path)

    def chat_rates(self, model):
        """
        (input, output) dollars per million tokens for a chat model.
        """

        if model not in self.models:
            raise BudgetError(
                f'no price recorded for model {model!r} in '
                f'{self.source}; refusing to dispatch paid requests '
                f'whose cost cannot be bounded'
            )
        entry = self.models[model]
        return float(entry['input_per_million']), float(
            entry['output_per_million'])

    def embedding_rate(self, model):
        """
        Dollars per million tokens for an embedding model.
        """

        if model not in self.embeddings:
            raise BudgetError(
                f'no embedding price recorded for {model!r} in '
                f'{self.source}'
            )
        return float(self.embeddings[model]['per_million'])

    def call_bound(self, model, max_input_tokens, max_output_tokens):
        """
        The most one chat request can cost, given explicit ceilings.

        Both ceilings must be real numbers. An unbounded output length
        on a reasoning model is an unbounded bill: the smoke's calls
        spent roughly 2500 output tokens each on hidden reasoning, and
        nothing in the request stopped that from being 25,000.
        """

        if max_input_tokens <= 0 or max_output_tokens <= 0:
            raise BudgetError(
                f'token ceilings must be positive to bound a request; '
                f'got input={max_input_tokens}, output='
                f'{max_output_tokens}'
            )
        rate_in, rate_out = self.chat_rates(model)
        return (max_input_tokens * rate_in
                + max_output_tokens * rate_out) / 1e6

    def embedding_bound(self, model, texts, max_tokens_each):
        """
        The most a batch of embeddings can cost.
        """

        if max_tokens_each <= 0:
            raise BudgetError('embedding token ceiling must be positive')
        return texts * max_tokens_each * self.embedding_rate(model) / 1e6


def _lock(path):
    """
    Best-effort exclusive lock around one ledger transaction.

    `flock` where the platform has it, an exclusive-create lock file
    otherwise. **Neither is guaranteed across nodes on a parallel
    filesystem, and this does not establish mutual exclusion there.**

    An earlier version of this docstring claimed a lost lock would only
    over-hold. That was wrong: read-modify-write means a lost update can
    erase a reservation entirely while both jobs proceed as though they
    held it. Short transactions shrink the window -- one per run at
    startup rather than one per request -- but shrinking a window is not
    closing it.

    Until locking on the target filesystem is verified, treat concurrent
    admission as unproven: submit paid runs so that admissions do not
    coincide, or serialize them through one coordinator. `status()`
    reports overspent runs, which is the detection of last resort.
    """

    class _Guard:
        def __enter__(self):
            self.handle = open(path + '.lock', 'a+')
            try:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
                self.flocked = True
            except (ImportError, OSError):
                self.flocked = False
                # No flock: spin on an exclusive create instead.
                for _ in range(600):
                    try:
                        fd = os.open(path + '.spinlock',
                                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        break
                    except FileExistsError:
                        time.sleep(0.1)
                else:
                    raise BudgetError(
                        f'could not acquire the ledger lock at {path}; '
                        f'if no other run holds it, remove '
                        f'{path}.spinlock'
                    )
            return self

        def __exit__(self, *exc):
            if not self.flocked:
                try:
                    os.remove(path + '.spinlock')
                except FileNotFoundError:
                    pass
            self.handle.close()
            return False

    return _Guard()


class CostLedger:
    """
    The shared pool, on disk, with reservations held against it.
    """

    def __init__(self, path):
        self.path = path

    # -- lifecycle ---------------------------------------------------

    @staticmethod
    def initialize(path, allowance, note='', prior_spend=0.0):
        """
        Create the pool. `prior_spend` books already-incurred costs.

        Money spent before the ledger existed is still money spent, so
        it is booked as a settled entry rather than forgiven. Refuses
        to overwrite an existing ledger.
        """

        if os.path.exists(path):
            raise BudgetError(
                f'{path} already exists; refusing to reset a ledger '
                f'that may be holding live reservations'
            )
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        entries = []
        if prior_spend:
            entries.append({
                'run_id': 'prior-spend',
                'reserved': prior_spend,
                'actual': prior_spend,
                'state': 'settled',
                'note': note or 'spend incurred before this ledger',
                'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
        payload = {
            'allowance_usd': float(allowance),
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                         time.gmtime()),
            'note': note,
            'entries': entries,
        }
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def _read(self):
        if not os.path.exists(self.path):
            raise BudgetError(
                f'no ledger at {self.path}; create it with '
                f'scripts/budget_status.py --init'
            )
        with open(self.path, encoding='utf-8') as handle:
            return json.load(handle)

    def _write(self, payload):
        temporary = f'{self.path}.tmp{os.getpid()}'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, self.path)

    # -- accounting --------------------------------------------------

    @staticmethod
    def _totals(payload):
        settled = sum(e['actual'] for e in payload['entries']
                      if e['state'] == 'settled')
        held = sum(e['reserved'] for e in payload['entries']
                   if e['state'] == 'open')
        return settled, held

    @staticmethod
    def _available(payload):
        """Subtract stored dollar amounts without binary-float cent errors."""
        allowance = payload['allowance_usd']
        if type(allowance) not in (int, float) or not math.isfinite(allowance):
            raise BudgetError('Allowance must be a finite dollar amount')
        liabilities = sum((Decimal(str(
            e['actual'] if e['state'] == 'settled' else e['reserved']))
            for e in payload['entries']
            if e['state'] in ('settled', 'open')), Decimal(0))
        return float(Decimal(str(allowance)) - liabilities)

    def status(self):
        """
        What the pool looks like right now.
        """

        payload = self._read()
        settled, held = self._totals(payload)
        return {
            'path': self.path,
            'allowance_usd': payload['allowance_usd'],
            'settled_usd': settled,
            'reserved_usd': held,
            'available_usd': self._available(payload),
            'open_reservations': [
                {'run_id': e['run_id'], 'reserved': e['reserved'],
                 'at': e['at'], 'note': e.get('note', '')}
                for e in payload['entries'] if e['state'] == 'open'
            ],
            # Runs that settled above their reservation. Any entry here
            # means the guard did not hold for that run and the cause
            # needs finding before another paid launch.
            'overspent_runs': [
                {'run_id': e['run_id'], 'reserved': e['reserved'],
                 'actual': e['actual']}
                for e in payload['entries'] if e.get('overspent')
            ],
        }

    def increase_allowance(self, allowance, note=''):
        """
        Raise the project allowance without releasing or rewriting holds.

        This is local accounting, not a purchase of provider credits.
        Repeating the same increase is idempotent.
        """

        if not math.isfinite(allowance) or allowance <= 0:
            raise BudgetError('Allowance must be finite and positive')
        with _lock(self.path):
            payload = self._read()
            previous = payload['allowance_usd']
            if allowance < previous:
                raise BudgetError('An increase cannot lower the allowance')
            if allowance == previous:
                return
            payload.setdefault('allowance_history', []).append({
                'previous_usd': previous, 'allowance_usd': allowance,
                'note': note,
                'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            payload['allowance_usd'] = float(allowance)
            self._write(payload)

    def reserve(self, run_id, dollars, note=''):
        """
        Hold `dollars` for this run, or refuse the run outright.

        Refusing here is the entire point. A run that starts and is
        stopped by a cap partway through has spent money on an arm that
        cannot be compared with the others.
        """

        if dollars <= 0:
            raise BudgetError(f'reservation must be positive, got {dollars}')
        with _lock(self.path):
            payload = self._read()
            settled, held = self._totals(payload)
            available = self._available(payload)
            if dollars > available:
                raise BudgetError(
                    f'{run_id} needs ${dollars:.2f} in the worst case '
                    f'but only ${available:.2f} of the '
                    f'${payload["allowance_usd"]:.2f} allowance is '
                    f'free (${settled:.2f} spent, ${held:.2f} held by '
                    f'other runs). Not starting a run that cannot '
                    f'finish.'
                )
            if any(e['run_id'] == run_id and e['state'] == 'open'
                   for e in payload['entries']):
                raise BudgetError(
                    f'{run_id} already holds an open reservation'
                )
            payload['entries'].append({
                'run_id': run_id,
                'reserved': float(dollars),
                'actual': None,
                'state': 'open',
                'note': note,
                'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            self._write(payload)
            return available - dollars

    def settle(self, run_id, actual):
        """
        Replace this run's reservation with what it actually spent.
        """

        with _lock(self.path):
            payload = self._read()
            for entry in payload['entries']:
                if entry['run_id'] == run_id and entry['state'] == 'open':
                    entry['actual'] = float(actual)
                    entry['state'] = 'settled'
                    entry['settled_at'] = time.strftime(
                        '%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    # A run that spent more than it reserved has escaped
                    # the guard, and the ledger has to say so loudly
                    # rather than quietly absorbing the difference. It is
                    # still recorded -- refusing the settlement would
                    # lose the only record of what was actually spent --
                    # but it is flagged, and status() surfaces it.
                    entry['overspent'] = float(actual) > entry['reserved']
                    self._write(payload)
                    return entry
            raise BudgetError(
                f'no open reservation for {run_id} to settle'
            )


def reservation_for(args, scheduled_slots, prices):
    """
    What one run must set aside, and the manifest fields explaining it.

    Extracted from the trainer so the arithmetic can be checked without
    a network call or an API key: the trainer builds its teachers
    before it reaches its budget, so an integration test of this logic
    would fail on a missing key long before it reached the part under
    test.

    `scheduled_slots` is the query-window bound, or None. Raises rather
    than guessing when neither bound exists -- the worst case would
    then be every step of every environment, which is not a bound
    anyone would authorize.
    """

    if getattr(args, 'online_rule_calls', 0) > 0:
        # llm_rules_online advises densely for free; only these calls pay.
        max_queries = int(args.online_rule_calls)
    elif args.query_budget > 0:
        max_queries = int(args.query_budget)
    elif scheduled_slots:
        max_queries = int(scheduled_slots)
    else:
        raise BudgetError(
            'a budgeted run needs a bound on consultations: pass '
            '--query-budget, or --query-windows. Without one the worst '
            'case is every step of every environment and nothing can '
            'be reserved.'
        )

    per_call = prices.call_bound(
        args.teacher_model, args.max_input_tokens, args.max_output_tokens
    )
    # Every attempt the code may make, because a billed attempt that
    # returns nothing leaves no usage behind to measure. SDK retries are
    # disabled under a budget, so this cap is the whole of it.
    teacher = per_call * max_queries * max(1, int(args.max_attempts))
    embedding = 0.0
    if (args.explanation != 'none'
            and getattr(args, 'explanation_target', 'embedding') == 'embedding'):
        embedding = prices.embedding_bound(
            args.embed_model, max_queries, args.max_embed_tokens
        )
    worst_case = teacher + embedding
    return worst_case, {
        'ledger': args.budget_ledger,
        'price_table': args.price_table,
        'price_recorded_on': prices.recorded_on,
        'max_consultations': max_queries,
        'max_input_tokens': args.max_input_tokens,
        'max_output_tokens': args.max_output_tokens,
        'max_attempts': args.max_attempts,
        'sdk_retries_disabled': True,
        'per_call_worst_case_usd': per_call,
        'teacher_reserved_usd': teacher,
        'embedding_reserved_usd': embedding,
        'reserved_usd': worst_case,
    }
