"""
The shared allowance: what it refuses, and what it holds.

Every test here is about a refusal or an accounting identity, because
those are the only two things a budget guard does. A guard that lets a
run start and stops it halfway has not protected anything -- it has
bought an arm that cannot be compared with the others -- so the
interesting behaviour is all at the boundary before the first paid
request.

No test here makes a network call or needs an API key. The reservation
arithmetic lives in `reservation_for` precisely so it can be checked
without one: the trainer constructs its teachers before it reaches its
budget, so an integration test would fail on a missing key long before
reaching the code under test.
"""

import json

import pytest

from teachers.budget import (
    BudgetError, CostLedger, PriceTable, reservation_for,
)

TABLE = {
    'recorded_on': '2026-09-09',
    'provenance': 'test fixture',
    'models': {'gpt-5-mini': {'input_per_million': 0.25,
                              'output_per_million': 2.00}},
    'embeddings': {'text-embedding-3-small': {'per_million': 0.02}},
}


# --- Prices ---------------------------------------------------------

def test_an_unknown_model_stops_paid_dispatch():
    """
    `estimate_dollars` returns 0.0 for a model it does not know, which
    is right for reporting and catastrophic for a guard: an unpriced
    model would appear free and no cap would ever fire.
    """

    with pytest.raises(BudgetError, match='no price recorded'):
        PriceTable(TABLE).chat_rates('gpt-5-nano')


def test_an_undated_price_table_is_refused():
    """
    A price with no date is not evidence of what anything costs now.
    """

    with pytest.raises(BudgetError, match='recorded_on'):
        PriceTable({'models': {}})


def test_a_request_without_ceilings_cannot_be_bounded():
    """
    An unbounded output length on a reasoning model is an unbounded
    bill, so it may not be reserved for.
    """

    with pytest.raises(BudgetError, match='ceilings must be positive'):
        PriceTable(TABLE).call_bound('gpt-5-mini', 4000, 0)


def test_the_call_bound_is_the_stated_arithmetic():
    """
    4000 input at $0.25/M plus 3500 output at $2.00/M.
    """

    bound = PriceTable(TABLE).call_bound('gpt-5-mini', 4000, 3500)
    assert bound == pytest.approx(0.001 + 0.007)


def test_the_bound_exceeds_what_the_smoke_actually_spent():
    """
    A ceiling below observed spend would be a cap, not a bound. The
    2026-09-09 smoke measured ~$0.0053 per consultation.
    """

    assert PriceTable(TABLE).call_bound(
        'gpt-5-mini', 4000, 3500) > 0.0053


# --- The pool -------------------------------------------------------

def test_prior_spend_is_booked_not_forgiven(tmp_path):
    """
    Money spent before the ledger existed is still money spent.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0, prior_spend=0.65)
    status = CostLedger(path).status()
    assert status['settled_usd'] == pytest.approx(0.65)
    assert status['available_usd'] == pytest.approx(99.35)


def test_reserving_holds_against_the_pool(tmp_path):
    """
    A hold reduces what other runs may take, before anything is spent.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 30.0)
    status = ledger.status()
    assert status['reserved_usd'] == pytest.approx(30.0)
    assert status['settled_usd'] == 0
    assert status['available_usd'] == pytest.approx(70.0)


def test_a_run_that_cannot_finish_is_refused_before_it_starts(tmp_path):
    """
    The central refusal. Two runs each under their own cap can still
    exceed a shared pool, which is exactly what a per-process guard
    cannot see.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 60.0)
    ledger.reserve('run-b', 30.0)
    with pytest.raises(BudgetError, match='Not starting a run'):
        ledger.reserve('run-c', 30.0)


def test_settling_releases_the_hold_and_records_the_truth(tmp_path):
    """
    A run that reserved 30 and spent 4 returns 26 to the pool.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 30.0)
    ledger.settle('run-a', 4.25)
    status = ledger.status()
    assert status['settled_usd'] == pytest.approx(4.25)
    assert status['reserved_usd'] == 0
    assert status['available_usd'] == pytest.approx(95.75)


def test_a_crashed_run_keeps_holding_its_reservation(tmp_path):
    """
    Deliberate. A request that timed out may still have been billed, so
    an unsettled run must not silently return money to the pool; it is
    reconciled by hand, with a stated amount.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('crashed', 30.0)
    assert CostLedger(path).status()['available_usd'] == pytest.approx(70.0)
    assert [e['run_id'] for e in
            CostLedger(path).status()['open_reservations']] == ['crashed']


def test_one_run_cannot_hold_two_reservations(tmp_path):
    """
    A restart that re-reserves would double-count the same run.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 1.0)
    with pytest.raises(BudgetError, match='already holds'):
        ledger.reserve('run-a', 1.0)


def test_initializing_over_a_live_ledger_is_refused(tmp_path):
    """
    Re-initializing would discard holds belonging to running jobs.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    with pytest.raises(BudgetError, match='already exists'):
        CostLedger.initialize(path, 100.0)


def test_settling_an_unknown_run_is_refused(tmp_path):
    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    with pytest.raises(BudgetError, match='no open reservation'):
        CostLedger(path).settle('never-reserved', 1.0)


def test_the_ledger_survives_a_reread(tmp_path):
    """
    It is a file, and other processes read it.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    CostLedger(path).reserve('run-a', 12.5)
    payload = json.loads((tmp_path / 'ledger.json').read_text())
    assert payload['entries'][0]['reserved'] == 12.5
    assert payload['entries'][0]['state'] == 'open'


# --- What a run reserves --------------------------------------------

class _Args:
    """
    The handful of fields `reservation_for` reads.
    """

    def __init__(self, **over):
        self.query_budget = 240
        self.teacher_model = 'gpt-5-mini'
        self.max_input_tokens = 4000
        self.max_output_tokens = 3500
        self.max_attempts = 3
        self.explanation = 'shuffled'
        self.embed_model = 'text-embedding-3-small'
        self.max_embed_tokens = 512
        self.budget_ledger = 'x.json'
        self.price_table = 'p.json'
        self.__dict__.update(over)


def test_an_unbounded_consultation_count_is_refused():
    """
    With neither a query budget nor windows, the worst case is every
    step of every environment. That is not a bound anyone authorized.
    """

    with pytest.raises(BudgetError, match='needs a bound'):
        reservation_for(_Args(query_budget=0), None, PriceTable(TABLE))


def test_query_windows_alone_are_a_sufficient_bound():
    """
    The schedule caps consultations even with no explicit budget.
    """

    worst, manifest = reservation_for(
        _Args(query_budget=0), 240, PriceTable(TABLE))
    assert manifest['max_consultations'] == 240
    assert worst > 0


def test_every_permitted_attempt_is_reserved(tmp_path):
    """
    The reservation covers the attempt cap, not an expected number.

    An earlier version reserved 1.25 attempts per call because
    reserving the cap looked implausible and priced the grid out of the
    allowance. That is an expectation, and an expectation is not a
    limit: with the SDK client retrying underneath the outer cap, one
    consultation was demonstrated reaching a transport nine times.
    """

    worst, manifest = reservation_for(_Args(), None, PriceTable(TABLE))
    per_call = PriceTable(TABLE).call_bound('gpt-5-mini', 4000, 3500)
    assert manifest['teacher_reserved_usd'] == pytest.approx(
        per_call * 240 * 3), 'attempts are not reserved in full'
    assert worst > 240 * 0.0053, 'below what the smoke actually spent'
    assert manifest['sdk_retries_disabled'] is True


def test_reserving_scales_with_the_attempt_cap():
    """
    Lowering the cap is what buys a bigger grid, and it must show up in
    the reservation rather than only in a comment.
    """

    three, _ = reservation_for(_Args(max_attempts=3), None,
                               PriceTable(TABLE))
    one, _ = reservation_for(_Args(max_attempts=1), None,
                             PriceTable(TABLE))
    assert three > one


def test_settling_over_the_reservation_is_flagged(tmp_path):
    """
    The guard failing silently is worse than the guard failing.

    Scenario: initialize $100, reserve $60 and $40, then settle the
    first run at $80. An unguarded ledger accepts this and reports a
    negative balance with nothing marked. It is still recorded -- refusing would
    destroy the only record of what was spent -- but it is flagged.
    """

    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 60.0)
    ledger.reserve('run-b', 40.0)
    ledger.settle('run-a', 80.0)

    status = ledger.status()
    assert status['overspent_runs'], (
        'a run settled at $80 against a $60 reservation and nothing '
        'in the ledger says so'
    )
    assert status['overspent_runs'][0]['run_id'] == 'run-a'
    assert status['settled_usd'] == pytest.approx(80.0)


def test_a_run_within_its_reservation_is_not_flagged(tmp_path):
    path = str(tmp_path / 'ledger.json')
    CostLedger.initialize(path, 100.0)
    ledger = CostLedger(path)
    ledger.reserve('run-a', 60.0)
    ledger.settle('run-a', 12.0)
    assert ledger.status()['overspent_runs'] == []


def test_an_arm_without_a_head_reserves_nothing_for_embeddings():
    """
    R1 pays for text and never embeds it.
    """

    _, manifest = reservation_for(
        _Args(explanation='none'), None, PriceTable(TABLE))
    assert manifest['embedding_reserved_usd'] == 0


def test_aggregate_reconciled_history_is_not_charged_twice(tmp_path):
    path = tmp_path / 'ledger.json'
    CostLedger.initialize(str(path), 203.13, prior_spend=35)
    book = CostLedger(str(path))
    book.reserve('active', 93.58)
    payload = json.loads(path.read_text())
    payload['entries'].append(dict(
        run_id='historical', state='reconciled_in_aggregate',
        reserved=210.5484416))
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    assert book.status()['available_usd'] == 74.55
    assert path.read_bytes() == before
    book.reserve('new', 74.55)
    assert book.status()['available_usd'] == 0
    with pytest.raises(BudgetError):
        book.reserve('over', .01)


def test_the_manifest_records_what_the_bound_was_built_from():
    """
    A number nobody can reconstruct is not accounting.
    """

    _, manifest = reservation_for(_Args(), None, PriceTable(TABLE))
    for field in ('price_recorded_on', 'max_consultations',
                  'max_input_tokens', 'max_output_tokens',
                  'max_attempts', 'sdk_retries_disabled',
                  'per_call_worst_case_usd', 'teacher_reserved_usd',
                  'embedding_reserved_usd', 'reserved_usd'):
        assert field in manifest, f'{field} missing from the manifest'
