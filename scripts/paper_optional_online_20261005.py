"""Main-weight online advice paired with already submitted fresh controls.

No API request is made by cell construction.
"""

from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
import math
from pathlib import Path
import re

from advising.uniform_queries import UniformQueries
from monitoring.evaluation_diagnostics import diagnostic_milestones
from scripts import run_fix_wave_20260929 as fw
from scripts.run_explanation_formats_20260925 import derived_of
from scripts.run_rule_bank_pilot_20260928 import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_optional_online_20261005_v1'
PRICES = 'configs/prices_paper_optionals_2026-10-05.json'
MODEL = 'gpt-5-mini-2025-08-07'
SOURCES = ('dk_fresh', 'mr_fresh')
REPS = range(80, 90)
CALLS = {('doorkey_8x8', 'none'): 62,
         ('doorkey_8x8', 'count'): 62,
         ('multiroom_n6', 'none'): 36,
         ('multiroom_n6', 'count'): 67}
TIME = '1-12:00:00'
FAILURE_TOLERANCE = .05


def allocation(calls):
    """
    Round each child's complete token bound upward to a dollar cent.
    """

    bound = Decimal(calls) * Decimal('.006596')
    return float(bound.quantize(Decimal('.01'), rounding=ROUND_CEILING))


def cells(root=ROOT, ledger_root=None):
    """
    Change only the advising channel and bounded paid-call parameters.
    """

    root = Path(root)
    result = []
    for suite in SOURCES:
        for rep in REPS:
            for bonus in ('none', 'count'):
                base = fw.arm_args(suite, 'rules_weak', rep, bonus, root)
                bank = fw.read(root / base.rule_bank)
                count = sum(stage['calls'] for stage in bank['cost'].values())
                if count != CALLS[(base.task, bonus)]:
                    raise ValueError('Selected bank consultation cost changed')
                ledger = (str((Path(ledger_root) / str(len(result)) /
                               'budget.json').resolve())
                          if ledger_root else 'unfunded-check-only')
                args = replace(
                    base, experiment_id=f'{STUDY}_{suite}_{bonus}',
                    teacher='llm_scoped', teacher_model=MODEL,
                    teacher_stream=False, rule_bank='', rule_bank_sha256='',
                    uniform_queries=True, query_budget=count,
                    advice_budget=count, budget_ledger=ledger,
                    price_table=PRICES, max_input_tokens=10_000,
                    max_output_tokens=2048, max_attempts=1,
                    consultation_failure_limit=.3,
                    consultation_failure_min_samples=10,
                    offline_summary_only=False, journal_paid_only=False)
                fw.cell(result, STUDY, 'online_main', args, rep)
                result[-1]['source_suite'] = suite
                result[-1]['allocation_usd'] = allocation(count)
    return result


def validate_learning(run, record):
    """
    Validate complete regular and initial/early teacher-off evaluations.
    """

    run = Path(run)
    metrics = fw.validate(run, record)
    args = fw.args_of(record)
    geometry = derived_of(args)
    points = diagnostic_milestones(
        args.diagnostic_eval_frames, geometry['batch_size'],
        geometry['num_iterations'])
    early = read_jsonl(run / 'diagnostic_evaluations.jsonl')
    regular = read_jsonl(run / 'evaluations.jsonl')
    for row in regular + early:
        keys = ['success_rate']
        if args.eval_sampled:
            keys.append('sampled_success_rate')
        if any(type(row.get(k)) not in (int, float)
               or not math.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in keys):
            raise ValueError('Invalid success measurement')
    if not re.fullmatch('[0-9a-f]{64}', metrics['initial_sha256']):
        raise ValueError('Invalid initial-policy hash')
    if ([r['iteration'] for r in early] != sorted(points)
            or any(r['global_step'] != r['iteration'] *
                   geometry['batch_size']
                   or r['requested_frames'] != points[r['iteration']]
                   or r['teacher_on'] or r['episodes'] != args.eval_episodes
                   or r['seed_base'] != args.seed + 50_000
                   or r['policy_sha256_before'] != r['policy_sha256_after']
                   or not (r['rng_isolated'] or
                           r['reused_regular_evaluation'])
                   for r in early)
            or (0 in points and (
                early[0]['training_episodes'] != 0
                or early[0]['policy_sha256_before'] !=
                metrics['initial_sha256']))):
        raise ValueError('Dense evaluation contract differs')
    return metrics


def validate(run, record):
    """
    Keep planned slots, actual consultations and response failures separate.
    """

    run = Path(run)
    metrics = validate_learning(run, record)
    summary = fw.read(run / 'run_summary.json')
    latest = summary['latest']
    args = record['args']
    rows = read_jsonl(run / 'consultations.jsonl')
    planned = args['query_budget']
    consulted = latest['teacher_total_queries']
    if (type(consulted) is not int or not 0 <= consulted <= planned
            or len(rows) != consulted
            or [r['consultation'] for r in rows] !=
            list(range(1, consulted + 1))):
        raise ValueError('Consultation journal count differs')
    parsed = fw.args_of(record)
    geometry = derived_of(parsed)
    iterations = geometry['num_iterations']
    active = [r for r in range(iterations)
              if fw.ppo.distill_coef(r + 1, iterations, parsed) > 0]
    schedule = UniformQueries(active, parsed.num_steps, parsed.num_envs,
                              planned, parsed.seed)
    for row in rows:
        if (any(type(row.get(k)) is not int
                for k in ('rollout', 'step', 'env'))
                or not 0 <= row['step'] < parsed.num_steps
                or not 0 <= row['env'] < parsed.num_envs):
            raise ValueError('Invalid consultation slot coordinates')
        schedule.record(row['rollout'], row['step'], row['env'])
    if latest['uniform_query_schedule'] != schedule.manifest():
        raise ValueError('Uniform query schedule differs from journal/recipe')
    failed = transport = delivered = unknown = 0
    dollars = 0.0
    for row in rows:
        meta = row['metadata']
        if (row['teacher'] != args['teacher']
                or row['teacher_model'] != args['teacher_model']
                or meta.get('service_tier') not in (None, 'default')
                or (meta.get('response_model') is not None and
                    meta['response_model'] != args['teacher_model'])
                or (not meta.get('failed') and
                    meta.get('response_model') != args['teacher_model'])):
            raise ValueError('Served model/tier or journal teacher differs')
        if (type(meta.get('attempt_count')) is not int
                or not 0 <= meta['attempt_count'] <= 1
                or meta.get('budget_violation')):
            raise ValueError('Unbounded request or service tier')
        for name, cap in (('tokens_in', args['max_input_tokens']),
                          ('tokens_out', args['max_output_tokens'])):
            if type(row[name]) is not int or not 0 <= row[name] <= cap:
                raise ValueError('Response token bound differs')
        value = row['dollars']
        if value is None:
            unknown += 1
        elif (type(value) not in (int, float)
              or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid paid cost evidence')
        else:
            dollars += value
        failed += bool(meta.get('failed'))
        transport += bool(meta.get('failed') and
                          meta.get('outcome') == 'request_failure')
        delivered += bool(row['delivered'])
    reserved = planned * .006596
    cost = latest['teacher_cost_dollars']
    if (not math.isfinite(cost) or not 0 <= cost <= reserved + 1e-8
            or dollars > reserved + 1e-8):
        raise ValueError('Paid run exceeded its complete token bound')
    budget = latest['budget']
    if (not math.isclose(budget['reserved_usd'], reserved,
                        rel_tol=0, abs_tol=1e-8)
            or budget['max_consultations'] != planned
            or budget['max_attempts'] != 1
            or not budget['sdk_retries_disabled']):
        raise ValueError('Frozen paid reservation differs')
    counters = latest['consultations']
    if (latest['advising']['num_asked'] != consulted
            or latest['advising']['num_delivered'] != delivered
            or latest['advisor_control']['teacher_labels'] != delivered
            or counters['records'] != consulted
            or counters['records_written'] != consulted
            or counters['failures'] != failed
            or counters['unknown_cost_records'] != unknown
            or not math.isclose(cost, dollars, rel_tol=0, abs_tol=1e-8)
            or not math.isclose(budget['actual_teacher_usd'], cost,
                                rel_tol=0, abs_tol=1e-8)
            or not math.isclose(budget['actual_usd'], cost,
                                rel_tol=0, abs_tol=1e-8)):
        raise ValueError('Paid journal and final counters disagree')
    metrics.update(
        planned_slots=planned, observed_consultations=consulted,
        skipped_slots=planned - consulted, request_failures=transport,
        failed_responses=failed, delivered_labels=delivered,
        known_journal_dollars=dollars, unknown_cost_records=unknown,
        dose_admissible=(consulted > 0 and
                         transport / planned <= FAILURE_TOLERANCE))
    return metrics


def primary_contrasts():
    """
    Declare all four bank-minus-online main-weight contrasts together.
    """

    return [(task, bonus, 'rules_weak', 'online_main')
            for task, bonus in CALLS]
