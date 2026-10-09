"""Validate completed native free-teacher runs against their planned Args.

This checks persisted software evidence, not scientific conclusions.
"""

import hashlib
import csv
import json
import math
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

from advising.uniform_queries import UniformQueries
from algos.ppo_distill import Agent, Args, distill_coef, is_symbolic_mode


def _require(condition, message):
    """Raise one predictable exception for an invalid artifact."""
    if not condition:
        raise ValueError(message)


def _number(value, name, minimum=0, maximum=math.inf, integer=False):
    """Reject booleans, missing data and nonfinite metric values."""
    valid = (type(value) in (int, float) and math.isfinite(value)
             and minimum <= value <= maximum)
    if integer:
        valid = valid and type(value) is int
    _require(valid, f'Invalid {name}: {value!r}')
    return value


def _read(path):
    """Treat missing or malformed JSON as invalid run evidence."""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        _require(isinstance(value, dict), 'Expected an object')
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read {path.name}: {error}') from error


def _rows(path):
    """Read actual records without synthesizing missing observations."""
    try:
        rows = [json.loads(line) for line in
                path.read_text(encoding='utf-8').splitlines() if line.strip()]
        _require(all(isinstance(row, dict) for row in rows),
                 'Expected object records')
        return rows
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read {path.name}: {error}') from error


def resolved_args(args):
    """Resolve current native defaults and training-derived fields."""
    values = asdict(args) if is_dataclass(args) else dict(args)
    native = Args(**values)
    native.batch_size = native.num_envs * native.num_steps
    native.minibatch_size = native.batch_size // native.num_minibatches
    native.num_iterations = native.total_timesteps // native.batch_size
    return native


def evaluation_steps(args):
    """Compute the native interval, milestone and terminal evaluation set."""
    milestones = {int(n) // args.batch_size for n in
                  args.eval_frame_milestones.split(',') if n.strip()}
    return [i * args.batch_size for i in range(1, args.num_iterations + 1)
            if args.eval_interval > 0 and (
                i % args.eval_interval == 0 or i in milestones
                or i == args.num_iterations)]


@lru_cache(maxsize=16)
def _policy_schema(shape, symbolic, recurrent, dual_value):
    """Resolve native tensor names and shapes without allocating weights."""
    envs = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=shape),
        single_action_space=SimpleNamespace(n=7))
    with torch.device('meta'):
        model = Agent(envs, symbolic=symbolic, recurrent=recurrent,
                      dual_value=dual_value)
    return {key: (tuple(value.shape), value.dtype)
            for key, value in model.state_dict().items()}


def _checkpoint(run, args, summary):
    """Load safe tensor data and hash its exact values, not pickle bytes."""
    try:
        state = torch.load(run / 'agent.pt', map_location='cpu',
                           weights_only=True)
    except Exception as error:
        raise ValueError(f'Cannot load final policy: {error}') from error
    _require(isinstance(state, dict) and bool(state), 'Missing final policy')
    schema = _policy_schema(tuple(summary['obs_shape']),
                            is_symbolic_mode(args.obs_mode),
                            args.recurrent, args.dual_value)
    _require(set(state) == set(schema), 'Final policy tensor keys differ')
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        _require(isinstance(value, torch.Tensor) and value.numel() > 0,
                 f'Invalid final policy tensor {key}')
        _require((tuple(value.shape), value.dtype) == schema[key],
                 f'Final policy tensor shape or dtype differs: {key}')
        # NumPy avoids dispatching a thread pool per tensor when the
        # report scans hundreds of checkpoints in a CPU-only process.
        array = value.detach().cpu().numpy()
        _require(bool(np.isfinite(array).all()),
                 f'Nonfinite final policy tensor {key}')
        digest.update(key.encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def _curve(run, args, summary, latest):
    """Check both teacher-off policies at every promised evaluation."""
    rows = _rows(run / 'evaluations.jsonl')
    steps = evaluation_steps(args)
    _require(len(steps) >= 2 and [r.get('global_step') for r in rows] == steps,
             'Incomplete, duplicate or off-schedule evaluation curve')
    last_wall = last_episodes = 0
    curves = {'greedy': [], 'sampled': []}
    for row in rows:
        _number(row.get('iteration'), 'evaluation iteration',
                1, args.num_iterations, integer=True)
        _require(row.get('teacher_on') is False
                 and row.get('episodes') == args.eval_episodes
                 and row.get('seed_base') == args.seed + 50_000
                 and row.get('iteration') * args.batch_size
                 == row['global_step'], 'Teacher-off evaluation identity')
        last_wall = _number(row.get('wall_time_sec'), 'evaluation wall time',
                            last_wall, summary['wall_time_sec'])
        last_episodes = _number(row.get('training_episodes'),
                                'evaluation training episodes', last_episodes,
                                summary['total_episodes'], integer=True)
        _number(row.get('mean_return'), 'evaluation return', -math.inf)
        _number(row.get('mean_length'), 'evaluation length', 1)
        for mode, field in (('greedy', 'success_rate'),
                            ('sampled', 'sampled_success_rate')):
            value = _number(row.get(field), field, 0, 1)
            _require(math.isclose(value * args.eval_episodes,
                                  round(value * args.eval_episodes),
                                  abs_tol=1e-8),
                     'Success rate is not an episode-count fraction')
            curves[mode].append(value)
    _require(latest.get('eval_success_rate') == curves['greedy'][-1]
             and latest.get('eval_sampled_success_rate')
             == curves['sampled'][-1]
             and latest.get('eval_mean_return') == rows[-1]['mean_return'],
             'Final evaluation disagrees with summary')
    result = {'evaluations': len(rows), 'evaluation_steps': steps,
              'curve': [dict(step=row['global_step'],
                             greedy=row['success_rate'],
                             sampled=row['sampled_success_rate'])
                        for row in rows]}
    for mode, values in curves.items():
        y = np.asarray(values, dtype=float)
        result[f'{mode}_auc'] = float(np.sum(
            np.diff(steps) * (y[:-1] + y[1:]) / 2)
            / (steps[-1] - steps[0]))
        result[f'final_{mode}_success'] = float(y[-1])
    return result


def _consultations(run, args, summary, latest, active):
    """Reconcile consultation, delivery, timing and free-cost evidence."""
    advisor = latest.get('advising', {})
    journal = latest.get('consultations', {})
    control = latest.get('advisor_control', {})
    count = _number(latest.get('teacher_total_queries'), 'teacher queries',
                    integer=True)
    delivered = _number(advisor.get('num_delivered'), 'deliveries',
                        0, count, integer=True)
    abstains = _number(latest.get('teacher_total_abstains'), 'abstains',
                       0, count, integer=True)
    _require(advisor.get('num_asked') == count
             and journal.get('records') == count
             and delivered + abstains == count
             and control.get('teacher_labels') == delivered,
             'Query, advisor, consultation and label counts disagree')
    _require(advisor.get('advisor') == args.advisor
             and advisor.get('advice_budget') == args.advice_budget
             and advisor.get('query_budget') == args.query_budget
             and advisor.get('num_free') == 0
             and advisor.get('num_env_steps') == summary['global_step'],
             'Advisor configuration or exposure differs')
    for field in ('num_steps', 'num_withheld'):
        _number(advisor.get(field), field, integer=True)
    _require(advisor['num_steps'] >= count
             and advisor['num_withheld'] <= abstains,
             'Impossible advisor accounting')
    _require(not args.guidance or count <= args.query_budget,
             'Query cap exceeded')
    _require(not args.advice_budget or delivered <= args.advice_budget,
             'Advice cap exceeded')
    for field in ('teacher_total_declined', 'teacher_compute_units'):
        _number(latest.get(field), field, integer=True)
    _require(latest['teacher_total_declined'] == advisor['num_steps'] - count,
             'Declined consultation count differs from advisor exposure')
    for field in ('teacher_wall_time_s',):
        _number(latest.get(field), field)
    zero = ('teacher_total_aliased', 'teacher_total_gated',
            'teacher_cost_dollars', 'reference_labels')
    _require(all(latest.get(key) == 0 for key in zero)
             and latest.get('reference_teacher')
             == (args.teacher if args.teacher_stream else None),
             'Unexpected paid, reference or modified-label activity')
    reference_calls = _number(latest.get('reference_calls'), 'planner calls',
                              integer=True)
    reference_wall = _number(latest.get('reference_wall_time_s'),
                             'planner wall time')
    reference_compute = _number(latest.get('reference_compute_units'),
                                'planner compute units', integer=True)
    if args.teacher_stream:
        # NEXT_STEP resets replace the tick after each terminal action.
        # Dense bot synchronization queries every other active state.
        try:
            path = run / 'episodes.csv'
            with path.open(encoding='utf-8', newline='') as f:
                episodes = list(csv.DictReader(f))
            skips = sum(int(row['global_step']) // args.batch_size in active
                        for row in episodes)
        except (OSError, KeyError, ValueError) as error:
            raise ValueError('Missing stream episode evidence') from error
        _require(len(episodes) == summary['total_episodes']
                 and reference_calls == len(active) * args.batch_size - skips
                 and reference_calls >= count
                 and latest['teacher_compute_units'] == 0
                 and latest['teacher_wall_time_s'] == 0,
                 'Dense planner stream and selected-query accounting differ')
    else:
        _require(reference_calls == reference_wall == reference_compute == 0,
                 'Unexpected dense reference planner work')
    cost = advisor.get('cost', {})
    _require(all(cost.get(k) == 0 for k in
                 ('dollars', 'tokens_in', 'tokens_out'))
             and cost.get('compute_units') == latest['teacher_compute_units']
             and cost.get('wall_time_s') == latest['teacher_wall_time_s']
             and journal.get('failures') == 0
             and journal.get('unknown_cost_records') == 0
             and journal.get('failure_limit')
             == args.consultation_failure_limit
             and journal.get('minimum_before_stop')
             == args.consultation_failure_min_samples,
             'Free-teacher cost or consultation evidence differs')
    _require(control.get('rng_mode') == 'isolated_numpy_query_stream_v1'
             and control.get('query_seed') == args.seed + 90_117
             and control.get('sham') is False
             and control.get('teacher_instances')
             == (args.num_envs if args.guidance else 0)
             and control.get('query_order_draws')
             == len(active) * args.num_steps,
             'Isolated query RNG or teacher-instance evidence differs')
    first = control.get('first_label_global_step')
    if delivered:
        _number(first, 'first label step', args.num_envs,
                (max(active) + 1) * args.batch_size, integer=True)
        _require(first % args.num_envs == 0
                 and (first - 1) // args.batch_size in active,
                 'First label falls outside active rollout')
    else:
        _require(first is None, 'Unlabelled run reports a first label')
    if not args.guidance:
        _require(count == 0 and latest['teacher_total_declined'] == 0
                 and latest['teacher_compute_units'] == 0
                 and latest['teacher_wall_time_s'] == 0
                 and advisor['num_steps'] == 0,
                 'Teacher-free control contains teacher activity')

    # Finite free runs can retain each consultation cheaply. Older
    # summary-only runs still expose counts, but no invented row audit.
    path = run / 'consultations.jsonl'
    rows = _rows(path) if path.exists() else []
    written = 0 if args.offline_summary_only else count
    _require(journal.get('records_written') == written
             and len(rows) == written,
             'Missing or extra consultation records')
    slots = []
    label_steps = []
    previous_clock = -1
    for number, row in enumerate(rows, 1):
        rollout = _number(row.get('rollout'), 'consultation rollout',
                          0, args.num_iterations - 1, integer=True)
        step = _number(row.get('step'), 'consultation step',
                       0, args.num_steps - 1, integer=True)
        env = _number(row.get('env'), 'consultation env',
                      0, args.num_envs - 1, integer=True)
        clock = rollout * args.num_steps + step
        slot = clock * args.num_envs + env
        _require(rollout in active and clock >= previous_clock
                 and slot not in slots, 'Duplicate or inactive consultation')
        previous_clock = clock
        slots.append(slot)
        _require(row.get('consultation') == number
                 and row.get('sample_id')
                 == f'{summary["run_name"]}:{rollout + 1}:{step}:{env}'
                 and row.get('teacher') == args.teacher
                 and row.get('teacher_model') == args.teacher_model,
                 'Consultation identity differs')
        _number(row.get('episode'), 'consultation episode', integer=True)
        _number(row.get('student_action'), 'student action',
                0, 6, integer=True)
        if row.get('teacher_action') is not None:
            _number(row['teacher_action'], 'teacher action',
                    0, 6, integer=True)
        _require(type(row.get('delivered')) is bool,
                 'Invalid consultation delivery flag')
        if row['delivered']:
            label_steps.append((clock + 1) * args.num_envs)
        _require(all(row.get(k) == 0 for k in (
            'dollars', 'known_response_dollars', 'tokens_in', 'tokens_out')),
            'Nonzero or unknown free consultation cost')
        _number(row.get('wall_time_s'), 'consultation wall time')
        metadata = row.get('metadata', {})
        _require(not metadata.get('failed')
                 and not metadata.get('attempt_errors')
                 and metadata.get('cost_known', True)
                 and not metadata.get('budget_violation'),
                 'Unexpected consultation failure or API uncertainty')
    if rows:
        _require(sum(r['delivered'] for r in rows) == delivered
                 and (min(label_steps) if label_steps else None) == first
                 and math.isclose(sum(r['wall_time_s'] for r in rows),
                                  latest['teacher_wall_time_s'],
                                  rel_tol=1e-9, abs_tol=1e-9),
                 'Consultation rows disagree with summary')
    if args.uniform_queries:
        expected = UniformQueries(active, args.num_steps, args.num_envs,
                                  args.query_budget, args.seed).manifest()
        observed = latest.get('uniform_query_schedule', {})
        used = observed.get('observed', [])
        _require(isinstance(used, list)
                 and all(type(v) is int for v in used)
                 and len(used) == len(set(used)) == count
                 and set(used) <= set(expected['slots']),
                 'Uniform observed slots are duplicate or off-plan')
        expected['observed'] = used
        _require(observed == expected and (not rows or used == slots),
                 'Uniform schedule identity or journal positions differ')
    else:
        _require('uniform_query_schedule' not in latest,
                 'Unexpected uniform schedule')
    return dict(teacher_queries=count, advisor_queries=advisor['num_asked'],
                deliveries=delivered, queried_without_target=abstains,
                advisor_withheld=advisor['num_withheld'],
                teacher_wall_seconds=latest['teacher_wall_time_s'],
                teacher_compute_units=latest['teacher_compute_units'],
                reference_calls=reference_calls,
                reference_wall_seconds=reference_wall,
                reference_compute_units=reference_compute,
                total_planner_calls=(reference_calls if args.teacher_stream
                                     else count),
                api_cost_usd=0.0, consultation_rows=len(rows),
                consultation_timing_audited=not args.offline_summary_only)


def validate_run(run, args):
    """Return metrics only for complete, matched native rule-reference runs.

    ``args`` is an Args dataclass or a mapping of planned native settings.
    Short engineering workloads use the same checks as the frozen 10M
    protocol; the launcher separately owns its horizon and seed contract.
    """
    run = Path(run)
    args = resolved_args(args)
    pairs = {'doorkey_8x8': 'oracle', 'multiroom_n6': 'door_bfs',
             'keycorridor_s3r3': 'bot'}
    _require(pairs.get(args.task) == args.teacher
             and args.advisor_rng_isolation and not args.advisor_sham
             and args.advisor_no_teacher_peek
             and args.teacher_stream == (args.teacher == 'bot'
                                         and args.guidance)
             and not args.action_reference
             and not args.budget_ledger and not args.max_cost_dollars
             and not args.advice_replay and args.explanation == 'none'
             and args.consequence == 'none'
             and not args.diagnostic_stop_rollouts
             and args.peek == 'none' and args.imitation_weighting == 'none'
             and args.advice_evidence_gate == 'none'
             and args.advice_keep_fraction == 1 and not args.advice_alias_rate
             and not args.query_windows and args.query_interval == 1
             and args.eval_sampled and args.eval_episodes > 0
             and args.record_initial_policy
             and (not args.guidance or args.query_budget > 0),
             'Unsupported rule-reference run configuration')
    summary = _read(run / 'run_summary.json')
    endpoint = args.num_iterations * args.batch_size
    _require(summary.get('status') == 'completed'
             and summary.get('global_step') == endpoint
             and summary.get('args') == asdict(args),
             'Incomplete endpoint or different resolved settings')
    _number(summary.get('wall_time_sec'), 'training wall time', 1e-12)
    _number(summary.get('total_episodes'), 'training episodes', integer=True)
    _number(summary.get('total_successes'), 'training successes',
            0, summary['total_episodes'], integer=True)
    latest = summary.get('latest', {})
    for field in ('learning_rate', 'value_loss', 'intrinsic_value_loss',
                  'policy_loss', 'entropy', 'approx_kl', 'clipfrac',
                  'distill_loss', 'distill_coef'):
        _number(latest.get(field), field, -math.inf)
    _require(latest['distill_coef']
             == distill_coef(args.num_iterations, args.num_iterations, args),
             'Final distillation coefficient differs')
    active = [r for r in range(args.num_iterations)
              if distill_coef(r + 1, args.num_iterations, args) > 0]
    metrics = _consultations(run, args, summary, latest, active)
    metrics.update(_curve(run, args, summary, latest))
    try:
        initial = (run / 'initial_policy.sha256').read_text().strip()
    except OSError as error:
        raise ValueError('Missing initial policy hash') from error
    _require(re.fullmatch('[0-9a-f]{64}', initial) is not None
             and (not args.expected_initial_policy_sha256
                  or initial == args.expected_initial_policy_sha256),
             'Invalid or mismatched initial policy hash')
    metrics.update(initial_policy_sha256=initial,
                   final_policy_sha256=_checkpoint(run, args, summary),
                   wall_seconds=summary['wall_time_sec'],
                   actual_transitions=endpoint)
    metrics.update(auc=metrics['greedy_auc'],
                   final=metrics['final_greedy_success'],
                   initial_sha256=initial, queries=metrics['teacher_queries'])
    return metrics
