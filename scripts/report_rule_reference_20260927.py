"""Report validated prospective rule-reference rows without rerunning work.

The launcher validates artifacts before calling ``build_report``. A complete
row has task, bonus, arm, seed, index, status='complete', auc, final,
wall_seconds (possibly None), queries, deliveries, initial_sha256, and curve.
Each curve point has step, greedy, and optionally sampled. Metrics describe
teacher-off evaluation. Missing outcomes are never filled with zero.
Optional planner resource fields are exported separately from selected queries.
"""

import csv
import json
import math
import statistics
import platform
from importlib.metadata import version
from collections import Counter
from pathlib import Path

from scipy.stats import t


PAIRS = 20
TASK_SEEDS = {
    'doorkey_8x8': 12_800_000,
    'multiroom_n6': 12_900_000,
    'keycorridor_s3r3': 13_000_000,
}
TASK_LABELS = {
    'doorkey_8x8': 'DoorKey 8x8',
    'multiroom_n6': 'MultiRoom N6',
    'keycorridor_s3r3': 'KeyCorridor S3R3',
}
BONUSES = ('none', 'count')
CORE_ARMS = ('none', 'entropy480', 'probability480', 'random480')
CORE_CONTRASTS = (
    ('entropy480', 'none'),
    ('probability480', 'none'),
    ('random480', 'none'),
    ('probability480', 'entropy480'),
    ('random480', 'entropy480'),
)
PLANNER_METRICS = (
    'reference_calls', 'reference_wall_seconds', 'reference_compute_units',
    'total_planner_calls', 'teacher_wall_seconds', 'teacher_compute_units',
)
COLORS = {
    'none': '#333333', 'entropy480': '#0072B2',
    'probability480': '#D55E00', 'random480': '#009E73',
    'random120': '#CC79A7', 'random60': '#E69F00',
    'random30': '#56B4E9', 'random15': '#8B6D43',
}


def planned_arms(task, bonus):
    """Return the frozen core and task-specific exploratory conditions."""
    extra = []
    if bonus == 'count' or task == 'keycorridor_s3r3':
        extra.append('random120')
    if task == 'multiroom_n6' and bonus == 'count':
        extra.extend(('random15', 'random30', 'random60'))
    return (*CORE_ARMS, *extra)


def planned_seeds(task):
    """Keep exactly the original twenty seeds; do not top up missing work."""
    return [TASK_SEEDS[task] + 100 * i for i in range(PAIRS)]


def interval(values):
    """Compute an ordinary 95% t interval with training seeds as units."""
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Statistics require finite values.')
    n = len(values)
    mean = statistics.mean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    half = float(t.ppf(.975, n - 1)) * se if se is not None else None
    return {
        'n': n, 'mean': mean, 'sd': sd, 'standard_error': se,
        'ci95_low': mean - half if half is not None else None,
        'ci95_high': mean + half if half is not None else None,
        'zero_empirical_variance': sd == 0 if sd is not None else None,
    }


def paired_pvalue(differences):
    """Use a two-sided paired t test, including explicit constant limits.

    All-zero differences have p=1. Constant nonzero differences have the
    limiting p=0 and a degenerate sample CI; this convention does not imply
    zero population uncertainty. Fewer than two pairs have no test.
    """
    summary = interval(differences)
    if summary['n'] < 2:
        return None
    if summary['sd'] == 0:
        return 1.0 if summary['mean'] == 0 else 0.0
    statistic = summary['mean'] / summary['standard_error']
    return float(2 * t.sf(abs(statistic), summary['n'] - 1))


def holm_adjust(pvalues):
    """Return monotone Holm adjusted p-values in their original order."""
    values = [float(value) for value in pvalues]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('Holm requires finite p-values between zero and one.')
    adjusted = [None] * len(values)
    running = 0.0
    order = sorted(range(len(values)), key=values.__getitem__)
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _complete(row):
    return row['status'] == 'complete'


def _prepare_rows(rows):
    """Check the interface and retain all planned missing cells explicitly."""
    supplied = {}
    indices = set()
    for source in rows:
        row = dict(source)
        key = tuple(row[name] for name in ('task', 'bonus', 'arm', 'seed'))
        task, bonus, arm, seed = key
        if (task not in TASK_SEEDS or bonus not in BONUSES
                or arm not in planned_arms(task, bonus)
                or seed not in planned_seeds(task)):
            raise ValueError(f'Unplanned row identity: {key}.')
        if key in supplied:
            raise ValueError(f'Duplicate row identity: {key}.')
        index = row.get('index')
        if index is not None:
            if index in indices:
                raise ValueError(f'Duplicate cell index: {index}.')
            indices.add(index)
        if _complete(row):
            for name in ('auc', 'final', 'queries', 'deliveries'):
                value = row.get(name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f'Complete row lacks finite {name}.')
            if not row.get('initial_sha256') or not row.get('curve'):
                raise ValueError('Complete row lacks hash or learning curve.')
            wall = row.get('wall_seconds')
            if wall is not None and (not math.isfinite(wall) or wall < 0):
                raise ValueError('Invalid observed wall time.')
            for name in PLANNER_METRICS:
                value = row.setdefault(name, None)
                if value is not None and (not math.isfinite(value)
                                          or value < 0):
                    raise ValueError(
                        f'Invalid observed planner metric {name}.')
            steps = [point['step'] for point in row['curve']]
            if steps != sorted(set(steps)):
                raise ValueError('Curve steps must be unique and increasing.')
            for point in row['curve']:
                for metric in ('greedy', 'sampled'):
                    value = point.get(metric)
                    if metric == 'sampled' and value is None:
                        continue
                    if (value is None or not math.isfinite(value)
                            or not 0 <= value <= 1):
                        raise ValueError('Invalid teacher-off curve value.')
        else:
            # A failed or partial artifact is not a completed outcome.
            for name in ('auc', 'final', 'queries', 'deliveries',
                         'wall_seconds', 'initial_sha256', *PLANNER_METRICS):
                row[name] = None
            row['curve'] = []
        supplied[key] = row
    result = []
    for task in TASK_SEEDS:
        for bonus in BONUSES:
            for arm in planned_arms(task, bonus):
                for seed in planned_seeds(task):
                    key = (task, bonus, arm, seed)
                    result.append(supplied.get(key, {
                        'task': task, 'bonus': bonus, 'arm': arm,
                        'seed': seed, 'index': None, 'status': 'unattempted',
                        'auc': None, 'final': None, 'wall_seconds': None,
                        'queries': None, 'deliveries': None,
                        'initial_sha256': None, 'curve': [],
                        **dict.fromkeys(PLANNER_METRICS),
                    }))
    return result


def _paired(rows, task, bonus, treatment, control, metric, kind):
    """Join on seed before differencing; never subtract independent means."""
    by_arm = {
        arm: {row['seed']: row for row in rows
              if row['task'] == task and row['bonus'] == bonus
              and row['arm'] == arm and _complete(row)}
        for arm in (treatment, control)
    }
    seeds = sorted(by_arm[treatment].keys() & by_arm[control].keys())
    differences = []
    for seed in seeds:
        left, right = by_arm[treatment][seed], by_arm[control][seed]
        if left['initial_sha256'] != right['initial_sha256']:
            raise ValueError('Compared seed has different initial policies.')
        differences.append(float(left[metric]) - float(right[metric]))
    return {
        'task': task, 'bonus': bonus, 'kind': kind, 'metric': metric,
        'treatment': treatment, 'control': control,
        'contrast': f'{treatment}_minus_{control}',
        'family': f'{task}/{bonus}/{kind}', 'seeds': seeds,
        'differences': differences, **interval(differences),
        'planned_pairs': PAIRS, 'p_two_sided': None,
        'p_holm_within5': None, 'p_holm_global30': None,
        'family_inference_available': False,
        'global_inference_available': False,
        'interpretation': 'descriptive; ordinary 95% paired t interval',
        'noninferiority_or_equivalence_established': False,
    }


def _contrasts(rows):
    contrasts, families = [], []
    for task in TASK_SEEDS:
        for bonus in BONUSES:
            core = [_paired(rows, task, bonus, left, right, metric, 'core')
                    for left, right in CORE_CONTRASTS
                    for metric in ('auc', 'final')]
            primary = [row for row in core if row['metric'] == 'auc']
            ready = all(row['seeds'] == planned_seeds(task)
                        for row in primary)
            families.append({
                'task': task, 'bonus': bonus, 'planned_contrasts': 5,
                'planned_pairs': PAIRS,
                'complete_pairs_per_contrast': [row['n'] for row in primary],
                'inference_available': ready,
                'status': 'complete' if ready else 'incomplete_descriptive',
            })
            if ready:
                raw = [paired_pvalue(row['differences']) for row in primary]
                for row, p, adjusted in zip(primary, raw, holm_adjust(raw)):
                    row.update(
                        p_two_sided=p, p_holm_within5=adjusted,
                        family_inference_available=True,
                        interpretation='prespecified paired AUC test; '
                        'Holm within five; independent review still required',
                    )
            contrasts.extend(core)
            for arm in planned_arms(task, bonus)[len(CORE_ARMS):]:
                for metric in ('auc', 'final'):
                    row = _paired(rows, task, bonus, arm, 'random480', metric,
                                  'exploratory_low_dose')
                    row['interpretation'] = (
                        'exploratory low-dose difference; no equivalence '
                        'or noninferiority claim; ordinary 95% paired t CI'
                    )
                    contrasts.append(row)
    primary = [row for row in contrasts
               if row['kind'] == 'core' and row['metric'] == 'auc']
    global_ready = all(row['family_inference_available'] for row in primary)
    if global_ready:
        adjusted = holm_adjust([row['p_two_sided'] for row in primary])
        for row, p in zip(primary, adjusted):
            row.update(p_holm_global30=p, global_inference_available=True)
    return contrasts, families, global_ready


def _planner_summary(complete, metric, planned):
    """Keep unknown planner work distinct from observed zero work."""
    values = [row[metric] for row in complete if row.get(metric) is not None]
    return {
        **interval(values),
        'unknown_complete_cells': len(complete) - len(values),
        'known_complete_total': sum(values) if values else None,
        'total_if_all_planned_complete': sum(values)
        if len(values) == planned else None,
    }


def _aggregate(rows):
    """Summarize each arm and unsmoothed evaluation point across seeds."""
    arms, curves = [], []
    for task in TASK_SEEDS:
        for bonus in BONUSES:
            for arm in planned_arms(task, bonus):
                selected = [row for row in rows if row['task'] == task
                            and row['bonus'] == bonus and row['arm'] == arm]
                complete = [row for row in selected if _complete(row)]
                identity = {'task': task, 'bonus': bonus, 'arm': arm}
                summary = {
                    **identity, 'planned': PAIRS, 'complete': len(complete),
                    'missing': PAIRS - len(complete),
                    'status_counts': dict(Counter(
                        row['status'] for row in selected)),
                }
                for metric in ('auc', 'final', 'wall_seconds', 'queries',
                               'deliveries'):
                    values = [row[metric] for row in complete
                              if row.get(metric) is not None]
                    summary[metric] = interval(values)
                summary['unknown_wall_seconds'] = (
                    len(complete) - summary['wall_seconds']['n'])
                for metric in PLANNER_METRICS:
                    summary[metric] = _planner_summary(complete, metric, PAIRS)
                arms.append(summary)
                grids = {tuple(point['step'] for point in row['curve'])
                         for row in complete}
                if len(grids) > 1:
                    raise ValueError('Arm evaluation step grids differ.')
                if not complete:
                    continue
                for i, step in enumerate(next(iter(grids))):
                    for metric in ('greedy', 'sampled'):
                        values = [row['curve'][i][metric] for row in complete
                                  if row['curve'][i].get(metric) is not None]
                        curves.append({
                            **identity, 'step': step, 'metric': metric,
                            **interval(values),
                        })
    return arms, curves


def _write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict))
                else value for key, value in row.items()
            })


def _plots(report, destination):
    """Export separate task figures with plain and count panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = []
    for task in TASK_SEEDS:
        for metric in ('greedy', 'sampled'):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
            for axis, bonus in zip(axes, BONUSES):
                plotted = False
                for arm in planned_arms(task, bonus):
                    points = [row for row in report['learning_curves']
                              if row['task'] == task and row['bonus'] == bonus
                              and row['arm'] == arm and row['metric'] == metric
                              and row['n'] > 0]
                    if not points:
                        continue
                    plotted = True
                    x = [point['step'] / 1e6 for point in points]
                    y = [point['mean'] for point in points]
                    counts = sorted({point['n'] for point in points})
                    nlabel = (str(counts[0]) if len(counts) == 1
                              else str(counts))
                    suffix = (' (exploratory)'
                              if arm not in CORE_ARMS else '')
                    axis.plot(x, y, color=COLORS[arm],
                              linestyle='--' if suffix else '-',
                              label=f'{arm}{suffix}, n={nlabel}/20')
                    band = [i for i, point in enumerate(points)
                            if point['ci95_low'] is not None]
                    if band:
                        axis.fill_between(
                            [x[i] for i in band],
                            [max(0., points[i]['ci95_low']) for i in band],
                            [min(1., points[i]['ci95_high']) for i in band],
                            color=COLORS[arm], alpha=.12,
                        )
                if plotted:
                    axis.legend(fontsize=7, loc='best')
                else:
                    axis.text(.5, .5, 'No validated completed curves',
                              ha='center', va='center',
                              transform=axis.transAxes)
                axis.set(title='Plain PPO' if bonus == 'none' else 'Count-PPO',
                         xlabel='Training transitions (millions)', ylim=(0, 1))
                axis.grid(alpha=.2)
            axes[0].set_ylabel(f'Teacher-off {metric} success')
            fig.suptitle(f'{TASK_LABELS[task]}: prospective rule reference')
            fig.text(.5, .02,
                     'Unsmoothed means; pointwise 95% t intervals across '
                     'training seeds, clipped to [0, 1].\n'
                     'Bands are descriptive, not simultaneous. No CI for '
                     'n < 2. Fixed 20 planned seeds; '
                     'incomplete arms retained.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .09, 1, .95))
            for extension in ('png', 'pdf'):
                name = f'{task}_{metric}.{extension}'
                fig.savefig(destination / name, dpi=180)
                paths.append(name)
            plt.close(fig)
    return paths


def build_report(rows, output_dir=None):
    """Describe validated rows from the fixed suite and optionally export.

    AUC is the sole formal endpoint. Final success, wall time, query counts,
    delivered advice, and low-dose effects are descriptive. Partial families
    retain ordinary CIs but withhold tests until all planned pairs exist.
    Statistical availability does not establish independent clearance.
    """
    rows = _prepare_rows(rows)
    contrasts, families, global_ready = _contrasts(rows)
    arms, curves = _aggregate(rows)
    complete = [row for row in rows if _complete(row)]
    known_wall = [row['wall_seconds'] for row in complete
                  if row.get('wall_seconds') is not None]
    report = {
        'analysis_runtime': dict(python=platform.python_version(),
                                 scipy=version('scipy'),
                                 matplotlib=version('matplotlib')),
        'schema_version': 1, 'study': 'rule_reference_20260927_v1',
        'planned_cells': len(rows), 'planned_pairs': PAIRS,
        'complete_cells': len(complete),
        'missing_cells': len(rows) - len(complete),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'runtime': {
            'known_complete_cells': len(known_wall),
            'unknown_complete_cells': len(complete) - len(known_wall),
            'known_complete_wall_seconds': sum(known_wall)
            if known_wall else None,
            'suite_total_wall_seconds': sum(known_wall)
            if len(known_wall) == len(rows) else None,
        },
        'planner_work': {
            metric: _planner_summary(complete, metric, len(rows))
            for metric in PLANNER_METRICS
        },
        'primary_metric': 'auc', 'families': families,
        'global30_inference_available': global_ready,
        'conclusions_independently_reviewed': False,
        'overall_winner_claim': False,
        'notes': [
            'Fixed 20 fresh paired seeds per task/background; no optional '
            'stopping, seed replacement, or data-dependent sample expansion.',
            'Seed is the replication unit. AUC/final CIs are ordinary 95% '
            'paired t intervals; curve bands are pointwise across-seed t CIs.',
            'Core AUC p-values are two-sided, Holm-adjusted within five '
            'contrasts per stratum and across all thirty when complete. '
            'No formal family inference while any planned pair is missing.',
            'Final success and low-dose effects are descriptive. Low-dose '
            'families are separate exploratory comparisons against random480; '
            'no equivalence or noninferiority conclusion is provided.',
            'Zero empirical variance gives a degenerate sample CI. The '
            'constant-difference t-test limit is p=1 for all zeros and p=0 '
            'for a nonzero constant; population uncertainty is not zero.',
            'Missing outcomes and runtime remain unknown. Zero queries in a '
            'validated no-advice run are real zeros, not missing data.',
            'Selected queries and delivered advice are distinct from planner '
            'work. Stream reference calls can exceed the selected query cap. '
            'Total planner calls are taken from validated instrumentation, '
            'not inferred by adding selected queries to reference calls. '
            'Planner wall seconds are sums of observed per-run work, not '
            'elapsed suite time; unavailable resource fields stay unknown.',
            'This prospective study uses isolated advisor RNG and fresh '
            'seeds. Historical paid-teacher cohorts are not paired controls; '
            'this report establishes neither rule superiority over GPT nor '
            'novelty or an overall winner from screening strata.',
            'Zero API fees does not mean zero compute cost. Artifact '
            'validation and scientific review are separate evidence states.',
        ],
        'arms': arms, 'contrasts': contrasts, 'learning_curves': curves,
        'individual_seeds': [{key: value for key, value in row.items()
                              if key != 'curve'} for row in rows],
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_csv(destination / 'individual_seeds.csv',
                   report['individual_seeds'],
                   ('task', 'bonus', 'arm', 'seed', 'index', 'status', 'auc',
                    'final', 'queries', 'deliveries', 'wall_seconds',
                    'initial_sha256', *PLANNER_METRICS))
        _write_csv(destination / 'paired_contrasts.csv', contrasts,
                   list(contrasts[0]))
        _write_csv(destination / 'learning_curves.csv', curves,
                   ('task', 'bonus', 'arm', 'step', 'metric', 'n', 'mean',
                    'sd', 'standard_error', 'ci95_low', 'ci95_high',
                    'zero_empirical_variance'))
        report['figures'] = _plots(report, destination)
        (destination / 'report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n',
            encoding='utf-8',
        )
    return report
