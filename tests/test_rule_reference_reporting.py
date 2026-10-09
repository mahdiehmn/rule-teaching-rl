"""Exercise fixed-family inference, seed pairing, missingness, and exports."""

import csv
import json
import math
import statistics

import pytest
from scipy.stats import t, ttest_1samp

from scripts import report_rule_reference_20260927 as report


def make_rows(tasks=None, bonuses=None, n=20, extra=False):
    """Make correlated seed outcomes whose treatment effects are known."""
    rows = []
    for task in tasks or report.TASK_SEEDS:
        for bonus in bonuses or report.BONUSES:
            arms = (report.planned_arms(task, bonus) if extra
                    else report.CORE_ARMS)
            for arm in arms:
                for i, seed in enumerate(report.planned_seeds(task)[:n]):
                    base = .1 + .02 * i
                    effect = {
                        'none': 0., 'entropy480': .05 + .001 * i,
                        'probability480': .025 + .0005 * i,
                        'random480': .08 - .001 * i,
                    }.get(arm, -.01)
                    value = base + effect
                    cap = 0 if arm == 'none' else int(''.join(
                        char for char in arm if char.isdigit()))
                    rows.append({
                        'task': task, 'bonus': bonus, 'arm': arm, 'seed': seed,
                        'index': len(rows), 'status': 'complete',
                        'auc': value, 'final': value + .1,
                        'wall_seconds': 100. + i, 'queries': cap,
                        'deliveries': cap, 'initial_sha256': f'seed-{seed}',
                        'curve': [{'step': 100, 'greedy': base, 'sampled': 0.},
                                  {'step': 200, 'greedy': value,
                                   'sampled': value / 2}],
                    })
    return rows


def contrast(result, treatment='entropy480', control='none', metric='auc',
             task='doorkey_8x8', bonus='none'):
    return next(row for row in result['contrasts']
                if (row['task'], row['bonus'], row['treatment'],
                    row['control'], row['metric'])
                == (task, bonus, treatment, control, metric))


def test_paired_effect_sign_ci_and_correlated_seed_unit():
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'])
    result = report.build_report(list(reversed(rows)))
    actual = contrast(result)
    differences = [.05 + .001 * i for i in range(20)]
    expected_mean = statistics.mean(differences)
    expected_se = statistics.stdev(differences) / math.sqrt(20)
    half = float(t.ppf(.975, 19)) * expected_se
    assert actual['mean'] == pytest.approx(expected_mean)
    assert actual['ci95_low'] == pytest.approx(expected_mean - half)
    assert actual['ci95_high'] == pytest.approx(expected_mean + half)
    assert actual['p_two_sided'] == pytest.approx(
        float(ttest_1samp(differences, 0).pvalue))
    assert actual['n'] == 20
    assert actual['seeds'] == report.planned_seeds('doorkey_8x8')
    assert actual['family_inference_available']
    assert not result['global30_inference_available']
    assert actual['p_holm_global30'] is None
    assert contrast(result, 'probability480', 'entropy480')['mean'] < 0
    # Shared seed variation must cancel before the interval is computed.
    raw_sd = statistics.stdev(row['auc'] for row in rows
                             if row['arm'] == 'entropy480')
    assert actual['sd'] < raw_sd / 10


def test_holm_order_ties_and_stepdown_monotonicity():
    assert report.holm_adjust([.03, .01, .04, .011, 1.]) == pytest.approx(
        [.09, .05, .09, .05, 1.])
    assert report.holm_adjust([.01, .01, 0., 1.]) == pytest.approx(
        [.03, .03, 0., 1.])
    assert report.holm_adjust([]) == []
    with pytest.raises(ValueError, match='finite'):
        report.holm_adjust([float('nan')])


def test_all_six_strata_required_for_global_thirty_tests():
    rows = make_rows()
    result = report.build_report(rows)
    core = [row for row in result['contrasts']
            if row['kind'] == 'core' and row['metric'] == 'auc']
    assert len(core) == 30
    assert result['global30_inference_available']
    expected = report.holm_adjust([row['p_two_sided'] for row in core])
    assert [row['p_holm_global30'] for row in core] == pytest.approx(expected)
    assert result['complete_cells'] == 480
    assert not result['conclusions_independently_reviewed']
    assert not result['overall_winner_claim']
    for family in result['families']:
        subset = [row for row in core if row['task'] == family['task']
                  and row['bonus'] == family['bonus']]
        assert [row['p_holm_within5'] for row in subset] == pytest.approx(
            report.holm_adjust([row['p_two_sided'] for row in subset]))


def test_one_missing_core_cell_withholds_entire_local_family():
    rows = make_rows()
    rows = [row for row in rows if not (
        row['task'] == 'doorkey_8x8' and row['bonus'] == 'none'
        and row['arm'] == 'probability480'
        and row['seed'] == report.planned_seeds('doorkey_8x8')[-1])]
    result = report.build_report(rows)
    unaffected = contrast(result)
    assert unaffected['n'] == 20
    assert unaffected['ci95_low'] is not None
    assert unaffected['p_two_sided'] is None
    assert unaffected['p_holm_within5'] is None
    assert not unaffected['family_inference_available']
    assert contrast(result, 'probability480')['n'] == 19
    assert contrast(result, bonus='count')['family_inference_available']
    assert not result['global30_inference_available']
    assert all(row['p_holm_global30'] is None for row in result['contrasts'])


def test_partial_pairs_join_by_seed_instead_of_row_position():
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'], n=3)
    rows = [row for row in rows if not (
        row['arm'] == 'entropy480' and row['seed'] == 12_800_000)
        and not (row['arm'] == 'none' and row['seed'] == 12_800_200)]
    actual = contrast(report.build_report(rows))
    assert actual['seeds'] == [12_800_100]
    assert actual['mean'] == pytest.approx(.051)
    assert actual['ci95_low'] is None
    assert actual['p_two_sided'] is None


@pytest.mark.parametrize('values,mean,p', [
    ([0.] * 20, 0., 1.), ([.25] * 20, .25, 0.),
    ([-.25] * 20, -.25, 0.),
])
def test_constant_and_zero_differences_have_finite_explicit_limits(
        values, mean, p):
    actual = report.interval(values)
    assert actual['ci95_low'] == actual['ci95_high'] == mean
    assert actual['zero_empirical_variance']
    assert report.paired_pvalue(values) == p
    json.dumps(actual, allow_nan=False)


def test_missing_outcomes_runtime_and_single_seed_ci_are_not_zero():
    empty = report.build_report([])
    assert empty['planned_cells'] == empty['missing_cells'] == 620
    assert empty['complete_cells'] == 0
    assert empty['runtime']['known_complete_wall_seconds'] is None
    assert contrast(empty)['mean'] is None
    assert all(row['auc'] is None for row in empty['individual_seeds'])
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'], n=1)
    for row in rows:
        row['wall_seconds'] = None
        row['auc'] = row['final'] = 0.
    result = report.build_report(rows)
    assert contrast(result)['mean'] == 0.
    assert contrast(result)['ci95_low'] is None
    assert result['runtime']['unknown_complete_cells'] == 4
    assert result['runtime']['known_complete_wall_seconds'] is None
    assert result['runtime']['suite_total_wall_seconds'] is None
    control = next(row for row in result['individual_seeds']
                   if row['status'] == 'complete' and row['arm'] == 'none')
    assert control['queries'] == control['deliveries'] == 0


def test_failed_partial_metrics_never_enter_completed_outcomes():
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'], n=1)
    rows[0].update(status='failed', auc=.99, final=.99,
                   reference_calls=3000, total_planner_calls=3000)
    result = report.build_report(rows)
    failed = next(row for row in result['individual_seeds']
                  if row['status'] == 'failed')
    assert failed['auc'] is None
    assert failed['final'] is None
    assert failed['reference_calls'] is None
    assert failed['total_planner_calls'] is None
    assert result['complete_cells'] == 3
    assert contrast(result)['n'] == 0


def test_stream_planner_work_remains_separate_from_selected_queries():
    rows = make_rows(tasks=['keycorridor_s3r3'], bonuses=['count'], n=2)
    selected = [row for row in rows if row['arm'] == 'random480']
    for i, row in enumerate(selected):
        row.update(reference_calls=3000 + 100 * i,
                   reference_wall_seconds=10. + i,
                   reference_compute_units=6000 + 200 * i,
                   total_planner_calls=3000 + 100 * i,
                   teacher_wall_seconds=0., teacher_compute_units=0)
    result = report.build_report(rows)
    arm = next(row for row in result['arms']
               if row['task'] == 'keycorridor_s3r3'
               and row['bonus'] == 'count' and row['arm'] == 'random480')
    assert arm['queries']['mean'] == 480
    assert arm['deliveries']['mean'] == 480
    assert arm['reference_calls']['mean'] == 3050
    assert arm['reference_calls']['known_complete_total'] == 6100
    assert arm['total_planner_calls']['known_complete_total'] == 6100
    assert arm['teacher_wall_seconds']['mean'] == 0
    assert arm['teacher_compute_units']['known_complete_total'] == 0
    assert arm['reference_wall_seconds']['known_complete_total'] == 21
    assert arm['reference_compute_units']['known_complete_total'] == 12200
    # The other completed mock arms omit optional instrumentation.
    total = result['planner_work']['total_planner_calls']
    assert total['n'] == 2
    assert total['unknown_complete_cells'] == 6
    assert total['known_complete_total'] == 6100
    assert total['total_if_all_planned_complete'] is None
    uninstrumented = next(row for row in result['arms']
                          if row['task'] == 'keycorridor_s3r3'
                          and row['bonus'] == 'count' and row['arm'] == 'none')
    assert uninstrumented['reference_calls']['mean'] is None
    assert uninstrumented['reference_calls']['unknown_complete_cells'] == 2
    actual = next(row for row in result['individual_seeds']
                  if row['index'] == selected[0]['index'])
    assert actual['queries'] == 480
    assert actual['reference_calls'] == actual['total_planner_calls'] == 3000


def test_complete_planner_totals_require_every_planned_value():
    rows = make_rows(extra=True)
    for row in rows:
        row.update(dict.fromkeys(report.PLANNER_METRICS, 0))
        row['total_planner_calls'] = row['queries']
    expected = sum(row['queries'] for row in rows)
    result = report.build_report(rows)
    total = result['planner_work']['total_planner_calls']
    assert total['known_complete_total'] == expected
    assert total['total_if_all_planned_complete'] == expected
    assert total['unknown_complete_cells'] == 0
    rows[0]['total_planner_calls'] = None
    partial = report.build_report(rows)['planner_work']['total_planner_calls']
    assert partial['known_complete_total'] == expected
    assert partial['unknown_complete_cells'] == 1
    assert partial['total_if_all_planned_complete'] is None


def test_low_dose_and_final_endpoints_stay_descriptive_separate_families():
    result = report.build_report(make_rows(extra=True))
    assert result['complete_cells'] == 620
    assert result['runtime']['suite_total_wall_seconds'] > 0
    extra = [row for row in result['contrasts']
             if row['kind'] == 'exploratory_low_dose']
    assert len(extra) == 14
    assert all(row['control'] == 'random480' for row in extra)
    assert all('exploratory_low_dose' in row['family'] for row in extra)
    assert all(row['p_two_sided'] is None for row in extra)
    assert all(row['ci95_low'] is not None for row in extra)
    assert all(not row['noninferiority_or_equivalence_established']
               for row in result['contrasts'])
    assert contrast(result, metric='final')['p_two_sided'] is None


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'Duplicate row'), ('unplanned', 'Unplanned row'),
    ('mismatched_hash', 'different initial'),
    ('mismatched_grid', 'step grids differ'),
    ('nonfinite', 'finite auc'),
    ('invalid_planner_work', 'Invalid observed planner metric'),
])
def test_reject_ambiguous_pairing_or_malformed_interface(mutation, match):
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'], n=2)
    if mutation == 'duplicate':
        rows.append(dict(rows[0]))
    elif mutation == 'unplanned':
        rows[0]['seed'] += 1
    elif mutation == 'mismatched_hash':
        rows[0]['initial_sha256'] = 'wrong-policy'
    elif mutation == 'mismatched_grid':
        rows[0]['curve'][0]['step'] += 1
    elif mutation == 'nonfinite':
        rows[0]['auc'] = float('nan')
    elif mutation == 'invalid_planner_work':
        rows[0]['reference_wall_seconds'] = float('nan')
    with pytest.raises(ValueError, match=match):
        report.build_report(rows)


def test_export_keeps_unclipped_intervals_and_all_seed_missingness(tmp_path):
    rows = make_rows(tasks=['doorkey_8x8'], bonuses=['none'], n=2)
    # The plotted band clips at bounds, while saved estimates remain raw.
    rows[0]['curve'][0]['greedy'] = 0.
    rows[1]['curve'][0]['greedy'] = 1.
    rows[0].update(dict.fromkeys(report.PLANNER_METRICS, 0))
    rows[2].update(reference_calls=3000, reference_wall_seconds=10.,
                   reference_compute_units=6000, total_planner_calls=3000,
                   teacher_wall_seconds=0., teacher_compute_units=0)
    result = report.build_report(rows, tmp_path)
    point = next(row for row in result['learning_curves']
                 if row['arm'] == 'none' and row['step'] == 100
                 and row['metric'] == 'greedy')
    assert point['ci95_low'] < 0
    assert point['ci95_high'] > 1
    assert point['mean'] == .5
    assert len(result['figures']) == 12
    assert all((tmp_path / name).stat().st_size > 1000
               for name in result['figures'])
    stored = json.loads((tmp_path / 'report.json').read_text())
    assert stored['complete_cells'] == 8
    with (tmp_path / 'individual_seeds.csv').open(newline='') as handle:
        seeds = list(csv.DictReader(handle))
    assert len(seeds) == 620
    assert sum(row['status'] == 'unattempted' for row in seeds) == 612
    assert all(row['auc'] == '' for row in seeds
               if row['status'] == 'unattempted')
    exported = next(row for row in seeds if row['index'] == '2')
    assert exported['queries'] == '480'
    assert exported['total_planner_calls'] == '3000'
    assert exported['reference_calls'] == '3000'
    assert exported['reference_wall_seconds'] == '10.0'
    assert exported['reference_compute_units'] == '6000'
    assert exported['teacher_wall_seconds'] == '0.0'
    assert exported['teacher_compute_units'] == '0'
    assert all(row['total_planner_calls'] == '' for row in seeds
               if row['status'] == 'unattempted')
    assert (tmp_path / 'paired_contrasts.csv').is_file()
    assert (tmp_path / 'learning_curves.csv').is_file()
