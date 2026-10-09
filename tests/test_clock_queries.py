"""Common-clock scheduling: unit semantics and real short trainer runs."""

import json

import numpy as np
import pytest
import torch

from advising.clock_queries import ClockQueries
from algos import ppo_distill as ppo


def test_schedule_is_shared_unique_and_inside_active_rollouts():
    a = ClockQueries('random', range(3, 40), 16, 50, seed=7)
    b = ClockQueries('entropy', range(3, 40), 16, 50, seed=7)
    assert a.times == b.times                  # common across selectors
    assert len(set(a.times)) == 50
    assert all(3 <= r < 40 and 0 <= s < 16 for r, s in a.times)
    with pytest.raises(ValueError):
        ClockQueries('random', range(2), 4, 9, seed=0)   # over capacity


def test_selectors_skip_unreplaced_slots_and_pick_max_entropy():
    rng = np.random.default_rng(0)
    clock = ClockQueries('entropy', range(10), 8, 3, seed=1)
    (r0, s0), (r1, s1), (r2, s2) = clock.times
    assert clock.choose(r0, s0, [], None, 7, rng) is None
    entropy = np.array([.1, .9, .5, .9])
    chosen = clock.choose(r1, s1, [0, 2, 3], entropy, 7, rng)
    assert chosen == 3                         # env 1 is not eligible
    clock.outcome(True, True, 100)
    ties = clock.choose(r2, s2, [1, 3], entropy, 7, rng)
    assert ties in (1, 3) and clock.events[-1]['ties'] == 2
    with pytest.raises(ValueError):
        clock.choose(r2, s2, [1], entropy, 7, rng)       # already used
    manifest = clock.manifest()
    assert manifest['skipped_no_eligible'] == 1
    assert manifest['delivered_labels'] == 1 and manifest['not_reached'] == 0


def test_random_selector_is_uniform_over_eligible():
    rng = np.random.default_rng(3)
    counts = {0: 0, 2: 0, 5: 0}
    clock = ClockQueries('random', range(200), 10, 1500, seed=2)
    for r, s in clock.times:
        counts[clock.choose(r, s, [0, 2, 5], None, 7, rng)] += 1
    assert all(430 < c < 570 for c in counts.values())


def clock_args(selector, advisor='unlimited', **extra):
    base = dict(task='doorkey_8x8', seed=11, total_timesteps=4096,
                num_envs=4, num_steps=64, num_minibatches=4,
                update_epochs=1, eval_interval=8, eval_episodes=1,
                obs_mode='symbolic', recurrent=True, dual_value=True,
                bonus='none', guidance=True, teacher='oracle',
                advisor=advisor, importance_source='none',
                query_budget=10, advice_budget=10, query_clock=selector,
                mistake_threshold=.2 if advisor == 'mistake' else 0.0,
                advisor_rng_isolation=True, advisor_no_teacher_peek=True,
                distill_coef_start=1.0, distill_coef_min=.01,
                record_initial_policy=True, experiment_id='clock_eng')
    base.update(extra)
    return ppo.Args(**base)


@pytest.mark.parametrize('selector,advisor', [
    ('random', 'unlimited'), ('entropy', 'unlimited'),
    ('entropy', 'mistake')])
def test_real_trainer_consults_once_per_clock_time(selector, advisor,
                                                   tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    ppo.train(clock_args(selector, advisor))
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())['latest']
    manifest = summary['clock_query_schedule']
    assert manifest['selector'] == selector
    assert manifest['planned_times'] == 10
    chosen = [e for e in manifest['events'] if e['chosen'] is not None]
    assert manifest['consultations'] == len(chosen) <= 10
    assert all(e['consulted'] for e in chosen)
    assert manifest['delivered_labels'] <= manifest['consultations']
    if advisor == 'unlimited':
        assert manifest['delivered_labels'] == manifest['consultations']
    if selector == 'entropy':
        assert all(e['chosen_entropy'] >= e['pool_mean_entropy'] - 1e-9
                   for e in chosen)
    assert summary['consultations']['records'] == manifest['consultations']


@pytest.mark.parametrize('field,value', [
    ('uniform_queries', True), ('importance_source', 'entropy'),
    ('advisor_rng_isolation', False), ('advisor', 'importance'),
    ('query_clock', 'top')])
def test_clock_rejects_mixed_schedules(field, value, tmp_path, monkeypatch):
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = clock_args('random')
    setattr(args, field, value)
    with pytest.raises(ValueError):
        ppo.train(args)
