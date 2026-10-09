"""Check finite prospective coverage and duplicate-safe submission."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts import run_rule_reference_20260927 as launch
from scripts.report_rule_reference_20260927 import (
    planned_arms, planned_seeds,
)


def test_complete_menu_and_reporter_agree():
    cells = launch.menu()
    assert len(cells) == 620
    assert [c['index'] for c in cells] == list(range(620))
    assert Counter(c['task'] for c in cells) == {
        'doorkey_8x8': 180, 'multiroom_n6': 240,
        'keycorridor_s3r3': 200,
    }
    expected = {(task, bonus, arm, seed)
                for task in launch.TASKS for bonus in ('none', 'count')
                for arm in planned_arms(task, bonus)
                for seed in planned_seeds(task)}
    assert {(c['task'], c['bonus'], c['arm'], c['seed'])
            for c in cells} == expected


def test_learner_is_identical_across_advisors():
    mutable = {
        'experiment_id', 'guidance', 'teacher_stream', 'advisor',
        'importance_source', 'mistake_threshold', 'query_budget',
        'advice_budget', 'uniform_queries',
    }
    cells = launch.menu()
    for task in launch.TASKS:
        for bonus in ('none', 'count'):
            rows = [c for c in cells if c['task'] == task
                    and c['bonus'] == bonus and c['replicate'] == 0]
            frozen = [{k: v for k, v in c['args'].items()
                       if k not in mutable} for c in rows]
            assert all(a == frozen[0] for a in frozen)


def test_no_paid_path_or_sparse_stateful_bot():
    for c in launch.menu():
        a = c['args']
        assert a['teacher'] == launch.TASKS[c['task']][0]
        assert a['teacher_model'] == a['budget_ledger'] == ''
        assert a['max_cost_dollars'] == 0
        assert a['teacher_stream'] == (a['guidance'] and a['teacher'] == 'bot')
        assert a['advisor_no_teacher_peek'] and a['advisor_rng_isolation']
        assert not a['offline_summary_only']
        assert not a['query_windows'] and a['query_interval'] == 1
        assert a['total_timesteps'] // (a['num_envs'] * a['num_steps']) == 9765
        assert a['advice_budget'] == a['query_budget']
        assert a['uniform_queries'] == c['arm'].startswith('random')


def test_unthrottled_command():
    command = launch.submission_command(Path('/research/batch'))
    assert '--array=0-619' in command
    assert not any('%' in s for s in command if s.startswith('--array'))


def test_submission_receipt_prevents_repetition(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'verify', lambda p: {})
    calls = []
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda command, **kw: calls.append(command) or '123\n')
    launch.submit(tmp_path)
    launch.submit(tmp_path)
    assert len(calls) == 1
    assert (tmp_path / 'SUBMITTED_JOB').read_text().strip() == '123'


def test_ambiguous_submission_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'verify', lambda p: {})
    calls = []
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda command, **kw: calls.append(command) or '???')
    with pytest.raises(ValueError, match='receipt'):
        launch.submit(tmp_path)
    with pytest.raises(FileExistsError):
        launch.submit(tmp_path)
    assert len(calls) == 1


def test_artifact_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        launch.inside(tmp_path, '../escape')
    with pytest.raises(ValueError):
        launch.inside(tmp_path, '..\\escape')


def test_args_defaults_roundtrip():
    for c in launch.menu():
        assert asdict(launch.Args(**c['args'])) == c['args']


def test_report_connects_native_metrics_to_fixed_inventory(tmp_path,
                                                         monkeypatch):
    from scripts import rule_reference_validation as validator
    from scripts import report_rule_reference_20260927 as reporting

    cell = launch.menu()[0]
    batch = tmp_path / 'batch'
    directory = batch / 'cells/0'
    directory.mkdir(parents=True)
    manifest = dict(cells=[cell], commit='a' * 40)
    launch.write(batch / 'manifest.json', manifest)
    launch.write(directory / 'dispatch.json', dict(
        cell=cell, commit=manifest['commit'],
        manifest_sha256=launch.digest(batch / 'manifest.json')))
    metrics = dict(
        auc=0.0, final=0.0, initial_sha256='b' * 64, queries=0,
        deliveries=0, wall_seconds=3.0, total_planner_calls=0,
        curve=[dict(step=204800, greedy=0.0, sampled=0.0),
               dict(step=9999360, greedy=0.0, sampled=0.0)])
    launch.write(directory / 'exit.json', dict(
        artifact_status='terminal_contract_validated', returncode=0,
        runs=['code/results/runs/test'], metrics=metrics))
    monkeypatch.setattr(launch, 'verify', lambda path: manifest)
    monkeypatch.setattr(validator, 'validate_run', lambda *args: metrics)
    original = reporting.build_report
    monkeypatch.setattr(reporting, 'build_report',
                        lambda rows, **kwargs: original(rows))
    report = launch.report(batch, tmp_path / 'output')
    assert report['planned_cells'] == 620
    assert report['complete_cells'] == 1
    assert report['missing_cells'] == 619
    assert not report['global30_inference_available']
