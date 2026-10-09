"""Preparation, missing-evidence and one-time dispatch regression checks."""

from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile

import pytest

from scripts import run_plan_repair_20260927 as launch


SOURCE = (Path(__file__).resolve().parents[2] / 'vlm-rl-bench/results'
          '/vulcan_sync/data/explanation_lessons'
          '/explanation_formats_20260925_v1')


@pytest.fixture(scope='module')
def cells():
    hashes = dict.fromkeys((*launch.PATHS.values(),
                           'corpora/corrected_bank.json'), 'a' * 64)
    return launch.menu(hashes)


def test_frozen_menu_changes_targets_without_online_teacher(cells):
    assert len(cells) == 20
    assert {c['seed'] for c in cells} == set(launch.SEEDS)
    allowed = {'experiment_id', 'repair_arm', 'format_bank',
               'format_bank_sha256', 'lesson_mode', 'lesson_targets',
               'lesson_scale'}
    for seed in launch.SEEDS:
        group = [c for c in cells if c['seed'] == seed]
        assert [c['arm'] for c in group] == list(launch.ARMS)
        baseline = group[0]['args']
        for cell in group:
            a = cell['args']
            assert not a['guidance'] and a['query_budget'] == 0
            assert a['lesson_action_coef'] == 0
            assert a['advisor_rng_isolation'] and not a['advisor_sham']
            assert a['bonus'] == 'count' and a['obs_mode'] == 'symbolic'
            d = launch.derived_of(launch.Args(**a))
            assert d['num_iterations'] * d['batch_size'] == 9_999_360
            assert {k for k in a if a[k] != baseline[k]} <= allowed


@pytest.mark.skipif(not SOURCE.exists(), reason='Saved source unavailable')
def test_preparation_derives_real_bank_and_rejects_tampering(
        tmp_path, monkeypatch):
    """Exercise actual frozen inputs; mock only Git archive transport."""
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, 'w') as z:
        for name in launch.REQUIRED:
            z.writestr(name, b'# archived test fixture\n')
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda *a, **k: 'a' * 40 if k.get('text')
                        else packed.getvalue())
    batch = launch.prepare(tmp_path, SOURCE)
    manifest = launch.verify(batch)
    assert len(manifest['cells']) == 20
    assert launch.prepare(tmp_path, SOURCE) == batch
    report = launch.read(batch / 'corpora/target_report.json')
    assert report['splits']['train']['correction']['changed_cases'] == 55
    assert report['splits']['audit']['correction']['changed_cases'] == 18
    path = batch / 'corpora/corrected_bank.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='Frozen input changed'):
        launch.verify(batch)


@pytest.mark.parametrize('path', ['../outside', '/outside', '..\\outside'])
def test_input_escape_is_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        launch.inside(tmp_path, path)


def test_transfer_paths_only_change_root(cells):
    original = cells[0]['args']
    linux, windows = deepcopy(original), deepcopy(original)
    for name in (*launch.PATHS, 'format_bank'):
        linux[name] = '/project/batch/' + original[name]
        windows[name] = 'F:\\audit\\batch\\' + original[name]
    assert launch.portable_settings(linux) == original
    assert launch.portable_settings(windows) == original
    windows['format_bank'] = 'F:/somewhere_else/bank.json'
    with pytest.raises(ValueError):
        launch.portable_settings(windows)


def test_submission_is_one_unthrottled_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'verify', lambda _: {})
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        return '12345;cluster\n'

    monkeypatch.setattr(launch.subprocess, 'check_output', scheduler)
    launch.submit(tmp_path)
    launch.submit(tmp_path)
    assert len(calls) == 1
    assert '--array=0-19' in calls[0]
    assert '%' not in next(x for x in calls[0] if x.startswith('--array='))
    assert (tmp_path / 'SUBMITTED_JOB').read_text().strip() == '12345'


def test_ambiguous_submission_refuses_automatic_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'verify', lambda _: {})
    calls = []

    def scheduler(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('connection interrupted after possible submission')

    monkeypatch.setattr(launch.subprocess, 'check_output', scheduler)
    with pytest.raises(RuntimeError):
        launch.submit(tmp_path)
    with pytest.raises(FileExistsError):
        launch.submit(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('completed,missing_dispatch', [(0, False),
                         (16, False), (20, False), (20, True)])
def test_nomination_requires_all_pairs_and_linked_dispatch(
        tmp_path, monkeypatch, cells, completed, missing_dispatch):
    manifest = {'cells': cells, 'commit': 'a' * 40}
    launch.write(tmp_path / 'manifest.json', manifest)
    monkeypatch.setattr(launch, 'verify', lambda _: manifest)
    metrics = {}
    for c in cells[:completed]:
        directory = tmp_path / 'cells' / str(c['index'])
        directory.mkdir(parents=True)
        run = f"code/results/runs/{c['index']}"
        values = dict(auc=.6 if c['arm'] == 'corrected' else .2,
                      initial_sha256='a' * 64)
        metrics[str((tmp_path / run).resolve())] = values
        launch.write(directory / 'exit.json', dict(
            artifact_status='terminal_contract_validated', returncode=0,
            runs=[run], metrics=values))
        if not missing_dispatch:
            launch.write(directory / 'dispatch.json', dict(
                cell=c, args=c['args'], commit=manifest['commit'],
                manifest_sha256=launch.digest(tmp_path / 'manifest.json')))
    monkeypatch.setattr(launch, 'validate_run',
                        lambda path, _: metrics[str(path)])
    if missing_dispatch:
        with pytest.raises(FileNotFoundError):
            launch.report(tmp_path)
    else:
        report = launch.report(tmp_path)
        assert report['completed'] == completed
        assert report['useful_semantic_candidate'] == (completed == 20)
        assert report['correction_nomination'] == (completed == 20)
        assert sum('auc' in row for row in report['runs']) == completed


def test_source_pin_change_rejected(tmp_path):
    name = next(iter(launch.PINS))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'unexpected': 'source'}))
    with pytest.raises(ValueError, match='fingerprint'):
        launch.check_source(tmp_path)
