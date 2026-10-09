"""
Tests for RunTracker's summary writing.

These exist because of a measured failure mode, not a hypothetical
one. Three of 655 concurrent cluster jobs died with FileNotFoundError
from os.replace on run_summary.json.tmp -- transient metadata failures
on the shared parallel filesystem, at unrelated iterations, with a
single-threaded writer. One of them was 1h49m into a 46-hour run.

The rule the tests below pin down: a PERIODIC snapshot is monitoring
and must never kill a training run, while the FINAL write is the
result itself and must fail loudly, because a finished run that wrote
no summary is indistinguishable from a crashed one.
"""

import argparse
import json
import os

import pytest

from monitoring.metrics import RunTracker


def make_tracker(run_dir):
    """
    Build a minimal tracker writing into the given directory.
    """

    args = argparse.Namespace(task='doorkey_5x5', seed=0)
    return RunTracker(
        run_dir=str(run_dir),
        run_name='test_run',
        algo='ppo_test',
        args=args,
        obs_shape=(7, 7, 3),
        action_space='Discrete(7)',
    )


def test_write_summary_survives_transient_replace_failure(
    tmp_path, monkeypatch
):
    """
    A snapshot that fails once must retry and still land on disk.
    """

    tracker = make_tracker(tmp_path / 'run')

    # Fail the first os.replace, then let every later call through, so
    # the test exercises the retry rather than the fallback.
    real_replace = os.replace
    calls = {'n': 0}

    def flaky_replace(src, dst):
        calls['n'] += 1
        if calls['n'] == 1:
            raise FileNotFoundError(2, 'No such file or directory', src)
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', flaky_replace)
    monkeypatch.setattr('time.sleep', lambda _seconds: None)

    tracker.write_summary(status='running', global_step=1024)

    assert calls['n'] >= 2, 'the failed replace should have been retried'
    with open(tracker.summary_path, encoding='utf-8') as f:
        assert json.load(f)['global_step'] == 1024


def test_periodic_snapshot_does_not_raise_when_writing_is_impossible(
    tmp_path, monkeypatch
):
    """
    A snapshot that cannot be written at all must warn, not raise.
    """

    tracker = make_tracker(tmp_path / 'run')

    # Break both the atomic path and the in-place fallback, which is
    # the worst case: nothing about the summary can be persisted.
    monkeypatch.setattr(
        os, 'replace',
        lambda src, dst: (_ for _ in ()).throw(OSError('replace failed'))
    )
    monkeypatch.setattr('time.sleep', lambda _seconds: None)

    real_open = open

    def broken_open(path, *rest, **kwargs):
        if str(path).endswith('run_summary.json'):
            raise OSError('write failed')
        return real_open(path, *rest, **kwargs)

    monkeypatch.setattr('builtins.open', broken_open)

    # Training must continue; the run is hours of compute and the
    # snapshot is not the result.
    tracker.write_summary(status='running', global_step=2048)


def test_final_write_raises_when_writing_is_impossible(
    tmp_path, monkeypatch
):
    """
    close() must raise rather than report a run that wrote nothing.
    """

    tracker = make_tracker(tmp_path / 'run')

    monkeypatch.setattr(
        os, 'replace',
        lambda src, dst: (_ for _ in ()).throw(OSError('replace failed'))
    )
    monkeypatch.setattr('time.sleep', lambda _seconds: None)

    real_open = open

    def broken_open(path, *rest, **kwargs):
        if str(path).endswith('run_summary.json'):
            raise OSError('write failed')
        return real_open(path, *rest, **kwargs)

    monkeypatch.setattr('builtins.open', broken_open)

    with pytest.raises(OSError):
        tracker.close(status='completed', global_step=4096)
