"""Concurrent view comparisons must never share writable artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from algos import ppo_babyai


def test_simultaneous_views_and_repeats_have_separate_outputs(
    tmp_path, monkeypatch,
):
    """
    Freeze time and launch repeated views with the same task and seed.

    Writing a marker in every output directory detects overwrites as well
    as duplicate path strings. Same-view retries must also stay separate.
    """

    monkeypatch.setattr(ppo_babyai.time, 'time', lambda: 1234567890)
    views = ['symbolic', 'partial', 'symbolic_historical', 'historical'] * 4

    def allocate(item):
        index, view = item
        args = SimpleNamespace(task='gotoseq', seed=0, obs_mode=view)
        path = Path(ppo_babyai.create_run_directory(args, tmp_path))
        (path / 'marker.txt').write_text(str(index), encoding='utf-8')
        return path

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(allocate, enumerate(views)))
    assert len(set(paths)) == len(views)
    for index, path in enumerate(paths):
        assert f'ppo_babyai_{views[index]}' in path.name
        assert path.parent == tmp_path / 'results' / 'runs'
        assert (path / 'marker.txt').read_text() == str(index)
