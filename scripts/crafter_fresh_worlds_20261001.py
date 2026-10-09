"""Evaluate all thirty saved Crafter v2 students on fifty fresh worlds.

Prospective candidate: freeze requires separate
clearance of the complete learning cohort. This module never trains.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats
import torch

from scripts import audit_crafter_learning_20261001 as audit
from scripts.reproduce_crafter_endpoints_20261001 import state_hash, write


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'crafter_fresh_worlds_20261001_v1'
PROTOCOL = ROOT / 'research/crafter_fresh_worlds_protocol_2026-10-01.md'
REPRO = ROOT / 'results/crafter_endpoint_reproduction_20261001'
REPRO_MANIFEST = (
    '2a93fc002ccb86dd9c7e7b42e536ac4088cb57775c6963c02b0ee4c29bb243af')
REPRO_REPORT = (
    'e808c778c30be64c1e74004bceffdc233d294eacb10857428f833f29c246e21d')
LAYOUTS = tuple(range(44000000, 44000050))
CAP = 3000
MAX_SECONDS = 7200
STOP = datetime(2026, 10, 1, 16, 30, tzinfo=timezone.utc).timestamp()


def cell_order():
    """Rotate arm position within each fixed ascending training-seed block."""

    cells = []
    for seed in audit.SEEDS:
        offset = (seed - 1) % 3
        arms = audit.ARMS[offset:] + audit.ARMS[:offset]
        cells.extend((arm, seed) for arm in arms)
    return cells


def runtime():
    """Bind the interpreter and directly relevant installed libraries."""

    return dict(python=sys.version, packages={name: version(name) for name in
                ('torch', 'numpy', 'scipy', 'crafter', 'opensimplex',
                 'gymnasium')})


def forecast(rows):
    """Apply the predeclared scheduling allowance without target outcomes."""

    lengths = [audit.read(Path(row['folder']) / 'run_summary.json')
               ['final']['mean_steps'] for row in rows]
    assert len(lengths) == 30
    assert all(np.isfinite(v) and 1 <= v <= CAP for v in lengths)
    return 60. + .003 * len(LAYOUTS) * sum(lengths)


def admit(seconds, now):
    """Require the forecast to fit both the elapsed and absolute limits."""

    assert np.isfinite(seconds) and 0 < seconds <= MAX_SECONDS
    assert now + seconds <= STOP, 'Forecast no longer fits before16:30UTC'


def archive_identity():
    """Reuse only the original code whose endpoint replay was accepted."""

    assert audit.digest(REPRO / 'manifest.json') == REPRO_MANIFEST
    assert audit.digest(REPRO / 'evaluation/report.json') == REPRO_REPORT
    manifest = audit.read(REPRO / 'manifest.json')
    report = audit.read(REPRO / 'evaluation/report.json')
    assert report['manifest_sha256'] == REPRO_MANIFEST
    assert report['all_reproduced'] is True
    assert manifest['python'] == sys.version
    assert manifest['torch'] == torch.__version__
    assert manifest['numpy'] == np.__version__
    assert version('crafter') == '1.8.3'
    source = Path(manifest['source'])
    for name, expected in manifest['archived_sha256'].items():
        assert audit.digest(source / name) == expected
    return source, manifest['archived_sha256']


def freeze(source, learning_report, clearance, out):
    """Pin every model only after the full learning result is reviewed."""

    source, out = Path(source).resolve(), Path(out).resolve()
    learning_report = Path(learning_report).resolve()
    clearance = Path(clearance).resolve()
    report, review = audit.read(learning_report), audit.read(clearance)
    assert report['complete'] == 30 and len(report['rows']) == 30
    assert Path(report['source']).resolve() == source
    assert review['accepted'] is True
    assert review['report_sha256'] == audit.digest(learning_report)
    assert review['study'] == 'crafter_learning_20260930'
    assert review['reviewer']
    assert review['scope'] == 'complete_30_run_learning_result'
    by_key = {(r['arm'], r['seed']): r for r in report['rows']}
    assert len(by_key) == 30 and set(by_key) == set(cell_order())
    cells = []
    for arm, seed in cell_order():
        folder = source / f'results/crafter_learning_20260930/{arm}_s{seed}'
        validated = audit.validate_run(folder, arm, seed)
        assert validated['artifact_sha256'] == (
            by_key[arm, seed]['artifact_sha256'])
        cells.append(dict(arm=arm, seed=seed, folder=str(folder),
                          artifact_sha256=validated['artifact_sha256']))
    archived, archived_hashes = archive_identity()
    from crafter.constants import achievements
    seconds = forecast(cells)
    admit(seconds, time.time())
    write(out / 'manifest.json', dict(
        study=STUDY, frozen_utc=datetime.now(timezone.utc).isoformat(),
        status='exploratory_fresh_world_endpoint_evaluation',
        source=str(archived), source_sha256=archived_hashes, cells=cells,
        achievement_names=list(achievements),
        learning_report=str(learning_report),
        learning_report_sha256=audit.digest(learning_report),
        clearance=str(clearance), clearance_sha256=audit.digest(clearance),
        layouts=list(LAYOUTS), episode_cap=CAP, episodes=1500,
        action_mode='sampled', generator_seed='training_seed',
        generator_reset='once_per_model_carried_across_worlds',
        advisor_on=False, llm_calls=0, policy_updates=0, threads=1,
        max_attempts=1, max_seconds=MAX_SECONDS, absolute_stop_unix=STOP,
        forecast_seconds=seconds, runtime=runtime(),
        code_sha256=audit.digest(__file__),
        auditor_sha256=audit.digest(audit.__file__),
        replay_helper_sha256=audit.digest(
            ROOT / 'scripts/reproduce_crafter_endpoints_20261001.py'),
        protocol_sha256=audit.digest(PROTOCOL)))
    print(f'Frozen 30 models / 1500 episodes; forecast {seconds / 60:.1f}min.')


def validate_manifest(path):
    """Reject altered inputs and protocol choices before run or reporting."""

    m = audit.read(path)
    assert m['study'] == STUDY and m['code_sha256'] == audit.digest(__file__)
    assert m['protocol_sha256'] == audit.digest(PROTOCOL)
    assert m['auditor_sha256'] == audit.digest(audit.__file__)
    assert m['replay_helper_sha256'] == audit.digest(
        ROOT / 'scripts/reproduce_crafter_endpoints_20261001.py')
    assert m['runtime'] == runtime()
    assert m['learning_report_sha256'] == audit.digest(m['learning_report'])
    assert m['clearance_sha256'] == audit.digest(m['clearance'])
    assert [(c['arm'], c['seed']) for c in m['cells']] == cell_order()
    assert m['layouts'] == list(LAYOUTS) and m['episode_cap'] == CAP
    assert m['episodes'] == 1500 and m['action_mode'] == 'sampled'
    assert m['generator_seed'] == 'training_seed'
    assert m['generator_reset'] == 'once_per_model_carried_across_worlds'
    assert m['advisor_on'] is False and m['policy_updates'] == 0
    assert m['llm_calls'] == 0 and m['threads'] == 1
    assert m['max_attempts'] == 1 and m['max_seconds'] == MAX_SECONDS
    assert m['absolute_stop_unix'] == STOP
    source, hashes = archive_identity()
    from crafter.constants import achievements
    assert m['achievement_names'] == list(achievements)
    assert Path(m['source']) == source and m['source_sha256'] == hashes
    for cell in m['cells']:
        for name, expected in cell['artifact_sha256'].items():
            assert audit.digest(Path(cell['folder']) / name) == expected
    return m


def check_time(deadline):
    """Stop with retained receipts instead of silently extending the batch."""

    if time.time() >= deadline:
        raise TimeoutError('Fixed fresh-world evaluation deadline reached')


def components(source):
    """Import the evaluated stack and script helpers from the archive."""

    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    # Namespace packages follow the new path; verify actual imports below.
    from algos.ppo_crafter import Net
    from envs import crafter_symbolic as cs
    from crafter.constants import achievements
    allowed = {'scripts', __name__, audit.__name__,
               'scripts.reproduce_crafter_endpoints_20261001'}
    prefixes = ('algos', 'envs', 'scripts', 'teachers', 'advising',
                'monitoring')
    # Reject cached local dependencies; a new CLI process must own a run.
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in prefixes and name not in allowed:
            filename = getattr(module, '__file__', None)
            if filename is None:
                paths = tuple(getattr(module, '__path__', ()))
                assert paths and Path(paths[0]).is_relative_to(source), name
            else:
                assert Path(filename).resolve().is_relative_to(source), name
    return Net, cs, achievements


def episode(net, env, rng, deadline):
    """Execute only frozen policy samples, retaining a compact action trace."""

    obs = env.reset()
    done, steps, reward_total = False, 0, 0.
    trace = hashlib.sha256()
    while not done and steps < CAP:
        if steps % 100 == 0:
            check_time(deadline)
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs[0][None]),
                            torch.as_tensor(obs[1][None]))
            assert torch.isfinite(logits).all() and logits.shape == (1, 17)
            action = int(torch.multinomial(logits.softmax(-1), 1,
                                           generator=rng))
        trace.update(bytes([action]))
        obs, reward, done, _ = env.step(action)
        assert np.isfinite(reward)
        reward_total += reward
        steps += 1
    return dict(steps=steps, achievements=env.achievements(),
                died=bool(env.env._player.health <= 0), native_done=done,
                evaluation_cap_reached=steps == CAP,
                reward=reward_total, action_sha256=trace.hexdigest())


def aggregate(episodes, names):
    """Recompute endpoint summaries from all fixed-panel episode records."""

    assert len(episodes) == 50
    assert [e['layout'] for e in episodes] == list(LAYOUTS)
    assert len(names) == 22 and len(set(names)) == 22
    for e in episodes:
        assert type(e['steps']) is int and 1 <= e['steps'] <= CAP
        assert type(e['died']) is bool and type(e['native_done']) is bool
        assert e['evaluation_cap_reached'] == (e['steps'] == CAP)
        assert e['native_done'] or e['evaluation_cap_reached']
        assert not e['died'] or e['native_done']
        assert np.isfinite(e['reward'])
        assert e['achievements'] == sorted(set(e['achievements']))
        assert set(e['achievements']) <= set(names)
        assert len(e['action_sha256']) == 64
        int(e['action_sha256'], 16)
    total = sum(len(e['achievements']) for e in episodes)
    rates = {name: 2 * sum(name in e['achievements'] for e in episodes)
             for name in names}
    return dict(achievement_total=total, mean_achievements=total / 50,
                rates=rates, score=float(np.expm1(np.mean(np.log1p(
                    list(rates.values()))))),
                mean_steps=sum(e['steps'] for e in episodes) / 50,
                transitions=sum(e['steps'] for e in episodes),
                deaths=sum(e['died'] for e in episodes))


def run(manifest, out):
    """Run one bounded attempt; retain failures and never resume silently."""

    out = Path(out).resolve()
    m = validate_manifest(manifest)
    admit(m['forecast_seconds'], time.time())
    binding = audit.digest(manifest)
    started = time.time()
    write(out / 'start.json', dict(manifest_sha256=binding,
                                   started_unix=started))
    deadline = min(started + MAX_SECONDS, STOP)
    try:
        torch.set_num_threads(1)
        Net, cs, achievements = components(m['source'])
        pool = []
        for layout in LAYOUTS:
            check_time(deadline)
            pool.extend(cs.world_pool([layout]))
        for cell in m['cells']:
            check_time(deadline)
            key = f'{cell["arm"]}_s{cell["seed"]}'
            identity = dict(manifest_sha256=binding, arm=cell['arm'],
                            seed=cell['seed'], checkpoint_sha256=(
                                cell['artifact_sha256']['final_model.pt']))
            write(out / 'attempts' / f'{key}.start.json', dict(
                **identity, started_unix=time.time()))
            try:
                tensors = torch.load(Path(cell['folder']) / 'final_model.pt',
                                     map_location='cpu', weights_only=True)
                assert isinstance(tensors, dict) and tensors
                assert all(isinstance(t, torch.Tensor)
                           and torch.isfinite(t).all()
                           for t in tensors.values())
                net = Net()
                net.load_state_dict(tensors, strict=True)
                net.eval()
                for parameter in net.parameters():
                    parameter.requires_grad_(False)
                before = state_hash(net)
                rng = torch.Generator().manual_seed(cell['seed'])
                rows = []
                journal_path = out / 'episodes' / f'{key}.jsonl'
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                with journal_path.open('x', encoding='utf-8') as journal:
                    for layout, world in zip(LAYOUTS, pool):
                        check_time(deadline)
                        row = dict(**identity, layout=layout,
                                   **episode(net, cs.CrafterSymbolic(
                                       [world], seed=0), rng, deadline))
                        journal.write(json.dumps(row, allow_nan=False) + '\n')
                        journal.flush()
                        rows.append(row)
                after = state_hash(net)
                assert before == after
                write(out / 'cells' / f'{key}.json', dict(
                    **identity, policy_before=before, policy_after=after,
                    episodes=50, advisor_on=False, policy_updates=0,
                    journal_sha256=audit.digest(journal_path),
                    achievement_names=list(achievements),
                    summary=aggregate(rows, achievements)))
                write(out / 'attempts' / f'{key}.end.json', dict(
                    **identity, status='completed', ended_unix=time.time()))
                print(f'Completed {key}: 50 fresh-world episodes.', flush=True)
            except Exception as exc:
                write(out / 'attempts' / f'{key}.end.json', dict(
                    **identity, status='failed', ended_unix=time.time(),
                    error_type=type(exc).__name__, error=str(exc)))
                raise
        write(out / 'end.json', dict(manifest_sha256=binding,
                                     status='completed', cells=30,
                                     ended_unix=time.time()))
    except Exception as exc:
        write(out / 'end.json', dict(manifest_sha256=binding, status='failed',
                                     ended_unix=time.time(),
                                     error_type=type(exc).__name__,
                                     error=str(exc)))
        raise


def contrasts(rows):
    """Use ten paired integer-count differences, preserving null variance."""

    grid = {(r['arm'], r['seed']): r for r in rows}
    assert len(rows) == len(grid) == 30 and set(grid) == set(cell_order())
    result = []
    for guided, control in (('rules_weak', 'none'), ('rules', 'none'),
                            ('rules_weak', 'rules')):
        values = np.array([grid[guided, s]['achievement_total']
                           - grid[control, s]['achievement_total']
                           for s in audit.SEEDS], dtype=float) / 50
        interval = audit.interval(values)
        p = None if interval['zero_empirical_variance'] else float(
            stats.ttest_1samp(values, 0).pvalue)
        result.append(dict(contrast=f'{guided} - {control}', n=10,
                           role='primary' if control == 'none'
                           and guided == 'rules_weak' else 'descriptive',
                           differences=values.tolist(), p_two_sided=p,
                           **interval))
    return result


def report(manifest, evaluation, out):
    """Require all1500 raw records and validate them before any inference."""

    m, evaluation, out = (validate_manifest(manifest), Path(evaluation),
                          Path(out))
    binding = audit.digest(manifest)
    start, end = (audit.read(evaluation / 'start.json'),
                  audit.read(evaluation / 'end.json'))
    assert start['manifest_sha256'] == end['manifest_sha256'] == binding
    assert end['status'] == 'completed' and end['cells'] == 30
    assert end['ended_unix'] - start['started_unix'] <= MAX_SECONDS + 1
    assert end['ended_unix'] <= STOP + 1
    keys = {f'{c["arm"]}_s{c["seed"]}' for c in m['cells']}
    assert {p.stem for p in (evaluation / 'cells').glob('*.json')} == keys
    assert {p.stem for p in (evaluation / 'episodes').glob('*.jsonl')} == keys
    assert {p.name for p in (evaluation / 'attempts').glob('*.json')} == {
        f'{key}.{suffix}.json' for key in keys for suffix in ('start', 'end')}
    rows, all_episodes = [], []
    for cell in m['cells']:
        key = f'{cell["arm"]}_s{cell["seed"]}'
        saved = audit.read(evaluation / 'cells' / f'{key}.json')
        identity = dict(manifest_sha256=binding, arm=cell['arm'],
                        seed=cell['seed'], checkpoint_sha256=(
                            cell['artifact_sha256']['final_model.pt']))
        receipts = [audit.read(evaluation / 'attempts' / f'{key}.{s}.json')
                    for s in ('start', 'end')]
        assert all(all(r[k] == v for k, v in identity.items())
                   for r in (saved, *receipts))
        assert receipts[1]['status'] == 'completed'
        assert start['started_unix'] <= receipts[0]['started_unix'] <= (
            receipts[1]['ended_unix']) <= end['ended_unix']
        assert saved['policy_before'] == saved['policy_after']
        assert saved['episodes'] == 50 and saved['advisor_on'] is False
        assert saved['policy_updates'] == 0
        assert saved['achievement_names'] == m['achievement_names']
        journal = evaluation / 'episodes' / f'{key}.jsonl'
        assert saved['journal_sha256'] == audit.digest(journal)
        episodes = [json.loads(line) for line in journal.read_text(
            encoding='utf-8').splitlines()]
        assert all(all(e[k] == v for k, v in identity.items())
                   for e in episodes)
        actual = aggregate(episodes, saved['achievement_names'])
        assert actual == saved['summary']
        rows.append(dict(arm=cell['arm'], seed=cell['seed'], **actual))
        all_episodes.extend(episodes)
    assert len(all_episodes) == 1500
    write(out / 'report.json', dict(
        study=STUDY, manifest_sha256=binding, cells=30, episodes=1500,
        rows=rows, contrasts=contrasts(rows),
        transitions=sum(r['transitions'] for r in rows),
        elapsed_seconds=end['ended_unix'] - start['started_unix'],
        independent_review='pending',
        scope='Exploratory final-policy difference conditional on 50 fresh '
        'shared worlds; ten reused training seeds; not learning AUC'))
    write(out / 'episodes.json', all_episodes)
    print('All 30 cells/1500 episodes validated; independent review pending.')


def main():
    """Separate freeze, bounded collection and complete-panel reporting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'report'))
    parser.add_argument('--source')
    parser.add_argument('--learning-report')
    parser.add_argument('--clearance')
    parser.add_argument('--manifest')
    parser.add_argument('--evaluation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.source, args.learning_report, args.clearance, args.out)
    elif args.action == 'run':
        run(args.manifest, args.out)
    else:
        report(args.manifest, args.evaluation, args.out)


if __name__ == '__main__':
    main()
