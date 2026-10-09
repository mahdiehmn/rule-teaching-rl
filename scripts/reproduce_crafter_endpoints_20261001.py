"""Reproduce the first seed's saved Crafter endpoints without any advisor."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from scripts import audit_crafter_learning_20261001 as audit


def write(path, value):
    """Keep validation receipts immutable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def state_hash(net):
    """Detect any unintended policy mutation during reproduction."""

    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def freeze(source, out):
    """Pin all three arms of seed1 before any reproduction outcome."""

    source, out = Path(source).resolve(), Path(out).resolve()
    # The live producer is developing v3. Preserve it and reproduce using a
    # newly materialized original commit, explicitly not a launch-time archive.
    code = out / 'original_code'
    code.mkdir(parents=True, exist_ok=False)
    archived = subprocess.check_output([
        'git', '-c', f'safe.directory={source.as_posix()}', 'archive',
        audit.SOURCE_COMMIT, 'algos', 'envs', 'scripts', 'teachers',
        'advising', 'monitoring', 'configs', audit.BANK,
        'research/crafter_learning_protocol_2026-09-30.md'], cwd=source)
    with tarfile.open(fileobj=io.BytesIO(archived)) as bundle:
        bundle.extractall(code, filter='data')
    identities = audit.source_identity(code, repository=source)
    archived_hashes = {p.relative_to(code).as_posix(): audit.digest(p)
                       for p in code.rglob('*') if p.is_file()}
    live_hashes = {name: audit.digest(source / name)
                   for name in audit.SOURCE_FILES}
    rows = []
    for arm in audit.ARMS:
        folder = source / f'results/crafter_learning_20260930/{arm}_s1'
        validated = audit.validate_run(folder, arm, 1)
        rows.append(dict(arm=arm, seed=1, folder=str(folder),
                         artifact_sha256=validated['artifact_sha256']))
    write(out / 'manifest.json', dict(
        study='crafter_endpoint_reproduction_20261001',
        purpose='Artifact reproduction, not new learning or transfer evidence',
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        source=str(code), producer_repository=str(source),
        source_identity=identities, archived_sha256=archived_hashes,
        producer_live_sha256=live_hashes, cells=rows,
        source_scope='Original c935b98 materialized for this audit; '
        'not an immutable historical training-launch archive',
        layouts=list(range(41000000, 41000010)), cap=3000,
        action_mode='sampled', action_rng_seed=1, teacher_on=False,
        policy_updates=0, threads=1, max_elapsed_seconds=3600,
        max_attempts_per_cell=1, python=sys.version,
        torch=torch.__version__, numpy=np.__version__,
        runner_sha256=audit.digest(__file__),
        auditor_sha256=audit.digest(audit.__file__)))
    print('Frozen three seed1 checkpoints; 30 original-world episodes.')


def run(manifest_path, out):
    """Use only the policy forward pass and independently aggregate episodes."""

    m = audit.read(manifest_path)
    out = Path(out).resolve()
    assert m['runner_sha256'] == audit.digest(__file__)
    assert m['auditor_sha256'] == audit.digest(audit.__file__)
    assert m['source_identity'] == audit.source_identity(
        m['source'], repository=m['producer_repository'])
    for name, expected in m['archived_sha256'].items():
        assert audit.digest(Path(m['source']) / name) == expected
    assert m['python'] == sys.version and m['torch'] == torch.__version__
    assert m['numpy'] == np.__version__
    assert [(c['arm'], c['seed']) for c in m['cells']] == [
        (arm, 1) for arm in audit.ARMS]
    assert m['layouts'] == list(range(41000000, 41000010))
    assert m['cap'] == 3000 and m['teacher_on'] is False
    assert m['policy_updates'] == 0 and m['max_attempts_per_cell'] == 1
    assert m['action_mode'] == 'sampled' and m['action_rng_seed'] == 1
    assert m['threads'] == 1 and m['max_elapsed_seconds'] == 3600
    torch.set_num_threads(1)
    sys.path.insert(0, m['source'])
    # Import architecture and observation adapters only, never train/evaluate
    # or a rule/advisor function. The loop below owns actions and metrics.
    from algos.ppo_crafter import Net
    from envs import crafter_symbolic as cs
    from crafter.constants import achievements
    assert Path(sys.modules['algos.ppo_crafter'].__file__).resolve() == (
        Path(m['source']) / 'algos/ppo_crafter.py').resolve()
    start_path = out / 'start.json'
    if not start_path.exists():
        write(start_path, dict(started_unix=time.time(),
                               manifest_sha256=audit.digest(manifest_path)))
    start = audit.read(start_path)
    assert start['manifest_sha256'] == audit.digest(manifest_path)
    deadline = start['started_unix'] + 3600
    pool = cs.world_pool(m['layouts'])
    results = []
    for cell in m['cells']:
        arm = cell['arm']
        folder = Path(cell['folder'])
        for name, expected in cell['artifact_sha256'].items():
            assert audit.digest(folder / name) == expected
        final_path = out / f'{arm}.json'
        if final_path.exists():
            saved_result = audit.read(final_path)
            assert saved_result['manifest_sha256'] == audit.digest(manifest_path)
            assert saved_result['checkpoint_sha256'] == (
                cell['artifact_sha256']['final_model.pt'])
            assert saved_result['arm'] == arm and saved_result['seed'] == 1
            assert saved_result['episodes'] == 10
            assert saved_result['teacher_on'] is False
            assert saved_result['policy_updates'] == 0
            assert saved_result['policy_sha256_before'] == (
                saved_result['policy_sha256_after'])
            results.append(saved_result)
            continue
        if time.time() >= deadline:
            raise TimeoutError('Endpoint reproduction incomplete at fixed cap')
        write(out / f'{arm}.start.json', dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint_sha256=cell['artifact_sha256']['final_model.pt']))
        state = torch.load(folder / 'final_model.pt', map_location='cpu',
                           weights_only=True)
        assert isinstance(state, dict) and state
        assert all(isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                   for v in state.values())
        net = Net()
        net.load_state_dict(state, strict=True)
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        before = state_hash(net)
        rng = torch.Generator().manual_seed(1)
        episodes = []
        with (out / f'{arm}.episodes.jsonl').open('x') as journal:
            for layout, world in zip(m['layouts'], pool):
                if time.time() >= deadline:
                    raise TimeoutError('Retain completed episodes at time cap')
                env = cs.CrafterSymbolic([world], seed=0)
                obs = env.reset()
                done, steps = False, 0
                trace = hashlib.sha256()
                while not done and steps < 3000:
                    with torch.no_grad():
                        logits, _ = net(torch.as_tensor(obs[0][None]),
                                        torch.as_tensor(obs[1][None]))
                    action = int(torch.multinomial(logits.softmax(-1), 1,
                                                   generator=rng))
                    trace.update(bytes([action]))
                    obs, _, done, _ = env.step(action)
                    steps += 1
                episode = dict(layout=layout, steps=steps,
                               achievements=env.achievements(),
                               died=bool(env.env._player.health <= 0),
                               action_sha256=trace.hexdigest())
                episodes.append(episode)
                journal.write(json.dumps(episode) + '\n')
                journal.flush()
        rates = {name: 100 * sum(name in e['achievements'] for e in episodes)
                 / 10 for name in achievements}
        actual = dict(mean_achievements=float(np.mean([
            len(e['achievements']) for e in episodes])), rates=rates,
            score=float(np.expm1(np.mean(np.log1p(list(rates.values()))))),
            mean_steps=float(np.mean([e['steps'] for e in episodes])),
            deaths=sum(e['died'] for e in episodes))
        saved = audit.read(folder / 'run_summary.json')['final']
        checks = {key: (actual[key] == saved[key] if key == 'rates' else
                       bool(np.isclose(actual[key], saved[key], rtol=0,
                                       atol=1e-10))) for key in actual}
        after = state_hash(net)
        assert before == after
        result = dict(arm=arm, seed=1, reproduced=all(checks.values()),
                      manifest_sha256=audit.digest(manifest_path),
                      checkpoint_sha256=cell['artifact_sha256']['final_model.pt'],
                      checks=checks, actual=actual, teacher_on=False,
                      policy_updates=0, policy_sha256_before=before,
                      policy_sha256_after=after, episodes=10)
        write(final_path, result)
        results.append(result)
        print(f'{arm} seed1 endpoint reproduced: {result["reproduced"]}',
              flush=True)
    write(out / 'report.json', dict(
        manifest_sha256=audit.digest(manifest_path), results=results,
        all_reproduced=all(r['reproduced'] for r in results),
        scope='Three first-seed endpoints on original evaluation worlds; '
        'no new independent learning replication or AUC reconstruction'))


def main():
    """Separate artifact identity freeze from bounded reproduction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--source')
    parser.add_argument('--manifest')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.source, args.out)
    else:
        run(args.manifest, args.out)


if __name__ == '__main__':
    main()
