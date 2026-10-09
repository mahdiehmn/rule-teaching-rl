"""Independently audit the complete Crafter learning artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'c935b98'
ARMS = ('none', 'rules_weak', 'rules')
SEEDS = tuple(range(1, 11))
BANK = 'research/rule_banks/crafter_v2_20260930/self_checked.json'
BANK_HASH = '0c378a80191af395d332f810acacd969f40f5b7e300fcea294ba5b5f0f2a812c'
SOURCE_FILES = (
    'algos/ppo_crafter.py', 'envs/crafter_symbolic.py',
    'scripts/run_crafter_learning_20260930.py',
    'scripts/crafter_rules_pilot_20260928.py',
    'scripts/crafter_rules_v2_20260930.py',
    'scripts/conditional_rules_v3.py', BANK,
    'research/crafter_learning_protocol_2026-09-30.md')


def read(path):
    """Read complete persisted JSON without importing the producer."""

    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """Record byte identity of each inspected artifact."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_steps():
    """Derive the protocol's rollout-rounded evaluation clock independently."""

    steps, next_eval = [], 50000
    for update in range(1, 1000000 // 2048 + 1):
        step = update * 2048
        if step >= next_eval or update == 1000000 // 2048:
            steps.append(step)
            next_eval += 50000
    return steps


def source_identity(source, repository=None):
    """Check the current source and bank against the declared original commit."""

    source = Path(source).resolve()
    repository = source if repository is None else Path(repository).resolve()
    hashes = {}
    for name in SOURCE_FILES:
        archived = subprocess.check_output(
            ['git', '-c', f'safe.directory={repository.as_posix()}',
             'show', f'{SOURCE_COMMIT}:{name}'], cwd=repository)
        current = (source / name).read_bytes()
        assert archived.replace(b'\r\n', b'\n') == current.replace(
            b'\r\n', b'\n'), f'Source drift: {name}'
        hashes[name] = dict(current_bytes=digest(source / name),
                            committed_lf=hashlib.sha256(archived.replace(
                                b'\r\n', b'\n')).hexdigest())
    assert hashes[BANK]['committed_lf'] == BANK_HASH
    return hashes


def expected_args(arm, seed, folder):
    """Spell out the frozen contract rather than reuse mutable Args defaults."""

    return dict(
        seed=seed, total_steps=1000000, num_envs=16, num_steps=128,
        learning_rate=.0003, gamma=.99, gae_lambda=.95, update_epochs=4,
        num_minibatches=4, clip_coef=.2, ent_coef=.01, vf_coef=.5,
        max_grad_norm=.5, rule_bank='' if arm == 'none' else BANK,
        distill_start=1. if arm == 'rules' else .1,
        distill_min=.01 if arm == 'rules' else .001,
        distill_decay_end=.5, distill_off=.75, pool_size=200,
        world_seed0=40000000, eval_every=50000, eval_episodes=10,
        eval_cap=3000, eval_seed0=41000000, out=str(folder.resolve()))


def validate_run(folder, arm, seed):
    """Recompute every saved endpoint and primary curve statistic."""

    folder = Path(folder)
    summary = read(folder / 'run_summary.json')
    args = read(folder / 'args.json')
    assert args == expected_args(arm, seed, folder)
    assert summary['args'] == args
    assert summary['status'] == 'completed'
    assert summary['global_step'] == 999424
    assert summary['wall_time_sec'] > 0
    assert np.isclose(summary['sps'], 999424 / summary['wall_time_sec'])
    evals = [json.loads(line) for line in
             (folder / 'evaluations.jsonl').read_text().splitlines()
             if line.strip()]
    assert [e['step'] for e in evals] == expected_steps()
    from crafter.constants import achievements
    for episode_panel in evals:
        rates = episode_panel['rates']
        assert set(rates) == set(achievements) and len(rates) == 22
        for value in rates.values():
            assert np.isfinite(value) and 0 <= value <= 100
            assert np.isclose(value / 10., round(value / 10.), atol=1e-12)
        mean = sum(rates.values()) / 100.
        score = np.exp(np.mean(np.log1p(list(rates.values())))) - 1
        assert np.isclose(mean, episode_panel['mean_achievements'],
                          atol=1e-12, rtol=0)
        assert np.isclose(score, episode_panel['score'], atol=1e-12, rtol=0)
        assert 0 <= episode_panel['deaths'] <= 10
        assert 1 <= episode_panel['mean_steps'] <= 3000
    auc = float(np.mean([e['mean_achievements'] for e in evals]))
    assert np.isclose(auc, summary['auc'], atol=1e-12, rtol=0)
    assert summary['final'] == evals[-1]
    teacher = summary['teacher']
    assert teacher['llm_calls'] == 0 and teacher['rule_bank'] == (
        args['rule_bank'])
    expected_asked = 0 if arm == 'none' else 751616
    assert teacher['asked'] == expected_asked
    assert 0 <= teacher['labelled'] <= expected_asked
    if arm != 'none':
        assert teacher['labelled'] > 0
    files = ('args.json', 'run_summary.json', 'evaluations.jsonl',
             'final_model.pt')
    return dict(
        arm=arm, seed=seed, auc=auc,
        final=evals[-1]['mean_achievements'], final_score=evals[-1]['score'],
        curve=[e['mean_achievements'] for e in evals],
        rates=evals[-1]['rates'], teacher=teacher,
        wall_time_sec=summary['wall_time_sec'],
        artifact_sha256={name: digest(folder / name) for name in files})


def interval(values):
    """Use training seeds, with a separate flag for degenerate intervals."""

    values = np.asarray(values, dtype=float)
    constant = bool(np.all(values == values[0]))
    mean = float(values.mean())
    sem = 0. if constant else float(values.std(ddof=1) / np.sqrt(len(values)))
    width = float(stats.t.ppf(.975, len(values) - 1) * sem)
    return dict(mean=mean, ci95=[mean - width, mean + width],
                zero_empirical_variance=constant)


def paired(rows, guided, control):
    """Preserve the prespecified primary and label other contrasts secondary."""

    grid = {(r['arm'], r['seed']): r['auc'] for r in rows}
    differences = np.array([grid[guided, seed] - grid[control, seed]
                            for seed in SEEDS])
    estimate = interval(differences)
    p = (None if estimate['zero_empirical_variance'] else
         float(stats.ttest_1samp(differences, 0).pvalue))
    return dict(contrast=f'{guided} - {control}', n=10,
                differences=differences.tolist(), p_two_sided=p,
                role='primary' if guided == 'rules_weak' and
                control == 'none' else 'secondary_descriptive', **estimate)


def inventory(source, out):
    """Inspect completion separately from outcome estimation or promotion."""

    source, out = Path(source).resolve(), Path(out).resolve()
    study = source / 'results/crafter_learning_20260930'
    manifest = read(study / 'manifest.json')
    assert manifest['study'] == 'crafter_learning_20260930'
    assert manifest['bank'] == BANK and manifest['bank_sha256'] == BANK_HASH
    assert manifest['llm_calls'] == 0 and len(manifest['cells']) == 30
    assert {(c['arm'], c['seed']) for c in manifest['cells']} == {
        (arm, seed) for arm in ARMS for seed in SEEDS}
    status = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = study / f'{arm}_s{seed}'
            finished = (folder / 'run_summary.json').is_file() and (
                folder / 'final_model.pt').is_file()
            status.append(dict(
                arm=arm, seed=seed,
                state='complete_artifacts_present' if finished else
                'partial' if folder.exists() else 'unattempted'))
    receipt = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        producer='study runner', checker='independent audit',
        source=str(source), manifest_sha256=digest(study / 'manifest.json'),
        source_identity=source_identity(source), status=status,
        complete=sum(r['state'] == 'complete_artifacts_present'
                     for r in status))
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = out / f'inventory_{stamp}.json'
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(f'{receipt["complete"]}/30 complete artifact sets; {path}')
    return receipt


def report(source, out):
    """Refuse partial-cohort statistical reporting, then export audited curves."""

    source, out = Path(source).resolve(), Path(out).resolve()
    receipt = inventory(source, out)
    assert receipt['complete'] == 30, 'Complete all 30 cells before inference'
    study = source / 'results/crafter_learning_20260930'
    rows = [validate_run(study / f'{arm}_s{seed}', arm, seed)
            for seed in SEEDS for arm in ARMS]
    contrasts = [paired(rows, x, y) for x, y in (
        ('rules_weak', 'none'), ('rules', 'none'), ('rules_weak', 'rules'))]
    result = dict(
        **receipt, rows=rows, contrasts=contrasts, eval_steps=expected_steps(),
        summary='Complete exploratory learning cohort; ten paired training '
        'seeds; 95% intervals conditional on ten fixed evaluation worlds',
        checkpoint_behavior='pending separate final-policy reproduction',
        initial_policy_hashes='not persisted by original trainer',
        original_raw_evaluation_episodes='not persisted by original trainer',
        provenance_limit='Current source matches c935b98; no immutable '
        'launch-time source archive or per-run source hash was saved')
    with (out / 'independent_report.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (out / 'per_seed.csv').open('x', newline='') as handle:
        columns = ['arm', 'seed', 'auc', 'final', 'final_score', 'wall_time_sec']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    plot(rows, out)
    print(json.dumps(contrasts, indent=2))


def plot(rows, out):
    """Show all three arms with training-seed 95% pointwise intervals."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for arm, label, color in (
            ('none', 'PPO alone', '#777777'),
            ('rules_weak', 'Rules, weak imitation (primary)', '#2166ac'),
            ('rules', 'Rules, stronger imitation (secondary)', '#d95f02')):
        curves = np.array([r['curve'] for r in rows if r['arm'] == arm])
        assert curves.shape == (10, 20)
        means = curves.mean(0)
        widths = stats.t.ppf(.975, 9) * curves.std(0, ddof=1) / np.sqrt(10)
        xs = np.array(expected_steps()) / 1e6
        ax.fill_between(xs, np.maximum(0, means - widths),
                        np.minimum(22, means + widths), color=color, alpha=.13)
        ax.plot(xs, means, color=color, linewidth=2, label=label)
    ax.set(xlabel='Training environment transitions (millions)',
           ylabel='Teacher-free achievements per episode',
           title='Crafter symbolic variant: exploratory learning study')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .015, '10 training seeds; 10 fixed evaluation worlds; '
             'sampled student-only actions; pointwise 95% t intervals.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'crafter_learning_ci95.{suffix}', dpi=180)
    plt.close(fig)


def main():
    """Audit read-only source/results and write only to the chosen new output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'report'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'inventory':
        inventory(args.source, args.out)
    else:
        report(args.source, args.out)


if __name__ == '__main__':
    main()
