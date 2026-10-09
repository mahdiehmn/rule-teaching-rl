"""Analyze the Phase 1 advisors at 120 GPT calls on three tasks.

Reads synced artifacts only. Primary:
greedy-success AUC paired against no advice per task x student, 95% t
intervals, Holm over that family's advisor arms (protocol 2026-10-08).
`--partial` plots whatever exists and never computes contrasts.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from scripts import run_phase1_lowcap120 as study

FULL = 9_999_360
TASK_TITLES = {'doorkey': 'DoorKey-8x8', 'multiroom': 'MultiRoom-N6',
               'keycorridor': 'KeyCorridor-S3R3'}
STUDENTS = (('none', 'PPO'), ('count', 'Count-PPO'))
# Paper palette; teal re-stepped to #00869e to pass the chroma floor.
ARMS = (('none', 'No advice', '#687381', '--', None),
        ('entropy120', 'Entropy importance', '#00869e', '-.', None),
        ('probability120', 'Probability correction', '#833bb1', '-', 'D'),
        ('random120', 'Random query times', '#c85b17', ':', None))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_dir(batch, args):
    # Selective advisors append e.g. `_importance_i-entropy_b120` to the id.
    eid, seed = args['experiment_id'], args['seed']
    hits = [p for p in (batch / 'code/results/runs').glob(f'*{eid}*__{seed}__*')
            if f'{eid}__{seed}__' in p.name or f'{eid}_' in p.name]
    return hits[0] if len(hits) == 1 else None


def curve(directory):
    path = directory / 'evaluations.jsonl'
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = sorted({r['global_step']: r for r in rows
                   if r.get('teacher_on') is False}.values(),
                  key=lambda r: r['global_step'])
    if not rows:
        return None
    return (np.array([r['global_step'] for r in rows], dtype=float),
            np.array([r['success_rate'] for r in rows], dtype=float))


def auc(x, y, upto=None):
    if upto is not None:
        keep = x <= upto
        x, y = x[keep], y[keep]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def collect(data, partial):
    """One record per (suite, bonus, arm, seed) from new and reused cells."""
    records, problems = [], []
    adv = data / 'advising_strength'
    for suite in study.STUDIES:
        manifest = study.make_manifest(suite, data=data)
        entries = [(adv / manifest['experiment_id'], c['args'], c['strategy'],
                    c['bonus'], c['seed']) for c in manifest['cells']]
        for r in manifest['reused_cohorts']:
            old = read(adv / r['batch'] / 'manifest.json')
            cell = next(c for c in old['cells'] if c['index'] == r['index'])
            entries.append((adv / r['batch'], cell['args'], r['arm'],
                            r['bonus'], cell['args']['seed']))
        for batch, args, arm, bonus, seed in entries:
            directory = run_dir(batch, args) if batch.is_dir() else None
            points = curve(directory) if directory else None
            if points is None:
                problems.append(f'missing {suite}/{bonus}/{arm}/{seed}')
                continue
            x, y = points
            complete = x[-1] == FULL
            if not complete:
                problems.append(f'incomplete {suite}/{bonus}/{arm}/{seed} at {int(x[-1])}')
                if not partial:
                    continue
            initial = directory / 'initial_policy.sha256'
            records.append(dict(
                suite=suite, bonus=bonus, arm=arm, seed=seed, x=x, y=y,
                complete=complete, last_step=int(x[-1]),
                initial=initial.read_text().strip() if initial.exists() else '',
                auc=auc(x, y) if complete else math.nan,
                auc_5m=auc(x, y, 5_000_000) if x[-1] >= 5_000_000 else math.nan,
                final=float(y[-1]), run=directory.name))
    return records, problems


def contrasts(records):
    """Paired advisor-minus-none AUC per task x student; Holm per family."""
    by = defaultdict(dict)
    for r in records:
        by[(r['suite'], r['bonus'], r['arm'])][r['seed']] = r
    rows = []
    for suite in study.STUDIES:
        for bonus, _ in STUDENTS:
            control = by[(suite, bonus, 'none')]
            family = []
            for arm, *_ in ARMS[1:]:
                treated = by.get((suite, bonus, arm), {})
                seeds = sorted(set(control) & set(treated))
                if len(seeds) != 5:
                    raise ValueError(f'{suite}/{bonus}/{arm}: {len(seeds)} pairs')
                # New cells ran on Fir, reused controls on Vulcan: pairing is
                # by training seed; identical initial policies are reported.
                same_init = sum(control[s]['initial'] == treated[s]['initial']
                                for s in seeds)
                d = np.array([treated[s]['auc'] - control[s]['auc'] for s in seeds])
                mean, sd = float(d.mean()), float(d.std(ddof=1))
                if sd == 0:
                    lo = hi = mean
                    p = math.nan
                else:
                    half = stats.t.ppf(.975, 4) * sd / math.sqrt(5)
                    lo, hi = mean - half, mean + half
                    p = float(stats.ttest_1samp(d, 0).pvalue)
                family.append(dict(task=suite, student=bonus, arm=arm, n=5,
                                   mean_diff=mean, ci_low=lo, ci_high=hi, p=p,
                                   positive=int((d > 0).sum()),
                                   same_initial_policy=f'{same_init}/5'))
            tested = sorted((r for r in family if not math.isnan(r['p'])),
                            key=lambda r: r['p'])
            running = 0.0
            for rank, r in enumerate(tested):
                running = max(running, min(1.0, r['p'] * (len(tested) - rank)))
                r['holm_p'] = running
            for r in family:
                r.setdefault('holm_p', math.nan)
            rows += family
    return rows


def plot(records, out, partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.0), sharey=True)
    handles = []
    for col, suite in enumerate(study.STUDIES):
        for row, (bonus, student) in enumerate(STUDENTS):
            ax = axes[row, col]
            for arm, label, color, style, marker in ARMS:
                runs = [r for r in records if (r['suite'], r['bonus'], r['arm'])
                        == (suite, bonus, arm)]
                if not runs:
                    continue
                n = min(len(r['x']) for r in runs)
                x = runs[0]['x'][:n] / 1e6
                ys = np.stack([r['y'][:n] for r in runs]) * 100
                mean = ys.mean(axis=0)
                (h,) = ax.plot(x, mean, color=color, linestyle=style,
                               linewidth=2, marker=marker, markersize=3,
                               markevery=(1, 6), label=label)
                if len(runs) > 1:
                    half = stats.t.ppf(.975, len(runs) - 1) * ys.std(
                        axis=0, ddof=1) / math.sqrt(len(runs))
                    ax.fill_between(x, np.clip(mean - half, 0, 100),
                                    np.clip(mean + half, 0, 100),
                                    color=color, alpha=.07, linewidth=0)
                if label not in [k.get_label() for k in handles]:
                    handles.append(h)
            ax.set_xlim(.2048, 10)
            ax.set_ylim(-3, 103)
            ax.set_xticks([2, 4, 6, 8, 10])
            ax.grid(color='#e4e6e9', linewidth=.6)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(TASK_TITLES[suite], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{student}\nsuccess (%)')
    fig.supxlabel('Training transitions (millions); 120 GPT calls per guided run; '
                  'five paired seeds' + ('; PARTIAL' if partial else ''),
                  fontsize=9)
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'phase1_lowcap120.{ext}', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=study.ROOT / 'results')
    parser.add_argument('--out', type=Path,
                        default=study.ROOT / 'docs/assets/phase1_lowcap120')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, problems = collect(args.data, args.partial)
    for p in problems:
        print('WARN', p)
    if problems and not args.partial:
        raise SystemExit('Incomplete artifacts; use --partial for a descriptive look')
    with (args.out / 'runs.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'student', 'arm', 'seed', 'auc', 'auc_5m', 'final',
                    'last_step', 'initial_policy_sha256', 'run'])
        for r in sorted(records, key=lambda r: (r['suite'], r['bonus'], r['arm'], r['seed'])):
            w.writerow([r['suite'], r['bonus'], r['arm'], r['seed'], r['auc'],
                        r['auc_5m'], r['final'], r['last_step'], r['initial'], r['run']])
    groups = defaultdict(list)
    for r in records:
        groups[(r['suite'], r['bonus'], r['arm'])].append(r)
    with (args.out / 'groups.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'student', 'arm', 'n', 'mean_auc', 'mean_final'])
        for key in sorted(groups):
            g = groups[key]
            w.writerow([*key, len(g), np.nanmean([r['auc'] for r in g]),
                        np.mean([r['final'] for r in g])])
            print(f'{key[0]:12} {key[1]:6} {key[2]:15} n={len(g)} '
                  f'AUC={np.nanmean([r["auc"] for r in g]):.3f} '
                  f'final={np.mean([r["final"] for r in g]):.3f}')
    if not args.partial:
        rows = contrasts(records)
        with (args.out / 'contrasts.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f'{r["task"]:12} {r["student"]:6} {r["arm"]:15} '
                  f'd={r["mean_diff"]:+.3f} [{r["ci_low"]:+.3f},{r["ci_high"]:+.3f}] '
                  f'holm_p={r["holm_p"]:.4f} pos={r["positive"]}/5 '
                  f'init={r["same_initial_policy"]}')
    plot(records, args.out, args.partial)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
