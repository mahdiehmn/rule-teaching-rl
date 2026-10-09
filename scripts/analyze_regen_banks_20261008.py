"""Analyze the MultiRoom/KeyCorridor bank regeneration (protocol 2026-10-08).

Reads only validated exit records: the
fresh cohort's no-advice and selected-bank cells (reused controls) and the
regenerated-bank cells, paired by training seed within task x student.
As in the DoorKey study (O4): every bank's mean AUC and paired difference
to no advice, the average regenerated-minus-none difference with a crossed
percentile bootstrap (banks and seeds resampled separately), and the
selected bank's paired difference reported apart. `--partial` summarizes
whatever has finished and computes no interval.

  python -m scripts.analyze_regen_banks_20261008 --fresh <data> \
      --regen <data> [--out <dir>] [--partial]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

STUDY = 'fix_wave_20260929_v1'
TASKS = {'multiroom': ('mr_fresh', 'mr_regen', 'rules_weak'),
         'keycorridor': ('kc_fresh', 'kc_regen', 'rules_mem_weak')}
STUDENTS = (('none', 'PPO'), ('count', 'Count-PPO'))
BANKS = tuple(f'regen{k}' for k in range(5))
COLORS = dict(none='#687381', selected='#c85b17', regen='#00869e')


def records(batch):
    """Validated cells of one batch: (bonus, arm, seed) -> metrics."""
    manifest = json.loads((batch / 'manifest.json').read_text())
    out, missing = {}, []
    for cell in manifest['cells']:
        path = batch / 'cells' / str(cell['index']) / 'exit.json'
        exit_ = json.loads(path.read_text()) if path.exists() else {}
        key = (cell['bonus'], cell['arm'], cell['seed'])
        if exit_.get('artifact_status') == 'terminal_contract_validated':
            out[key] = exit_['metrics']
        else:
            missing.append(key)
    return out, missing


def paired(a, b):
    d = np.array(a) - np.array(b)
    if len(d) < 2 or np.ptp(d) == 0:
        return dict(mean=float(d.mean()), low=None, high=None, p=None,
                    positive=int((d > 0).sum()), n=len(d))
    half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return dict(mean=float(d.mean()), low=float(d.mean() - half),
                high=float(d.mean() + half),
                p=float(stats.ttest_1samp(d, 0).pvalue),
                positive=int((d > 0).sum()), n=len(d))


def analyze(fresh_root, regen_root, partial):
    report, rows = {}, []
    for task, (fresh_suite, regen_suite, selected) in TASKS.items():
        fresh, _ = records(Path(fresh_root) / 'fix_wave' / STUDY / fresh_suite)
        regen_batch = Path(regen_root) / 'fix_wave' / STUDY / regen_suite
        regen, missing = (records(regen_batch) if regen_batch.exists()
                          else ({}, ['batch not synced']))
        for bonus, student in STUDENTS:
            seeds = sorted(s for (b, a, s) in fresh if b == bonus and a == 'none')
            none = {s: fresh[(bonus, 'none', s)] for s in seeds}
            arms = {'selected': {s: fresh[(bonus, selected, s)] for s in seeds
                                 if (bonus, selected, s) in fresh}}
            for bank in BANKS:
                arms[bank] = {s: regen[(bonus, bank, s)] for s in seeds
                              if (bonus, bank, s) in regen}
            entry = dict(none_mean_auc=float(np.mean(
                [none[s]['auc'] for s in seeds])), arms={})
            same_init = []
            for arm, got in arms.items():
                common = [s for s in seeds if s in got]
                if not common:
                    continue
                same_init += [got[s]['initial_sha256'] == none[s]['initial_sha256']
                              for s in common]
                entry['arms'][arm] = dict(
                    n=len(common), mean_auc=float(np.mean(
                        [got[s]['auc'] for s in common])),
                    mean_final=float(np.mean([got[s]['final'] for s in common])),
                    vs_none=paired([got[s]['auc'] for s in common],
                                   [none[s]['auc'] for s in common]))
                for s in common:
                    rows.append(dict(task=task, student=student, arm=arm, seed=s,
                                     auc=got[s]['auc'], none_auc=none[s]['auc'],
                                     final=got[s]['final'],
                                     same_initial=got[s]['initial_sha256']
                                     == none[s]['initial_sha256']))
            entry['initial_policy_matches_control'] = (
                f'{sum(same_init)}/{len(same_init)}')
            complete = all(len(arms[b]) == len(seeds) for b in BANKS)
            if complete:
                delta = np.array([[arms[b][s]['auc'] - none[s]['auc']
                                   for s in seeds] for b in BANKS])
                rng = np.random.default_rng(20261008)
                draws = [float(delta[np.ix_(rng.integers(0, 5, 5),
                                            rng.integers(0, len(seeds),
                                                         len(seeds)))].mean())
                         for _ in range(10000)]
                entry['regenerated_minus_none'] = dict(
                    mean=float(delta.mean()),
                    crossed_bootstrap_ci95=np.quantile(draws, [.025, .975]).tolist(),
                    banks_with_positive_mean=int((delta.mean(axis=1) > 0).sum()))
            elif not partial:
                raise SystemExit(f'{task}/{student}: regenerated cells missing '
                                 f'({len(missing)} unfinished)')
            report[f'{task}/{student}'] = entry
    return report, rows


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.2), sharey=True)
    for ax, (key, entry) in zip(axes, report.items()):
        labels = ['No advice', 'Selected'] + [f'Regen {k + 1}' for k in range(5)]
        names = ['none', 'selected', *BANKS]
        for i, name in enumerate(names):
            if name == 'none':
                y, lo, hi = entry['none_mean_auc'], None, None
            elif name in entry['arms']:
                arm = entry['arms'][name]
                y = arm['mean_auc']
                v = arm['vs_none']
                lo = None if v['low'] is None else entry['none_mean_auc'] + v['low']
                hi = None if v['high'] is None else entry['none_mean_auc'] + v['high']
            else:
                continue
            color = COLORS['none' if name == 'none' else
                           'selected' if name == 'selected' else 'regen']
            ax.plot([i], [y], 'o', color=color, markersize=8)
            if lo is not None:
                ax.plot([i, i], [max(lo, 0), min(hi, 1)], color=color, lw=2)
        ax.axhline(entry['none_mean_auc'], color=COLORS['none'], lw=1, ls='--')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_title(key.replace('/', ' / ').replace('multiroom', 'MultiRoom-N6')
                     .replace('keycorridor', 'KeyCorridor-S3R3'), fontsize=9)
        ax.set_ylim(-.03, 1.03)
        ax.grid(axis='y', color='#e4e6e9', lw=.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Greedy-success AUC')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(Path(out) / f'regen_banks.{ext}', dpi=200)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--fresh', type=Path, required=True)
    cli.add_argument('--regen', type=Path, required=True)
    cli.add_argument('--out', type=Path, default=Path('docs/assets/regen_banks'))
    cli.add_argument('--partial', action='store_true')
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report, rows = analyze(args.fresh, args.regen, args.partial)
    (args.out / 'report.json').write_text(json.dumps(report, indent=1))
    if rows:
        with (args.out / 'runs.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(report, indent=1))
    plot(report, args.out)


if __name__ == '__main__':
    main()
