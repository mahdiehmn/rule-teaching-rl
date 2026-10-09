"""Exploratory Crafter learning study: PPO with and without the v2 rules.

Protocol, frozen
before any run: research/crafter_learning_protocol_2026-09-30.md. The v2
offline gate was INCONCLUSIVE (+1.27 achievements over random against a
+1.5 bar); the study was run anyway and is labelled exploratory.

Cells: arms x seeds 1-10, each a run of algos/ppo_crafter.py (1M steps,
symbolic Crafter, plain PPO). Arms differ only in the teaching channel:

  none        no teacher
  rules_weak  the chosen v2 bank, imitation .1 -> .001 (primary)
  rules       the same bank at the paper's weight, 1 -> .01

Same seed, same training worlds, same initial network: the arms pair by
seed. Runs execute locally in parallel worker processes; nothing calls an
LLM.

  run      every missing cell (--workers N)
  report   paired AUC of teacher-free mean achievements, final Crafter
           score and achievement rates, and the learning-curve figure
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'crafter_learning_20260930'
OUT = ROOT / 'results' / STUDY
BANK = 'research/rule_banks/crafter_v2_20260930/self_checked.json'
SEEDS = tuple(range(1, 11))
ARMS = {'none': dict(rule_bank=''),
        'rules_weak': dict(rule_bank=BANK, distill_start=0.1,
                           distill_min=0.001),
        'rules': dict(rule_bank=BANK, distill_start=1.0, distill_min=0.01)}
CONTRASTS = (('rules_weak', 'none'), ('rules', 'none'),
             ('rules_weak', 'rules'))
PRIMARY = ('rules_weak', 'none')


def cells():
    return [dict(arm=arm, seed=seed, **cfg) for seed in SEEDS
            for arm, cfg in ARMS.items()]


def cell_dir(cell, out=OUT):
    return Path(out) / f"{cell['arm']}_s{cell['seed']}"


def command(cell, out=OUT):
    argv = [sys.executable, '-m', 'algos.ppo_crafter',
            '--seed', str(cell['seed']), '--out', str(cell_dir(cell, out))]
    for key in ('rule_bank', 'distill_start', 'distill_min'):
        if key in cell:
            argv += ['--' + key.replace('_', '-'), str(cell[key])]
    return argv


def run_one(cell, out=OUT):
    target = cell_dir(cell, out)
    if (target / 'run_summary.json').exists():
        return cell, 'done before'
    if target.exists():
        return cell, 'incomplete directory exists; left untouched'
    log = Path(out) / 'logs' / f"{cell['arm']}_s{cell['seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as handle:
        code = subprocess.run(command(cell, out), cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    return cell, f'exit {code}'


def run(out=OUT, workers=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank = (ROOT / BANK).read_bytes().replace(b'\r\n', b'\n')
    manifest = dict(study=STUDY, bank=BANK,
                    bank_sha256=hashlib.sha256(bank).hexdigest(),
                    cells=cells(), llm_calls=0)
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['cells'] != \
            manifest['cells']:
        raise ValueError('Frozen cells differ from this runner')
    path.write_text(json.dumps(manifest, indent=1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell, status in pool.map(run_one, cells(), [out] * len(cells())):
            print(f"{cell['arm']:10s} seed {cell['seed']:2d}: {status}",
                  flush=True)


def load(out=OUT):
    rows = {}
    for cell in cells():
        path = cell_dir(cell, out) / 'run_summary.json'
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        if s.get('status') != 'completed':
            continue
        evals = [json.loads(line) for line in
                 (cell_dir(cell, out) / 'evaluations.jsonl').read_text()
                 .splitlines() if line.strip()]
        rows.setdefault(cell['arm'], {})[cell['seed']] = dict(
            auc=s['auc'], final=s['final']['mean_achievements'],
            score=s['final']['score'], evals=evals,
            labelled=s['teacher']['labelled'], asked=s['teacher']['asked'],
            hours=s['wall_time_sec'] / 3600)
    return rows


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k]['auc'] - b[k]['auc'] for k in keys])
    n = len(d)
    if n < 2:
        return dict(n=n)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n))
    p = (float(2 * student_t.sf(abs(mean) / (sd / np.sqrt(n)), n - 1))
         if sd > 0 else (1.0 if mean == 0 else 0.0))
    return dict(n=n, mean=mean, ci95=[mean - half, mean + half],
                positive=int((d > 0).sum()), p=p)


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_rule_speed_20260929 import (BLUE, GREY, GRID, INK,
                                                  INK2, ORANGE)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor='white')
    for arm, label, color in (('none', 'PPO alone', GREY),
                              ('rules', 'LLM rules, paper weight', ORANGE),
                              ('rules_weak', 'LLM rules, a tenth of it',
                               BLUE)):
        seeds = rows.get(arm, {})
        if not seeds:
            continue
        curves = [[e['mean_achievements'] for e in s['evals']]
                  for s in seeds.values()]
        n = min(len(c) for c in curves)
        ys = np.array([c[:n] for c in curves])
        x = np.array([e['step'] for e in next(iter(seeds.values()))
                      ['evals'][:n]]) / 1e6
        mean = ys.mean(0)
        se = ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else 0 * mean
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=.15,
                        linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2,
                label=f'{label} (n={len(ys)})')
    ax.set_title('Crafter (symbolic), plain PPO: teacher-free achievements '
                 'per episode,\nmean and one standard error over paired '
                 'seeds', fontsize=10, color=INK, loc='left')
    ax.set_xlabel('environment steps (millions)', color=INK2)
    ax.set_ylabel('achievements per episode', color=INK2)
    ax.grid(color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


def report(out=OUT):
    rows = load(out)
    result = dict(arms={}, contrasts=[])
    for arm, seeds in rows.items():
        result['arms'][arm] = dict(
            n=len(seeds),
            auc=float(np.mean([s['auc'] for s in seeds.values()])),
            final=float(np.mean([s['final'] for s in seeds.values()])),
            score=float(np.mean([s['score'] for s in seeds.values()])),
            label_rate=float(np.mean([s['labelled'] / s['asked']
                                      for s in seeds.values()
                                      if s['asked']] or [0])),
            hours=float(np.median([s['hours'] for s in seeds.values()])))
        a = result['arms'][arm]
        print(f"{arm:10s} n={a['n']:2d}  AUC {a['auc']:.2f}  final "
              f"{a['final']:.2f} achievements  score {a['score']:.2f}  "
              f"labelled {a['label_rate']:.0%}  {a['hours']:.2f} h")
    for x, y in CONTRASTS:
        if x in rows and y in rows:
            c = dict(contrast=f'{x} - {y}', **paired(rows[x], rows[y]))
            result['contrasts'].append(c)
            if c['n'] > 1:
                print(f"{c['contrast']:20s} {c['mean']:+.2f} "
                      f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] "
                      f"{c['positive']}/{c['n']} positive, p {c['p']:.3g}")
    (Path(out) / 'report.json').write_text(json.dumps(result, indent=1))
    plot(rows, Path(out) / 'crafter_learning_curves.png')
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('run', 'report'))
    cli.add_argument('--out', type=Path, default=OUT)
    cli.add_argument('--workers', type=int, default=10)
    args = cli.parse_args()
    if args.action == 'run':
        run(args.out, args.workers)
    else:
        report(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
