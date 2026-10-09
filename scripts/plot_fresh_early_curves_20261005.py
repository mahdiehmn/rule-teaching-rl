"""Learning curves from step 0 for addendum 15's fresh cohort (figure data).

Reads only validated cells of the fresh
suites (results/fix_wave/<study>/<suite>/cells/<i>/exit.json), merges each
run's measured early evaluations (diagnostic_evaluations.jsonl, from step 0)
with its regular evaluations (evaluations.jsonl, 204,800 to 5M), and plots
the mean with pointwise 95% t-intervals over training seeds for every
task, student and arm. The x axis is symmetric-log: linear up to the first
regular evaluation, so the early rise is visible, logarithmic beyond.
Nothing is imputed: a step is plotted only where every seed was measured.
These are secondary displays; the primary statistics stay in the report.

    python -m scripts.plot_fresh_early_curves_20261005 --mode greedy
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'fix_wave_20260929_v1'
SUITES = ('dk_fresh', 'mr_fresh', 'kc_fresh', 'dk_fresh_cc', 'mr_fresh_cc',
          'kc_fresh_cc', 'mr_fresh_cross')
TASKS = (('doorkey_8x8', 'DoorKey-8x8'), ('multiroom_n6', 'MultiRoom-N6'),
         ('keycorridor_s3r3', 'KeyCorridor-S3R3'))
STUDENTS = (('none', 'PPO'), ('count', 'Count-PPO'))
# (label, colour, line style); colours follow the paper_20261004 figures.
STYLE = {'none': ('Without advice', '#536677', '--'),
         'rules_weak': ('Full rule bank', '#C45A20', '-'),
         'rules_mem_weak': ('Full rule bank', '#C45A20', '-'),
         'rules_mem_weak_noprog': ('Progress conditions removed', '#237B83',
                                   '-.'),
         'rules_weak_cc': ('Deranged targets', '#7A4E9C', ':'),
         'rules_mem_weak_cc': ('Deranged targets', '#7A4E9C', ':'),
         'rules_weak_xbank': ("Other student's bank", '#8C6D1F',
                              (0, (5, 1, 1, 1)))}
FIRST_REGULAR = 204_800
KEY = {'greedy': 'success_rate', 'sampled': 'sampled_success_rate'}


def read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding='utf-8').splitlines() if line]


def diagnostic_problems(run, rows):
    """Why a run's early evaluations cannot be used, if any reason: every
    requested frame measured once, teacher off, the configured episode
    count, and the student's parameters unchanged by the evaluation."""
    args = json.loads((Path(run) / 'run_summary.json').read_text(
        encoding='utf-8'))['args']
    wanted = sorted(int(f) for f in str(
        args.get('diagnostic_eval_frames', '')).split(',') if f.strip())
    got = sorted(int(f) for row in rows for f in row['requested_frames'])
    problems = []
    if got != wanted:
        problems.append(f'frames {got} != requested {wanted}')
    for row in rows:
        if row.get('teacher_on'):
            problems.append(f"teacher on at {row['global_step']}")
        if row.get('episodes') != args.get('eval_episodes'):
            problems.append(f"episodes {row.get('episodes')} at "
                            f"{row['global_step']}")
        if row.get('policy_sha256_before') != row.get('policy_sha256_after'):
            problems.append(f"policy changed at {row['global_step']}")
    return problems


def curve(run, mode):
    """One run's measured points (early diagnostics, then regular ones)
    and the reasons, if any, that its early points are unusable."""
    key = KEY[mode]
    points = {}
    extra = Path(run) / 'diagnostic_evaluations.jsonl'
    rows = read_jsonl(extra) if extra.exists() else []
    problems = diagnostic_problems(run, rows)
    for row in rows:
        points[int(row['global_step'])] = float(row[key])
    for row in read_jsonl(Path(run) / 'evaluations.jsonl'):
        points[int(row['global_step'])] = float(row[key])
    return dict(sorted(points.items())), problems


def load(root, suites=SUITES, mode='greedy', study=STUDY, problems=None):
    """{(task, bonus, arm): {seed: {step: success}}} from validated cells
    whose early evaluations are complete; other runs are left out and
    their reasons appended to `problems` (a list), when given."""
    out = {}
    for suite in suites:
        batch = Path(root) / 'results/fix_wave' / study / suite
        if not (batch / 'manifest.json').exists():
            continue
        cells = json.loads((batch / 'manifest.json').read_text(
            encoding='utf-8'))['cells']
        for cell in cells:
            exit_path = batch / 'cells' / str(cell['index']) / 'exit.json'
            if not exit_path.exists():
                continue
            saved = json.loads(exit_path.read_text(encoding='utf-8'))
            if saved.get('artifact_status') != 'terminal_contract_validated':
                continue
            run = batch / saved['runs'][0]
            key = (cell['task'], cell['bonus'], cell['arm'])
            points, reasons = curve(run, mode)
            if reasons:
                if problems is not None:
                    problems.append(dict(suite=suite, cell=cell['index'],
                                         seed=cell['seed'], reasons=reasons))
                continue
            out.setdefault(key, {})[cell['seed']] = points
    return out


def summarize(seeds):
    """Mean and pointwise 95% t-interval on the steps every seed has."""
    steps = sorted(set.intersection(*(set(c) for c in seeds.values())))
    y = np.array([[c[s] for s in steps] for c in seeds.values()])
    mean = y.mean(axis=0)
    n = len(y)
    half = (student_t.ppf(.975, n - 1) * y.std(axis=0, ddof=1) / np.sqrt(n)
            if n > 1 else np.zeros_like(mean))
    return np.array(steps), mean, half, n


def early_area(steps, mean, end=FIRST_REGULAR):
    """Normalized area under the mean curve from step 0 to `end`."""
    keep = steps <= end
    s, m = steps[keep], mean[keep]
    if len(s) < 2 or s[0] != 0:
        return None
    return float(np.sum(np.diff(s) * (m[:-1] + m[1:]) / 2) / (s[-1] - s[0]))


def write_table(data, path):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        out = csv.writer(stream)
        out.writerow(['task', 'bonus', 'arm', 'n_seeds', 'step', 'mean',
                      'ci95_low', 'ci95_high'])
        for (task, bonus, arm), seeds in sorted(data.items()):
            steps, mean, half, n = summarize(seeds)
            for s, m, h in zip(steps, mean, half):
                out.writerow([task, bonus, arm, n, int(s), f'{m:.4f}',
                              f'{m - h:.4f}', f'{m + h:.4f}'])


def plot(data, path, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey=True)
    handles = {}
    for row, (bonus, student) in enumerate(STUDENTS):
        for col, (task, name) in enumerate(TASKS):
            ax = axes[row][col]
            for (tk, bn, arm), seeds in sorted(data.items()):
                if (tk, bn) != (task, bonus) or arm not in STYLE:
                    continue
                label, color, style = STYLE[arm]
                steps, mean, half, n = summarize(seeds)
                ax.fill_between(steps, np.clip(mean - half, 0, 1),
                                np.clip(mean + half, 0, 1), color=color,
                                alpha=.17, lw=0)
                line, = ax.plot(steps, mean, color=color, ls=style, lw=1.4,
                                label=f'{label}')
                handles.setdefault(label, line)
            ax.set_xscale('symlog', linthresh=FIRST_REGULAR, linscale=1.0)
            ax.set_xlim(0, 5_000_000)
            ax.set_ylim(-.02, 1.02)
            ax.axvline(FIRST_REGULAR, color='#8A949D', ls=':', lw=.7)
            ax.set_xticks([0, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6])
            ax.set_xticklabels(['0', '.1', '.2', '.5', '1', '2', '5'])
            ax.grid(axis='y', color='#DCE1E5', lw=.55)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f'{student}\nsuccess ({mode})')
    fig.supxlabel('Environment transitions (millions; linear to 0.2, '
                  'logarithmic beyond)', fontsize=7.5, y=.1)
    fig.legend(handles.values(), handles.keys(), loc='lower center',
               ncol=len(handles), frameon=False, fontsize=7,
               bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT)
    cli.add_argument('--mode', choices=tuple(KEY), default='greedy')
    cli.add_argument('--out', type=Path,
                     default=ROOT / 'docs/assets/fresh_cohort_2026-10-05')
    args = cli.parse_args()
    problems = []
    data = load(args.root, mode=args.mode, problems=problems)
    for p in problems:
        print(f"EXCLUDED {p['suite']} cell {p['cell']} seed {p['seed']}: "
              + '; '.join(p['reasons']))
    if not data:
        raise SystemExit('No validated fresh-cohort cells found')
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(data, args.out / f'curves_{args.mode}.csv')
    plot(data, args.out / f'early_curves_{args.mode}.png', args.mode)
    summary = {}
    for key, seeds in sorted(data.items()):
        steps, mean, _half, n = summarize(seeds)
        summary['/'.join(key)] = dict(n_seeds=n, first_step=int(steps[0]),
                                      early_area=early_area(steps, mean))
    (args.out / f'early_summary_{args.mode}.json').write_text(
        json.dumps(dict(groups=summary, excluded=problems), indent=1) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
