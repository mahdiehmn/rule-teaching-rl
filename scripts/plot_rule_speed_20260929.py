"""Learning speed of the six cells: no teacher vs LLM rules at two weights.

Reads synced runs only (no training, no
API). The claim of the rule-teaching paper is efficiency: how fast the
student learns, not whether it eventually can. So this plots teacher-off
success against environment steps and reports steps to 80% and 95%
success and success over the first 1M and 2M steps, paired by seed.

Two cohorts (--cohort):
  dev      the fix wave's replicates (5-14 KeyCorridor, 5-9 DoorKey and
           MultiRoom); the paper-weight rules on DoorKey and MultiRoom come
           from the rule-bank confirmation batches on the same seeds (the
           fix wave's re-run no-teacher cells reproduce that study's
           exactly, so they pair)
  confirm  the fresh replicates 30-39 of protocol addenda 2 and 3, every
           arm from the fix wave's *_confirm suites

KeyCorridor's rules are the memory rules of addendum 3 (rules_mem,
rules_mem_weak); its rules without memory (rules_weak) are drawn dotted.

    python -m scripts.plot_rule_speed_20260929 --data <vulcan_sync/data> \\
        --cohort confirm
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASKS = (('doorkey_8x8', 'DoorKey-8x8', 'dk', 'doorkey'),
         ('multiroom_n6', 'MultiRoom-N6', 'mr', 'multiroom'),
         ('keycorridor_s3r3', 'KeyCorridor-S3R3', 'kc', 'keycorridor'))
STUDENTS = (('none', 'plain PPO'), ('count', 'PPO + count bonus'))
PAPER_RULES = {('doorkey_8x8', 'none'): 'llm_rules_blind_strict',
               ('doorkey_8x8', 'count'): 'llm_rules_blind_strict',
               ('multiroom_n6', 'none'): 'llm_rules_scoped',
               ('multiroom_n6', 'count'): 'llm_rules_blind_strict'}
# validated with dataviz/scripts/validate_palette.js (blue, orange);
# the baseline is neutral
GREY, ORANGE, BLUE = '#898781', '#eb6834', '#2a78d6'
ARMS = (('none', 'no teacher', GREY),
        ('rules', 'LLM rules, paper weight (1 to .01)', ORANGE),
        ('rules_weak', 'LLM rules, a tenth of it (.1 to .001)', BLUE))
# KeyCorridor: the memory rules take the rule arms' places; the rules
# without memory at a tenth of the weight are drawn dotted
KC_ARMS = (('none', 'no teacher', GREY, '-'),
           ('rules_mem', 'LLM rules, paper weight (1 to .01)', ORANGE, '-'),
           ('rules_mem_weak', 'LLM rules, a tenth of it (.1 to .001)', BLUE,
            '-'),
           ('rules_weak', 'KeyCorridor rules without the unlocking memory, '
            'a tenth', BLUE, ':'))
KEEP = ('none', 'rules', 'rules_weak', 'rules_mem', 'rules_mem_weak')
SUITES = {'dev': {'doorkey_8x8': ('dk',), 'multiroom_n6': ('mr',),
                  'keycorridor_s3r3': ('kc', 'kc_mem')},
          'confirm': {'doorkey_8x8': ('dk_confirm',),
                      'multiroom_n6': ('mr_confirm',),
                      'keycorridor_s3r3': ('kc_confirm', 'kc_mem_confirm')}}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e1e0d9'


def arms_of(task):
    if task == 'keycorridor_s3r3':
        return KC_ARMS
    return tuple((*a, '-') for a in ARMS)


def curve(summary):
    rows = [json.loads(line) for line in
            (summary.parent / 'evaluations.jsonl').read_text().splitlines()
            if line.strip()]
    return (np.array([r['global_step'] for r in rows], float),
            np.array([r['success_rate'] for r in rows], float))


def load(data, cohort='dev'):
    """(task, bonus, arm) -> {seed: (steps, success)}, completed runs."""
    out = defaultdict(dict)
    fix = Path(data) / 'fix_wave/fix_wave_20260929_v1'
    for task, suites in SUITES[cohort].items():
        for suite in suites:
            for p in (fix / suite).rglob('run_summary.json'):
                s = json.loads(p.read_text())
                if s.get('status') != 'completed':
                    continue
                a = s['args']
                arm = a['experiment_id'].split(f"_{a['bonus']}_", 1)[1]
                if arm in KEEP:
                    out[(task, a['bonus'], arm)][a['seed']] = curve(p)
    if cohort != 'dev':
        return out
    # the paper-weight rules on DoorKey and MultiRoom: rule-bank study,
    # restricted to the seeds the fix wave ran
    for task, _name, _suite, study in TASKS:
        if task == 'keycorridor_s3r3':
            continue
        batch = (Path(data) / 'efficiency' /
                 f'rule_bank_confirm_{study}_20260928_v1')
        for p in batch.rglob('run_summary.json'):
            s = json.loads(p.read_text())
            a = s['args']
            arm = PAPER_RULES.get((task, a['bonus']))
            if (s.get('status') == 'completed' and arm
                    and a['experiment_id'].endswith(f"_{a['bonus']}_{arm}")
                    and a['seed'] in out[(task, a['bonus'], 'none')]):
                out[(task, a['bonus'], 'rules')][a['seed']] = curve(p)
    return out


def first(x, y, th):
    hit = np.nonzero(y >= th)[0]
    return x[hit[0]] if len(hit) else np.nan


def window(x, y, horizon):
    m = x <= horizon
    return float(np.sum(np.diff(x[m]) * (y[m][:-1] + y[m][1:]) / 2)
                 / (x[m][-1] - x[m][0]))


def table(runs):
    rows = []
    for task, name, _s, _st in TASKS:
        for bonus, student in STUDENTS:
            for arm in KEEP:
                d = runs.get((task, bonus, arm), {})
                if not d:
                    continue
                t80 = np.array([first(*v, .8) for v in d.values()])
                t95 = np.array([first(*v, .95) for v in d.values()])
                rows.append(dict(
                    task=name, student=student, arm=arm, n=len(d),
                    to80=float(np.nanmedian(t80)) if np.any(~np.isnan(t80))
                    else None, reach80=int(np.sum(~np.isnan(t80))),
                    to95=float(np.nanmedian(t95)) if np.any(~np.isnan(t95))
                    else None, reach95=int(np.sum(~np.isnan(t95))),
                    first1m=float(np.mean([window(*v, 1e6)
                                           for v in d.values()])),
                    first2m=float(np.mean([window(*v, 2e6)
                                           for v in d.values()])),
                    auc=float(np.mean([window(*v, 1e12)
                                       for v in d.values()]))))
    return rows


def plot(runs, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True,
                             sharey=True, facecolor='white')
    for col, (task, name, _s, _st) in enumerate(TASKS):
        for row, (bonus, student) in enumerate(STUDENTS):
            ax = axes[row][col]
            for arm, label, color, style in arms_of(task):
                d = runs.get((task, bonus, arm), {})
                if not d:
                    continue
                x = next(iter(d.values()))[0]
                ys = np.array([v[1] for v in d.values()
                               if len(v[1]) == len(x)])
                mean = ys.mean(axis=0)
                se = ys.std(axis=0, ddof=1) / np.sqrt(len(ys)) \
                    if len(ys) > 1 else np.zeros_like(mean)
                if style == '-':
                    ax.fill_between(x / 1e6, np.clip(mean - se, 0, 1),
                                    np.clip(mean + se, 0, 1), color=color,
                                    alpha=.14, linewidth=0)
                ax.plot(x / 1e6, mean, color=color, linewidth=2,
                        linestyle=style, label=f'{label} (n={len(ys)})')
            ax.set_title(f'{name} · {student}', fontsize=10.5, color=INK,
                         loc='left')
            ax.grid(color=GRID, linewidth=.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.set_ylim(-.02, 1.02)
            if row == 1:
                ax.set_xlabel('environment steps (millions)', color=INK2)
            if col == 0:
                ax.set_ylabel('teacher-off success', color=INK2)
    handles, labels = axes[0][2].get_legend_handles_labels()
    labels = [lab.rsplit(' (n=', 1)[0] for lab in labels]
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle(title, y=.9, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def plot_zoom(runs, path, cohort):
    """KeyCorridor, count bonus: the first 1.5M steps, every seed."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor='white')
    for arm, label, color, style in KC_ARMS:
        d = runs.get(('keycorridor_s3r3', 'count', arm), {})
        if not d:
            continue
        for x, y in d.values():
            m = x <= 1.5e6
            ax.plot(x[m] / 1e6, y[m], color=color, alpha=.2, linewidth=1,
                    linestyle=style)
        x = next(iter(d.values()))[0]
        ys = np.array([v[1] for v in d.values() if len(v[1]) == len(x)])
        m = x <= 1.5e6
        ax.plot(x[m] / 1e6, ys.mean(axis=0)[m], color=color, linewidth=2.4,
                linestyle=style, label=label)
    seeds = 'fresh seeds 30-39' if cohort == 'confirm' else 'seeds 5-14'
    ax.set_title(f'KeyCorridor-S3R3 · PPO + count bonus, {seeds}: first '
                 '1.5M steps\nthick = mean, thin = single seeds',
                 fontsize=10, color=INK, loc='left')
    ax.set_xlabel('environment steps (millions)', color=INK2)
    ax.set_ylabel('teacher-off success', color=INK2)
    ax.grid(color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=True, facecolor='white', edgecolor=GRID, framealpha=1,
              fontsize=8.5, labelcolor=INK, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--cohort', choices=tuple(SUITES), default='dev')
    cli.add_argument('--out', type=Path,
                     default=Path('docs/assets/rule_speed_2026-09-30'))
    args = cli.parse_args()
    runs = load(args.data, args.cohort)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = ('fresh replicates 30-39' if args.cohort == 'confirm' else
             'development replicates (5-14 KeyCorridor, 5-9 others)')
    plot(runs, args.out / f'learning_curves_{args.cohort}.png',
         f'How fast each student learns, {seeds}: mean over paired seeds, '
         'band = 1 standard error')
    plot_zoom(runs, args.out / f'keycorridor_count_first_1p5M_'
              f'{args.cohort}.png', args.cohort)
    rows = table(runs)
    (args.out / f'speed_table_{args.cohort}.json').write_text(
        json.dumps(rows, indent=1))
    fmt = (lambda v, n, k: f'{v / 1e3:6.0f}k ({k}/{n})' if v is not None
           else f'  never ({k}/{n})')
    print(f"{'cell':34s} {'arm':15s} {'to 80%':>14s} {'to 95%':>14s} "
          f"{'first 1M':>8s} {'first 2M':>8s} {'5M AUC':>7s}")
    for r in rows:
        print(f"{r['task'] + ' / ' + r['student']:34s} {r['arm']:15s} "
              f"{fmt(r['to80'], r['n'], r['reach80']):>14s} "
              f"{fmt(r['to95'], r['n'], r['reach95']):>14s} "
              f"{r['first1m']:8.3f} {r['first2m']:8.3f} {r['auc']:7.3f}")
    print('wrote', args.out)


if __name__ == '__main__':
    raise SystemExit(main())
