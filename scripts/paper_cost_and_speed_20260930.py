"""Cost, time and learning speed: LLM rules against LLM action advice.

Read-only over synced runs: no training,
no API.

Per run of every arm the paper compares: wall-clock hours of the whole
training run, the teacher's own time (API waits for paid advice; rule
lookups for a rule bank), and LLM calls and dollars DURING training. Rule
banks also report their one-off writing cost (calls and dollars before
training, from the bank files). Runs are the ones the paper's report uses:
the rule-bank study's cells on replicates 5-19 with paid runs admitted by
dose fidelity (5%), selected by the same code; KeyCorridor's memory rules
come from the fix wave (protocol addendum 3) on replicates 5-14, the seeds
those studies share.

Also draws teacher-off learning curves of no teacher, the paper's rules
and the equal-call LLM advice on the same seeds.

    python -m scripts.paper_cost_and_speed_20260930 --data <vulcan_sync/data>
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts import report_rule_bank_paper_20260928 as rp  # noqa: E402
from scripts.plot_rule_speed_20260929 import (BLUE, GREY, GRID, INK,  # noqa: E402
                                              INK2, ORANGE, curve)

TASKS = (('doorkey_8x8', 'DoorKey-8x8'), ('multiroom_n6', 'MultiRoom-N6'),
         ('keycorridor_s3r3', 'KeyCorridor-S3R3'))
STUDENTS = (('none', 'plain PPO'), ('count', 'PPO + count bonus'))
# the paper's development-selected rule set per cell (report --selected)
SELECTED = {('doorkey_8x8', 'none'): 'llm_rules_blind_strict',
            ('doorkey_8x8', 'count'): 'llm_rules_blind_strict',
            ('multiroom_n6', 'none'): 'llm_rules_scoped',
            ('multiroom_n6', 'count'): 'llm_rules_blind_strict'}
BANK_FILE = {'llm_rules_blind_strict': 'blind_strict.json',
             'llm_rules_scoped': 'scoped.json'}
BANK_DIR = {'doorkey_8x8': 'research/rule_banks/v3_20260928',
            'multiroom_n6': 'research/rule_banks/multiroom_20260928'}
MEM_BANK = ('research/rule_banks/keycorridor_mem_20260929/'
            'self_checked_pooled_valid.json')
PURPLE = '#8e5bb5'
REPLICATES = (5, 20)


def bank_runs(efficiency, tolerance=0.05):
    """(task, bonus, arm) -> {seed: run dir}, exactly as the report admits.

    Mirrors report_rule_bank_paper_20260928.load (later batches replace
    earlier ones), keeping the run directory as well.
    """
    out = defaultdict(dict)
    for name in rp.BATCHES:
        batch = Path(efficiency) / name
        if not (batch / 'manifest.json').exists():
            continue
        for cell in rp.read(batch / 'manifest.json')['cells']:
            args = cell['args']
            r = (cell['seed'] - rp.SEED0[args['task']]) // 100
            if not REPLICATES[0] <= r < REPLICATES[1]:
                continue
            key = (args['task'], cell.get('bonus', args['bonus']),
                   cell['arm'])
            try:
                run = rp.find_run(batch / 'code/results/runs', args)
                rp.validate_run(run, args)
                paid = args['teacher'].startswith('llm')
                if paid and not (run / 'consultations.jsonl').exists():
                    continue
                lost = rp.request_failures(run) if paid else 0
                promised = rp.paid_calls(args) if paid else 0
                if lost and (not promised or lost / promised > tolerance):
                    continue
                out[key][cell['seed']] = run
            except (FileNotFoundError, ValueError, KeyError):
                continue
    return out


def memory_runs(data):
    """KeyCorridor memory rules (fix wave kc_mem, replicates 5-14)."""
    out = defaultdict(dict)
    batch = Path(data) / 'fix_wave/fix_wave_20260929_v1/kc_mem'
    for cell in json.loads((batch / 'manifest.json').read_text())['cells']:
        exit_path = batch / 'cells' / str(cell['index']) / 'exit.json'
        saved = json.loads(exit_path.read_text())
        if saved['artifact_status'] == 'terminal_contract_validated':
            out[(cell['task'], cell['bonus'], cell['arm'])][cell['seed']] = \
                batch / saved['runs'][0]
    return out


def run_costs(run):
    s = json.loads((Path(run) / 'run_summary.json').read_text())
    latest = s['latest']
    paid = s['args']['teacher'].startswith('llm')
    calls = latest.get('teacher_total_queries') or 0
    return dict(hours=s['wall_time_sec'] / 3600,
                teacher_s=latest.get('teacher_wall_time_s') or 0.0,
                calls=calls if paid else 0,
                dollars=latest.get('teacher_cost_dollars') or 0.0,
                per_call=(latest.get('teacher_wall_time_s') or 0) / calls
                if paid and calls else None)


def bank_cost(path):
    """One-off writing cost of a frozen rule bank: calls and dollars."""
    cost = json.loads(Path(path).read_text(encoding='utf-8')).get('cost', {})
    calls = sum(v.get('calls', 0) for v in cost.values())
    dollars = sum(v.get('dollars', v.get('usd', 0)) or 0
                  for v in cost.values())
    return calls, dollars


def arms_of(task, bonus):
    if task == 'keycorridor_s3r3':
        return (('none', 'no teacher', GREY),
                ('llm_action_online_eq', 'LLM action advice, equal calls',
                 PURPLE),
                ('rules_mem', 'LLM rules, paper weight', ORANGE),
                ('rules_mem_weak', 'LLM rules, a tenth of the weight', BLUE))
    return (('none', 'no teacher', GREY),
            ('llm_action_online_eq', 'LLM action advice, equal calls', PURPLE),
            (SELECTED[(task, bonus)], 'LLM rules, paper weight', ORANGE))


def table(runs, root):
    rows = []
    for task, name in TASKS:
        for bonus, student in STUDENTS:
            keys = [a for a, _, _ in arms_of(task, bonus)]
            keys += sorted(a for (t, b, a) in runs if (t, b) == (task, bonus)
                           and a.startswith('llm_action_online_480'))
            for arm in dict.fromkeys(keys):
                seeds = runs.get((task, bonus, arm), {})
                if not seeds:
                    continue
                c = [run_costs(r) for r in seeds.values()]
                row = dict(task=name, student=student, arm=arm, n=len(c),
                           hours=float(np.median([x['hours'] for x in c])),
                           teacher_s=float(np.mean([x['teacher_s']
                                                    for x in c])),
                           calls=float(np.mean([x['calls'] for x in c])),
                           dollars=float(np.mean([x['dollars'] for x in c])))
                per = [x['per_call'] for x in c if x['per_call']]
                row['per_call_s'] = float(np.mean(per)) if per else None
                bank = (MEM_BANK if arm.startswith('rules_mem') else
                        f'{BANK_DIR[task]}/{BANK_FILE[arm]}'
                        if arm in BANK_FILE else None)
                if bank:
                    row['bank_calls'], row['bank_dollars'] = bank_cost(
                        Path(root) / bank)
                rows.append(row)
    return rows


def plot(runs, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True,
                             sharey=True, facecolor='white')
    for col, (task, name) in enumerate(TASKS):
        for row, (bonus, student) in enumerate(STUDENTS):
            ax = axes[row][col]
            for arm, label, color in arms_of(task, bonus):
                seeds = runs.get((task, bonus, arm), {})
                curves = [curve(Path(r) / 'run_summary.json')
                          for r in seeds.values()]
                if not curves:
                    continue
                x = curves[0][0]
                ys = np.array([y for xx, y in curves if len(y) == len(x)])
                mean = ys.mean(axis=0)
                se = (ys.std(axis=0, ddof=1) / np.sqrt(len(ys))
                      if len(ys) > 1 else np.zeros_like(mean))
                ax.fill_between(x / 1e6, np.clip(mean - se, 0, 1),
                                np.clip(mean + se, 0, 1), color=color,
                                alpha=.13, linewidth=0)
                ax.plot(x / 1e6, mean, color=color, linewidth=2,
                        label=label)
            ax.set_title(f'{name} · {student}', fontsize=10.5, color=INK,
                         loc='left')
            ax.grid(color=GRID, linewidth=.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.set_ylim(-.02, 1.02)
            if row == 1:
                ax.set_xlabel('environment steps (millions)', color=INK2)
            if col == 0:
                ax.set_ylabel('teacher-off success', color=INK2)
    handles, labels = axes[0][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle('Rules against the same number of LLM calls spent on action '
                 'advice: mean over paired seeds (replicates 5-19; '
                 'KeyCorridor 5-14), band = 1 standard error', y=.925,
                 fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--root', type=Path, default=Path('.'))
    cli.add_argument('--out', type=Path,
                     default=Path('docs/assets/rule_speed_2026-09-30'))
    args = cli.parse_args()
    runs = bank_runs(args.data / 'efficiency')
    mem = memory_runs(args.data)
    for key, seeds in mem.items():
        runs[key] = seeds
    # KeyCorridor comparators on the memory rules' seeds only (5-14)
    mem_seeds = {s for seeds in mem.values() for s in seeds}
    for key in list(runs):
        if key[0] == 'keycorridor_s3r3' and not key[2].startswith(
                'rules_mem'):
            runs[key] = {s: r for s, r in runs[key].items()
                         if s in mem_seeds}
    args.out.mkdir(parents=True, exist_ok=True)
    plot(runs, args.out / 'rules_vs_advice.png')
    rows = table(runs, args.root)
    (args.out / 'cost_and_time.json').write_text(json.dumps(rows, indent=1))
    print(f"{'cell':34s} {'arm':30s} {'n':>3s} {'run h':>6s} "
          f"{'LLM calls':>9s} {'$ train':>8s} {'teacher s':>9s} "
          f"{'s/call':>6s} {'bank calls/$':>13s}")
    for r in rows:
        bank = (f"{r['bank_calls']:4d} ${r['bank_dollars']:.3f}"
                if 'bank_calls' in r else '')
        per = f"{r['per_call_s']:6.1f}" if r['per_call_s'] else '     -'
        print(f"{r['task'] + ' / ' + r['student']:34s} {r['arm']:30s} "
              f"{r['n']:3d} {r['hours']:6.2f} {r['calls']:9.0f} "
              f"{r['dollars']:8.3f} {r['teacher_s']:9.0f} {per} {bank:>13s}")
    print('wrote', args.out)


if __name__ == '__main__':
    raise SystemExit(main())
