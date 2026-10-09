"""Crafter wave on Vulcan: progress ablation, the stronger student, a replication.

Protocol, frozen
before any run: research/crafter_vulcan_protocol_2026-10-02.md. Every setting
is the confirmed v3b study's (symbolic Crafter, PPO, 1M steps, 200 seeded
training worlds per seed, the same 10 held-out evaluation worlds, evaluation
every 50k steps, imitation .1 -> .001); only the arms below differ. No LLM call.

Arms, all on fresh seeds 31-40 (no Crafter run has used them):
  none                 plain PPO, no teacher
  rules_v3b_weak       plain PPO + the v3b bank (confirmed on seeds 21-30)
  rules_noprog_weak    plain PPO + the v3b bank minus its progress clauses
                       (scripts/progress_ablation_20261002.py)
  count_none           the count student (count_coef .01), no teacher
  count_rules_v3b      the count student + the v3b bank

    python -m scripts.run_crafter_vulcan_20261002 check
    python -m scripts.run_crafter_vulcan_20261002 prepare      # on Vulcan
    python -m scripts.run_crafter_vulcan_20261002 run-cell --index I
    python -m scripts.run_crafter_vulcan_20261002 report --root <results root>
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'crafter_vulcan_20261002'
PROTOCOL = 'research/crafter_vulcan_protocol_2026-10-02.md'
BANK = 'research/rule_banks/crafter_v3_20261001/v3b_preconditions.json'
NOPROG = ('research/rule_banks/progress_ablation_20261002/'
          'crafter_v3b_noprogress.json')
COUNT_COEF = 0.01
SEEDS = tuple(range(31, 41))
WEAK = dict(distill_start=0.1, distill_min=0.001)
ARMS = {
    'none': dict(rule_bank=''),
    'rules_v3b_weak': dict(rule_bank=BANK, **WEAK),
    'rules_noprog_weak': dict(rule_bank=NOPROG, **WEAK),
    'count_none': dict(rule_bank='', count_coef=COUNT_COEF),
    'count_rules_v3b': dict(rule_bank=BANK, count_coef=COUNT_COEF, **WEAK),
}
# Primary family (Holm over two): the progress clauses, and the rules for
# the stronger student. Replication: the confirmed v3b contrast on new seeds.
PRIMARY = (('rules_v3b_weak', 'rules_noprog_weak'),
           ('count_rules_v3b', 'count_none'))
REPLICATION = ('rules_v3b_weak', 'none')
SECONDARY = (('count_none', 'none'), ('rules_noprog_weak', 'none'),
             ('count_rules_v3b', 'rules_v3b_weak'))


def cells():
    return [dict(index=k, arm=arm, seed=seed, **cfg) for k, (seed, (arm, cfg))
            in enumerate((s, a) for s in SEEDS for a in ARMS.items())]


def batch_dir(root=ROOT):
    # results/<family>/<batch>, the layout scripts/sync_vulcan_selected.py
    # fetches with --batch crafter_vulcan/crafter_vulcan_20261002
    return Path(root) / 'results' / 'crafter_vulcan' / STUDY


def cell_dir(cell, root=ROOT):
    return batch_dir(root) / f"{cell['arm']}_s{cell['seed']}"


def command(cell, root=ROOT):
    argv = [sys.executable, '-u', '-m', 'algos.ppo_crafter',
            '--seed', str(cell['seed']), '--out', str(cell_dir(cell, root))]
    for key in ('rule_bank', 'distill_start', 'distill_min', 'count_coef'):
        if key in cell:
            argv += ['--' + key.replace('_', '-'), str(cell[key])]
    return argv


def prepare(root=ROOT):
    """Write the manifest once; refuse a dirty tree or a changed plan."""
    root = Path(root).resolve()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                     text=True).strip()
    out = batch_dir(root)
    manifest = dict(study=STUDY, protocol=PROTOCOL, commit=commit,
                    cells=cells(), llm_calls=0)
    path = out / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['cells'] != manifest['cells']:
            raise ValueError('An earlier manifest has different cells')
        print(f'{STUDY}: already prepared at {old["commit"][:12]}')
        return out
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    print(f'{STUDY}: prepared {len(manifest["cells"])} cells at {commit[:12]}')
    return out


def run_cell(index, root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((batch_dir(root) / 'manifest.json').read_text())
    cell = manifest['cells'][index]
    if cell != cells()[index]:
        raise ValueError('Cell differs from the frozen manifest')
    target = cell_dir(cell, root)
    if (target / 'run_summary.json').exists():
        print('already complete:', target.name)
        return 0
    if target.exists():
        raise FileExistsError(f'{target} exists without a summary; an '
                              'earlier attempt must be inspected first')
    log = batch_dir(root) / 'logs' / f"{cell['arm']}_s{cell['seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log.open('w') as handle:
        code = subprocess.run(command(cell, root), cwd=root, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    (batch_dir(root) / 'exits').mkdir(exist_ok=True)
    (batch_dir(root) / 'exits' / f'{index}.json').write_text(json.dumps(dict(
        cell=cell, returncode=code, started=started,
        finished=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        completed=(target / 'run_summary.json').exists())))
    return code


def paired(a, b, label):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    if len(d) < 2:
        return dict(contrast=label, n=len(d))
    half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else (
        1.0 if d.mean() == 0 else 0.0)
    return dict(contrast=label, n=len(d), mean=float(d.mean()),
                ci95=[float(d.mean() - half), float(d.mean() + half)],
                positive=int((d > 0).sum()), p=p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def report(root=ROOT):
    auc, final, score = {}, {}, {}
    for cell in cells():
        path = cell_dir(cell, root) / 'run_summary.json'
        if path.exists():
            s = json.loads(path.read_text())
            auc.setdefault(cell['arm'], {})[cell['seed']] = s['auc']
            final.setdefault(cell['arm'], {})[cell['seed']] = \
                s['final']['mean_achievements']
            score.setdefault(cell['arm'], {})[cell['seed']] = \
                s['final']['score']
    print({arm: len(v) for arm, v in auc.items()}, 'complete runs per arm')
    for arm in ARMS:
        if arm in auc:
            print(f"  {arm:18s} AUC {np.mean(list(auc[arm].values())):.3f}  "
                  f"final {np.mean(list(final[arm].values())):.2f}  "
                  f"score {np.mean(list(score[arm].values())):.2f}")
    result = dict(primary=[paired(auc.get(a, {}), auc.get(b, {}),
                                  f'{a} - {b}') for a, b in PRIMARY])
    ps = [c.get('p') for c in result['primary']]
    if all(p is not None for p in ps):
        for c, q in zip(result['primary'], holm(ps)):
            c['p_holm'] = q
            c['verdict'] = 'HELPS' if c['mean'] > 0 and q < .05 else (
                'NO HARM' if c['ci95'][0] > -.05 else 'NOT SHOWN')
    rep = paired(auc.get(REPLICATION[0], {}), auc.get(REPLICATION[1], {}),
                 ' - '.join(REPLICATION))
    if rep.get('p') is not None:
        rep['replicated'] = bool(rep['mean'] > 0 and rep['p'] < .05)
    result['replication'] = rep
    result['secondary'] = [paired(auc.get(a, {}), auc.get(b, {}),
                                  f'{a} - {b}') for a, b in SECONDARY]
    print(json.dumps(result, indent=1))
    (batch_dir(root) / 'report.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('check', 'prepare', 'run-cell',
                                        'report'))
    cli.add_argument('--index', type=int)
    cli.add_argument('--root', type=Path, default=ROOT)
    args = cli.parse_args()
    if args.action == 'check':
        for bank in (BANK, NOPROG):
            if not (ROOT / bank).exists():
                raise FileNotFoundError(bank)
        print(f'PASS {STUDY}: {len(cells())} cells, arms {list(ARMS)}, '
              f'seeds {SEEDS[0]}-{SEEDS[-1]}')
        return 0
    if args.action == 'prepare':
        prepare(args.root)
        return 0
    if args.action == 'run-cell':
        return run_cell(args.index, args.root)
    report(args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
