"""Crafter v3b confirmation: fresh seeds 21-30, no teacher vs rules_v3b_weak.

Protocol, frozen before any run:
research/crafter_v3b_confirm_protocol_2026-10-01.md. Every setting is the
exploratory study's; the bank is the v3b bank that passed its offline gate
and its development round (rules_v3b_weak - none +0.81, 5/5, seeds 1-5).

  run      the 20 runs (--workers N)
  report   the confirmatory contrast and the figure
"""

import argparse
import json
from pathlib import Path

from scripts import run_crafter_learning_20260930 as exp

STUDY = 'crafter_v3b_confirm_20261001'
OUT = exp.ROOT / 'results' / STUDY
BANK = 'research/rule_banks/crafter_v3_20261001/v3b_preconditions.json'
SEEDS = tuple(range(21, 31))
ARMS = {'none': dict(rule_bank=''),
        'rules_v3b_weak': dict(rule_bank=BANK, distill_start=0.1,
                               distill_min=0.001)}


def cells():
    return [dict(arm=arm, seed=seed, **cfg) for seed in SEEDS
            for arm, cfg in ARMS.items()]


def run(out=OUT, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(dict(
        study=STUDY, bank=BANK, cells=cells(), llm_calls=0), indent=1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell, status in pool.map(exp.run_one, cells(),
                                     [out] * len(cells())):
            print(f"{cell['arm']:15s} seed {cell['seed']:2d}: {status}",
                  flush=True)


def report(out=OUT):
    rows = {}
    for cell in cells():
        path = exp.cell_dir(cell, out) / 'run_summary.json'
        if path.exists():
            s = json.loads(path.read_text())
            rows.setdefault(cell['arm'], {})[cell['seed']] = dict(
                auc=s['auc'], final=s['final']['mean_achievements'],
                score=s['final']['score'], evals=[
                    json.loads(line) for line in (exp.cell_dir(cell, out) /
                                                  'evaluations.jsonl')
                    .read_text().splitlines() if line.strip()])
    contrast = exp.paired(rows.get('rules_v3b_weak', {}), rows.get('none', {}))
    contrast['confirmed'] = bool(contrast.get('n', 0) > 1
                                 and contrast['mean'] > 0
                                 and contrast['p'] < .05)
    print(json.dumps(dict(contrast=contrast, n_per_arm={
        a: len(s) for a, s in rows.items()}), indent=1))
    (Path(out) / 'report.json').write_text(json.dumps(dict(
        contrast=contrast), indent=1))
    if rows:
        renamed = {('rules_weak' if a == 'rules_v3b_weak' else a): s
                   for a, s in rows.items()}
        exp.plot(renamed, Path(out) / 'crafter_v3b_confirm_curves.png')
    return contrast


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('run', 'report'))
    cli.add_argument('--workers', type=int, default=8)
    args = cli.parse_args()
    run(OUT, args.workers) if args.action == 'run' else report(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
