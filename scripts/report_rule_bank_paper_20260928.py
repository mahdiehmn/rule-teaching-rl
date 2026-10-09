"""Paired report across every rule-bank learning batch (read-only).

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The arms of one (task, student) cell are
spread over several frozen batches: the DoorKey plain-PPO pilot and its
blind extension, the MultiRoom study, the KeyCorridor and DoorKey
Count-PPO studies, the feature-replay controls and the paid equal-budget
online controls. They share seeds and learner settings, so this report
pairs them by seed. It refuses a pair whose initial policies differ.

Each batch's manifest must match its sha256 sidecar. With `--strict`,
every archived source file must also match (use after a full transfer).
Runs are validated with the pilot's terminal and teacher-off evaluation
contract. Missing or failed runs are listed, never counted as zero.

  python -m scripts.report_rule_bank_paper_20260928 \\
      --efficiency <dir holding the batch folders> [--out report.json] \\
      [--replicates 5 20]      # the fresh confirmation cohort only
      [--selected]             # rule set chosen on 0-4, reported on 5-19

`--selected` (the goal is faster learning, not rule precision) treats the
check level as a per-cell setting.
For each (task, student) it takes whichever of the checked and unchecked
rule sets has the higher mean AUC on the development replicates 0-4, or no
rules if neither beats the no-teacher arm there. The choice is then scored
on replicates 5-19 against every comparator, with Holm-adjusted two-sided
paired-t p-values over all of those contrasts together.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from scripts.run_plan_repair_20260927 import digest, read
from scripts.run_rule_bank_pilot_20260928 import (find_run, paired,
                                                  validate_run)
from scripts.run_rule_bank_study_20260928 import (paid_calls,
                                                  request_failures)

BATCHES = ('rule_bank_pilot_20260928_v1',
           'rule_bank_pilot_20260928_blind_v1',
           'rule_bank_multiroom_20260928_v1',
           'rule_bank_doorkey_count_20260928_v2',
           'rule_bank_keycorridor_20260928_v1',
           'rule_bank_doorkey_controls_20260928_v1',
           'rule_bank_multiroom_controls_20260928_v1',
           'rule_bank_doorkey_online_20260928_v1',
           'rule_bank_multiroom_online_20260928_v1',
           'rule_bank_keycorridor_online_20260928_v1',
           *(f'rule_bank_{kind}_{task}_20260928_v1'
             for kind in ('confirm', 'confirm_online', 'budget',
                          'budget_online')
             for task in ('doorkey', 'multiroom', 'keycorridor')),
           # reruns of paid cells hit by the credit outage come AFTER
           # their originals; the originals are never paired
           'rule_bank_rerun_paid_20260928_v2',
           # completion reruns of cells no batch covered (2026-09-29)
           'rule_bank_complete_paid_eq_20260929_v1',
           'rule_bank_complete_paid_480_20260929_v1',
           # deferred batches, and the third wave (every remaining axis)
           *(f'rule_bank_{kind}_20260928_v1'
             for task in ('doorkey', 'multiroom', 'keycorridor')
             for kind in (f'budget_online_{task}_b',
                          f'confirm_shuffled_{task}',
                          f'sweep_online_{task}', f'schedule_online_{task}',
                          f'variants_{task}', f'online_rules_{task}')),
           # fourth wave: valid-action KeyCorridor rules, replicates 20-29
           'rule_bank_valid_keycorridor_20260928_v1')
SEED0 = {'doorkey_8x8': 14_500_000, 'multiroom_n6': 14_600_000,
         'keycorridor_s3r3': 14_700_000}
V = 'llm_rules_blind_strict'
CONTRASTS = (
    # primary (confirmation plan, 2026-09-28)
    (V, 'none'),
    ('llm_rules_scoped', 'none'),
    (V, 'llm_action_online_eq'),        # equal LLM calls, action only
    (V, 'llm_action_replay'),
    (V, 'llm_action_predicate_replay'),  # labels reused on same features
    (V, 'llm_rules_random_subset'),
    # mechanism and secondary
    (V, 'llm_rules_scoped'),
    (V, 'llm_rules_shuffled'),
    ('llm_rules_scoped', 'llm_action_online_eq'),
    ('llm_rules_scoped', 'llm_action_replay'),
    ('llm_rules_scoped', 'llm_action_predicate_replay'),
    ('llm_action_online_eq', 'none'),
    ('llm_action_replay', 'none'),
    ('llm_action_predicate_replay', 'none'),
    # budget and dose
    (V, 'llm_action_online_480'),
    ('llm_rules_scoped', 'llm_action_online_480'),
    ('llm_action_online_480', 'none'),
    ('llm_action_online_480', 'llm_action_online_eq'),
    ('llm_action_online_480_entropy', 'none'),
    (V, 'llm_action_online_480_entropy'),
    ('llm_rules_scoped', 'llm_rules_scoped_cap480'),
    ('llm_rules_scoped_cap480', 'llm_action_online_480'),
    ('llm_rules_scoped_cap480', 'none'),
    ('llm_rules_scoped_12', 'none'),
    ('llm_rules_blind_strict_12', 'none'),
    ('llm_rules_scoped', 'llm_rules_scoped_12'),
    (V, 'llm_rules_blind_strict_12'),
    # third wave: online budgets and schedules
    *((f'llm_action_online_{n}', 'none') for n in (15, 30, 120)),
    *((V, f'llm_action_online_{n}') for n in (15, 30, 120)),
    *((f'llm_action_online_480_{s}', 'none')
      for s in ('entropy', 'mistake', 'regular')),
    *((f'llm_action_online_480_{s}', 'llm_action_online_480')
      for s in ('entropy', 'mistake', 'regular')),
    *((V, f'llm_action_online_480_{s}')
      for s in ('mistake', 'regular')),
    # third wave: weaker rule variants
    ('llm_rules_direct', 'none'),
    ('llm_rules_scope_checked', 'none'),
    ('llm_rules_scoped', 'llm_rules_direct'),
    (V, 'llm_rules_scope_checked'),
    # third wave: rules written during training
    ('llm_rules_online_blind', 'none'),
    ('llm_rules_online_raw', 'none'),
    ('llm_rules_online_blind', 'llm_rules_online_raw'),
    ('llm_rules_online_blind', 'llm_action_online_eq'),
    ('llm_rules_online_raw', 'llm_action_online_eq'),
    ('llm_rules_online_blind', V),
    # fourth wave (--replicates 20 30)
    ('llm_rules_scoped_valid', 'none'),
    ('llm_rules_scoped_valid', 'llm_rules_scoped'),
)


def load(efficiency, strict=False, replicates=(0, 20), tolerance=0.0):
    rows, problems = {}, []
    for name in BATCHES:
        batch = Path(efficiency) / name
        if not (batch / 'manifest.json').exists():
            problems.append(f'{name}: batch not present')
            continue
        if digest(batch / 'manifest.json') != \
                (batch / 'manifest.sha256').read_text().strip():
            raise ValueError(f'{name}: manifest identity differs')
        manifest = read(batch / 'manifest.json')
        if strict:
            for rel, expected in manifest['source_hashes'].items():
                if digest(batch / 'code' / rel) != expected:
                    raise ValueError(f'{name}: archived source changed: '
                                     f'{rel}')
        for cell in manifest['cells']:
            args = cell['args']
            r = (cell['seed'] - SEED0[args['task']]) // 100
            if not replicates[0] <= r < replicates[1]:
                continue
            key = (args['task'], cell.get('bonus', args['bonus']),
                   cell['arm'])
            try:
                run = find_run(batch / 'code/results/runs', args)
                metrics = validate_run(run, args)
                # A paid run whose LLM calls failed (for example while the
                # API account had no credit) trained on fewer labels than
                # its arm promises: list it, never pair it.
                # Truncated or malformed replies are the teacher's own
                # abstentions and stay; calls that never reached the model
                # (the 2026-09-28 credit outage) disqualify the run, whose
                # cell is rerun in rule_bank_rerun_paid_20260928_v2.
                # request_failures() finds none in a journal that is not
                # there, so a paid batch synced with --no-journals would
                # pass every outage-hit run: refuse instead.
                paid = args['teacher'].startswith('llm')
                if paid and not (run / 'consultations.jsonl').exists():
                    raise FileNotFoundError(
                        'consultations.jsonl missing; sync paid batches '
                        'without --no-journals')
                lost = request_failures(run) if paid else 0
                # Admission by dose fidelity, declared in
                # research/reviews/paid_dose_tolerance_and_deviations_
                # 2026-09-28.md. tolerance=0 is the original rule and stays
                # the default, so nothing changes unless it is asked for.
                # A run within `tolerance` of its arm's promised call count
                # trained on that arm's dose; below that it did not.
                promised = paid_calls(args) if paid else 0
                if lost and (not promised
                             or lost / promised > tolerance):
                    raise ValueError(
                        f'{lost} of {promised} paid calls never reached '
                        f'the model (tolerance {tolerance:.2%})')
                rows.setdefault(key, {})[cell['seed']] = metrics
            except (FileNotFoundError, ValueError, KeyError) as error:
                problems.append(f"{name} cell {cell['index']} "
                                f"({key[2]}, {cell['seed']}): {error}")
    return rows, problems


def report(efficiency, strict=False, replicates=(0, 20), tolerance=0.0):
    rows, problems = load(efficiency, strict, replicates, tolerance)
    result = dict(cells={}, problems=problems, replicates=list(replicates),
                  failure_tolerance=tolerance)
    for task, bonus in sorted({(t, b) for t, b, _ in rows}):
        arms = {arm: dict(
            n=len(by_seed),
            mean_auc=sum(m['auc'] for m in by_seed.values()) / len(by_seed),
            mean_labels=sum(m['labels_delivered'] for m in by_seed.values())
            / len(by_seed),
            mean_api_dollars=sum(m['api_dollars'] for m in by_seed.values())
            / len(by_seed))
            for (t, b, arm), by_seed in rows.items()
            if (t, b) == (task, bonus)}
        contrasts = [paired(rows.get((task, bonus, a), {}),
                            rows.get((task, bonus, c), {}), f'{a} - {c}')
                     for a, c in CONTRASTS]
        result['cells'][f'{task}/{bonus}'] = dict(
            arms=arms, contrasts=[c for c in contrasts if c['n']])
    return result


DEVELOP, CONFIRM = (0, 5), (5, 20)
CANDIDATES = {'llm_rules_blind_strict': 'checked',
              'llm_rules_scoped': 'unchecked'}
COMPARATORS = ('none', 'llm_action_online_eq', 'llm_action_replay',
               'llm_action_predicate_replay', 'llm_rules_random_subset')


def cohort(rows, first, stop):
    """The rows whose replicate index lies in [first, stop)."""
    return {key: {s: m for s, m in by_seed.items()
                  if first <= (s - SEED0[key[0]]) // 100 < stop}
            for key, by_seed in rows.items()}


def holm(pvalues):
    """Holm step-down adjustment; None stays None and is not counted."""
    order = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    adjusted, running = [None] * len(pvalues), 0.0
    for rank, (p, i) in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p))
        adjusted[i] = running
    return adjusted


def contrast(a, b, label):
    """paired() plus a two-sided paired-t p-value."""
    out = paired(a, b, label)
    delta = np.array([a[k]['auc'] - b[k]['auc'] for k in sorted(set(a) &
                                                                 set(b))])
    if len(delta) < 2:
        out['p'] = None
    elif np.allclose(delta, delta[0]):
        out['p'] = 1.0 if delta[0] == 0 else 0.0
    else:
        out['p'] = float(ttest_1samp(delta, 0).pvalue)
    return out


def selected(efficiency, strict=False, tolerance=0.0):
    """Choose the rule set per cell on DEVELOP; score it on CONFIRM."""
    rows, problems = load(efficiency, strict, (DEVELOP[0], CONFIRM[1]),
                          tolerance)
    develop, confirm = cohort(rows, *DEVELOP), cohort(rows, *CONFIRM)

    def mean(table, key):
        by_seed = table.get(key, {})
        return (sum(m['auc'] for m in by_seed.values()) / len(by_seed)
                if by_seed else None)

    cells, family = {}, []
    for task, bonus in sorted({(t, b) for t, b, _ in rows}):
        dev = {arm: mean(develop, (task, bonus, arm))
               for arm in (*CANDIDATES, 'none')}
        scored = [a for a in CANDIDATES if dev[a] is not None]
        best = max(scored, key=dev.get) if scored else None
        use = best if best and dev['none'] is not None and \
            dev[best] > dev['none'] else None
        cell = dict(develop_auc=dev, chosen=use,
                    chosen_kind=CANDIDATES.get(use, 'rules off'),
                    contrasts=[])
        if use:
            for other in COMPARATORS:
                c = contrast(confirm.get((task, bonus, use), {}),
                             confirm.get((task, bonus, other), {}),
                             f'{CANDIDATES[use]} rules - {other}')
                if c['n']:
                    cell['contrasts'].append(c)
                    family.append(c)
        elif best:
            # What the rules would have done: shown, never claimed.
            cell['not_used'] = contrast(
                confirm.get((task, bonus, best), {}),
                confirm.get((task, bonus, 'none'), {}),
                f'{CANDIDATES[best]} rules - none (not used)')
        cells[f'{task}/{bonus}'] = cell
    for c, p in zip(family, holm([c['p'] for c in family])):
        c['p_holm'] = p
    return dict(develop=list(DEVELOP), confirm=list(CONFIRM),
                holm_family_size=sum(c['p'] is not None for c in family),
                cells=cells, problems=problems)


def print_selected(result):
    print(f"Rule set chosen on replicates {result['develop']}, reported on "
          f"{result['confirm']}; Holm over "
          f"{result['holm_family_size']} contrasts")
    for name, cell in result['cells'].items():
        dev = ', '.join(f'{a} {v:.3f}' for a, v in cell['develop_auc'].items()
                        if v is not None)
        print(f"== {name}: {cell['chosen_kind']}   (develop AUC: {dev})")
        for c in cell['contrasts'] + ([cell['not_used']]
                                      if 'not_used' in cell else []):
            ci = ('' if c['ci95'] is None else
                  f" [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
            p = '' if c.get('p_holm') is None else f"  p_holm {c['p_holm']:.2g}"
            print(f"   {c['contrast']:48s} {c['mean']:+.3f}{ci} "
                  f"{c['positive']}/{c['n']} positive{p}")
    print(f"{len(result['problems'])} missing or invalid runs/batches")


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--efficiency', type=Path, default=Path(
        'results/efficiency'))
    cli.add_argument('--strict', action='store_true')
    cli.add_argument('--replicates', type=int, nargs=2, default=(0, 20),
                     metavar=('FIRST', 'STOP'),
                     help='replicate range; 5 20 is the fresh confirmation '
                          'cohort')
    cli.add_argument('--selected', action='store_true',
                     help='rule set chosen per cell on replicates 0-4, '
                          'reported on 5-19 with Holm; ignores --replicates')
    cli.add_argument('--failure-tolerance', type=float, default=0.0,
                     metavar='FRACTION',
                     help='admit a paid run that lost at most this fraction '
                          'of its arm\'s promised calls (dose fidelity). '
                          '0 is the original rule and the default; the '
                          'declared levels are 0, .05 (primary) and .30. '
                          'See research/reviews/'
                          'paid_dose_tolerance_and_deviations_2026-09-28.md')
    cli.add_argument('--out', type=Path)
    args = cli.parse_args()
    if args.selected:
        result = selected(args.efficiency, args.strict,
                          args.failure_tolerance)
        print(f'== paid-run admission: dose-fidelity tolerance '
              f'{args.failure_tolerance:.2%}')
        print_selected(result)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1))
        return 0
    result = report(args.efficiency, args.strict, tuple(args.replicates),
                    args.failure_tolerance)
    print(f"== paid-run admission: dose-fidelity tolerance "
          f"{args.failure_tolerance:.2%}")
    for cell, value in result['cells'].items():
        print(f'== {cell}')
        for arm, a in sorted(value['arms'].items(),
                             key=lambda kv: -kv[1]['mean_auc']):
            print(f"   {arm:30s} n={a['n']} AUC {a['mean_auc']:.3f} "
                  f"labels {a['mean_labels']:9.0f} "
                  f"API ${a['mean_api_dollars']:.3f}")
        for c in value['contrasts']:
            ci = ('' if c['ci95'] is None else
                  f" [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
            print(f"   {c['contrast']:58s} {c['mean']:+.3f}{ci} "
                  f"{c['positive']}/{c['n']} positive")
    print(f"{len(result['problems'])} missing or invalid runs/batches")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
