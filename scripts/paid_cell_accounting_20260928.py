"""Paid-cell accounting: lost calls, admission by dose fidelity, cost to finish.

Read-only. Makes no API call, submits nothing, touches no ledger and writes
nothing unless --out is given. It answers three questions for every paid
batch of the rule-bank study:

  * how many consultations each run lost to calls that never reached the
    model (request_failures: metadata.failed and outcome request_failure);
  * how many cells are admitted at each tolerance in the declared rule
    (research/reviews/paid_dose_tolerance_and_deviations_2026-09-28.md);
  * how many calls and dollars remain to complete what is still missing.

The tolerances are fixed by that note and are not options here, so a run of
this script cannot quietly become a search over admission rules.

Usage, from the repository root on the cluster:

    python -m scripts.paid_cell_accounting_20260928
    python -m scripts.paid_cell_accounting_20260928 --out accounting.json
"""
import argparse
import json
from pathlib import Path

from scripts.run_plan_repair_20260927 import read
from scripts.run_rule_bank_pilot_20260928 import find_run
from scripts.run_rule_bank_study_20260928 import (
    EXPECTED_USD_PER_CALL, ROOT, SPECS, batch_dir, paid_calls,
    request_failures)

# Declared in the dose-tolerance note. Primary is the middle level; a
# conclusion counts as supported only if it holds at all three.
TOLERANCES = (0.0, 0.05, 0.30)
PRIMARY = 0.05


def judge(batch, cell, promised):
    """State and lost-call fraction of one cell's run, or why it has none.

    The verdict carries promised_calls itself, so admits() can take it
    directly: rerun_coverage() once passed a verdict without it and
    crashed on the cluster with KeyError: 'promised_calls'.
    """
    try:
        run = find_run(batch / 'code/results/runs', cell['args'])
    except FileNotFoundError:
        return dict(promised_calls=promised, state='not_started',
                    lost_calls=None, lost_fraction=None)
    if not (Path(run) / 'consultations.jsonl').exists():
        # A paid batch synced with --no-journals cannot be judged:
        # request_failures() returns 0 for an outage-hit run.
        return dict(promised_calls=promised, state='no_journal',
                    lost_calls=None, lost_fraction=None)
    lost = request_failures(run)
    return dict(promised_calls=promised, state='ran', lost_calls=lost,
                lost_fraction=(lost / promised) if promised else 0.0)


def rerun_coverage(root=ROOT, tolerance=PRIMARY):
    """What the rerun batch already gives, and what it will give.

    The report pairs the rerun and never the original, so an original whose
    rerun is admitted is NOT residual. Without this the accounting bills for
    cells already collected: the DoorKey online_eq arm showed 20 missing
    cells while the report had n=14 for it.

    A rerun that is queued but has not finished is a different case again:
    the cell is genuinely missing NOW, but relaunching it would pay twice.
    Returns (supplied, queued) as sets of (origin study, origin index).
    """
    supplied, queued = set(), set()
    for spec in (s for s in SPECS.values() if s.get('rerun')):
        batch = batch_dir(root, spec)
        if not (batch / 'manifest.json').exists():
            continue
        for cell in read(batch / 'manifest.json')['cells']:
            origin = cell.get('origin')
            if not origin:
                continue
            key = (origin['study'], origin['index'])
            verdict = judge(batch, cell, paid_calls(cell['args']))
            if admits(verdict, tolerance):
                supplied.add(key)
            elif not (batch / 'cells' / str(cell['index'])
                      / 'exit.json').exists():
                # Queued only while the rerun has not ended: v2 ended with
                # 72 cells refused at start, which are missing, not queued.
                queued.add(key)
    return supplied, queued


def inspect(root=ROOT, tolerance=PRIMARY):
    """One record per paid cell of every launched paid batch."""
    supplied, queued = rerun_coverage(root, tolerance)
    out = []
    for name, spec in sorted(SPECS.items()):
        if not spec.get('paid') or spec.get('rerun'):
            continue
        batch = batch_dir(root, spec)
        if not (batch / 'manifest.json').exists():
            continue
        manifest = read(batch / 'manifest.json')
        for cell in manifest['cells']:
            args = cell['args']
            promised = paid_calls(args)
            record = dict(batch=name, study=manifest['study'],
                          index=cell['index'], seed=cell['seed'],
                          arm=cell['arm'], task=args['task'],
                          bonus=cell.get('bonus', args.get('bonus')),
                          promised_calls=promised,
                          superseded=(manifest['study'],
                                      cell['index']) in supplied,
                          awaiting_rerun=(manifest['study'],
                                          cell['index']) in queued)
            out.append(dict(record, **judge(batch, cell, promised)))
    return out


def admits(record, tolerance):
    """Whether the declared rule admits this run at this tolerance."""
    if record['state'] != 'ran':
        return False
    if not record['promised_calls']:
        return True
    return record['lost_fraction'] <= tolerance


def is_residual(record, tolerance=PRIMARY):
    """Missing now: not superseded by a rerun, and not admitted itself."""
    if record.get('superseded'):
        return False
    return record['state'] != 'ran' or not admits(record, tolerance)


def needs_launch(record, tolerance=PRIMARY):
    """Missing AND nothing already queued to collect it.

    This is the only set worth paying for. A cell whose rerun is queued is
    missing from today's report but must not be relaunched.
    """
    return (is_residual(record, tolerance)
            and not record.get('awaiting_rerun'))


def summarise(records):
    """Per batch and overall: states, admission per tolerance, residual."""
    batches = {}
    totals = dict(cells=0, ran=0, not_started=0, no_journal=0)
    for r in records:
        b = batches.setdefault(r['batch'], dict(
            cells=0, ran=0, not_started=0, no_journal=0,
            admitted={str(t): 0 for t in TOLERANCES},
            lost_calls=0, arms=[]))
        b['cells'] += 1
        totals['cells'] += 1
        b[r['state']] += 1
        totals[r['state']] += 1
        if r['state'] == 'ran':
            b['lost_calls'] += r['lost_calls']
        for t in TOLERANCES:
            b['admitted'][str(t)] += admits(r, t)
        if r['arm'] not in b['arms']:
            b['arms'] = sorted(b['arms'] + [r['arm']])

    # Residual: cells that never started, plus cells the PRIMARY rule
    # refuses. Those are the ones that still have to be collected.
    residual = [r for r in records if is_residual(r, PRIMARY)]
    calls = sum(r['promised_calls'] for r in residual)
    launch = [r for r in records if needs_launch(r, PRIMARY)]
    launch_calls = sum(r['promised_calls'] for r in launch)
    # What zero tolerance would demand, for comparison: it is the rule the
    # 145-cell rerun batch was built from.
    strict = [r for r in records if is_residual(r, 0.0)]
    strict_calls = sum(r['promised_calls'] for r in strict)
    return dict(
        batches=batches, totals=totals, tolerances=list(TOLERANCES),
        primary=PRIMARY,
        residual=dict(cells=len(residual), calls=calls,
                      usd_bound=round(calls * EXPECTED_USD_PER_CALL, 2)),
        to_launch=dict(cells=len(launch), calls=launch_calls,
                       usd_bound=round(launch_calls
                                       * EXPECTED_USD_PER_CALL, 2)),
        awaiting_rerun=sum(1 for r in records
                           if is_residual(r, PRIMARY)
                           and r.get('awaiting_rerun')),
        residual_strict=dict(cells=len(strict), calls=strict_calls,
                             usd_bound=round(strict_calls
                                             * EXPECTED_USD_PER_CALL, 2)),
        saved_by_rule=dict(
            cells=len(strict) - len(residual),
            usd_bound=round((strict_calls - calls)
                            * EXPECTED_USD_PER_CALL, 2)))


def missing_by_arm(records):
    """Cells worth launching, grouped by task/student/arm.

    Excludes anything a queued rerun will supply, so this table can be read
    straight into a launch command without paying twice.
    """
    out = {}
    for r in records:
        if needs_launch(r, PRIMARY):
            key = '{}/{}/{}'.format(r['task'], r['bonus'], r['arm'])
            entry = out.setdefault(key, dict(cells=0, calls=0, seeds=[]))
            entry['cells'] += 1
            entry['calls'] += r['promised_calls']
            entry['seeds'].append(r['seed'])
    for entry in out.values():
        entry['seeds'] = sorted(entry['seeds'])
        entry['usd_bound'] = round(entry['calls'] * EXPECTED_USD_PER_CALL, 2)
    return out


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--root', type=Path, default=Path(ROOT))
    cli.add_argument('--out', type=Path)
    args = cli.parse_args()

    records = inspect(args.root)
    if not records:
        print('No launched paid batch found under', args.root)
        return 1
    summary = summarise(records)
    per_arm = missing_by_arm(records)
    totals = summary['totals']

    print('== paid cells: {} (ran {}, not started {}, no journal {})'.format(
        totals['cells'], totals['ran'], totals['not_started'],
        totals['no_journal']))
    print('   admission tolerances {}, primary {}'.format(
        summary['tolerances'], summary['primary']))
    print()
    head = '{:44s} {:>5s} {:>4s} {:>5s}'.format('batch', 'cells', 'ran',
                                                'lost')
    for t in TOLERANCES:
        head += ' {:>8s}'.format('t=' + str(t))
    print(head)
    for name, b in sorted(summary['batches'].items()):
        line = '{:44s} {:5d} {:4d} {:5d}'.format(
            name, b['cells'], b['ran'], b['lost_calls'])
        for t in TOLERANCES:
            line += ' {:8d}'.format(b['admitted'][str(t)])
        print(line)

    print()
    r = summary['residual']
    rs = summary['residual_strict']
    saved = summary['saved_by_rule']
    print('== still to collect at the primary rule (t={}): '
          '{} cells, {} calls, bound ${}'.format(
              PRIMARY, r['cells'], r['calls'], r['usd_bound']))
    print('   under the old zero-tolerance rule: '
          '{} cells, {} calls, bound ${}'.format(
              rs['cells'], rs['calls'], rs['usd_bound']))
    print('   the declared rule avoids: {} cells, bound ${}'.format(
        saved['cells'], saved['usd_bound']))
    print('   of the missing, {} already have a rerun QUEUED: do not '
          'relaunch those'.format(summary['awaiting_rerun']))
    tl = summary['to_launch']
    print('== WORTH LAUNCHING: {} cells, {} calls, bound ${}'.format(
        tl['cells'], tl['calls'], tl['usd_bound']))

    if per_arm:
        print()
        print('== residual by task/student/arm')
        for key, entry in sorted(per_arm.items()):
            print('   {:52s} {:3d} cells ${:6.2f}  seeds {}'.format(
                key, entry['cells'], entry['usd_bound'], entry['seeds']))

    if totals['no_journal']:
        print()
        print('WARNING: paid runs without consultations.jsonl cannot be '
              'judged. Re-sync those batches WITHOUT --no-journals.')

    if args.out:
        args.out.write_text(json.dumps(
            dict(summary=summary, by_arm=per_arm, records=records), indent=1))
        print()
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
