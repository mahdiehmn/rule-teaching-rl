"""Study inventory: is every planned study run, and does every claim have data?

Read-only. Makes no API call, submits nothing and writes nothing unless
--out is given. For every batch the paper report reads (its BATCHES list),
it classifies each cell the way report_rule_bank_paper_20260928.load()
does, in the same order of checks:

  valid        the report can use it
  not_started  no run directory yet
  incomplete   a run exists but has not reached its endpoint
  invalid      settings or the teacher-off evaluation contract differ
  duplicate    more than one run directory matches the cell
  no_journal   paid run without consultations.jsonl: cannot be judged
  lost_calls   paid run lost more than the declared share of its calls

Reruns merge with their originals by (task, student, arm, seed), as in the
report, so a cell counts as covered if ANY batch holds a valid run for it.
Where squeue is available, a cell whose batch is still queued or running is
in flight, so "missing" separates cells to wait for from cells to act on.

Usage, from the repository root on the cluster:

    python -m scripts.study_inventory_20260929
    python -m scripts.study_inventory_20260929 --out inventory.json
"""
import argparse
from collections import Counter, defaultdict
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.paid_cell_accounting_20260928 import PRIMARY
from scripts.report_rule_bank_paper_20260928 import BATCHES
from scripts.run_plan_repair_20260927 import read
from scripts.run_rule_bank_pilot_20260928 import find_run, validate_run
from scripts.run_rule_bank_study_20260928 import (SPECS, paid_calls,
                                                  request_failures)

REPO = os.environ.get('VLM_RL_BENCH_CLUSTER_REPO',
                      '/project/<allocation>/<user>/vlm-rl-bench')
CELLS = (('doorkey_8x8', 'none'), ('doorkey_8x8', 'count'),
         ('multiroom_n6', 'none'), ('multiroom_n6', 'count'),
         ('keycorridor_s3r3', 'none'), ('keycorridor_s3r3', 'count'))
HEAD = ('DK/PPO', 'DK/cnt', 'MR/PPO', 'MR/cnt', 'KC/PPO', 'KC/cnt')

# What each arm is for, in the order the paper argues it.
GROUPS = (
    ('core: rules and the no-teacher baseline',
     ('none', 'llm_rules_scoped', 'llm_rules_blind_strict')),
    ('P3: equal-call LLM action advice [paid]', ('llm_action_online_eq',)),
    ('P4/P5: replayed action labels',
     ('llm_action_replay', 'llm_action_predicate_replay')),
    ('P6: random rule subset', ('llm_rules_random_subset',)),
    ('content control: shuffled rules', ('llm_rules_shuffled',)),
    ('budget: 480-call action advice [paid]', ('llm_action_online_480',)),
    ('budget: label dose and bank size',
     ('llm_rules_scoped_cap480', 'llm_rules_scoped_12',
      'llm_rules_blind_strict_12')),
    ('sweep: 15/30/120-call advice [paid]',
     ('llm_action_online_15', 'llm_action_online_30',
      'llm_action_online_120')),
    ('schedules [paid; cancelled, deviation D3]',
     ('llm_action_online_480_entropy', 'llm_action_online_480_mistake',
      'llm_action_online_480_regular')),
    ('variants: weaker rule forms',
     ('llm_rules_direct', 'llm_rules_scope_checked')),
    ('online rules [paid; exploratory, deviation D2]',
     ('llm_rules_online_blind', 'llm_rules_online_raw')),
    ('valid-action KeyCorridor rules, replicates 20-29',
     ('llm_rules_scoped_valid',)),
)
# study directory -> the spec name the launcher takes
SPEC_OF = {spec['study']: name for name, spec in SPECS.items()
           if 'study' in spec}
# Cancelled on 2026-09-28 (deviation D3): absent on purpose.
CANCELLED = frozenset(
    SPECS[f'schedule_online_{task}']['study']
    for task in ('doorkey', 'multiroom', 'keycorridor')
    if f'schedule_online_{task}' in SPECS)
WAITING = ('not_started', 'incomplete')


def classify(batch, cell, tolerance=PRIMARY):
    """One cell's state, checked in the order the report's load() uses."""
    args = cell['args']
    try:
        run = find_run(Path(batch) / 'code/results/runs', args)
    except FileNotFoundError as error:
        # find_run reports how many directories matched: 0 is not started,
        # 2 or more is an ambiguity the report also refuses.
        return 'not_started' if str(error).startswith('0 ') else 'duplicate'
    try:
        validate_run(run, args)
    except ValueError as error:
        return 'incomplete' if 'Incomplete' in str(error) else 'invalid'
    except (FileNotFoundError, KeyError):
        return 'incomplete'
    paid = str(args.get('teacher', '')).startswith('llm')
    if paid and not (Path(run) / 'consultations.jsonl').exists():
        return 'no_journal'
    if paid:
        lost, promised = request_failures(run), paid_calls(args)
        if lost and (not promised or lost / promised > tolerance):
            return 'lost_calls'
    return 'valid'


def active_jobs():
    """Base ids of jobs still queued or running, or None without squeue."""
    if not shutil.which('squeue'):
        return None
    user = os.environ.get('USER') or getpass.getuser()
    out = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i'],
                         capture_output=True, text=True, check=False).stdout
    # Array tasks print as 1212430_53 and pending ranges as 1219881_[5-9%5].
    return {token.split('_')[0] for token in out.split() if token}


def batch_status(row):
    n, valid = row['cells'], row['states']['valid']
    if row['batch'] in CANCELLED:
        return 'cancelled'
    if n and valid == n:
        return 'complete'
    if row['running']:
        return 'running'
    if row['job'] is None and not valid:
        return 'NOT SUBMITTED'
    if row['running'] is None and row['job']:
        return 'submitted (no squeue here)'
    return 'ENDED INCOMPLETE'


def inventory(efficiency, tolerance=PRIMARY, jobs=None):
    """Per-batch rows, and every cell's entries across batches."""
    batches, cells = [], defaultdict(list)
    for name in dict.fromkeys(BATCHES):
        batch = Path(efficiency) / name
        row = dict(batch=name, spec=SPEC_OF.get(name), job=None,
                   running=None, cells=0, states=Counter())
        if not (batch / 'manifest.json').exists():
            # A cancelled study must never come back as something to
            # launch: schedule_online is paid (~$86) and was cancelled on
            # purpose. It once did, when its batch was not on disk.
            row['status'] = ('cancelled' if name in CANCELLED
                             else 'NOT PREPARED')
            batches.append(row)
            continue
        marker = batch / 'SUBMITTED_JOB'
        if marker.exists():
            row['job'] = marker.read_text().strip()
            if jobs is not None:
                row['running'] = row['job'] in jobs
        for cell in read(batch / 'manifest.json')['cells']:
            args = cell['args']
            key = (args['task'], cell.get('bonus', args.get('bonus')),
                   cell['arm'], cell['seed'])
            state = classify(batch, cell, tolerance)
            row['states'][state] += 1
            row['cells'] += 1
            cells[key].append(dict(batch=name, state=state,
                                   running=bool(row['running'])))
        row['status'] = batch_status(row)
        batches.append(row)
    return batches, cells


def coverage(cells):
    """Per (task, student, arm): planned seeds and where each one stands."""
    out = defaultdict(lambda: dict(planned=0, valid=0, in_flight=0,
                                   stuck=0, cancelled=0, why=Counter()))
    for (task, bonus, arm, _seed), entries in cells.items():
        c = out[(task, bonus, arm)]
        c['planned'] += 1
        if any(e['state'] == 'valid' for e in entries):
            c['valid'] += 1
        elif any(e['running'] and e['state'] in WAITING for e in entries):
            c['in_flight'] += 1
        elif all(e['batch'] in CANCELLED for e in entries):
            c['cancelled'] += 1
        else:
            c['stuck'] += 1
            # the most informative reason: the last batch tried, which is
            # the rerun when there is one
            c['why'][entries[-1]['state']] += 1
    return out


def cell_text(c):
    """have/planned, then ! act, ~ wait, x cancelled on purpose."""
    if not c:
        return '-'
    text = f"{c['valid']}/{c['planned']}"
    if c['valid'] == c['planned']:
        return text
    if c['stuck']:
        return text + '!'
    if c['in_flight']:
        return text + '~'
    if c['cancelled']:
        return text + 'x'
    return text


def launch_line(row):
    if row['batch'] in CANCELLED:
        return None
    if not row['spec']:
        return '(no study spec: produced by another launcher)'
    line = f"bash scripts/launch_rule_bank_study.sh {REPO} {row['spec']}"
    if SPECS[row['spec']].get('paid'):
        line += '   # PAID: run the accounting first'
    return line


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--efficiency', type=Path,
                     default=Path('results/efficiency'))
    cli.add_argument('--out', type=Path)
    args = cli.parse_args()

    jobs = active_jobs()
    batches, cells = inventory(args.efficiency, PRIMARY, jobs)
    cov = coverage(cells)

    print(f'== coverage: valid/planned seeds, admission tolerance '
          f'{PRIMARY:.0%}')
    print('   !  some seeds ended with no usable run and nothing is queued:'
          ' ACT')
    print('   ~  the rest are queued or running: WAIT')
    print('   x  cancelled on purpose        -  not planned for this cell')
    if jobs is None:
        print('   (no squeue here: in-flight cells cannot be told apart'
              ' from stuck ones)')
    shown = set()
    for label, arms in GROUPS:
        print()
        print(label)
        print(f"   {'arm':32s}" + ''.join(f'{h:>10s}' for h in HEAD))
        for arm in arms:
            shown.add(arm)
            line = f'   {arm:32s}'
            for task, bonus in CELLS:
                line += f'{cell_text(cov.get((task, bonus, arm))):>10s}'
            print(line)
    other = sorted({arm for _t, _b, arm in cov} - shown)
    if other:
        print()
        print('other arms found on disk')
        for arm in other:
            line = f'   {arm:32s}'
            for task, bonus in CELLS:
                line += f'{cell_text(cov.get((task, bonus, arm))):>10s}'
            print(line)

    stuck = sorted((k, c) for k, c in cov.items() if c['stuck'])
    if stuck:
        print()
        print('== why seeds are stuck (state of the last batch tried)')
        for (task, bonus, arm), c in stuck:
            reasons = ', '.join(f'{s} {n}' for s, n in c['why'].most_common())
            print(f'   {task}/{bonus}/{arm}: {c["stuck"]} stuck ({reasons})')

    print()
    print('== batches that are not complete')
    done = 0
    for row in batches:
        if row['status'] in ('complete', 'cancelled'):
            done += 1
            continue
        states = ', '.join(f'{s} {n}' for s, n in row['states'].most_common())
        job = f"job {row['job']}" if row['job'] else 'no job'
        print(f"   {row['status']:26s} {row['batch']}  [{job}]  "
              f"{row['cells']} cells: {states or '-'}")
        line = launch_line(row)
        if line and row['status'] in ('NOT SUBMITTED', 'NOT PREPARED'):
            print(f'      {line}')
    print(f'   ({done} batches complete or cancelled on purpose)')

    if args.out:
        args.out.write_text(json.dumps(dict(
            batches=[dict(r, states=dict(r['states'])) for r in batches],
            coverage={'/'.join(k): dict(c, why=dict(c['why']))
                      for k, c in cov.items()}), indent=1))
        print()
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
