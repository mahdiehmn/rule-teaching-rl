"""Release the budget holds of ENDED paid rule-bank runs (dry run first).

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The paid controls reserve their worst case in
the shared ledger when they start. By design the trainer keeps that hold
open after the run ends ("retaining ... reservation for provider
reconciliation"): a timed-out attempt may have been billed without
reporting usage. A cancelled or crashed run never reaches that point at
all. Holds therefore pile up, and new paid batches are refused although
little money was spent.

Every paid call in these studies is a single attempt (no retries). A call
rejected with HTTP 429 (no credit, or a rate limit) is not billed. So:

  completed run   settled at its measured dollars, plus the per-call worst
                  case for every other failed call whose usage is unknown
  ended run       (its Slurm job is gone but it never completed: cancelled
                  or crashed) settled at the dollars its journal recorded,
                  the same unknown-usage bound, and one more worst-case
                  call for a request that may have been in flight

Runs still queued or running keep their holds. Every settlement is
printed; nothing changes without --apply.

  cd <a worktree of this commit>
  python -m scripts.settle_paid_runs_20260928 --repo <repo> [--apply]
"""

import argparse
import glob
import json
from pathlib import Path

from scripts.run_rule_bank_study_20260928 import job_active
from teachers.budget import CostLedger, PriceTable


def journal(run):
    """Known dollars and failed calls whose billing is unknown (not 429)."""
    path = Path(run) / 'consultations.jsonl'
    dollars, unknown = 0.0, 0
    if not path.exists():
        return dollars, unknown
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            dollars += row.get('dollars') or 0.0
            meta = row.get('metadata', {})
            if not meta.get('failed') or meta.get('usage_known', True):
                continue
            statuses = {e.get('http_status')
                        for e in meta.get('attempt_errors') or []}
            if statuses != {429}:
                unknown += 1
    return dollars, unknown


def unknown_cost_failures(run):
    return journal(run)[1]


def per_call_bound(code, args):
    prices = PriceTable.load(str(Path(code) / args['price_table']))
    return (prices.call_bound(args['teacher_model'], args['max_input_tokens'],
                              args['max_output_tokens'])
            * max(1, int(args.get('max_attempts', 1))))


def dispatched_jobs(repo):
    """(experiment_id, seed) -> Slurm job id of the cell that ran it."""
    jobs = {}
    for path in glob.glob(f'{repo}/results/efficiency/rule_bank_*/cells/*/'
                          'dispatch.json'):
        record = json.loads(Path(path).read_text())
        args = record['cell']['args']
        jobs[(args['experiment_id'], args['seed'])] = record['slurm_job_id']
    return jobs


def settlements(repo, ledger, active=job_active):
    runs = {Path(p).parent.name: Path(p).parent for p in glob.glob(
        f'{repo}/results/efficiency/rule_bank_*/code/results/runs/*/'
        'run_summary.json')}
    jobs = None
    out = []
    for entry in ledger.status()['open_reservations']:
        run = runs.get(entry['run_id'])
        if run is None:
            continue
        summary = json.loads((run / 'run_summary.json').read_text())
        args = summary['args']
        known, unknown = journal(run)
        bound = per_call_bound(run.parents[2], args)
        if summary.get('status') == 'completed':
            budget = summary['latest'].get('budget', {})
            measured = float(budget.get('actual_teacher_usd', known))
            state, in_flight = 'completed', 0
        else:
            jobs = dispatched_jobs(repo) if jobs is None else jobs
            job = jobs.get((args['experiment_id'], args['seed']))
            if job is None or active(job):
                continue                      # still queued or running
            measured, state, in_flight = known, 'ended', 1
        out.append(dict(run_id=entry['run_id'], reserved=entry['reserved'],
                        state=state, measured=measured, unknown_calls=unknown,
                        settle=min(entry['reserved'], measured
                                   + (unknown + in_flight) * bound)))
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--repo', type=Path, required=True)
    cli.add_argument('--ledger', type=Path)
    cli.add_argument('--apply', action='store_true')
    args = cli.parse_args()
    ledger = CostLedger(str(args.ledger or
                            args.repo / 'results/budget_ledger.json'))
    rows = settlements(args.repo, ledger)
    for r in rows:
        print(f"{r['state']:9s} {r['run_id'][:84]}  held ${r['reserved']:.2f}"
              f" -> ${r['settle']:.4f} (unknown-cost calls "
              f"{r['unknown_calls']})")
    released = sum(r['reserved'] - r['settle'] for r in rows)
    print(f'{len(rows)} ended paid runs; releases ${released:.2f}; '
          f"available now ${ledger.status()['available_usd']:.2f}")
    if not args.apply:
        print('Dry run: nothing changed. Re-run with --apply to settle.')
        return 0
    for r in rows:
        ledger.settle(r['run_id'], r['settle'])
    print(f"Settled. Available now ${ledger.status()['available_usd']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
