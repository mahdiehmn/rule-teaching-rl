"""
Create, inspect and reconcile the shared paid-model allowance.

The allowance is one pool across every paid run. This is the only place
it is created or repaired by hand; runs themselves only reserve against
it and settle what they spent.

    # once, booking what the smoke already cost
    python -m scripts.budget_status --init 100 --prior-spend 0.65 \
        --note 'AAMAS explanation pilot; smoke 2026-09-09'

    python -m scripts.budget_status                    # where it stands
    python -m scripts.budget_status --reconcile RUN_ID # after a crash

A crashed run leaves its reservation open on purpose: a request that
timed out may still have been billed, so returning the money
automatically would understate what has been spent. Reconciling is a
deliberate act, and it asks for the amount actually spent so the pool
records the truth rather than assuming zero.
"""

import argparse
import json

from teachers.budget import CostLedger

DEFAULT_PATH = 'results/budget_ledger.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', default=DEFAULT_PATH)
    parser.add_argument('--init', type=float, default=None,
                        metavar='ALLOWANCE_USD',
                        help='create the pool with this allowance')
    parser.add_argument('--prior-spend', type=float, default=0.0,
                        help='money already spent before the ledger '
                             'existed; booked as settled, not forgiven')
    parser.add_argument('--increase-allowance', type=float,
                        help='raise the existing project ceiling, preserving '
                             'all holds; does not buy provider credit')
    parser.add_argument('--note', default='')
    parser.add_argument('--reconcile', default='',
                        metavar='RUN_ID',
                        help='close a reservation left open by a run '
                             'that did not finish')
    parser.add_argument('--actual', type=float, default=None,
                        help='dollars that run really spent; required '
                             'with --reconcile')
    args = parser.parse_args()

    ledger = CostLedger(args.ledger)

    if args.increase_allowance is not None:
        if args.init is not None or args.reconcile:
            parser.error('Increase separately from initialization/settlement')
        if not 0 < args.increase_allowance <= 203.13:
            parser.error('The authorized project ceiling is $203.13 '
                         '(user decision 2026-09-17)')
        ledger.increase_allowance(args.increase_allowance, args.note)

    if args.init is not None:
        CostLedger.initialize(args.ledger, args.init, note=args.note,
                              prior_spend=args.prior_spend)
        print(f'created {args.ledger} with ${args.init:.2f}')
        if args.prior_spend:
            print(f'booked ${args.prior_spend:.2f} of prior spend')

    if args.reconcile:
        if args.actual is None:
            raise SystemExit(
                '--reconcile needs --actual: state what the run spent. '
                'If it truly made no billable request, pass --actual 0, '
                'but check the provider dashboard first -- a request '
                'that failed after generation was still billed.'
            )
        ledger.settle(args.reconcile, args.actual)
        print(f'settled {args.reconcile} at ${args.actual:.4f}')

    status = ledger.status()
    print(json.dumps(status, indent=2))
    if status['available_usd'] < 0:
        print('\n!! the pool is oversubscribed; stop launching paid runs')
    for entry in status['open_reservations']:
        print(f"  held: {entry['run_id']}  ${entry['reserved']:.2f}  "
              f"since {entry['at']}")


if __name__ == '__main__':
    main()
