"""Write one evidence row per consultation, independently of delivery."""

import json
from pathlib import Path


class ConsultationJournal:
    """Preserve failed responses and stop on a predeclared failure rate."""

    def __init__(self, path, failure_limit=0.0, minimum=40, write_rows=True,
                 paid_only=False):
        if not 0 <= failure_limit <= 1 or minimum < 1:
            raise ValueError('Invalid consultation failure limit/minimum')
        self.path = Path(path)
        self.failure_limit = failure_limit
        self.minimum = minimum
        self.count = self.failures = 0
        self.unknown_cost = 0
        self.write_rows = write_rows
        # A teacher that advises for free at every step but pays for a few
        # calls writes rows only for the calls that used tokens or failed.
        self.paid_only = paid_only
        self.written = 0

    def record(self, cost, advice, delivered, **identity):
        """Write before returning a stop reason; unknown costs stay null."""
        metadata = dict(cost.metadata)
        response_known = metadata.get('cost_known', True)
        known = response_known and not metadata.get('attempt_errors')
        self.count += 1
        self.failures += bool(metadata.get('failed'))
        self.unknown_cost += not known
        row = {
            **identity, 'consultation': self.count,
            'teacher_action': metadata.get(
                'teacher_action', getattr(advice, 'action', None)),
            'delivered': bool(delivered),
            'tokens_in': cost.tokens_in, 'tokens_out': cost.tokens_out,
            'dollars': cost.dollars if known else None,
            'known_response_dollars': cost.dollars if response_known else None,
            'wall_time_s': cost.wall_time_s, 'metadata': metadata,
        }
        if self.write_rows and (not self.paid_only or cost.tokens_in
                                or cost.tokens_out or metadata.get('failed')):
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            self.written += 1
        if metadata.get('budget_violation'):
            return 'Unexpected API service tier; evidence saved'
        if (self.failure_limit and self.count >= self.minimum
                and self.failures / self.count > self.failure_limit):
            return (f'Consultation failures {self.failures}/{self.count} '
                    f'exceed {self.failure_limit:.0%}; evidence saved')
        return None

    def stats(self):
        return {'records': self.count, 'failures': self.failures,
                'records_written': self.written,
                'unknown_cost_records': self.unknown_cost,
                'failure_limit': self.failure_limit,
                'minimum_before_stop': self.minimum}
