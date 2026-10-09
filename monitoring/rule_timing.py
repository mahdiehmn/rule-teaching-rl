"""Read-only rule applicability and retained-label diagnostics.

Selection remains in EvidenceGate; this
observer must never provide labels, rewards, actions or gate verdicts.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np


class RuleTimingDiagnostics:
    """
    Summarize each rollout and save bounded examples from real students.
    """

    def __init__(self, run_dir, shape, teachers):
        self.path = Path(run_dir)
        self.shape = shape
        self.pending = {}
        self.examples = Counter()
        self.total_emitted = 0
        self.total_retained = 0
        for teacher in teachers:
            if teacher.mode == 'replay':
                raise ValueError('Timing diagnostics require rule conditions')
            teacher.capture_details = True

    def record(self, step, env, teacher, student_action, episode):
        """
        Retain the executor's pre-transition decision until filtering.
        """

        details = teacher.last_details
        if details is None:
            raise ValueError('Rule timing needs captured scoped decisions')
        self.pending[(step, env)] = {
            'phase': details['predicates'].get('door_unlocked', 'unknown'),
            # Advisors may discard the Advice object on abstention.
            # Read the executor snapshot to preserve its actual reason.
            'status': details['status'],
            'action': details['action'],
            'student_action': int(student_action),
            'episode': int(episode),
            'rule_ids': details['rule_ids'],
            'predicates': details['predicates'],
        }

    def finalize(self, mask, rollout, global_step, coefficient):
        """
        Audit the fixed mask reused by every PPO epoch of this rollout.
        """

        retained = np.asarray(mask) > 0
        if retained.shape != self.shape:
            raise ValueError('Rule timing mask has the wrong shape')
        emitted = np.zeros(self.shape, dtype=bool)
        phases = {}
        examples = []
        for (step, env), decision in self.pending.items():
            phase = decision['phase']
            row = phases.setdefault(phase, {
                'queried': 0, 'advised': 0, 'no_rule': 0,
                'conflict': 0, 'retained': 0,
                'emitted_actions': Counter(),
                'retained_actions': Counter(),
                'matching_rules': Counter(),
            })
            status, action = decision['status'], decision['action']
            row['queried'] += 1
            row[status] += 1
            row['matching_rules'].update(map(str, decision['rule_ids']))
            emitted[step, env] = action is not None
            if action is not None:
                row['emitted_actions'][str(action)] += 1
            if retained[step, env]:
                row['retained'] += 1
                row['retained_actions'][str(action)] += 1
            # Bounded exemplars cover both progress phases and abstention.
            # They describe sampled student paths, not their prevalence.
            signature = (phase, status, action)
            if self.examples[signature] < 3:
                self.examples[signature] += 1
                examples.append(dict(
                    rollout=rollout, step=step, env=env,
                    retained=bool(retained[step, env]), **decision))
        if np.any(retained & ~emitted):
            raise ValueError('Retained rule label lacks an emitted target')
        count = int(retained.sum())
        self.total_emitted += int(emitted.sum())
        self.total_retained += count
        self._append('rule_timing.jsonl', dict(
            rollout=int(rollout), global_step=int(global_step),
            coefficient=float(coefficient), emitted=int(emitted.sum()),
            retained=count, phases=phases,
            total_emitted=self.total_emitted,
            total_retained=self.total_retained))
        for row in examples:
            self._append('rule_timing_examples.jsonl', row)
        self.pending.clear()

    def optimization(self, rollout, labeled_minibatches, minibatches,
                     loss_sum):
        """
        Report exposure to the auxiliary loss across actual PPO updates.
        """

        self._append('rule_timing_updates.jsonl', dict(
            rollout=int(rollout), labeled_minibatches=labeled_minibatches,
            minibatches=minibatches,
            mean_auxiliary_loss=loss_sum / max(1, minibatches)))

    def _append(self, name, row):
        with (self.path / name).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
