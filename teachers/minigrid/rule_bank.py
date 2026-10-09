"""A frozen LLM rule bank acting as a free, student-side teacher.

The bank was written by the LLM before training, by a fixed number of
paid consultations (and scope-check calls). During training nothing is
called. At each visited state the teacher reads ONLY the student's own
7x7 view (the symbolic image the policy sees) and returns one of:

  replay   the LLM's own `action_now` from a consultation, reused only
           when the current view is IDENTICAL to that consultation's
           view (action-only control: no explanation, no reuse).
  direct   the LLM's rules applied by their conditions alone
           (exceptions ignored); conflicting rules abstain.
  scoped   the full v3 scope semantics: conditions observed, exceptions
           observed false, unknown attributes never assumed; conflicts
           abstain.

Abstention leaves the state unlabelled. No simulator planner, oracle or
repair of the LLM's actions is involved.

The bank's `observer` names the predicates its rules are written in:
`doorkey_v3` (the default, for banks without the field), `multiroom_v1`
(scripts/conditional_rules_multiroom.py) or `keycorridor_v1`
(scripts/conditional_rules_keycorridor.py). All read the same 7x7 view.
`keycorridor_mem_v1` (scripts/conditional_rules_keycorridor_mem.py) adds
the progress fact `door_unlocked` (the locked door has been unlocked this
episode), read from the episode (no locked door left).
"""

import hashlib
import importlib
import json
from pathlib import Path
import time

from teachers.base import Advice, BaseTeacher, Cost

MODES = ('replay', 'direct', 'scoped')
OBSERVERS = {'doorkey_v3': ('scripts.conditional_rules_v3', 'observe_v3'),
             'multiroom_v1': ('scripts.conditional_rules_multiroom',
                              'observe_mr'),
             'keycorridor_v1': ('scripts.conditional_rules_keycorridor',
                                'observe_kc'),
             'keycorridor_mem_v1': ('scripts.conditional_rules_keycorridor_mem',
                                    'observe_kc_mem')}
# Observers that also read the agent's memory of its own actions from the
# episode, because the view alone cannot show it: observe(image, env).
STATEFUL = {'keycorridor_mem_v1'}
# Fix-wave addendum 6 (ablation): each vocabulary read from the full map
# instead of the student's view (teachers/minigrid/teacher_view.py); these
# read the env too. A bank selects one through its `observer` field.
from teachers.minigrid import teacher_view as _teacher_view  # noqa: E402
OBSERVERS.update(_teacher_view.OBSERVERS)
STATEFUL |= set(_teacher_view.OBSERVERS)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(path):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    if bank.get('mode') not in MODES:
        raise ValueError(f'Rule bank mode must be one of {MODES}')
    rules = [(dict(r['condition']), int(r['action']),
              tuple(tuple(e) for e in r['exceptions']))
             for r in bank.get('rules', [])]
    replay = {k: int(v) for k, v in bank.get('replay', {}).items()}
    return bank['mode'], rules, replay


class RuleBankTeacher(BaseTeacher):
    """Free teacher: the frozen LLM bank, read from the student's view."""

    def __init__(self, bank_path, seed=0):
        super().__init__(teacher_id=f'rule_bank:{file_sha256(bank_path)[:12]}',
                         seed=seed)
        self.mode, self.rules, self.replay = load_bank(bank_path)
        observer = json.loads(Path(bank_path).read_text(
            encoding='utf-8')).get('observer', 'doorkey_v3')
        if observer not in OBSERVERS:
            raise ValueError(f'Rule bank observer must be one of '
                             f'{sorted(OBSERVERS)}')
        module, name = OBSERVERS[observer]
        self.observe = getattr(importlib.import_module(module), name)
        self.stateful = observer in STATEFUL
        self.stats = {'advised': 0, 'no_rule': 0, 'conflict': 0}
        # Optional read-only diagnostics reuse this exact observation.
        self.capture_details = False
        self.last_details = None

    def _advise(self, image, env=None):
        from scripts import conditional_rules_v3 as v3
        if self.mode == 'replay':
            key = hashlib.sha256(image.tobytes()).hexdigest()
            action = self.replay.get(key)
            return (action, 'advised') if action is not None else \
                (None, 'no_rule')
        pred = self.observe(image, env) if self.stateful else \
            self.observe(image)
        rules = (self.rules if self.mode == 'scoped' else
                 [(c, a, ()) for c, a, _ in self.rules])
        action, status = v3.advise(rules, pred)
        if self.capture_details:
            self.last_details = {
                'predicates': dict(pred),
                'action': action, 'status': status,
                'rule_ids': [i for i, rule in enumerate(rules)
                             if v3.executable(rule, pred)],
            }
        return action, status

    def recommend(self, state, context=None):
        t0 = time.perf_counter()
        image = state.gen_obs()['image']
        action, status = self._advise(image, state)
        self.stats[status] += 1
        return Advice(
            action=None if action is None else int(action),
            confidence=1.0, explanation=status,
            teacher_id=self.teacher_id, call_id=self._next_call_id(),
            cost=Cost(wall_time_s=time.perf_counter() - t0,
                      compute_units=1,
                      metadata={} if action is not None else
                      {'abstain': status}))
