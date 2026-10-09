"""LLM rules written DURING training, from the states the student visits.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The frozen rule banks were written before
training from sampled situations. This teacher writes its bank while the
student learns, spending the same kind of paid calls:

- A fixed budget of calls. Consultation times are drawn uniformly over the
  first 70% of the run (the distillation weight stays positive to 75%).
  With blind checking, half the budget consults and half checks.
- At a consultation, the LLM gets the offline consultation prompt for the
  state the student is in (ScopedConsultTeacher.request). Its action_now
  labels that state. Its rule, with blind checking, is shown to nobody: the
  same LLM answers up to 5 situations from the student's OWN earlier
  experience where the rule would fire (the offline blind prompt), and the
  rule joins the bank only if every answer equals its action. A rule with
  no such situation yet is dropped, as offline. Without blind checking
  every rule joins.
- Between consultations the student gets free advice from the growing bank
  under the same scope semantics (conditions observed, exceptions observed
  false, unknown never assumed, conflicts abstain).

The per-environment teachers share one OnlineRuleState: the budget, the
schedule, the bank, a reservoir of visited-state records (one every
RECORD_EVERY teacher calls) and a log of every consultation and check,
saved in the run summary. The teacher sees the full state; the student
its 7x7 view.
"""

import numpy as np

from scripts import conditional_rules_v3 as v3
from teachers.base import Advice, Cost
from teachers.minigrid.llm_general import render_ascii_map
from teachers.minigrid.llm_scoped import ScopedConsultTeacher

ACTIVE = .7            # consultations inside the first 70% of the run
RECORD_EVERY = 64      # teacher calls between reservoir records
CAPACITY = 1000        # reservoir of visited states for blind checks
K_SITUATIONS = v3.K_SITUATIONS
MAX_FAILED_IN_A_ROW = 3
BLIND_VERSION = {'scoped_rule_v3': 'blind_check_v3',
                 'scoped_rule_mr_v1': 'blind_check_mr_v1',
                 'scoped_rule_kc_v1': 'blind_check_kc_v1'}


def rule_json(rule):
    condition, action, exceptions = rule
    return dict(condition=condition, action=action,
                exceptions=[list(e) for e in exceptions])


class OnlineRuleState:
    """What every environment's teacher shares within one run."""

    def __init__(self, calls, total_timesteps, seed, blind=True):
        self.calls, self.blind = int(calls), bool(blind)
        self.rng = np.random.default_rng([int(seed), 20260928])
        consults = self.calls // 2 if self.blind else self.calls
        horizon = max(consults, int(ACTIVE * total_timesteps))
        self.slots = sorted(int(t) for t in self.rng.choice(
            horizon, consults, replace=False))
        self.next = self.spent = self.steps = self.recorded = 0
        self.failed_in_a_row = 0
        self.rules, self.log, self.reservoir = [], [], []

    def paid(self, ok):
        """Stop the run after repeated failed paid calls (e.g. no credit):
        training on without labels would silently break the comparison."""
        self.failed_in_a_row = 0 if ok else self.failed_in_a_row + 1
        if self.failed_in_a_row >= MAX_FAILED_IN_A_ROW:
            raise RuntimeError(f'{self.failed_in_a_row} paid LLM calls failed '
                               'in a row; stopping instead of training '
                               'without the teacher')

    def due(self):
        return (self.next < len(self.slots) and self.spent < self.calls
                and self.steps >= self.slots[self.next])

    def remember(self, record):
        """Uniform reservoir over every recorded visited state."""
        self.recorded += 1
        if len(self.reservoir) < CAPACITY:
            self.reservoir.append(record)
        else:
            j = int(self.rng.integers(self.recorded))
            if j < CAPACITY:
                self.reservoir[j] = record

    def summary(self):
        return dict(calls_budget=self.calls, calls_spent=self.spent,
                    blind=self.blind, consultations=len(self.log),
                    rules_kept=len(self.rules),
                    rules=[rule_json(r) for r in self.rules], log=self.log)


class OnlineRuleTeacher(ScopedConsultTeacher):
    """Free advice from a bank the LLM writes and checks during training."""

    def __init__(self, env_id, state, model='gpt-5-mini-2025-08-07', seed=0,
                 timeout=300.0):
        super().__init__(env_id, model=model, seed=seed, strict=False,
                         timeout=timeout)
        self.teacher_id = f'llm_rules_online:{model}'
        self.state = state
        self.blind_version = BLIND_VERSION[self.version]

    def _record(self, u, pred):
        return dict(full_map=render_ascii_map(u),
                    facts=self.module.teacher_facts(u),
                    pose=[int(u.agent_pos[0]), int(u.agent_pos[1]),
                          int(u.agent_dir)],
                    pred=pred, native=self.module.native_key(u))

    def recommend(self, state, context=None):
        st = self.state
        st.steps += 1
        pred = self.observe(state.gen_obs()['image'])
        if st.steps % RECORD_EVERY == 0:
            st.remember(self._record(state, pred))
        if st.due():
            st.next += 1
            return self._consult(state)
        action, status = v3.advise(st.rules, pred)
        return Advice(action=action, confidence=1.0, explanation=status,
                      teacher_id=self.teacher_id,
                      call_id=self._next_call_id(),
                      cost=Cost(compute_units=1, metadata={}
                                if action is not None
                                else {'abstain': status}))

    def _blind_body(self, situations):
        body = v3.body(self.module.blind_prompt(situations),
                       v3.blind_schema(), self.blind_version)
        ceiling = self.request_ceiling()
        body['model'] = self.model
        if ceiling:
            body['max_output_tokens'] = ceiling
        return body

    @staticmethod
    def request_ceiling():
        import os
        return int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '0') or 0)

    def _consult(self, u):
        st = self.state
        data, cost = self.call(self.request(u))
        st.spent += 1
        st.paid(data is not None)
        entry = dict(step=st.steps)
        if data is None:
            entry['outcome'] = 'failed'
            st.log.append(entry)
            return Advice(action=None, teacher_id=self.teacher_id,
                          call_id=self._next_call_id(), cost=cost)
        now, rule = self.module.parse(data)
        entry.update(outcome='consulted', action_now=now,
                     rule=None if rule is None else rule_json(rule))
        if rule is not None and st.blind:
            native = self.module.native_key(u)
            matches = [r for r in st.reservoir if r['native'] != native
                       and v3.executable(rule, r['pred'])]
            if not matches:
                entry['check'] = 'no_situation'        # dropped, as offline
            elif st.spent >= st.calls:
                entry['check'] = 'no_budget'
            else:
                pick = st.rng.choice(len(matches), min(K_SITUATIONS,
                                                       len(matches)),
                                     replace=False)
                situations = [matches[i] for i in sorted(pick)]
                answer, check = self.call(self._blind_body(situations))
                st.spent += 1
                st.paid(answer is not None)
                cost = cost + check
                if answer is None:
                    entry['check'] = 'failed'
                else:
                    said = {a['situation']: a['best_action']
                            for a in answer['answers']}
                    agree = [said.get(i) == rule[1]
                             for i in range(len(situations))]
                    entry.update(check='kept' if all(agree) else 'dropped',
                                 agreement=sum(agree),
                                 situations=len(situations))
                    if all(agree):
                        st.rules.append(rule)
        elif rule is not None:
            entry['check'] = 'unchecked_kept'
            st.rules.append(rule)
        st.log.append(entry)
        cost.metadata.update(teacher_action=now, online_rule=entry)
        return Advice(action=now, confidence=1.0, explanation='consulted',
                      teacher_id=self.teacher_id,
                      call_id=self._next_call_id(), cost=cost)
