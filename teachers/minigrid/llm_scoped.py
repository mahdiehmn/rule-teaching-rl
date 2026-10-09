"""Online LLM action advice through the scoped-rule consultation prompt.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The frozen rule banks spend a fixed number of
LLM calls before training. This teacher is their equal-budget action-only
control: the same number of calls, spent DURING training on states the
student actually visits, each labelling only that state.

Each consultation renders the student's current state exactly as the
offline consultations did (full map, object facts, pose and the student's
predicates, from the task's own module) and sends the identical request:
same model snapshot, low reasoning effort, output ceiling and JSON schema
(`scripts/conditional_rules_v3.body`). Only `action_now` becomes the
label. The rule in the reply is recorded in the cost metadata and never
used. The teacher sees the full state; the student sees its 7x7 view, as
in every rule-bank study.

A failed, incomplete or malformed reply abstains (no label) and its
billed tokens are still counted. A prompt the reservation does not cover
is refused before sending.
"""

import importlib
import json
import os
import re
import time

from teachers.base import (Advice, BaseTeacher, Cost, build_openai_client,
                           call_with_cold_start_retry)
from teachers.minigrid.llm_general import (estimated_tokens,
                                           input_token_ceiling,
                                           render_ascii_map)
from teachers.reliability import failure_details

# env id -> (task module, predicate key its prompt reads, observer,
#            the offline consultation's schema name)
TASKS = {
    'MiniGrid-DoorKey-8x8-v0': ('scripts.conditional_rules_v3', 'v3',
                                'observe_v3', 'scoped_rule_v3'),
    'MiniGrid-MultiRoom-N6-v0': ('scripts.conditional_rules_multiroom',
                                 'pred', 'observe_mr', 'scoped_rule_mr_v1'),
    'BabyAI-KeyCorridorS3R3-v0': ('scripts.conditional_rules_keycorridor',
                                  'pred', 'observe_kc', 'scoped_rule_kc_v1'),
}


def price_model(model):
    """Price a dated snapshot at its family's rate (gpt-5-mini-2025-...)."""
    return re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model)


class ScopedConsultTeacher(BaseTeacher):
    """Equal-budget control: the consultation prompt, action label only."""

    def __init__(self, env_id, model='gpt-5-mini-2025-08-07', seed=0,
                 strict=False, timeout=300.0):
        super().__init__(teacher_id=f'llm_scoped:{model}', seed=seed)
        if env_id not in TASKS:
            raise ValueError(f'llm_scoped has no consultation prompt for '
                             f'{env_id!r}')
        name, self.key, observer, self.version = TASKS[env_id]
        self.module = importlib.import_module(name)
        self.observe = getattr(self.module, observer)
        self.env_id, self.model, self.strict = env_id, model, strict
        self.client = build_openai_client(timeout=timeout, max_retries=0)
        self.num_api_calls = 0
        self.num_failures = 0

    def request(self, u):
        """The offline consultation request for the live state `u`."""
        from scripts import conditional_rules_v3 as v3
        state = {'full_map': render_ascii_map(u),
                 'facts': self.module.teacher_facts(u),
                 'pose': [int(u.agent_pos[0]), int(u.agent_pos[1]),
                          int(u.agent_dir)],
                 self.key: self.observe(u.gen_obs()['image'])}
        schema = (v3.consult_schema() if self.key == 'v3'
                  else self.module._rule_schema(True))
        body = v3.body(self.module.consult_prompt(state), schema,
                       self.version)
        body['model'] = self.model
        ceiling = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '0') or 0)
        if ceiling:
            body['max_output_tokens'] = ceiling
        return body

    def recommend(self, state, context=None):
        data, cost = self.call(self.request(state))
        if data is None:
            return Advice(action=None, teacher_id=self.teacher_id,
                          call_id=self._next_call_id(), cost=cost)
        action = int(data['action_now'])
        cost.metadata.update(
            teacher_action=action, rule_unused=None if data['abstain'] else
            dict(condition=data['condition'], action=data['action'],
                 exceptions=data['exceptions']))
        return Advice(action=action, confidence=1.0,
                      explanation=str(data.get('rationale', '')),
                      teacher_id=self.teacher_id,
                      call_id=self._next_call_id(), cost=cost)

    def call(self, body):
        """Send one frozen request: (validated JSON or None, its cost)."""
        from teachers.minigrid.llm_prompts import estimate_dollars
        t0 = time.perf_counter()
        cost = Cost(model=self.model)
        ceiling = input_token_ceiling()
        text = body['input'][0]['content']
        estimate = (estimated_tokens(text)
                    + estimated_tokens(json.dumps(body['text'])) + 1024)
        if ceiling and estimate > ceiling:
            self.num_failures += 1
            return None, self._failed(t0, cost, dict(
                outcome='input_bound', estimated_tokens=estimate,
                attempt_count=0, usage_known=True, cost_known=True))
        attempts, errors, response = 0, [], None

        def send():
            nonlocal attempts
            attempts += 1
            try:
                return self.client.responses.create(**body)
            except Exception as exc:
                details = failure_details(exc)
                details.pop('message', None)
                errors.append(details)
                raise
        try:
            self.num_api_calls += 1
            response = call_with_cold_start_retry(send)
            usage = getattr(response, 'usage', None)
            cost.tokens_in = int(getattr(usage, 'input_tokens', 0) or 0)
            cost.tokens_out = int(getattr(usage, 'output_tokens', 0) or 0)
            tier = getattr(response, 'service_tier', None)
            cost.metadata = dict(
                outcome='valid', response_id=getattr(response, 'id', None),
                response_model=getattr(response, 'model', None),
                response_status=getattr(response, 'status', None),
                service_tier=tier, usage_known=usage is not None,
                cost_known=usage is not None and tier in (None, 'default'))
            cost.dollars = estimate_dollars(price_model(self.model),
                                            cost.tokens_in, cost.tokens_out)
            if tier not in (None, 'default'):
                cost.metadata['budget_violation'] = 'service_tier'
                raise ValueError('Response did not use Standard service')
            if getattr(response, 'status', None) not in (None, 'completed'):
                cost.metadata['outcome'] = 'incomplete_response'
                raise ValueError('Response was not completed')
            data = json.loads(response.output_text)
            from jsonschema import Draft202012Validator
            Draft202012Validator(body['text']['format']['schema']).validate(
                data)
        except Exception as exc:
            self.num_failures += 1
            meta = dict(attempt_count=attempts, attempt_errors=errors,
                        error_type=type(exc).__name__)
            if response is None:
                meta.update(outcome='request_failure', usage_known=False,
                            cost_known=False)
            elif cost.metadata.get('outcome') == 'valid':
                meta['outcome'] = 'schema_failure'
            self._failed(t0, cost, meta)
            if self.strict:
                raise RuntimeError(f'llm_scoped call failed: '
                                   f'{type(exc).__name__}') from exc
            return None, cost
        cost.wall_time_s = time.perf_counter() - t0
        cost.metadata.update(attempt_count=attempts, attempt_errors=errors)
        return data, cost

    @staticmethod
    def _failed(t0, cost, metadata):
        cost.wall_time_s = time.perf_counter() - t0
        cost.metadata.update(metadata, failed=True)
        return cost
