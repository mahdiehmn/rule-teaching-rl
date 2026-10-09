"""Online equal-budget LLM action teacher, with a mocked client (no API)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from envs.registry import build_env
from teachers.minigrid import llm_scoped

PRICES = 'configs/prices_llm_scoped_2026-09-28.json'


class FakeResponses:
    def __init__(self, action=2):
        self.bodies, self.action = [], action

    def create(self, **body):
        self.bodies.append(body)
        answer = dict(action_now=self.action, abstain=True, condition={},
                      action=0, exceptions=[], rationale='mock')
        return SimpleNamespace(
            id='resp', model=body['model'], status='completed',
            service_tier='default', output_text=json.dumps(answer),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500))


@pytest.fixture
def fake(monkeypatch):
    responses = FakeResponses()
    monkeypatch.setattr(llm_scoped, 'build_openai_client',
                        lambda **_: SimpleNamespace(responses=responses))
    monkeypatch.setattr(llm_scoped, 'call_with_cold_start_retry',
                        lambda fn: fn())
    monkeypatch.setenv('LLM_MAX_OUTPUT_TOKENS', '8192')
    return responses


@pytest.mark.parametrize('task, module, key', [
    ('doorkey_8x8', 'scripts.conditional_rules_v3', 'v3'),
    ('multiroom_n6', 'scripts.conditional_rules_multiroom', 'pred'),
    ('keycorridor_s3r3', 'scripts.conditional_rules_keycorridor', 'pred')])
def test_live_request_is_the_offline_consultation_request(fake, task,
                                                         module, key):
    import importlib
    from envs.registry import TASKS as ENV_IDS
    mod = importlib.import_module(module)
    env = build_env(task, seed=3, obs_mode='symbolic')
    env.reset(seed=3)
    u = env.unwrapped
    teacher = llm_scoped.ScopedConsultTeacher(ENV_IDS[task])
    body = teacher.request(u)
    record = mod.state_record(u, 'live', 'x', 3, 'live') if key == 'pred' \
        else mod.state_record(u, 'live', 'x', 3, 'live')
    offline = mod.consult_prompt(record)
    assert body['input'][0]['content'] == offline
    assert body['model'] == 'gpt-5-mini-2025-08-07'
    assert body['reasoning'] == {'effort': 'low'}
    assert body['max_output_tokens'] == 8192
    assert body['text']['format']['name'] == teacher.version
    env.close()


def test_only_the_action_label_is_used(fake, monkeypatch):
    from envs.registry import TASKS as ENV_IDS
    import jsonschema
    monkeypatch.setattr(jsonschema.Draft202012Validator, 'validate',
                        lambda self, data: None)
    env = build_env('multiroom_n6', seed=3, obs_mode='symbolic')
    env.reset(seed=3)
    teacher = llm_scoped.ScopedConsultTeacher(ENV_IDS['multiroom_n6'])
    advice = teacher.recommend(env.unwrapped)
    assert advice.action == 2
    assert advice.cost.dollars == pytest.approx(1000 * .25e-6 + 500 * 2e-6)
    assert advice.cost.metadata['rule_unused'] is None
    env.close()


def test_budgeted_training_consults_exactly_the_cap(fake, monkeypatch,
                                                    tmp_path):
    import jsonschema
    from teachers.budget import CostLedger
    monkeypatch.setattr(jsonschema.Draft202012Validator, 'validate',
                        lambda self, data: None)
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 5.0, note='test pool')
    ppo.train(ppo.Args(
        task='doorkey_8x8', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='none',
        guidance=True, teacher='llm_scoped',
        teacher_model='gpt-5-mini-2025-08-07', advisor='unlimited',
        uniform_queries=True, query_budget=3, advice_budget=3,
        advisor_rng_isolation=True, advisor_no_teacher_peek=False,
        record_initial_policy=True, budget_ledger=str(ledger),
        price_table=PRICES, max_input_tokens=10000, max_output_tokens=8192,
        max_attempts=1, experiment_id='llm_scoped_eng'))
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    assert len(fake.bodies) == 3
    assert latest['advising']['num_delivered'] == 3
    assert latest['teacher_cost_dollars'] == pytest.approx(
        3 * (1000 * .25e-6 + 500 * 2e-6))


def test_entropy_scheduled_training_consults_exactly_the_cap(
        fake, monkeypatch, tmp_path):
    import jsonschema
    from teachers.budget import CostLedger
    monkeypatch.setattr(jsonschema.Draft202012Validator, 'validate',
                        lambda self, data: None)
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 5.0, note='test pool')
    ppo.train(ppo.Args(
        task='multiroom_n6', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='count',
        guidance=True, teacher='llm_scoped',
        teacher_model='gpt-5-mini-2025-08-07', advisor='importance',
        importance_source='entropy', importance_threshold=0.0,
        uniform_queries=False, query_budget=3, advice_budget=3,
        advisor_rng_isolation=True, advisor_no_teacher_peek=False,
        record_initial_policy=True, budget_ledger=str(ledger),
        price_table=PRICES, max_input_tokens=10000, max_output_tokens=4096,
        max_attempts=1, experiment_id='llm_scoped_entropy_eng'))
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    assert len(fake.bodies) == 3
    assert latest['advising']['num_delivered'] == 3


class FakeOnline:
    """Consultations return a 'wall ahead: turn right' rule; checks agree."""

    def __init__(self):
        self.consults = self.checks = 0

    def create(self, **body):
        if body['text']['format']['name'].startswith('blind'):
            self.checks += 1
            n = body['input'][0]['content'].count('SITUATION ')
            answer = dict(answers=[dict(situation=i, best_action=1)
                                   for i in range(n)])
        else:
            self.consults += 1
            answer = dict(action_now=1, abstain=False, action=1,
                          condition={'front': 'wall'}, exceptions=[],
                          rationale='mock')
        return SimpleNamespace(
            id='resp', model=body['model'], status='completed',
            service_tier='default', output_text=json.dumps(answer),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500))


def test_online_rules_consult_check_and_then_advise_for_free(monkeypatch,
                                                             tmp_path):
    import jsonschema
    from teachers.budget import CostLedger
    from teachers.minigrid import llm_rules_online
    fake = FakeOnline()
    monkeypatch.setattr(llm_scoped, 'build_openai_client',
                        lambda **_: SimpleNamespace(responses=fake))
    monkeypatch.setattr(llm_scoped, 'call_with_cold_start_retry',
                        lambda fn: fn())
    monkeypatch.setattr(jsonschema.Draft202012Validator, 'validate',
                        lambda self, data: None)
    monkeypatch.setattr(llm_rules_online, 'RECORD_EVERY', 4)
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 5.0, note='test pool')
    ppo.train(ppo.Args(
        task='doorkey_8x8', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='none',
        guidance=True, teacher='llm_rules_online',
        teacher_model='gpt-5-mini-2025-08-07', advisor='unlimited',
        query_budget=0, advice_budget=0, online_rule_calls=6,
        online_rule_blind=True, journal_paid_only=True,
        advisor_rng_isolation=True, advisor_no_teacher_peek=False,
        record_initial_policy=True, budget_ledger=str(ledger),
        price_table=PRICES, max_input_tokens=20000, max_output_tokens=2048,
        max_attempts=1, experiment_id='llm_rules_online_eng'))
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    online = summary['latest']['online_rules']
    assert fake.consults == online['consultations'] == 3
    assert online['calls_spent'] == fake.consults + fake.checks <= 6
    assert online['rules_kept'] >= 1
    rows = (run / 'consultations.jsonl').read_text().splitlines()
    assert len(rows) == online['consultations']     # a row per paid event
    advising = summary['latest']['advising']
    assert advising['num_delivered'] > online['consultations']
    assert summary['latest']['teacher_cost_dollars'] == pytest.approx(
        online['calls_spent'] * (1000 * .25e-6 + 500 * 2e-6))


@pytest.mark.parametrize('schedule', ['mistake', 'regular'])
def test_paid_teacher_under_the_other_schedules(fake, monkeypatch, tmp_path,
                                                schedule):
    """Mistake-based and regularly spaced advice reach the paid teacher."""
    import jsonschema
    from scripts.run_rule_bank_study_20260928 import SCHEDULES
    from teachers.budget import CostLedger
    monkeypatch.setattr(jsonschema.Draft202012Validator, 'validate',
                        lambda self, data: None)
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 5.0, note='test pool')
    settings = dict(
        task='doorkey_8x8', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='none',
        guidance=True, teacher='llm_scoped',
        teacher_model='gpt-5-mini-2025-08-07', advisor='unlimited',
        uniform_queries=True, query_budget=3, advice_budget=3,
        advisor_rng_isolation=True, advisor_no_teacher_peek=False,
        record_initial_policy=True, budget_ledger=str(ledger),
        price_table=PRICES, max_input_tokens=10000, max_output_tokens=2048,
        max_attempts=1, experiment_id=f'llm_scoped_{schedule}_eng')
    settings.update(SCHEDULES[schedule])
    ppo.train(ppo.Args(**settings))
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    assert 1 <= len(fake.bodies) <= 3
    assert latest['advising']['num_delivered'] <= len(fake.bodies)
