"""KeyCorridor rule interface and oracle; rule-bank study cells (no API)."""

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from scripts import conditional_rules_keycorridor as kc
from teachers.minigrid.rule_bank import RuleBankTeacher, file_sha256

SEED = kc.PANEL_SEED0


def remaining(env):
    u = env.unwrapped
    doors, ball, _ = kc.world(u)
    return kc.cost_to_go(u, doors, ball, kc.abstract_state(u))


def test_oracle_routes_succeed_in_the_real_env_at_the_shortest_count():
    for seed in range(SEED, SEED + 12):
        env = kc.make_env(seed)
        shortest = remaining(env)
        env.close()
        assert len(kc.route(seed)) == shortest     # route() checks success


def test_optimal_actions_are_exactly_those_that_shorten_the_real_task():
    rng = np.random.default_rng(1)
    for seed in (SEED, SEED + 4):                   # 4 revisits a side door
        actions = kc.route(seed)
        for _ in range(6):
            env, _ = kc.make_state(seed, actions, rng)
            here, best = remaining(env), set(kc.optimal(env.unwrapped))
            for a in (0, 1, 2, 3, 4, 5):
                trial = copy.deepcopy(env)
                _, reward, done, _, _ = trial.step(a)
                after = (0 if reward > 0 else float('inf')) if done \
                    else remaining(trial)
                assert (a in best) == (after == here - 1)
            env.close()


def test_student_predicates_track_the_carried_key():
    env = kc.make_env(SEED)
    for a in kc.route(SEED):
        if a == kc.PICKUP:
            before = kc.observe_kc(env.unwrapped.gen_obs()['image'])
            env.step(a)
            break
        env.step(a)
    after = kc.observe_kc(env.unwrapped.gen_obs()['image'])
    assert before['front'] == 'key' and before['carrying'] == 'nothing'
    assert after['carrying'] == 'key' and after['front'] == 'empty'
    assert set(after) == set(kc.FIELDS)
    env.close()


def test_rule_bank_uses_the_keycorridor_observer(tmp_path):
    env = kc.make_env(SEED)
    bank = tmp_path / 'b.json'
    bank.write_text(json.dumps(dict(
        mode='scoped', observer='keycorridor_v1',
        rules=[dict(condition={'carrying': 'nothing'}, action=1,
                    exceptions=[])])))
    assert RuleBankTeacher(bank).recommend(env.unwrapped).action == 1
    env.close()


def test_real_keycorridor_training_with_a_rule_bank(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    bank = tmp_path / 'b.json'
    bank.write_text(json.dumps(dict(
        mode='scoped', observer='keycorridor_v1',
        rules=[dict(condition={'front': 'wall'}, action=1, exceptions=[])])))
    ppo.train(ppo.Args(
        task='keycorridor_s3r3', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='count',
        guidance=True, teacher='rule_bank', rule_bank=str(bank),
        rule_bank_sha256=file_sha256(bank), advisor='unlimited',
        query_budget=0, advice_budget=0, advisor_rng_isolation=True,
        record_initial_policy=True, experiment_id='rule_bank_kc_eng'))
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    assert latest['advising']['num_delivered'] > 0
    assert latest['teacher_cost_dollars'] == 0


@pytest.mark.parametrize('name', ['doorkey_count', 'keycorridor'])
def test_study_cells_change_only_the_teaching_channel(name):
    from scripts import run_rule_bank_study_20260928 as study
    spec = study.SPECS[name]
    if not all((Path(spec['banks']) / f'{b}.json').exists()
               for b in study.bank_names(spec)):
        pytest.skip('Banks not built yet')
    built = study.cells(spec)
    assert len(built) == len(study.arms_of(spec)) * 5 * len(spec['bonuses'])
    groups = {}
    for c in built:
        groups.setdefault((c['bonus'], c['seed']), {})[c['arm']] = c['args']
    keys = {'experiment_id', 'rule_bank', 'rule_bank_sha256'}
    for (bonus, _), arms in groups.items():
        guided = [a for n, a in arms.items() if n != 'none']
        for a in guided[1:]:
            assert {k for k in a if a[k] != guided[0][k]} <= keys
        assert not guided[0]['teacher_stream']
        assert not arms['none']['guidance']
        assert arms['none']['bonus'] == bonus == guided[0]['bonus']
        assert arms['none']['task'] == spec['task']


@pytest.mark.parametrize('name', ['doorkey_controls', 'multiroom_controls'])
def test_control_cells_pair_with_the_running_scoped_cells(name):
    from scripts import run_rule_bank_study_20260928 as study
    spec = study.SPECS[name]
    if not all((Path(spec['banks']) / f'{b}.json').exists()
               for b in study.bank_names(spec)):
        pytest.skip('Banks not built yet')
    built, reference = study.cells(spec), study.reference_cells(spec)
    assert len(built) == 10
    study.paired_cells(reference, built)
    changed = [dict(c, args=dict(c['args'], learning_rate=1e-3))
               for c in built[:1]]
    with pytest.raises(ValueError):
        study.paired_cells(reference, changed)


def test_manifest_spec_survives_the_json_round_trip():
    from scripts import run_rule_bank_study_20260928 as study
    for spec in study.SPECS.values():
        saved = json.loads(json.dumps(spec))
        assert saved == json.loads(json.dumps(study.spec_of(spec['study'])))


@pytest.mark.parametrize('name, calls', [('doorkey_online', 62),
                                         ('multiroom_online', 67),
                                         ('keycorridor_online', 61)])
def test_online_cells_match_the_bank_budget_and_pair(name, calls):
    from scripts import run_rule_bank_study_20260928 as study
    spec = study.SPECS[name]
    assert study.online_budget(spec) == calls
    built = study.cells(spec, ledger='/pool/ledger.json')
    assert len(built) == 10
    for c in built:
        a = c['args']
        assert a['teacher'] == 'llm_scoped' and a['uniform_queries']
        assert a['query_budget'] == a['advice_budget'] == calls
        assert a['budget_ledger'] == '/pool/ledger.json'
        assert a['teacher_model'] == 'gpt-5-mini-2025-08-07'
    study.paired_cells(study.reference_cells(spec), built, study.PAID_KEYS)
    with pytest.raises(ValueError):                 # a learner change
        study.paired_cells(study.reference_cells(spec), [dict(
            built[0], args=dict(built[0]['args'], learning_rate=1e-3))],
            study.PAID_KEYS)


def test_paid_admission_needs_the_pool_and_the_key(tmp_path):
    from scripts import run_rule_bank_study_20260928 as study
    from teachers.budget import BudgetError, CostLedger
    spec = study.SPECS['doorkey_online']
    key = tmp_path / '.env'
    key.write_text('OPENAI_API_KEY=sk-test\n')
    rich, poor = tmp_path / 'rich.json', tmp_path / 'poor.json'
    CostLedger.initialize(str(rich), 100.0)
    CostLedger.initialize(str(poor), 1.0)
    paid = study.paid_admission(spec, study.ROOT, rich, key)
    assert 10 < paid['worst_case_usd'] < 13        # 62 calls x 10 runs
    with pytest.raises(BudgetError):
        study.paid_admission(spec, study.ROOT, poor, key)
    with pytest.raises(ValueError):
        study.paid_admission(spec, study.ROOT, rich, tmp_path / 'none')


def second_wave():
    from scripts import run_rule_bank_study_20260928 as study
    return [n for n, s in study.SPECS.items() if s.get('seed_major')]


@pytest.mark.parametrize('name', second_wave())
def test_second_wave_cells_pair_and_run_seed_by_seed(name):
    from scripts import run_rule_bank_study_20260928 as study
    spec = study.SPECS[name]
    built = study.cells(spec, ledger='/pool/ledger.json')
    reps = list(study.replicates(spec))
    assert len(built) == len(study.arms_of(spec)) * len(reps) * len(
        spec['bonuses'])
    study.fresh_pairs(spec, built)
    seeds = [c['seed'] for c in built]
    assert seeds == sorted(seeds)                  # complete seeds first
    if name.startswith('confirm'):
        assert reps == list(range(5, 20))          # fresh replicates only
    for c in built:
        a = c['args']
        if c['arm'] in study.ONLINE_ARMS:
            calls, ceiling, schedule = study.ONLINE_ARMS[c['arm']]
            assert a['query_budget'] == (study.online_budget(spec)
                                         if calls == 'bank' else calls)
            assert a['max_output_tokens'] == spec.get('max_output_tokens', ceiling)
            assert a['advisor'] == study.SCHEDULES[schedule].get(
                'advisor', 'unlimited')
            assert a['budget_ledger'] == '/pool/ledger.json'
        elif c['arm'] in study.ONLINE_RULE_ARMS:
            assert a['online_rule_calls'] == study.online_budget(spec)
            assert a['journal_paid_only'] and a['query_budget'] == 0
            assert a['online_rule_blind'] == study.ONLINE_RULE_ARMS[c['arm']]
        elif spec.get('summary_only') and c['arm'] != 'none':
            assert a['offline_summary_only']     # 'none' has no teacher rows
        elif c['arm'] in study.CAPPED:
            assert a['uniform_queries'] and a['query_budget'] == 480


def test_second_wave_pairs_with_the_running_studies():
    """Replicates 0-4 of the budget arms reuse the running scoped cells."""
    from scripts import run_rule_bank_pilot_20260928 as pilot
    from scripts import run_rule_bank_multiroom_20260928 as multiroom
    from scripts import run_rule_bank_study_20260928 as study
    running = ([dict(c, bonus='none') for c in pilot.cells()]
               + study.cells(study.SPECS['doorkey_count'])
               + multiroom.cells() + study.cells(study.SPECS['keycorridor']))
    scoped = {(c['args']['task'], c['bonus'], c['seed']): c['args']
              for c in running if c['arm'] == 'llm_rules_scoped'}
    for task in ('doorkey', 'multiroom', 'keycorridor'):
        spec = study.SPECS[f'budget_{task}']
        base = {b: study.base_args(spec['task'], b) for b in spec['bonuses']}
        for bonus in spec['bonuses']:
            for r in range(5):
                fresh = asdict(study.arm_args(spec, 'llm_rules_scoped', r,
                                              bonus, *base[bonus],
                                              study.ROOT, ''))
                old = scoped[(spec['task'], bonus, fresh['seed'])]
                assert {k for k in old if old[k] != fresh[k]} == {
                    'experiment_id'}


def test_throttled_admission_reserves_only_the_peak(tmp_path):
    from scripts import run_rule_bank_study_20260928 as study
    from teachers.budget import BudgetError, CostLedger
    spec = study.SPECS['budget_online_doorkey']     # 10 runs, throttle 5
    key = tmp_path / '.env'
    key.write_text('OPENAI_API_KEY=sk-test\n')
    enough, short = tmp_path / 'enough.json', tmp_path / 'short.json'
    CostLedger.initialize(str(enough), 20.0)
    CostLedger.initialize(str(short), 10.0)
    paid = study.paid_admission(spec, study.ROOT, enough, key)
    assert paid['peak_reserved_usd'] == pytest.approx(
        paid['worst_case_usd'] / 2)
    assert 15 < paid['peak_reserved_usd'] < 17     # 5 x 480 x $0.0066
    assert paid['calls'] == 10 * 480
    with pytest.raises(BudgetError):
        study.paid_admission(spec, study.ROOT, short, key)


@pytest.mark.parametrize('task', ['doorkey_8x8', 'multiroom_n6',
                                  'keycorridor_s3r3'])
def test_confirmation_banks_come_from_the_frozen_replies(task):
    from scripts.build_confirm_banks_20260928 import FIRST, TASKS, subset
    from scripts.run_rule_bank_pilot_20260928 import rule_json
    module, out, banks, _ = TASKS[task]
    if not (out / 'blind_replies.jsonl').exists():
        pytest.skip('Local replies not present')
    _, parsed = module.consult_rules(out)
    raw = [rule for _, _, rule, _ in parsed if rule is not None]
    first = [rule_json(rule) for k, (_, _, rule, _) in enumerate(parsed)
             if k < FIRST and rule is not None]
    assert json.loads((banks / 'scoped_12.json').read_text())['rules'] \
        == first
    strict12 = json.loads((banks / 'blind_strict_12.json').read_text())
    assert all(rule in first for rule in strict12['rules'])
    size = len(json.loads((banks / 'blind_strict.json').read_text())['rules'])
    for r in (0, 7, 19):
        bank = json.loads((banks / f'random_subset_r{r}.json').read_text())
        assert bank['rules'] == [rule_json(raw[i])
                                 for i in subset(raw, size, r)]


def fake_paid_run(runs, args, rows, status='completed', budget=None):
    run = runs / f"x_{args['experiment_id']}__{args['seed']}__1"
    run.mkdir(parents=True)
    (run / 'run_summary.json').write_text(json.dumps(dict(
        args=args, status=status, latest=dict(budget=budget or {}))))
    (run / 'consultations.jsonl').write_text(
        ''.join(json.dumps(dict(metadata=m)) + '\n' for m in rows))
    return run


def test_reruns_are_exactly_the_runs_that_lost_calls(tmp_path):
    from scripts import run_rule_bank_study_20260928 as study
    spec = study.SPECS['doorkey_online']
    built = study.cells(spec, ledger='/pool/ledger.json')[:3]
    batch = study.batch_dir(tmp_path, spec)
    (batch / 'code/results/runs').mkdir(parents=True)
    (batch / 'manifest.json').write_text(json.dumps(dict(
        study=spec['study'], cells=built)))
    runs = batch / 'code/results/runs'
    lost = dict(failed=True, outcome='request_failure')
    fake_paid_run(runs, built[0]['args'], [lost, {'failed': False}])
    fake_paid_run(runs, built[1]['args'], [dict(failed=True,
                                                outcome='incomplete_response')])
    # built[2] never started
    rerun = study.rerun_cells(tmp_path)
    assert [c['origin'] for c in rerun] == [dict(study=spec['study'],
                                                 index=built[0]['index'])]
    assert rerun[0]['args'] == built[0]['args'] and rerun[0]['lost_calls'] == 1


def test_settling_ended_runs_bills_only_what_might_be_billed(tmp_path):
    import shutil
    from scripts import run_rule_bank_study_20260928 as study
    from scripts.settle_paid_runs_20260928 import per_call_bound, settlements
    from teachers.budget import CostLedger
    ledger_path = tmp_path / 'results/budget_ledger.json'
    ledger_path.parent.mkdir(parents=True)
    CostLedger.initialize(str(ledger_path), 20.0)
    ledger = CostLedger(str(ledger_path))
    batch = tmp_path / 'results/efficiency/rule_bank_x'
    runs = batch / 'code/results/runs'
    (batch / 'code/configs').mkdir(parents=True)
    shutil.copy(study.ROOT / study.ONLINE_PRICES,
                batch / 'code' / study.ONLINE_PRICES)
    base = dict(teacher='llm_scoped', teacher_model=study.ONLINE_MODEL,
                price_table=study.ONLINE_PRICES, max_input_tokens=10_000,
                max_output_tokens=2048, max_attempts=1)
    no_credit = dict(failed=True, usage_known=False,
                     attempt_errors=[dict(http_status=429)])
    timeout = dict(failed=True, usage_known=False,
                   attempt_errors=[dict(http_status=None)])
    done = fake_paid_run(runs, dict(base, experiment_id='a', seed=1),
                         [no_credit, timeout],
                         budget=dict(actual_teacher_usd=.10))
    cancelled = fake_paid_run(runs, dict(base, experiment_id='b', seed=2),
                              [no_credit], status='running')
    going = fake_paid_run(runs, dict(base, experiment_id='c', seed=3), [],
                          status='running')
    for i, (run, job) in enumerate(((cancelled, 11), (going, 12))):
        cell = batch / 'cells' / str(i)
        cell.mkdir(parents=True)
        args = json.loads((run / 'run_summary.json').read_text())['args']
        (cell / 'dispatch.json').write_text(json.dumps(dict(
            cell=dict(args=args), slurm_job_id=job)))
    for run in (done, cancelled, going):
        ledger.reserve(run.name, 3.0)
    bound = per_call_bound(batch / 'code', base)
    rows = {r['run_id']: r for r in settlements(
        tmp_path, ledger, active=lambda job: job == 12)}
    assert set(rows) == {done.name, cancelled.name}    # running keeps hold
    assert rows[done.name]['settle'] == pytest.approx(.10 + bound)
    assert rows[cancelled.name]['settle'] == pytest.approx(bound)  # in flight


def test_valid_action_bank_restricts_the_frozen_rules_on_fresh_seeds():
    from scripts import run_rule_bank_study_20260928 as study
    from scripts import valid_action_rules_20260928 as valid
    from teachers.minigrid.rule_bank import load_bank
    _, raw, _ = load_bank(kc.BANKS / 'scoped.json')
    _, bank, _ = load_bank(kc.BANKS / 'scoped_valid.json')
    assert bank == valid.restrict(raw, kc.FIELDS)
    for condition, action, _ in bank:     # no rule left that cannot act
        if action in valid.VALID_FRONT:
            assert condition['front'] in valid.VALID_FRONT[action]
        if action == valid.PICKUP:
            assert condition['carrying'] == 'nothing'
    spec = study.SPECS['valid_keycorridor']
    built = study.cells(spec)
    assert min(c['seed'] for c in built) == spec['seed0'] + 100 * 20
    guided = [c for c in built if c['arm'] == 'llm_rules_scoped_valid']
    assert len(guided) == 20 and all(
        c['args']['rule_bank'].endswith('/scoped_valid.json')
        and c['args']['offline_summary_only'] for c in guided)
