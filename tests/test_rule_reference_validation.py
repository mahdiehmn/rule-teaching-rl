"""Real native free-teacher runs and attacks on their saved artifacts."""

import copy
import json
import shutil
from dataclasses import asdict

import pytest
import torch

from advising import make_advisor
from advising.policy import probability_mistake
from algos import ppo_distill as ppo
from envs import registry
from envs.state import extract_doorkey_state
from scripts.rule_reference_validation import validate_run
from teachers.controlled_advice import CurrentAdvice, advisor_teacher


def _write(path, value):
    """Write a deliberate fixture mutation, preserving real source runs."""
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory):
    """Exercise real policies, optimizers, advisors, environments and bots."""
    root = tmp_path_factory.mktemp('native_rule_reference')
    runs = {}
    torch.set_num_threads(1)
    for task, teacher in (('doorkey_8x8', 'oracle'),
                          ('multiroom_n6', 'door_bfs'),
                          ('keycorridor_s3r3', 'bot')):
        for mode in ('none', 'entropy', 'probability', 'uniform'):
            target = root / task / mode
            guided = mode != 'none'
            values = dict(
                task=task, teacher=teacher, teacher_model='', seed=9_917_800,
                experiment_id=f'rule_smoke_{mode}', guidance=guided,
                teacher_stream=teacher == 'bot' and guided,
                bonus='count', dual_value=True, recurrent=True,
                obs_mode='symbolic', cuda=False,
                total_timesteps=1024 if teacher == 'bot' else 256,
                num_envs=2, num_steps=64 if teacher == 'bot' else 16,
                num_minibatches=2, update_epochs=1,
                eval_interval=4, eval_episodes=1, eval_sampled=True,
                record_initial_policy=True, offline_summary_only=False,
                advisor_rng_isolation=True, advisor_no_teacher_peek=True,
                query_budget=12 if guided else 0,
                advice_budget=12 if guided else 0,
                uniform_queries=mode == 'uniform',
                advisor=('importance' if mode == 'entropy' else
                         'mistake' if mode == 'probability' else 'unlimited'),
                importance_source=('entropy' if mode in
                                   ('entropy', 'probability') else 'none'),
                mistake_threshold=0.2 if mode == 'probability' else 0.0,
                learning_rate=0.00025, distill_coef_start=1.0,
                distill_coef_min=0.01)
            trace = []
            state = {}
            original_build = ppo.build_env
            original_teacher = ppo.make_teacher

            def observed_build(*args, **kwargs):
                env = original_build(*args, **kwargs)
                base = env.unwrapped
                item = {'generation': 0, 'last_action': None, 'actions': 0}
                state[id(base)] = item
                old_reset, old_step = base.reset, base.step

                def reset(*args, **kwargs):
                    result = old_reset(*args, **kwargs)
                    item.update(generation=item['generation'] + 1,
                                last_action=None, actions=0)
                    return result

                def step(action):
                    result = old_step(action)
                    item['last_action'] = int(action)
                    item['actions'] += 1
                    return result

                base.reset, base.step = reset, step
                return env

            def observed_teacher(*args, **kwargs):
                assert guided, 'No-guide arm instantiated a teacher'
                result = original_teacher(*args, **kwargs)
                if teacher != 'bot':
                    return result
                recommend = result.recommend
                previous = {'generation': None, 'actions': None}

                def observed_recommend(env, context=None):
                    item = state[id(env)]
                    fresh = previous['generation'] != item['generation']
                    assert context['new_episode'] is fresh
                    if not fresh:
                        assert item['actions'] == previous['actions'] + 1
                        assert context['last_action'] == item['last_action']
                    else:
                        assert item['actions'] == 0
                    previous.update(generation=item['generation'],
                                    actions=item['actions'])
                    trace.append(dict(env=id(env), **item, fresh=fresh))
                    return recommend(env, context)

                result.recommend = observed_recommend
                return result

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(ppo, '__file__',
                              str(target / 'algos/ppo_distill.py'))
                patch.setattr(ppo, 'build_env', observed_build)
                patch.setattr(registry, 'build_env', observed_build)
                patch.setattr(ppo, 'make_teacher', observed_teacher)
                args = ppo.Args(**values)
                ppo.train(args)
            path = next((target / 'results/runs').iterdir())
            _write(path / 'observed_bot_trace.json', trace)
            runs[task, mode] = path, asdict(args), trace
    return runs


@pytest.mark.parametrize('task', ['doorkey_8x8', 'multiroom_n6',
                                 'keycorridor_s3r3'])
@pytest.mark.parametrize('mode', ['none', 'entropy', 'probability', 'uniform'])
def test_actual_native_runs_validate(native_runs, task, mode):
    path, args, trace = native_runs[task, mode]
    metrics = validate_run(path, args)
    assert metrics['actual_transitions'] == args['total_timesteps']
    assert metrics['evaluations'] == 2
    assert 0 <= metrics['greedy_auc'] <= 1
    assert 0 <= metrics['sampled_auc'] <= 1
    assert metrics['api_cost_usd'] == 0
    assert metrics['teacher_queries'] <= args['query_budget']
    assert metrics['consultation_rows'] == metrics['teacher_queries']
    assert len(metrics['final_policy_sha256']) == 64
    baseline, baseline_args, _ = native_runs[task, 'none']
    assert metrics['initial_policy_sha256'] == validate_run(
        baseline, baseline_args)['initial_policy_sha256']
    if mode != 'none':
        assert metrics['teacher_queries'] > 0
    if task == 'keycorridor_s3r3' and mode != 'none':
        assert len(trace) == metrics['reference_calls']
        assert metrics['reference_calls'] > metrics['teacher_queries']
        assert any(not row['fresh'] for row in trace)
        assert sum(row['fresh'] for row in trace) > args['num_envs']
        assert metrics['reference_wall_seconds'] > 0
        assert metrics['teacher_wall_seconds'] == 0


def _change_summary(path, fields, value):
    """Mutate one field inside the copied genuine terminal artifact."""
    data = json.loads(path.read_text())
    cursor = data
    for field in fields[:-1]:
        cursor = cursor[field]
    cursor[fields[-1]] = value
    _write(path, data)


@pytest.mark.parametrize('fields,value', [
    (('status',), 'running'),
    (('global_step',), 224),
    (('args', 'learning_rate'), 0.001),
    (('args', 'batch_size'), 64),
    (('args', 'seed'), 2),
    (('latest', 'teacher_total_queries'), 13),
    (('latest', 'advising', 'num_asked'), 0),
    (('latest', 'advisor_control', 'teacher_labels'), 99),
    (('latest', 'advisor_control', 'query_seed'), 1),
    (('latest', 'advisor_control', 'query_order_draws'), 0),
    (('latest', 'advisor_control', 'first_label_global_step'), 256),
    (('latest', 'teacher_cost_dollars'), 0.01),
    (('latest', 'consultations', 'records_written'), 0),
    (('latest', 'consultations', 'unknown_cost_records'), 1),
    (('latest', 'eval_sampled_success_rate'), 0.123),
    (('latest', 'uniform_query_schedule', 'rng_seed'), 1),
    (('latest', 'reference_calls'), 1),
    (('latest', 'value_loss'), float('nan')),
    (('wall_time_sec',), float('inf')),
])
def test_rejects_terminal_mutations(native_runs, tmp_path, fields, value):
    source, args, _ = native_runs['doorkey_8x8', 'uniform']
    path = tmp_path / 'run'
    shutil.copytree(source, path)
    _change_summary(path / 'run_summary.json', fields, value)
    with pytest.raises(ValueError):
        validate_run(path, args)


@pytest.mark.parametrize('attack', [
    'missing_evaluation', 'duplicate_evaluation', 'wrong_eval_seed',
    'teacher_on', 'missing_sampled', 'nonfinite_sampled',
    'missing_consultation', 'duplicate_consultation', 'inactive_consultation',
    'nonzero_consultation_cost', 'initial_hash', 'missing_checkpoint',
    'nonfinite_checkpoint', 'missing_actor', 'stream_count',
])
def test_rejects_low_level_artifact_attacks(native_runs, tmp_path, attack):
    task = 'keycorridor_s3r3' if attack == 'stream_count' else 'doorkey_8x8'
    source, args, _ = native_runs[task, 'uniform']
    path = tmp_path / 'run'
    shutil.copytree(source, path)
    if 'evaluation' in attack or attack in (
            'wrong_eval_seed', 'teacher_on', 'missing_sampled',
            'nonfinite_sampled'):
        output = path / 'evaluations.jsonl'
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if attack == 'missing_evaluation':
            rows.pop(0)
        elif attack == 'duplicate_evaluation':
            rows.insert(0, copy.deepcopy(rows[0]))
        elif attack == 'wrong_eval_seed':
            rows[0]['seed_base'] += 1
        elif attack == 'teacher_on':
            rows[0]['teacher_on'] = True
        elif attack == 'missing_sampled':
            rows[0].pop('sampled_success_rate')
        else:
            rows[0]['sampled_success_rate'] = float('nan')
        output.write_text('\n'.join(map(json.dumps, rows)))
    elif 'consultation' in attack:
        output = path / 'consultations.jsonl'
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if attack == 'missing_consultation':
            rows.pop(0)
        elif attack == 'duplicate_consultation':
            rows.append(copy.deepcopy(rows[0]))
        elif attack == 'inactive_consultation':
            rows[-1]['rollout'] = 7
        else:
            rows[0]['dollars'] = 0.01
        output.write_text('\n'.join(map(json.dumps, rows)))
    elif attack == 'initial_hash':
        (path / 'initial_policy.sha256').write_text('broken')
    elif attack == 'stream_count':
        _change_summary(path / 'run_summary.json',
                        ('latest', 'reference_calls'), 12)
    elif attack == 'missing_checkpoint':
        (path / 'agent.pt').unlink()
    else:
        output = path / 'agent.pt'
        state = torch.load(output, weights_only=True)
        if attack == 'nonfinite_checkpoint':
            state[next(iter(state))].flatten()[0] = float('nan')
        else:
            state = {k: v for k, v in state.items()
                     if not k.startswith('actor.')}
        torch.save(state, output)
    with pytest.raises(ValueError):
        validate_run(path, args)


@pytest.mark.parametrize('task,teacher', [
    ('doorkey_8x8', 'oracle'), ('multiroom_n6', 'door_bfs'),
    ('keycorridor_s3r3', 'bot'),
])
def test_real_probability_gate_counts_query_before_withholding(task, teacher):
    """A confident student still spends its selection before withholding."""
    env = ppo.build_env(task, seed=9917800, obs_mode='symbolic')
    try:
        env.reset(seed=9917800)
        source = ppo.make_teacher(teacher, ppo.TASKS[task], 9917800)
        state = (extract_doorkey_state(env.unwrapped) if teacher == 'oracle'
                 else env.unwrapped)
        advice = source.recommend(state, {'new_episode': True})
        assert advice.action is not None
        if teacher == 'bot':
            source = CurrentAdvice(advice)
        wrapped = advisor_teacher(source, no_peek=True)
        assert not hasattr(wrapped, 'peek_action')
        assert not hasattr(wrapped, 'optimal_actions')
        advisor = make_advisor(
            'mistake', num_actions=7, importance_source='none',
            normalize_importance=False, advice_budget=2, query_budget=2,
            mistake_fn=probability_mistake(0.2))
        for probability, expected_delivery in ((0.9, False), (0.1, True)):
            probabilities = [(1 - probability) / 6] * 7
            probabilities[advice.action] = probability
            selected, cost = advisor.advise(
                wrapped, state, (advice.action + 1) % 7,
                {'action_probs': probabilities})
            assert (selected is not None) is expected_delivery
            assert cost.dollars == 0
        assert advisor.num_asked == 2
        assert advisor.num_delivered == advisor.num_withheld == 1
    finally:
        env.close()
