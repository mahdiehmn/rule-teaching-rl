"""Real CPU Count-PPO controls with offline oracle labels and zero APIs."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from advising import make_advisor
from advising.uniform_queries import UniformQueries
from algos import ppo_distill as ppo
from algos.advice_control import ShamAdvisor
from teachers.base import Advice


def policy_digest(agent):
    """
    Hash exact policy and critic tensors, independent of serialization.
    """

    digest = hashlib.sha256()
    for key, tensor in sorted(agent.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer):
    """
    Hash the exact Adam state tensors and scalar settings after updates.
    """

    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(optimizer.state_dict())
    return digest.hexdigest()


def run_training(root, monkeypatch, seed, **changes):
    """
    Run the real trainer and observe every sample and update read-only.
    """

    args = dict(
        task='doorkey_8x8', teacher='oracle', seed=seed,
        guidance=True, bonus='count', dual_value=True, recurrent=True,
        obs_mode='symbolic', cuda=False, total_timesteps=1024,
        num_envs=4, num_steps=32, num_minibatches=2, update_epochs=2,
        eval_interval=0, eval_sampled=False, record_initial_policy=True,
        offline_summary_only=True, uniform_queries=True, query_budget=1,
        distill_coef_start=1.0)
    args.update(changes)
    history = []
    samples = []
    agents = []
    optimizers = []
    permutations = []
    minibatches = []
    distill_masks = []
    original_agent = ppo.Agent
    original_summary = ppo.RunTracker.write_summary
    original_permutation = np.random.permutation
    original_shuffle = np.random.shuffle
    original_adam = ppo.optim.Adam
    original_loss = ppo.masked_distillation_loss

    class ObservedAgent(original_agent):
        """Read-only instrumentation; inference and updates stay real."""

        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            agents.append(self)

        def get_action_and_value(self, x, action=None, **kwargs):
            output = super().get_action_and_value(x, action, **kwargs)
            if action is None:
                # Keep state, action, value, logits and recurrent state,
                # so matching final weights cannot hide a divergence.
                digest = hashlib.sha256(x.cpu().numpy().tobytes())
                for value in output:
                    if value is not None:
                        digest.update(value.detach().cpu().numpy().tobytes())
                samples.append(digest.hexdigest())
            return output

    class ObservedAdam(original_adam):
        """Retain the real optimizer for read-only state snapshots."""

        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            optimizers.append(self)

    def record_summary(tracker, *values, **kwargs):
        if kwargs.get('status') == 'running' and 'extra' in kwargs:
            extra = kwargs['extra']
            keys = ('learning_rate', 'value_loss', 'intrinsic_value_loss',
                    'exploration', 'policy_loss', 'entropy', 'approx_kl',
                    'clipfrac', 'explained_variance', 'distill_loss')
            history.append({
                'step': kwargs['global_step'],
                'policy': policy_digest(agents[0]),
                'optimizer': optimizer_digest(optimizers[0]),
                # JSON also compares undefined explained variance
                # consistently when all extrinsic returns are zero.
                'metrics': json.dumps(
                    {key: extra[key] for key in keys}, sort_keys=True),
                'queries': extra['teacher_total_queries'],
            })
        return original_summary(tracker, *values, **kwargs)

    def record_permutation(size):
        permutations.append(size)
        return original_permutation(size)

    def record_shuffle(indices):
        original_shuffle(indices)
        minibatches.append(indices.tolist())

    def record_loss(ce, mask, normalization='labeled'):
        distill_masks.append(int(mask.sum()))
        return original_loss(ce, mask, normalization)

    def forbidden_teacher(*values, **kwargs):
        raise AssertionError('A zero-teacher arm instantiated a teacher')

    torch.set_num_threads(1)
    with monkeypatch.context() as patch:
        patch.setattr(ppo, '__file__', str(root / 'algos/ppo_distill.py'))
        patch.setattr(ppo, 'Agent', ObservedAgent)
        patch.setattr(ppo.optim, 'Adam', ObservedAdam)
        patch.setattr(ppo.RunTracker, 'write_summary', record_summary)
        patch.setattr(np.random, 'permutation', record_permutation)
        patch.setattr(np.random, 'shuffle', record_shuffle)
        patch.setattr(ppo, 'masked_distillation_loss', record_loss)
        if not args['guidance'] or args.get('advisor_sham'):
            patch.setattr(ppo, 'make_teacher', forbidden_teacher)
        ppo.train(ppo.Args(**args))

    path = next((root / 'results/runs').iterdir())
    summary = json.loads((path / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['global_step'] == 1024
    # Retain the read-only trace beside the normal trainer artifacts
    # so a reviewer can inspect the exact histories behind assertions.
    (path / 'rng_control_trace.json').write_text(json.dumps(dict(
        history=history, samples=samples, minibatches=minibatches,
        global_query_permutations=permutations,
        labeled_minibatch_counts=distill_masks), indent=2), encoding='utf-8')
    return dict(
        path=path, summary=summary, history=history, samples=samples,
        minibatches=minibatches,
        permutations=permutations, distill_masks=distill_masks,
        initial=(path / 'initial_policy.sha256').read_text(),
        final=torch.load(path / 'agent.pt', weights_only=True))


def learning_history(run):
    """
    Compare learning data separately from the intervention counters.
    """

    return [{key: value for key, value in row.items() if key != 'queries'}
            for row in run['history']]


@pytest.mark.parametrize('seed', [9917300, 9917400])
def test_real_uniform_sham_matches_no_teacher_and_oracle_changes_after_label(
        tmp_path, monkeypatch, seed):
    # The first sparse slot is in rollout 4 or 3 respectively. This
    # leaves several real PPO updates to expose pre-advice RNG drift.
    no_teacher = run_training(
        tmp_path / 'none', monkeypatch, seed,
        guidance=False, uniform_queries=False, advisor_rng_isolation=True)
    sham = run_training(
        tmp_path / 'sham', monkeypatch, seed,
        advisor_rng_isolation=True, advisor_sham=True)
    active = run_training(
        tmp_path / 'active', monkeypatch, seed, advisor_rng_isolation=True)
    declined = run_training(
        tmp_path / 'declined', monkeypatch, seed, uniform_queries=False,
        advisor_rng_isolation=True, advisor='importance', query_budget=0,
        importance_source='entropy', importance_threshold=2.0)
    legacy = run_training(tmp_path / 'legacy', monkeypatch, seed)
    explicit_off = run_training(
        tmp_path / 'explicit_off', monkeypatch, seed,
        advisor_rng_isolation=False, advisor_sham=False)

    assert len(no_teacher['history']) == 8
    assert len(no_teacher['samples']) == 256
    assert len({run['initial'] for run in
                (no_teacher, sham, active, declined,
                 legacy, explicit_off)}) == 1
    assert learning_history(sham) == learning_history(no_teacher)
    assert sham['samples'] == no_teacher['samples']
    assert sham['minibatches'] == no_teacher['minibatches']
    assert learning_history(declined) == learning_history(no_teacher)
    assert declined['samples'] == no_teacher['samples']
    assert declined['minibatches'] == no_teacher['minibatches']
    assert declined['summary']['latest']['teacher_total_queries'] == 0
    assert declined['summary']['latest']['teacher_total_declined'] == 768
    assert all(torch.equal(value, sham['final'][key])
               for key, value in no_teacher['final'].items())
    assert sham['distill_masks'] == no_teacher['distill_masks'] == []
    latest = sham['summary']['latest']
    control = latest['advisor_control']
    assert control['query_order_draws'] == 6 * 32
    assert control['teacher_instances'] == 0
    assert control['teacher_labels'] == 0
    assert control['first_label_global_step'] is None
    for key in ('teacher_total_queries', 'teacher_total_abstains',
                'teacher_cost_dollars', 'teacher_compute_units',
                'reference_calls', 'reference_labels'):
        assert latest[key] == 0
    assert latest['consultations']['records'] == 0
    assert latest['advising']['num_asked'] == 0
    assert latest['advising']['num_delivered'] == 0
    assert latest['advising']['sham']['slots_selected'] == 1
    assert latest['advising']['sham']['virtual_query_budget_remaining'] == 0
    assert not (sham['path'] / 'consultations.jsonl').exists()
    schedule = latest['uniform_query_schedule']
    assert schedule['observed'] == []
    assert schedule['sham_observed'] == schedule['slots']

    slot = schedule['slots'][0]
    prefix_rollouts = slot // 128
    prefix_steps = slot // 4 + 1
    assert prefix_rollouts >= 3
    assert learning_history(active)[:prefix_rollouts] == (
        learning_history(no_teacher)[:prefix_rollouts])
    assert active['samples'][:prefix_steps] == (
        no_teacher['samples'][:prefix_steps])
    assert active['minibatches'] == no_teacher['minibatches']
    assert all(row['queries'] == 0
               for row in active['history'][:prefix_rollouts])
    active_latest = active['summary']['latest']
    assert active_latest['advisor_control']['teacher_labels'] == 1
    assert active_latest['teacher_total_queries'] == 1
    assert active_latest['advisor_control']['first_label_global_step'] == (
        prefix_steps * 4)
    assert sum(active['distill_masks']) == 2
    assert any(not torch.equal(value, active['final'][key])
               for key, value in no_teacher['final'].items())

    # False is the old path: global permutations still occur and cause
    # policy divergence before the first consultation. Explicit false
    # reproduces the default, including all real training updates.
    assert legacy['permutations'] == [4] * (6 * 32)
    assert sham['permutations'] == active['permutations'] == []
    assert learning_history(legacy) == learning_history(explicit_off)
    assert legacy['samples'] == explicit_off['samples']
    assert legacy['minibatches'] == explicit_off['minibatches']
    assert legacy['minibatches'][:prefix_rollouts * 2] != (
        no_teacher['minibatches'][:prefix_rollouts * 2])
    assert all(row['queries'] == 0
               for row in legacy['history'][:prefix_rollouts])
    assert legacy['history'][0]['policy'] != no_teacher['history'][0]['policy']


@pytest.mark.parametrize('advisor', ['unlimited', 'early', 'importance'])
def test_sham_matches_real_teacher_free_scheduling_and_budget(advisor):
    # Real scheduling on an identical input stream must select the
    # same positions, including pacing updates and budget exhaustion.
    args = dict(advice_budget=13, query_budget=17, horizon=120,
                importance_fn=lambda state, context: context['importance'],
                importance_warmup=5, threshold=0.4)
    real = make_advisor(advisor, **args)
    sham = ShamAdvisor(make_advisor(advisor, **args))

    class ConstantTeacher:
        def recommend(self, state, context):
            return Advice(action=0, confidence=1.0)

    selected = 0
    for step in range(120):
        context = {'importance': (step % 17) / 17}
        real.note_env_steps(1)
        sham.note_env_steps(1)
        real.note_step(step)
        sham.note_step(step)
        advice, _ = real.advise(ConstantTeacher(), None, 1, context)
        accepted = sham.schedule(None, 1, context, step + 1)
        assert accepted == (advice is not None)
        selected += accepted
        assert real.threshold == sham.scheduler.threshold
    stats = sham.stats()
    assert selected == real.num_delivered == 13
    assert stats['sham']['slots_selected'] == 13
    assert stats['num_asked'] == stats['num_delivered'] == 0
    assert stats['sham']['virtual_advice_budget_remaining'] == 0
    assert stats['sham']['slots_declined'] == 107


@pytest.mark.parametrize('changes', [
    {'guidance': False}, {'advisor_rng_isolation': False},
    {'advisor': 'mistake'}, {'advisor': 'predictive'},
    {'importance_source': 'teacher_q'}, {'teacher_stream': True},
    {'action_reference': 'oracle'}, {'explanation': 'correct'},
    {'consequence': 'aligned'}, {'advice_replay': True},
    {'imitation_weighting': 'advisor'}, {'peek': 'exact'},
    {'budget_ledger': 'must-not-be-created.json'},
])
def test_sham_rejects_teacher_dependent_paths_before_setup(changes):
    args = asdict(ppo.Args(advisor_sham=True, advisor_rng_isolation=True))
    args.update(changes)
    with pytest.raises(ValueError, match='advisor_sham'):
        ppo.train(ppo.Args(**args))


def test_uniform_planner_does_not_advance_global_numpy_stream():
    np.random.seed(4821)
    before = np.random.get_state()
    planner = UniformQueries(range(6), 32, 4, 1, 9917300)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert planner.slots == (541,)
