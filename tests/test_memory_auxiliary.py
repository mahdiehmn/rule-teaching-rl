"""Check recurrent target alignment, private heads, and real PPO controls."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos import ppo_memory as memory


def agent():
    """
    Build the actual policy without giving the auxiliary an environment.
    """

    shape = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            0, 255, shape=(7, 7, 3), dtype=np.uint8),
        single_action_space=gym.spaces.Discrete(7),
    )
    return memory.ppo.Agent(shape, symbolic=True, recurrent=True,
                            dual_value=True)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def tiny_args(**overrides):
    """
    Keep manual hook tests small while resolving native rollout dimensions.
    """

    return replace(memory.Args(), total_timesteps=4, num_envs=2,
                   num_steps=2, num_minibatches=1, update_epochs=1,
                   batch_size=4, minibatch_size=4, num_iterations=1,
                   memory_audit_vector_steps=2, **overrides)


def sequence():
    """
    Observe different red-door states, then the same empty current image.
    """

    images = np.zeros((2, 2, 7, 7, 3), dtype=np.float32)
    images[..., 0] = 1
    images[0, 0, 3, 3] = [4, 0, 0]
    images[0, 1, 3, 3] = [4, 0, 2]
    return torch.tensor(images)


def test_vector_step_schedule_is_fixed_and_dense_by_default():
    args = replace(tiny_args(), num_iterations=10, num_steps=8,
                   total_timesteps=160)
    selected, info = memory.opportunity_schedule(args)
    assert selected is None and info['selected_vector_steps'] == 80
    assert info['maximum_transition_opportunities'] == 160
    finite = replace(args, memory_budget=7)
    first = memory.opportunity_schedule(finite)
    assert first == memory.opportunity_schedule(finite)
    assert len(first[0]) == 7
    assert first[1]['maximum_transition_opportunities'] == 14
    with pytest.raises(ValueError):
        memory.opportunity_schedule(replace(args, memory_budget=81))


def test_reciprocal_pairs_preserve_effective_marginals_and_masks():
    target = {}
    for name, width in memory.FIELDS.items():
        values = np.arange(7 * width).reshape(7, width) % 3
        mask = np.ones_like(values, dtype=bool)
        if name == 'memory':
            mask[::2, 2:] = False
            values[~mask] = memory.IGNORE_INDEX
        target[name], target[name + '_mask'] = values, mask
    paired = memory.paired_targets(
        target, [True] * 5 + [False] * 2, np.random.default_rng(44))
    donors = paired['donors']
    assert (donors >= 0).sum() == 4
    for i, donor in enumerate(donors):
        if donor >= 0:
            assert donor != i and donors[donor] == i
        for name in memory.FIELDS:
            if donor >= 0:
                expected = target[name + '_mask'][i] & target[
                    name + '_mask'][donor]
                assert np.array_equal(paired[name + '_mask'][i], expected)
                assert np.array_equal(
                    paired[name][i][expected],
                    paired[name + '_permuted'][donor][expected])
            else:
                assert not paired[name + '_mask'][i].any()
    for name in memory.FIELDS:
        assert np.array_equal(memory.class_counts(paired[name]),
                              memory.class_counts(paired[name + '_permuted']))


def test_batch_normalization_scales_with_real_supervision_density():
    logits = torch.zeros((4, 6, 3), requires_grad=True)
    targets = torch.full((4, 6), memory.IGNORE_INDEX, dtype=torch.long)
    targets[0, 0] = 1
    sparse = memory.batch_normalized_ce(logits, targets)
    assert float(sparse.detach()) == pytest.approx(np.log(3) / 24)
    targets[1, 0] = 1
    dense = memory.batch_normalized_ce(logits, targets)
    assert float(dense.detach()) == pytest.approx(2 * np.log(3) / 24)
    dense.backward()
    assert not logits.grad[2:].any()
    empty = torch.full_like(targets, memory.IGNORE_INDEX)
    assert memory.batch_normalized_ce(logits, empty).item() == 0


def test_head_and_schedule_initialization_do_not_consume_policy_rng(tmp_path):
    torch.set_num_threads(1)
    policy = agent()
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state()
    aux = memory.MemoryAuxiliary()
    aux.initialize(policy, tmp_path, torch.device('cpu'), tiny_args())
    assert torch.equal(torch.get_rng_state(), before_torch)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    expected = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        expected.update(name.encode() + b'\0')
        expected.update(value.numpy().tobytes())
    assert memory.policy_hash(policy) == expected.hexdigest()


@pytest.mark.parametrize('mode', ['combined', 'permuted', 'detached', 'image'])
def test_historical_door_targets_train_the_gru_except_in_controls(
        tmp_path, mode):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    policy = agent()
    aux = memory.MemoryAuxiliary()
    args = tiny_args(memory_mode=mode)
    aux.initialize(policy, tmp_path, torch.device('cpu'), args)
    images = sequence()
    aux.start_collection(1)
    for step in range(2):
        aux.observe(step, 2 * (step + 1), images[step],
                    torch.ones(2) if step == 0 else torch.zeros(2),
                    torch.zeros(2))
    assert (aux.targets['memory'][0] == memory.IGNORE_INDEX).all()
    expected = [2, 0] if mode == 'permuted' else [0, 2]
    assert aux.targets['memory'][1, :, 0].tolist() == expected
    before = aux.targets['memory'].clone()
    aux.begin_rollout(1)
    assert torch.equal(before, aux.targets['memory'])
    flat = images.reshape(4, 7, 7, 3)
    hidden, _ = policy.get_states(
        flat, core_state=policy.initial_core_state(2, torch.device('cpu')),
        episode_start=torch.tensor([1, 1, 0, 0], dtype=torch.float32))
    aux.set_batch(np.arange(4), hidden, flat)
    loss = aux.loss()
    record = rows(tmp_path / 'memory/gradients.jsonl')[0]
    if mode in ('combined', 'permuted'):
        assert record['memory_core_grad_l2'] > 0
        assert record['memory_encoder_grad_l2'] > 0
    else:
        assert record['memory_core_grad_l2'] == 0
    if mode == 'image':
        assert record['memory_encoder_grad_l2'] > 0
    if mode == 'detached':
        assert record['combined_encoder_grad_l2'] == 0
    loss.backward()
    aux.optimizer_step()
    aux.finish(4)
    finished = read(tmp_path / 'memory/finished.json')
    assert finished['counts']['observed_vector_steps'] == 2
    assert finished['counts']['paired_memory_fields'] == 2
    assert finished['counts']['memory_eligible_episodes'] == 2
    assert finished['counts']['knowledge_remembered_episodes'] == 2
    assert finished['eligible_episodes']['memory'] == [[0, 1], [1, 1]]
    assert np.sum(finished['training_confusion']['memory']) == 2
    assert np.sum(finished['training_confusion']['knowledge']) == 72
    assert finished['class_counts']['paired'] == finished[
        'class_counts']['permuted']


def test_collection_updates_history_at_unselected_steps(tmp_path):
    args = tiny_args(memory_budget=1)
    aux = memory.MemoryAuxiliary()
    aux.initialize(agent(), tmp_path, torch.device('cpu'), args)
    # Choose the second step explicitly to test the hook's observation
    # semantics; the schedule sampler itself is checked separately.
    aux.selected = frozenset({1})
    aux.start_collection(1)
    images = sequence()
    aux.observe(0, 2, images[0], torch.ones(2), torch.zeros(2))
    aux.observe(1, 4, images[1], torch.zeros(2), torch.zeros(2))
    assert aux.targets['memory'][1, :, 0].tolist() == [0, 2]
    assert aux.counts['observed_vector_steps'] == 2
    assert aux.counts['scheduled_vector_steps'] == 1


def test_real_short_ppo_and_detached_control_match_exactly(
        tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')

    def no_teacher(*args, **kwargs):
        raise AssertionError('The history pilot must not create a teacher')

    monkeypatch.setattr(memory.ppo, 'make_teacher', no_teacher)
    outputs = {}
    for mode in memory.MODES:
        directory = tmp_path / mode
        directory.mkdir()
        monkeypatch.chdir(directory)
        monkeypatch.setattr(memory.ppo, '__file__',
                            str(directory / 'algos/ppo_distill.py'))
        args = replace(
            memory.Args(), seed=19, memory_mode=mode, bonus='none',
            total_timesteps=128, num_envs=2, num_steps=8,
            num_minibatches=2, update_epochs=1, eval_interval=0,
            eval_episodes=1, memory_audit_vector_steps=4,
            experiment_id='history_software_check',
        )
        memory.ppo.train(args, auxiliary=memory.MemoryAuxiliary())
        path = next(directory.glob('results/runs/*/memory/finished.json'))
        output = read(path)
        outputs[mode] = output
        assert output['counts']['observed_vector_steps'] == 64
        assert output['counts']['transition_opportunities'] == 128
        assert output['class_counts']['paired'] == output[
            'class_counts']['permuted']
        assert read(path.parent / 'contract.json')['no_evaluation_head_input']
        native = (path.parent.parent / 'initial_policy.sha256').read_text()
        native = native.strip()
        assert native == output['initial_policy_sha256']
    assert len({v['initial_policy_sha256'] for v in outputs.values()}) == 1
    assert outputs['ppo']['final_policy_sha256'] == outputs[
        'detached']['final_policy_sha256']
    assert outputs['ppo']['target_stream_sha256'] == outputs[
        'detached']['target_stream_sha256']
    assert outputs['combined']['final_policy_sha256'] != outputs[
        'ppo']['final_policy_sha256']
    # A tracker-only hook must also match the native trainer without hooks.
    directory = tmp_path / 'unmodified'
    directory.mkdir()
    monkeypatch.setattr(memory.ppo, '__file__',
                        str(directory / 'algos/ppo_distill.py'))
    memory.ppo.train(args)
    checkpoint = next(directory.glob('results/runs/*/agent.pt'))
    assert memory.policy_hash(torch.load(checkpoint, weights_only=True)) == (
        outputs['ppo']['final_policy_sha256'])
