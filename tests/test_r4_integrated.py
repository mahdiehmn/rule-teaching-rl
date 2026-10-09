"""
R4 equivalence through the REAL trainer, not a standalone harness.

`test_r4_policy_equivalence.py` establishes the properties on a toy
policy: a detached head passes no gradient to shared features, and the
policy update survives that only if clipping covers policy parameters
alone and the head is constructed after the policy.

Those are necessary but they are claims about a miniature. This file
runs `ppo_distill.train` itself, with its actual optimizer, its actual
global-clipping call, its actual RNG order and its actual recurrent
update, and checks the property that matters end to end:

    with identical settings and seed, an R4 run must produce the same
    policy parameters as an R1 run.

If that holds, any R4-vs-R1 difference in a real experiment is
attributable to the auxiliary gradient into shared features -- which is
the only thing R4 is supposed to vary. If it fails, R4 is not a control.

Targets are controlled (deterministic, synthetic) rather than real LLM
text: this is a software-equivalence check, and using real text would
make it depend on a network call while testing nothing extra.
"""

import json
import hashlib
import re

import pytest
import torch

from algos import ppo_distill


def _short_args(explanation, tmp_path):
    """
    The smallest configuration that still exercises rollout + update.
    """

    return ppo_distill.Args(
        task='keycorridor_s3r3', obs_mode='symbolic', recurrent=True,
        teacher='bot', guidance=True, bonus='count', dual_value=True,
        cuda=False, total_timesteps=32, num_envs=2, num_steps=8,
        num_minibatches=1, update_epochs=1,
        eval_interval=99, eval_episodes=1, eval_sampled=False,
        gamma=0.999, seed=0,
        explanation=explanation, embed_dim=13, lambda_aux=100.0,
    )


def _run_capturing_policy(explanation, tmp_path, monkeypatch,
                          args_factory=None):
    """
    Run the real trainer and return the final policy parameters.

    `lambda_aux` is deliberately enormous. If the detached head could
    perturb the policy through the optimizer or the clipping norm, a
    huge auxiliary loss is what would expose it; a small one could hide
    a real coupling inside floating-point noise.
    """

    monkeypatch.setattr(ppo_distill, '__file__',
                        str(tmp_path / 'algos' / 'ppo_distill.py'))

    captured = {}
    original_agent = ppo_distill.Agent

    def capture(*args, **kwargs):
        agent = original_agent(*args, **kwargs)
        captured['agent'] = agent
        return agent

    monkeypatch.setattr(ppo_distill, 'Agent', capture)

    # A deterministic offline embedder: the same text always maps to
    # the same unit vector, so the two runs see identical targets and
    # no network call is made. This exercises the real target path
    # rather than reaching into the buffers.
    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            self.model = 'fake'
            self.dimension = 13

        @staticmethod
        def key_for(text):
            # Bot planner strings contain Python object addresses.
            # Those change between runs despite identical state/action
            # histories; they must not change controlled test targets.
            stable = re.sub(r'0x[0-9a-fA-F]+', '0xOBJECT', text)
            return hashlib.sha256(stable.encode()).hexdigest()

        def embed(self, texts):
            import numpy as np
            out = []
            for text in texts:
                seed = int(FakeEmbedder.key_for(text)[:8], 16)
                rng = np.random.default_rng(seed)
                vector = rng.normal(size=13).astype('float32')
                out.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
            return out

        def save(self):
            pass

        def stats(self, price_per_million=0.0):
            return {'embedding_model': 'fake', 'dimension': 13,
                    'price_per_million': price_per_million or None}

    monkeypatch.setattr(ppo_distill, 'EmbeddingProvider', FakeEmbedder)

    torch.set_num_threads(1)
    factory = args_factory or _short_args
    ppo_distill.train(factory(explanation, tmp_path))
    agent = captured['agent']
    return {name: p.detach().clone()
            for name, p in agent.named_parameters()}


def test_detached_head_leaves_the_real_policy_update_unchanged(
        tmp_path, monkeypatch):
    """
    The integrated check: R4 must reproduce R1's policy exactly.
    """

    r1 = _run_capturing_policy('none', tmp_path / 'r1', monkeypatch)
    r4 = _run_capturing_policy('detached', tmp_path / 'r4', monkeypatch)

    shared = set(r1) & set(r4)
    assert shared, 'no comparable parameters were captured'
    for name in sorted(shared):
        assert torch.allclose(r1[name], r4[name], atol=0, rtol=0), (
            f'R4 changed policy parameter {name!r} relative to R1; the '
            f'detached head is coupling into the policy update through '
            f'the optimizer or the clipping norm'
        )


def test_attached_head_does_change_the_real_policy(tmp_path, monkeypatch):
    """
    The complement, and the reason the test above is meaningful.

    R2 is *supposed* to move the policy. If an attached head left it
    identical too, the equivalence above would prove only that the
    auxiliary loss is inert in the integrated path.
    """

    r1 = _run_capturing_policy('none', tmp_path / 'r1b', monkeypatch)
    r2 = _run_capturing_policy('correct', tmp_path / 'r2', monkeypatch)

    shared = set(r1) & set(r2)
    differs = any(
        not torch.allclose(r1[n], r2[n], atol=0, rtol=0) for n in shared
    )
    assert differs, (
        'an attached explanation head did not change the policy at all; '
        'the auxiliary gradient is not reaching shared features'
    )


def test_every_explanation_arm_trains_and_saves(tmp_path, monkeypatch):
    """
    All four teacher arms must run end to end, not just typecheck.
    """

    for arm in ('none', 'correct', 'shuffled', 'detached'):
        # Reuse the same runner, so every arm goes through the real
        # trainer with the same offline embedder.
        _run_capturing_policy(arm, tmp_path / arm, monkeypatch)
        run = next((tmp_path / arm / 'results' / 'runs').iterdir())
        summary = json.loads((run / 'run_summary.json').read_text())
        assert summary['status'] == 'completed', f'arm {arm} did not finish'
        evaluations = [json.loads(line) for line in
                       (run / 'evaluations.jsonl').read_text().splitlines()]
        assert len(evaluations) == 1
        assert evaluations[0]['global_step'] == 32
        assert evaluations[0]['teacher_on'] is False
        assert evaluations[0]['success_rate'] == summary['latest'][
            'eval_success_rate']


def test_explanation_arm_is_validated():
    """
    A typo in the arm name must fail at startup, not silently train R1.
    """

    with pytest.raises(ValueError, match='none/correct/shuffled/detached'):
        ppo_distill.train(ppo_distill.Args(explanation='corect'))


def test_explanation_arm_requires_a_teacher():
    """
    An explanation arm with no teacher has nothing to explain.
    """

    with pytest.raises(ValueError, match='needs a teacher'):
        ppo_distill.train(
            ppo_distill.Args(explanation='correct', guidance=False))


# --- The configuration the launch review found broken ---

def _launch_shaped_args(explanation, tmp_path):
    """
    Eight environments and FOUR minibatches, as the smoke will run.

    The single-minibatch tests above passed while R2 crashed here,
    because `newhidden` is already in minibatch order and was being
    indexed a second time with the global batch indices.
    """

    return ppo_distill.Args(
        task='keycorridor_s3r3', obs_mode='symbolic', recurrent=True,
        teacher='bot', guidance=True, bonus='count', dual_value=True,
        cuda=False, total_timesteps=128, num_envs=8, num_steps=8,
        num_minibatches=4, update_epochs=1,
        eval_interval=99, eval_episodes=1, eval_sampled=False,
        gamma=0.999, seed=0,
        explanation=explanation, embed_dim=13, lambda_aux=1.0,
    )


@pytest.mark.parametrize('arm', ['none', 'correct', 'shuffled', 'detached'])
def test_every_arm_runs_with_four_minibatches(arm, tmp_path, monkeypatch):
    """
    The reproduction from the launch review: R2 raised IndexError here.
    """

    _run_capturing_policy(arm, tmp_path, monkeypatch,
                          args_factory=_launch_shaped_args)
    run = next((tmp_path / 'results' / 'runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'


def test_r4_equals_r1_under_the_launch_shape(tmp_path, monkeypatch):
    """
    Equivalence must hold in the configuration that will actually run,
    not only in the one-minibatch toy.
    """

    r1 = _run_capturing_policy('none', tmp_path / 'a', monkeypatch,
                               args_factory=_launch_shaped_args)
    r4 = _run_capturing_policy('detached', tmp_path / 'b', monkeypatch,
                               args_factory=_launch_shaped_args)
    for name in sorted(set(r1) & set(r4)):
        assert torch.allclose(r1[name], r4[name], atol=0, rtol=0), (
            f'R4 diverged from R1 at {name!r} with four minibatches'
        )


def test_explanation_records_are_written(tmp_path, monkeypatch):
    """
    A real run must leave the evidence behind: text, unique ids, phase,
    donor and eligibility. Without it the shuffle report does not exist.
    """

    _run_capturing_policy('shuffled', tmp_path, monkeypatch,
                          args_factory=_launch_shaped_args)
    run = next((tmp_path / 'results' / 'runs').iterdir())
    records = run / 'explanations' / 'explanation_records.jsonl'
    assert records.exists(), 'no explanation records were persisted'

    rows = [json.loads(line) for line in
            records.read_text().splitlines() if line.strip()]
    assert rows, 'records file is empty'
    ids = [r['sample_id'] for r in rows]
    assert len(set(ids)) == len(ids), 'transition ids must be unique'
    for row in rows:
        assert row['text'], 'source text must be retained'
        assert row['phase'] in ('seek_key', 'unlock_door', 'reach_target')
        assert row['arm'] == 'shuffled'
        assert row['queried_at_utc']


def test_arms_do_not_share_a_run_directory(tmp_path, monkeypatch):
    """
    R2 and R4 must be distinguishable by name, not only by timestamp.

    Every field the run name folds in -- task, teacher, guidance,
    bonus, dual value, advisor -- is identical across R1-R4, so before
    the arm tag existed two arms launched in the same second wrote into
    one directory, and the survivors could not be told apart without
    opening each summary. A six-arm sweep already lost 23 of 30 runs to
    exactly that. Both arms are run into one results root here, which
    is the configuration where the collision happens.
    """

    _run_capturing_policy('correct', tmp_path, monkeypatch,
                          args_factory=_launch_shaped_args)
    _run_capturing_policy('detached', tmp_path, monkeypatch,
                          args_factory=_launch_shaped_args)

    names = sorted(p.name for p in
                   (tmp_path / 'results' / 'runs').iterdir())
    assert len(names) == 2, (
        f'two arms produced {len(names)} run directories: {names}; '
        f'they collided'
    )
    assert any('x-correct' in n for n in names)
    assert any('x-detached' in n for n in names)


@pytest.mark.parametrize('arm', ['none', 'correct', 'detached'])
def test_audit_preserves_policy_and_records_auxiliary_gradient(
        arm, tmp_path, monkeypatch):
    """Logging must preserve actual updates, including the R4 zero path."""

    def audited(explanation, directory):
        args = _launch_shaped_args(explanation, directory)
        args.audit_explanations = True
        return args

    plain = _run_capturing_policy(arm, tmp_path / 'plain', monkeypatch,
                                  args_factory=_launch_shaped_args)
    traced = _run_capturing_policy(arm, tmp_path / 'traced', monkeypatch,
                                   args_factory=audited)
    assert plain.keys() == traced.keys()
    for name in plain:
        assert torch.equal(plain[name], traced[name]), name
    run = next((tmp_path / 'traced/results/runs').iterdir())
    directory = run / 'explanations'
    rows = [json.loads(line) for line in
            (directory / 'explanation_records.jsonl').read_text().splitlines()]
    raw = [json.loads(line) for line in
           (directory / 'collected_explanations.jsonl').read_text().splitlines()]
    assert len(rows) == len(raw) > 0
    for row, initial in zip(rows, raw):
        assert row['sample_id'] == initial['sample_id']
        assert row['features'] == initial['features']
        assert len(row['features']) == 512
        assert row['feature_stage'] == 'rollout_pre_update'
        assert row['mission']
        assert len(row['agent_pos']) == 2
        assert 0 <= row['agent_dir'] < 4
        assert row['full_grid'] and row['local_obs']
        assert row['student_action'] == row['executed_action']
        assert row['teacher_target']
    path = directory / 'auxiliary_updates.jsonl'
    if arm == 'none':
        assert not path.exists()
        return
    updates = [json.loads(line) for line in path.read_text().splitlines()]
    assert updates
    norms = [row['weighted_aux_hidden_grad_l2'] for row in updates]
    assert all(row['valid_targets'] > 0 for row in updates)
    if arm == 'detached':
        assert all(norm == 0 for norm in norms)
    else:
        assert any(norm > 0 for norm in norms)
