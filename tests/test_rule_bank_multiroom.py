"""MultiRoom rule interface, oracle and learning-study checks (no API)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from scripts import conditional_rules_multiroom as mr
from teachers.minigrid.door_bfs import DoorOnlyBFSTeacher
from teachers.minigrid.rule_bank import RuleBankTeacher, file_sha256

SEED = mr.PANEL_SEED0


def test_oracle_routes_reach_the_goal_in_the_shortest_count():
    for seed in range(SEED, SEED + 3):
        env = mr.make_env(seed)
        u = env.unwrapped
        start = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir))
        shortest = mr.distances(u)[start]
        env.close()
        assert len(mr.route(seed)) == shortest


def test_oracle_agrees_with_the_door_bfs_teacher():
    teacher, rng = DoorOnlyBFSTeacher(), np.random.default_rng(0)
    actions = mr.route(SEED)
    for _ in range(20):
        env, _ = mr.make_state(SEED, actions, rng)
        u = env.unwrapped
        advice = teacher.recommend(u).action
        assert advice is None or advice in mr.optimal(u)
        env.close()


def facing_a_closed_door():
    env = mr.make_env(SEED)
    for a in mr.route(SEED):
        if a == mr.TOGGLE:
            return env
        env.step(a)
    raise AssertionError('The route opens no door')


def test_student_predicates_come_from_the_view():
    env = facing_a_closed_door()
    u = env.unwrapped
    pred = mr.observe_mr(u.gen_obs()['image'])
    assert pred['front'] == 'door_closed'
    assert pred['closed_door_visible'] == 'yes'
    assert (pred['closed_door_fwd'], pred['closed_door_right']) == ('1', '0')
    assert mr.optimal(u) == (mr.TOGGLE,)
    env.step(mr.TOGGLE)
    pred = mr.observe_mr(u.gen_obs()['image'])
    assert pred['front'] == 'door_open' and pred['door_state'] == 'open'
    assert set(pred) == set(mr.FIELDS)
    env.close()


def write_bank(path, rules, observer='multiroom_v1'):
    path.write_text(json.dumps(dict(
        mode='scoped', observer=observer,
        rules=[dict(condition=c, action=a, exceptions=[list(e) for e in x])
               for c, a, x in rules])))
    return path


def test_rule_bank_uses_the_multiroom_observer(tmp_path):
    env = facing_a_closed_door()
    bank = write_bank(tmp_path / 'b.json',
                      [({'front': 'door_closed'}, mr.TOGGLE, ())])
    assert RuleBankTeacher(bank).recommend(env.unwrapped).action == mr.TOGGLE
    with pytest.raises(ValueError):
        RuleBankTeacher(write_bank(tmp_path / 'x.json', [], 'unknown_v0'))
    env.close()


def test_real_multiroom_training_with_a_rule_bank(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    bank = write_bank(tmp_path / 'b.json', [({'front': 'wall'}, 1, ())])
    ppo.train(ppo.Args(
        task='multiroom_n6', seed=9, total_timesteps=4096, num_envs=4,
        num_steps=64, update_epochs=1, eval_interval=8, eval_episodes=1,
        obs_mode='symbolic', recurrent=True, dual_value=True, bonus='count',
        guidance=True, teacher='rule_bank', rule_bank=str(bank),
        rule_bank_sha256=file_sha256(bank), advisor='unlimited',
        query_budget=0, advice_budget=0, advisor_rng_isolation=True,
        record_initial_policy=True, experiment_id='rule_bank_mr_eng'))
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    assert latest['advising']['num_delivered'] > 0
    assert latest['teacher_cost_dollars'] == 0


@pytest.mark.skipif(not (mr.BANKS / 'blind_strict.json').exists(),
                    reason='MultiRoom banks not built yet')
def test_study_cells_change_only_the_teaching_channel():
    from scripts import run_rule_bank_multiroom_20260928 as study
    built = study.cells()
    assert len(built) == 60
    groups = {}
    for c in built:
        groups.setdefault((c['bonus'], c['seed']), {})[c['arm']] = c['args']
    keys = {'experiment_id', 'rule_bank', 'rule_bank_sha256'}
    for arms in groups.values():
        guided = [a for n, a in arms.items() if n != 'none']
        for a in guided[1:]:
            assert {k for k in a if a[k] != guided[0][k]} <= keys
        assert not arms['none']['guidance']
        assert arms['none']['task'] == 'multiroom_n6'
    for bank in {c['args']['rule_bank'] for c in built if c['args']['rule_bank']}:
        assert json.loads(Path(bank).read_text())['observer'] == 'multiroom_v1'
