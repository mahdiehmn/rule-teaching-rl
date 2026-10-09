"""Rule-bank teacher and learning-pilot engineering checks (no API)."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from envs.registry import build_env
from minigrid.core.world_object import Key
from teachers.minigrid.rule_bank import RuleBankTeacher, file_sha256

BANKS = Path('research/rule_banks/v3_20260928')


def write_bank(path, mode, rules=(), replay=None):
    path.write_text(json.dumps(dict(
        mode=mode, rules=[dict(condition=c, action=a,
                               exceptions=[list(e) for e in x])
                          for c, a, x in rules], replay=replay or {})))
    return path


def test_advice_depends_only_on_the_student_view(tmp_path):
    bank = write_bank(tmp_path / 'b.json', 'scoped',
                      [({'front': 'wall'}, 1, ())])
    teacher = RuleBankTeacher(bank)
    env = build_env('doorkey_8x8', seed=3, obs_mode='symbolic')
    env.reset(seed=3)
    u = env.unwrapped
    first = teacher.recommend(u)
    # change hidden state far outside the view: the goal cell
    gx, gy = next((x, y) for x in range(u.width) for y in range(u.height)
                  if (c := u.grid.get(x, y)) is not None and c.type == 'goal')
    before = u.gen_obs()['image'].copy()
    u.grid.set(gx, gy, None)
    if np.array_equal(before, u.gen_obs()['image']):
        assert teacher.recommend(u).action == first.action
    env.close()


def test_modes_follow_scope_semantics(tmp_path):
    env = build_env('doorkey_8x8', seed=3, obs_mode='symbolic')
    env.reset(seed=3)
    u = env.unwrapped
    view = hashlib.sha256(u.gen_obs()['image'].tobytes()).hexdigest()
    replay = RuleBankTeacher(write_bank(tmp_path / 'r.json', 'replay',
                                        replay={view: 4}))
    assert replay.recommend(u).action == 4
    u.agent_dir = (u.agent_dir + 1) % 4
    assert replay.recommend(u).action is None            # view changed
    guarded = [({'carrying': 'nothing'}, 2, (('door_state', 'locked'),))]
    scoped = RuleBankTeacher(write_bank(tmp_path / 's.json', 'scoped',
                                        guarded))
    direct = RuleBankTeacher(write_bank(tmp_path / 'd.json', 'direct',
                                        guarded))
    from scripts.conditional_rules_v3 import observe_v3
    pred = observe_v3(u.gen_obs()['image'])
    if pred['door_state'] != 'open':                     # unknown or locked
        assert scoped.recommend(u).action is None
    assert direct.recommend(u).action == 2               # exceptions ignored
    conflict = RuleBankTeacher(write_bank(
        tmp_path / 'c.json', 'scoped',
        [({'carrying': 'nothing'}, 0, ()), ({'carrying': 'nothing'}, 1, ())]))
    advice = conflict.recommend(u)
    assert advice.action is None and advice.explanation == 'conflict'
    u.carrying = Key('yellow')
    assert conflict.recommend(u).explanation == 'no_rule'
    env.close()


def pilot_args(tmp_path, bank, **extra):
    base = dict(task='doorkey_8x8', seed=9, total_timesteps=4096,
                num_envs=4, num_steps=64, update_epochs=1, eval_interval=8,
                eval_episodes=1, obs_mode='symbolic', recurrent=True,
                dual_value=True, bonus='none', guidance=True,
                teacher='rule_bank', rule_bank=str(bank),
                rule_bank_sha256=file_sha256(bank), advisor='unlimited',
                query_budget=0, advice_budget=0, advisor_rng_isolation=True,
                record_initial_policy=True, experiment_id='rule_bank_eng')
    base.update(extra)
    return ppo.Args(**base)


def test_real_training_labels_only_where_the_bank_advises(tmp_path,
                                                          monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    bank = write_bank(tmp_path / 'b.json', 'scoped',
                      [({'front': 'wall'}, 1, ())])
    args = pilot_args(tmp_path, bank)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']
    advising = latest['advising']
    assert advising['num_asked'] > advising['num_delivered'] > 0
    assert latest['teacher_cost_dollars'] == 0
    bad = pilot_args(tmp_path, bank)
    bad.rule_bank_sha256 = '0' * 64
    with pytest.raises(ValueError):
        ppo.train(bad)


@pytest.mark.skipif(not (BANKS / 'scope_checked.json').exists(),
                    reason='Frozen banks not built yet')
def test_pilot_cells_change_only_the_teaching_channel():
    from scripts import run_rule_bank_pilot_20260928 as pilot
    cells = pilot.cells()
    assert len(cells) == 30
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c['seed'], {})[c['arm']] = c['args']
    for arms in by_seed.values():
        guided = [a for n, a in arms.items() if n != 'none']
        keys = {'experiment_id', 'rule_bank', 'rule_bank_sha256'}
        for a in guided[1:]:
            assert {k for k in a if a[k] != guided[0][k]} <= keys
        assert not arms['none']['guidance']
        assert arms['none']['total_timesteps'] == 5_000_000


@pytest.mark.skipif(not (BANKS / 'blind_strict.json').exists(),
                    reason='Extension banks not built yet')
def test_extension_cells_pair_with_the_scoped_cells():
    from scripts import run_rule_bank_pilot_20260928 as pilot
    main = pilot.cells()
    extension = pilot.cells(study=pilot.EXTENSION)
    assert len(extension) == 10
    pilot.paired_cells(main, extension)
    subsets = {c['args']['rule_bank'] for c in extension
               if c['arm'] == 'llm_rules_random_subset'}
    assert len(subsets) == 5                        # one per replicate
    ids = {c['args']['experiment_id'] for c in main + extension}
    assert len(ids) == len(pilot.ARMS) + len(pilot.EXTENSION_ARMS)
    changed = [dict(c, args=dict(c['args'], learning_rate=1e-3))
               for c in extension[:1]]
    with pytest.raises(ValueError):
        pilot.paired_cells(main, changed)


def test_extension_refuses_changed_trainer_sources():
    from scripts import run_rule_bank_pilot_20260928 as pilot
    old = {'algos/ppo_distill.py': 'a',
           'scripts/run_rule_bank_pilot_20260928.py': 'b',
           'research/reviews/note.md': 'c',
           f'{BANKS.as_posix()}/scoped.json': 'd',
           'tests/test_rule_bank_pilot.py': 'e'}
    pilot.same_trainer(old, dict(old, **{
        'scripts/run_rule_bank_pilot_20260928.py': 'x',
        'research/reviews/note.md': 'x', 'tests/test_rule_bank_pilot.py': 'x',
        f'{BANKS.as_posix()}/blind_strict.json': 'new'}))
    for key in ('algos/ppo_distill.py', f'{BANKS.as_posix()}/scoped.json'):
        with pytest.raises(ValueError):
            pilot.same_trainer(old, dict(old, **{key: 'x'}))
    with pytest.raises(ValueError):
        pilot.same_trainer(old, {k: v for k, v in old.items()
                                 if k != 'algos/ppo_distill.py'})


SOURCE = Path('results/conditional_rules_v3_20260928')


@pytest.mark.skipif(not ((SOURCE / 'blind_replies.jsonl').exists()
                         and (BANKS / 'blind_strict.json').exists()),
                    reason='Local blind-check replies not present')
def test_extension_banks_are_the_frozen_offline_selection():
    from scripts import conditional_rules_v3 as v3
    from teachers.minigrid.rule_bank import load_bank
    panels, parsed = v3.consult_rules(SOURCE)
    kept, _ = v3.blind_filtered(SOURCE, parsed, panels, strict=True)
    mode, rules, _ = load_bank(BANKS / 'blind_strict.json')
    assert mode == 'scoped' and rules == kept
    _, raw, _ = load_bank(BANKS / 'scoped.json')
    for r in range(5):
        _, subset, _ = load_bank(BANKS / f'random_subset_r{r}.json')
        assert len(subset) == len(kept)
        assert all(rule in raw for rule in subset)


def test_validation_accepts_the_settings_the_trainer_derives(tmp_path,
                                                           monkeypatch):
    """batch_size etc. are filled in at start-up; a finished run is valid."""
    from scripts.run_rule_bank_pilot_20260928 import validate_run
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    bank = write_bank(tmp_path / 'b.json', 'scoped',
                      [({'front': 'wall'}, 1, ())])
    args = pilot_args(tmp_path, bank)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    metrics = validate_run(run, asdict(args))
    assert 0 <= metrics['auc'] <= 1 and metrics['labels_delivered'] > 0
    wrong = dict(asdict(args), learning_rate=args.learning_rate * 2)
    with pytest.raises(ValueError, match='learning_rate'):
        validate_run(run, wrong)
