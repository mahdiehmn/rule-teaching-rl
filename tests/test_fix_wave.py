"""Fix wave: cell construction, channel purity, pairing and the combined path."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from algos import ppo_distill as ppo
from algos import ppo_potential as pot
from scripts import run_fix_wave_20260929 as fw
from scripts import run_rule_bank_study_20260928 as rb

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('suite, n', [('kc', 140), ('mr', 50), ('dk', 50),
                                      ('kc_long', 15), ('kc_llm', 20),
                                      ('mr_llm', 10), ('dk_llm', 10),
                                      ('kc_confirm', 60), ('mr_confirm', 60),
                                      ('dk_confirm', 60), ('kc_mem', 40),
                                      ('kc_mem_confirm', 40), ('dk16', 40),
                                      ('kc_s4', 40), ('dk_tv', 20),
                                      ('mr_tv', 20), ('kc_tv', 20),
                                      ('dk16_w', 10), ('kc_s4_w', 10),
                                      ('dk_act', 20), ('mr_act', 20),
                                      ('kc_act', 20), ('dk16_long', 40),
                                      ('mr10_long', 40), ('kc_s4_long', 40),
                                      ('kc_prog', 20), ('kc_s5_long', 40),
                                      ('kc_s6_long', 40), ('dk_cc', 20),
                                      ('mr_cc', 20), ('kc_cc', 20),
                                      ('dk_dense', 40), ('mr_dense', 40),
                                      ('kc_dense', 40), ('mr_cross', 20),
                                      ('dk_fresh', 40), ('mr_fresh', 40),
                                      ('kc_fresh', 60), ('dk_fresh_cc', 20),
                                      ('mr_fresh_cc', 20), ('kc_fresh_cc', 20),
                                      ('mr_fresh_cross', 20), ('dk_rl', 20),
                                      ('mr_rl', 20), ('kc_rl', 20),
                                      ('dk_rlfull', 20),
                                      ('mr_rlfull', 20),
                                      ('kc_rlfull', 20),
                                      ('dk16_rl_long', 20),
                                      ('mr10_rl_long', 20),
                                      ('kc_s4_rl_long', 20)])
def test_every_suite_builds_its_planned_cells(suite, n):
    cells = fw.suite_cells(suite)
    assert len(cells) == n
    assert [c['index'] for c in cells] == list(range(n))
    assert len({(c['bonus'], c['arm'], c['seed']) for c in cells}) == n


def test_seeds_are_the_rule_bank_studys_seeds():
    for suite, (task, _students, reps, _h) in fw.SUITES.items():
        seed0 = rb.SPECS[fw.BASE_SPEC[task]]['seed0']
        seeds = {c['seed'] for c in fw.suite_cells(suite)}
        assert seeds == {seed0 + 100 * r for r in reps}


@pytest.mark.parametrize('suite', ('kc', 'mr', 'dk'))
def test_none_equals_the_rule_bank_none_cell_but_its_name(suite):
    # The pairing premise: the re-run none is the rule-bank study's own
    # no-teacher cell, so the fix arms pair with that study by seed.
    task, students, reps, _h = fw.SUITES[suite]
    spec = rb.SPECS[fw.BASE_SPEC[task]]
    r = list(reps)[0]
    for bonus in students:
        none, guided = rb.base_args(task, bonus)
        theirs = asdict(rb.arm_args(spec, 'none', r, bonus, none, guided,
                                    rb.ROOT, ''))
        mine = asdict(fw.arm_args(suite, 'none', r, bonus))
        assert {k for k in theirs if theirs[k] != mine[k]} == {
            'experiment_id'}


def test_rule_arms_imitate_the_same_bank_as_the_paper_arm():
    for suite in ('kc', 'mr', 'dk'):
        task, students, reps, _h = fw.SUITES[suite]
        spec = rb.SPECS[fw.BASE_SPEC[task]]
        for bonus in students:
            rule_arm = fw.RULE_ARM[(task, bonus)]
            none, guided = rb.base_args(task, bonus)
            theirs = rb.arm_args(dict(spec, summary_only=True), rule_arm,
                                 list(reps)[0], bonus, none, guided,
                                 rb.ROOT, '')
            for arm in ('rules_plus_shaping', 'rules_weak'):
                mine = fw.arm_args(suite, arm, list(reps)[0], bonus)
                assert mine.rule_bank == theirs.rule_bank
                assert mine.rule_bank_sha256 == theirs.rule_bank_sha256
                assert mine.teacher == 'rule_bank'


def test_channels_are_what_each_arm_claims():
    kc = {(c['bonus'], c['arm']): c for c in fw.suite_cells('kc')
          if c['replicate'] == 5}
    for bonus in ('none', 'count'):
        a = {arm: kc[(bonus, arm)]['args'] for arm in fw.ARMS_OF['kc']}
        assert not a['none']['guidance']
        for arm in ('shaping_dense', 'shaping_dense_wrong', 'shaping_480'):
            assert not a[arm]['guidance']                 # no imitation
            assert not a[arm]['potential_with_guidance']
        assert a['shaping_dense']['potential_queries'] == -1
        assert a['shaping_480']['potential_queries'] == 480
        assert a['shaping_dense_wrong']['potential_wrong_rate'] == 1.0
        assert a['shaping_dense']['potential_wrong_rate'] == 0.0
        assert a['rules_plus_shaping']['guidance']        # both channels
        assert a['rules_plus_shaping']['potential_with_guidance']
        # The paper's rule arms imitate at 1 -> .01; the weak arm at a
        # tenth. (A first draft set the weak arm to 1 -> .01, identical
        # to the rules arm; this assertion caught it.)
        assert (a['rules']['distill_coef_start'],
                a['rules']['distill_coef_min']) == (1.0, 0.01)
        assert (a['rules_weak']['distill_coef_start'],
                a['rules_weak']['distill_coef_min']) == pytest.approx(
                    (0.1, 0.001))
        assert a['rules']['offline_summary_only']        # no GB journals
    trainers = {c['arm']: c['trainer'] for c in fw.suite_cells('kc')}
    assert trainers['rules'] == trainers['rules_weak'] == \
        trainers['none'] == 'algos.ppo_distill'
    assert trainers['rules_plus_shaping'] == 'algos.ppo_potential'


def test_long_suite_runs_three_times_the_horizon():
    assert {c['args']['total_timesteps'] for c in
            fw.suite_cells('kc_long')} == {15_000_000}
    assert {c['args']['total_timesteps'] for c in
            fw.suite_cells('kc')} == {5_000_000}


def test_builder_refuses_a_cell_that_leaks_outside_its_channel(monkeypatch):
    real = fw.arm_args

    def leaky(suite, arm, r, bonus, root=fw.ROOT):
        args = real(suite, arm, r, bonus, root)
        return replace(args, learning_rate=1e-3) if arm == \
            'shaping_dense' else args

    monkeypatch.setattr(fw, 'arm_args', leaky)
    with pytest.raises(ValueError, match='beyond its channel'):
        fw.suite_cells('dk')


def test_pure_potential_channel_still_rejects_guidance():
    args = fw.arm_args('kc', 'rules_plus_shaping', 5, 'none')
    with pytest.raises(ValueError):
        pot.validate_args(replace(args, potential_with_guidance=False))
    with pytest.raises(ValueError, match='rule-bank teacher'):
        pot.validate_args(replace(fw.arm_args('kc', 'shaping_dense', 5,
                                              'none'),
                                  potential_with_guidance=True))
    pot.validate_args(args)                               # combined: fine


def test_paired_refuses_different_initial_policies():
    a = {1: dict(auc=.5, initial_sha256='x')}
    b = {1: dict(auc=.4, initial_sha256='y')}
    assert fw.paired(a, b, 'x - y')['error'].startswith('1 seeds')
    same = fw.paired({1: dict(auc=.5, initial_sha256='x'),
                      2: dict(auc=.7, initial_sha256='z')},
                     {1: dict(auc=.4, initial_sha256='x'),
                      2: dict(auc=.6, initial_sha256='z')}, 'ok')
    assert same['n'] == 2 and same['positive'] == 2
    assert same['mean'] == pytest.approx(.1)


def test_report_on_an_empty_root_says_not_prepared(tmp_path, capsys):
    assert fw.report(tmp_path, list(fw.SUITES)) == {}
    assert capsys.readouterr().out.count('not prepared') == len(fw.SUITES)


def test_required_packet_files_exist():
    for name in fw.REQUIRED:
        assert (ROOT / name).exists(), name


def test_combined_arm_runs_end_to_end(tmp_path, monkeypatch):
    # The one new trainer path: rule-bank imitation and shaping together.
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc', 'rules_plus_shaping', 5, 'none'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args, auxiliary=pot.PotentialAuxiliary())
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    potential = json.loads((run / 'potential_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['latest']['advising']['num_delivered'] > 0   # imitation
    assert potential['queries_used'] > 0                        # shaping
    assert potential['wrong_messages'] == 0


# ------------------------------------------------ addendum 1: the LLM's plans

@pytest.mark.parametrize('suite, base', [('kc_llm', 'kc'), ('mr_llm', 'mr'),
                                         ('dk_llm', 'dk')])
def test_llm_plan_cells_differ_from_hand_coded_shaping_only_in_the_plan(
        suite, base):
    import json as js
    frozen = js.loads((ROOT / fw.PLAN_FILE).read_text(encoding='utf-8'))
    mine = {(c['bonus'], c['seed']): c['args']
            for c in fw.suite_cells(suite)}
    theirs = {(c['bonus'], c['seed']): c['args']
              for c in fw.suite_cells(base) if c['arm'] == 'shaping_dense'}
    assert set(mine) == set(theirs)              # same students and seeds
    for key, args in mine.items():
        diff = {k for k in args if args[k] != theirs[key][k]}
        assert diff == {'experiment_id', 'potential_plan',
                        'potential_plan_sha256'}
        assert args['potential_plan'] == fw.PLAN_FILE
        assert args['potential_plan_sha256'] ==             frozen['tasks'][args['task']]['sha256']
        assert not args['guidance'] and args['potential_wrong_rate'] == 0


def test_every_suite_has_a_report_group_and_a_time_limit():
    assert set(fw.GROUP) == set(fw.SUITES) == set(fw.TIME) ==         set(fw.ARMS_OF)
    # an addendum shares task, students, seeds and horizon with its group;
    # 'confirm' is its own group of fresh seeds, not a suite
    for suite, group in fw.GROUP.items():
        if group in fw.SUITES:
            # same task, seeds and horizon; an addendum may run a subset of
            # its group's students (addendum 7: plain PPO only)
            task, students, reps, horizon = fw.SUITES[suite]
            g_task, g_students, g_reps, g_horizon = fw.SUITES[group]
            assert (task, reps, horizon) == (g_task, g_reps, g_horizon)
            assert set(students) <= set(g_students)
        elif group == 'dense':
            # addendum 13's re-run of the confirmation cells
            assert suite.endswith('_dense')
        elif group == 'fresh':
            # addendum 15's fresh-seed cohort: core and optional suites
            assert suite in fw.FRESH_CORE + fw.FRESH_OPTIONAL
        else:
            # addendum 6's teacher-view suites join the confirmation seeds
            assert group == 'confirm' and suite.endswith(
                ('_confirm', '_tv', '_act', '_prog', '_cc', '_cross',
                 '_rl', '_rlfull', '_rlex'))


def _fake_batch(tmp_path, required=None, files=fw.REQUIRED_V1):
    """A prepared batch directory as an earlier commit would leave it."""
    from scripts.run_plan_repair_20260927 import digest
    batch = tmp_path / 'b'
    for name in files:
        path = batch / 'code' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    manifest = dict(study=fw.STUDY, suite='kc', commit='abc', cells=[],
                    source_hashes={n: digest(batch / 'code' / n)
                                   for n in files})
    if required is not None:
        manifest['required'] = list(required)
    (batch / 'manifest.json').write_text(json.dumps(manifest))
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text('abc')        # compared after strip()
    return batch


def test_a_batch_prepared_before_addendum_1_still_verifies(tmp_path):
    # Regression: adding the plan files to REQUIRED made every running
    # batch fail its identity check, which would abort the launcher and
    # crash the report on the fix wave.
    batch = _fake_batch(tmp_path)                  # no 'required' key
    assert fw.verify(batch, rebuild=False)['suite'] == 'kc'


def test_a_new_batch_must_carry_every_file_it_requires(tmp_path):
    batch = _fake_batch(tmp_path, required=fw.REQUIRED)
    with pytest.raises(ValueError, match='identity'):
        fw.verify(batch, rebuild=False)
    complete = _fake_batch(tmp_path / 'c', required=fw.REQUIRED,
                           files=fw.REQUIRED)
    assert fw.verify(complete, rebuild=False)['suite'] == 'kc'


# ------------------------------------ addendum 2: fresh-seed confirmation

def test_confirmation_seeds_are_fresh():
    """Replicates 30-39: no fix-wave suite and no rule-bank study used them
    (rule-bank replicates 0-19, valid-action KeyCorridor 20-29). Addendum
    6's teacher-view suites run on exactly these seeds, by design, to pair
    with the completed confirmation cells."""
    confirm = {c['seed'] for s in ('kc_confirm', 'mr_confirm', 'dk_confirm')
               for c in fw.suite_cells(s)}
    others = {c['seed'] for s in fw.SUITES
              if not s.endswith(('_confirm', '_tv', '_act', '_prog', '_cc',
                                 '_dense', '_cross', '_rl', '_rlfull',
                                 '_rlex'))
              for c in fw.suite_cells(s)}
    assert not confirm & others
    for suites in (('dk_tv', 'mr_tv', 'kc_tv'), ('dk_act', 'mr_act', 'kc_act')):
        if suites[0] == 'dk_act':
            kc = {c['seed'] for c in fw.suite_cells('kc_prog')}
            assert kc <= confirm
        assert {c['seed'] for s in suites
                for c in fw.suite_cells(s)} == confirm
    for suite in ('kc_confirm', 'mr_confirm', 'dk_confirm'):
        task = fw.SUITES[suite][0]
        seed0 = rb.SPECS[fw.BASE_SPEC[task]]['seed0']
        reps = {(c['seed'] - seed0) // 100 for c in fw.suite_cells(suite)}
        assert reps == set(range(30, 40))


def test_confirmation_arms_equal_the_fix_wave_arms_but_their_seed():
    for suite, base in (('kc_confirm', 'kc'), ('mr_confirm', 'mr'),
                        ('dk_confirm', 'dk')):
        for arm in ('none', 'rules_weak', 'rules'):
            if arm not in fw.ARMS_OF[base]:
                continue                # mr/dk ran no plain rules arm
            a = asdict(fw.arm_args(suite, arm, 30, 'none'))
            b = asdict(fw.arm_args(base, arm, 5, 'none'))
            assert {k for k in a if a[k] != b[k]} == {'experiment_id',
                                                      'seed'}


def test_holm_step_down():
    assert fw.holm([0.01, None, 0.04, 0.03]) == pytest.approx(
        [0.03, None, 0.06, 0.06])


def test_paired_reports_a_two_sided_p():
    a = {k: dict(auc=.5 + .01 * k, initial_sha256=str(k)) for k in range(5)}
    b = {k: dict(auc=.4, initial_sha256=str(k)) for k in range(5)}
    c = fw.paired(a, b, 'x')
    assert 0 < c['p'] < .01
    same = fw.paired(b, b, 'y')
    assert same['p'] == 1.0 and same['mean'] == 0



# ------------------------------- addendum 3: KeyCorridor rules with memory

@pytest.mark.parametrize('suite, base, r', [('kc_mem', 'kc', 5),
                                            ('kc_mem_confirm', 'kc_confirm',
                                             30)])
def test_memory_arms_change_only_the_bank_and_the_weight(suite, base, r):
    from teachers.minigrid.rule_bank import file_sha256
    for bonus in ('none', 'count'):
        rules = asdict(fw.arm_args(base, 'rules', r, bonus))
        weak = asdict(fw.arm_args(base, 'rules_weak', r, bonus))
        mem = asdict(fw.arm_args(suite, 'rules_mem', r, bonus))
        mem_weak = asdict(fw.arm_args(suite, 'rules_mem_weak', r, bonus))
        assert {k for k in mem if mem[k] != rules[k]} == {
            'experiment_id', 'rule_bank', 'rule_bank_sha256'}
        assert {k for k in mem_weak if mem_weak[k] != weak[k]} == {
            'experiment_id', 'rule_bank', 'rule_bank_sha256'}
        assert mem['rule_bank'] == mem_weak['rule_bank'] == fw.MEM_BANK
        assert mem['rule_bank_sha256'] == file_sha256(ROOT / fw.MEM_BANK)
        assert (mem['distill_coef_start'], mem['distill_coef_min']) == (
            1.0, 0.01)
        assert (mem_weak['distill_coef_start'],
                mem_weak['distill_coef_min']) == pytest.approx((0.1, 0.001))


def test_memory_bank_is_read_with_the_memory_observer():
    from teachers.minigrid.rule_bank import RuleBankTeacher
    bank = json.loads((ROOT / fw.MEM_BANK).read_text(encoding='utf-8'))
    assert bank['observer'] == 'keycorridor_mem_v1'
    assert bank['task'] == 'keycorridor_s3r3'
    assert RuleBankTeacher(ROOT / fw.MEM_BANK).stateful


def test_memory_arms_refuse_other_tasks():
    with pytest.raises(ValueError, match='KeyCorridor'):
        fw.arm_args('dk', 'rules_mem', 5, 'none')


def test_memory_arm_trains_end_to_end(tmp_path, monkeypatch):
    # The trainer hands the rule teacher the live episode, from which the
    # memory predicate is read; labels must actually be delivered.
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc_mem', 'rules_mem', 5, 'count'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['latest']['advising']['num_delivered'] > 0


def test_memory_family_is_reported_with_holm(tmp_path, capsys, monkeypatch):
    rows = {}
    for bonus in ('none', 'count'):
        for arm, base in (('none', .5), ('rules_mem', .7),
                          ('rules_mem_weak', .6)):
            rows[('keycorridor_s3r3', bonus, arm)] = {
                s: dict(auc=base + .01 * s, final=base,
                        initial_sha256=str(s)) for s in range(4)}
    for suite in ('kc_mem_confirm',):
        (tmp_path / 'results/fix_wave' / fw.STUDY / suite).mkdir(
            parents=True)
        (tmp_path / 'results/fix_wave' / fw.STUDY / suite /
         'manifest.json').write_text('{}')
    monkeypatch.setattr(fw, 'validated', lambda batch: (
        {'cells': [0] * 12}, rows))
    result = fw.report(tmp_path, ['kc_mem_confirm'])
    out = capsys.readouterr().out
    assert 'addendum 3 family' in out and 'Holm over 4 contrasts' in out
    family = [c for c in result['confirm'] if 'p_holm' in c]
    assert len(family) == 4


# ------------------------- addendum 4: the DoorKey bank on DoorKey-16x16

@pytest.mark.parametrize('arm', ('none', 'rules_weak'))
def test_transfer_cells_change_only_the_map_and_the_seed(arm):
    for bonus in ('none', 'count'):
        big = asdict(fw.arm_args('dk16', arm, 40, bonus))
        small = asdict(fw.arm_args('dk_confirm', arm, 30, bonus))
        assert {k for k in big if big[k] != small[k]} == {
            'experiment_id', 'seed', 'task'}
        assert big['task'] == 'doorkey_16x16'
        assert big['rule_bank'] == small['rule_bank']          # unchanged
        assert big['rule_bank_sha256'] == small['rule_bank_sha256']


def test_transfer_seeds_are_unused_by_every_other_suite():
    mine = {c['seed'] for c in fw.suite_cells('dk16')}
    # dk16_w (addendum 7) reruns these seeds by design, to pair with dk16
    others = {c['seed'] for s in fw.SUITES if s not in ('dk16', 'dk16_w')
              for c in fw.suite_cells(s)
              if c['task'] in ('doorkey_8x8', 'doorkey_16x16')}
    assert len(mine) == 10 and not mine & others


def test_large_doorkey_keeps_the_student_observation():
    from envs.registry import build_env
    small = build_env('doorkey_8x8', seed=1, obs_mode='symbolic')
    big = build_env('doorkey_16x16', seed=1, obs_mode='symbolic')
    assert (small.observation_space.shape == big.observation_space.shape
            == (7, 7, 3))
    big.reset(seed=1)
    assert big.unwrapped.width == 16 and big.unwrapped.max_steps == 2560
    small.close()
    big.close()


def test_transfer_arm_trains_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('dk16', 'rules_weak', 40, 'none'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['args']['task'] == 'doorkey_16x16'
    assert summary['latest']['advising']['num_delivered'] > 0
    assert summary['latest']['teacher_cost_dollars'] == 0


# ------------------ addendum 5: the KeyCorridor memory bank on S4R3

@pytest.mark.parametrize('arm, base', [('none', 'kc_confirm'),
                                       ('rules_mem_weak', 'kc_mem_confirm')])
def test_kc_transfer_cells_change_only_the_map_and_the_seed(arm, base):
    for bonus in ('none', 'count'):
        big = asdict(fw.arm_args('kc_s4', arm, 40, bonus))
        small = asdict(fw.arm_args(base, arm, 30, bonus))
        assert {k for k in big if big[k] != small[k]} == {
            'experiment_id', 'seed', 'task'}
        assert big['task'] == 'keycorridor_s4r3'
        if arm == 'rules_mem_weak':
            assert big['rule_bank'] == fw.MEM_BANK


def test_kc_transfer_memory_arm_trains_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc_s4', 'rules_mem_weak', 40, 'count'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['args']['task'] == 'keycorridor_s4r3'
    assert summary['latest']['advising']['num_delivered'] > 0


# ------------------------------- addendum 7: weak student, larger maps

@pytest.mark.parametrize('suite, base, arm, weak', [
    ('dk16_w', 'dk16', 'rules', 'rules_weak'),
    ('kc_s4_w', 'kc_s4', 'rules_mem', 'rules_mem_weak')])
def test_paper_weight_cells_differ_from_the_weak_cells_only_in_weight(
        suite, base, arm, weak):
    weak_cells = {c['seed']: c['args'] for c in fw.suite_cells(base)
                  if c['arm'] == weak and c['bonus'] == 'none'}
    cells = fw.suite_cells(suite)
    assert {c['bonus'] for c in cells} == {'none'}
    assert {c['seed'] for c in cells} == set(weak_cells)
    for c in cells:
        assert c['arm'] == arm
        twin = weak_cells[c['seed']]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed == {'experiment_id', 'distill_coef_start',
                           'distill_coef_min'}
        assert c['args']['distill_coef_start'] ==             10 * twin['distill_coef_start']


# ----------------- addendum 9: larger environments, 15M steps, fresh seeds

@pytest.mark.parametrize('suite, arm, base, task', [
    ('dk16_long', 'none', 'dk_confirm', 'doorkey_16x16'),
    ('dk16_long', 'rules_weak', 'dk_confirm', 'doorkey_16x16'),
    ('mr10_long', 'none', 'mr_confirm', 'multiroom_n10'),
    ('mr10_long', 'rules_weak', 'mr_confirm', 'multiroom_n10'),
    ('kc_s4_long', 'none', 'kc_confirm', 'keycorridor_s4r3'),
    ('kc_s4_long', 'rules_mem_weak', 'kc_mem_confirm', 'keycorridor_s4r3')])
def test_long_cells_change_only_map_seed_and_horizon(suite, arm, base, task):
    for bonus in ('none', 'count'):
        big = asdict(fw.arm_args(suite, arm, 50, bonus))
        small = asdict(fw.arm_args(base, arm, 30, bonus))
        assert {k for k in big if big[k] != small[k]} == {
            'experiment_id', 'seed', 'task', 'total_timesteps'}
        assert big['task'] == task and big['total_timesteps'] == 15_000_000
        assert big.get('rule_bank') == small.get('rule_bank')     # unchanged


def test_long_seeds_are_fresh():
    long_suites = ('dk16_long', 'mr10_long', 'kc_s4_long', 'kc_s5_long',
                   'kc_s6_long', 'dk16_rl_long', 'mr10_rl_long',
                   'kc_s4_rl_long')
    mine = {c['seed'] for s in long_suites for c in fw.suite_cells(s)}
    others = {c['seed'] for s in fw.SUITES if s not in long_suites
              for c in fw.suite_cells(s)}
    assert not mine & others


def test_ten_room_multiroom_keeps_the_student_observation():
    from envs.registry import build_env
    small = build_env('multiroom_n6', seed=1, obs_mode='symbolic')
    big = build_env('multiroom_n10', seed=1, obs_mode='symbolic')
    assert (small.observation_space.shape == big.observation_space.shape
            == (7, 7, 3))
    big.reset(seed=1)
    assert len(big.unwrapped.rooms) == 10 and big.unwrapped.max_steps == 200
    small.close()
    big.close()


def test_ten_room_multiroom_arm_trains_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('mr10_long', 'rules_weak', 50, 'none'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['args']['task'] == 'multiroom_n10'
    assert summary['latest']['advising']['num_delivered'] > 0


# ---------- addendum 10: progress-only ablation; addendum 11: S5R3 / S6R3

def test_noprogress_bank_is_the_offline_counterfactual_exactly():
    from scripts import analyze_executable_teaching_20261002 as offline
    from scripts import progress_ablation_20261002 as pa
    from teachers.minigrid.rule_bank import load_bank
    assert pa.check_banks()
    source = json.loads((ROOT / fw.MEM_BANK).read_text(encoding='utf-8'))
    _mode, rules, _replay = load_bank(ROOT / fw.NOPROG_BANK)
    assert rules == offline.parsed_rules(source, strip_progress=True)
    assert rules != offline.parsed_rules(source, strip_progress=False)


def test_progress_ablation_cells_differ_from_their_twins_only_in_the_bank():
    twins = {(c['bonus'], c['seed']): c['args']
             for c in fw.suite_cells('kc_mem_confirm')
             if c['arm'] == 'rules_mem_weak'}
    cells = fw.suite_cells('kc_prog')
    assert len(cells) == 20
    for c in cells:
        twin = twins[(c['bonus'], c['seed'])]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed == {'experiment_id', 'rule_bank', 'rule_bank_sha256'}
        assert c['args']['rule_bank'] == fw.NOPROG_BANK


@pytest.mark.parametrize('suite, task', [('kc_s5_long', 'keycorridor_s5r3'),
                                         ('kc_s6_long',
                                          'keycorridor_s6r3_babyai')])
@pytest.mark.parametrize('arm, base', [('none', 'kc_confirm'),
                                       ('rules_mem_weak', 'kc_mem_confirm')])
def test_larger_kc_cells_change_only_map_seed_and_horizon(suite, task, arm,
                                                          base):
    for bonus in ('none', 'count'):
        big = asdict(fw.arm_args(suite, arm, fw.SUITES[suite][2][0], bonus))
        small = asdict(fw.arm_args(base, arm, 30, bonus))
        assert {k for k in big if big[k] != small[k]} == {
            'experiment_id', 'seed', 'task', 'total_timesteps'}
        assert big['task'] == task and big['total_timesteps'] == 15_000_000


def test_larger_kc_maps_keep_the_student_observation():
    from envs.registry import build_env
    for task in ('keycorridor_s5r3', 'keycorridor_s6r3_babyai'):
        env = build_env(task, seed=1, obs_mode='symbolic')
        assert env.observation_space.shape == (7, 7, 3)
        env.close()


def test_s6_memory_arm_trains_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc_s6_long', 'rules_mem_weak', 70, 'count'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1)
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['args']['task'] == 'keycorridor_s6r3_babyai'


# ---------- addenda 12-14: content control, dense early evaluation, crossover

CONFIRMED = {'doorkey_8x8': ('dk_confirm', 'rules_weak'),
             'multiroom_n6': ('mr_confirm', 'rules_weak'),
             'keycorridor_s3r3': ('kc_mem_confirm', 'rules_mem_weak')}


def _twins(task):
    suite, arm = CONFIRMED[task]
    return {(c['bonus'], c['seed']): c['args'] for c in fw.suite_cells(suite)
            if c['arm'] == arm}


def test_content_control_banks_are_their_sources_with_deranged_actions():
    from scripts import content_control_banks_20261004 as cc
    assert cc.check_banks()
    assert cc.BANKS == fw.CC_BANKS
    assert all(cc.SIGMA[a] != a for a in cc.SIGMA)


@pytest.mark.parametrize('suite', ('dk_cc', 'mr_cc', 'kc_cc'))
def test_content_cells_differ_from_their_twins_only_in_bank_and_frames(
        suite):
    twins = _twins(fw.SUITES[suite][0])
    cells = fw.suite_cells(suite)
    assert len(cells) == 20
    for c in cells:
        twin = twins[(c['bonus'], c['seed'])]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed == {'experiment_id', 'rule_bank', 'rule_bank_sha256',
                           'diagnostic_eval_frames'}
        assert c['args']['rule_bank'] == fw.CC_BANKS[twin['rule_bank']]
        assert c['args']['diagnostic_eval_frames'] == fw.DENSE_FRAMES


@pytest.mark.parametrize('task', ('doorkey_8x8', 'multiroom_n6',
                                  'keycorridor_s3r3'))
def test_deranged_bank_labels_the_same_states_with_deranged_targets(task):
    """Coverage and abstention are unchanged; only the target moves."""
    import numpy as np
    from envs.registry import build_env
    from scripts import content_control_banks_20261004 as cc
    from teachers.minigrid.rule_bank import RuleBankTeacher
    deranged_banks = {c['args']['rule_bank'] for s in fw.SUITES
                      if s.endswith('_cc') and fw.SUITES[s][0] == task
                      for c in fw.suite_cells(s)}
    rng = np.random.default_rng(0)
    labelled = 0
    for deranged in deranged_banks:
        source = next(k for k, v in fw.CC_BANKS.items() if v == deranged)
        a = RuleBankTeacher(ROOT / source)
        b = RuleBankTeacher(ROOT / deranged)
        for episode in range(6):
            env = build_env(task, seed=episode, obs_mode='symbolic')
            env.reset(seed=episode)
            u = env.unwrapped
            for _ in range(120):
                image = u.gen_obs()['image']
                act_a, status_a = a._advise(image, u)
                act_b, status_b = b._advise(image, u)
                assert status_a == status_b
                if act_a is None:
                    assert act_b is None
                else:
                    labelled += 1
                    assert act_b == cc.SIGMA[int(act_a)] != act_a
                _, _, done, trunc, _ = env.step(int(rng.integers(6)))
                if done or trunc:
                    break
            env.close()
    assert labelled > 0


@pytest.mark.parametrize('suite, base', [('dk_dense', 'dk_confirm'),
                                         ('mr_dense', 'mr_confirm'),
                                         ('kc_dense', 'kc_confirm')])
def test_dense_cells_equal_the_confirmed_cells_but_the_extra_frames(suite,
                                                                   base):
    originals = {(c['bonus'], c['arm'], c['seed']): c['args']
                 for s in (base, 'kc_mem_confirm') for c in fw.suite_cells(s)}
    cells = fw.suite_cells(suite)
    assert len(cells) == 40
    for c in cells:
        twin = originals[(c['bonus'], c['arm'], c['seed'])]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed == {'experiment_id', 'diagnostic_eval_frames'}


def test_dense_frames_start_at_zero_and_fill_the_first_gap():
    from monitoring.evaluation_diagnostics import diagnostic_milestones
    args = fw.arm_args('dk_dense', 'rules_weak', 30, 'none')
    batch = args.num_envs * args.num_steps
    iterations = args.total_timesteps // batch
    points = diagnostic_milestones(fw.DENSE_FRAMES, batch, iterations)
    assert 0 in points
    frames = [f for v in points.values() for f in v]
    assert all(f < args.eval_interval * batch for f in frames)
    assert all(f % batch == 0 for f in frames)        # whole rollouts
    assert len(frames) == len(set(frames)) == 9


def test_crossover_cells_swap_only_the_bank():
    twins = _twins('multiroom_n6')
    own = {bonus: next(v['rule_bank'] for (b, _), v in twins.items()
                       if b == bonus) for bonus in ('none', 'count')}
    assert own['none'] != own['count']
    for c in fw.suite_cells('mr_cross'):
        twin = twins[(c['bonus'], c['seed'])]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed == {'experiment_id', 'rule_bank', 'rule_bank_sha256',
                           'diagnostic_eval_frames'}
        other = 'count' if c['bonus'] == 'none' else 'none'
        assert c['args']['rule_bank'] == own[other]


def test_crossover_refuses_other_tasks():
    with pytest.raises(ValueError, match='MultiRoom'):
        fw.arm_args('dk_cc', fw.XBANK_ARM, 30, 'none')


def test_reproduction_compares_curves_not_only_areas(capsys):
    def m(auc, curve, labels=7, initial='x'):
        return dict(auc=auc, initial_sha256=initial, final=0.0,
                    curve=curve, labels_delivered=labels)
    a, b = [[0, 0.0, 0.0], [10, 1.0, 1.0]], [[0, 1.0, 1.0], [10, 0.0, 0.0]]
    groups = {'dense': {('t', 'none', 'none'): {
                  1: m(.5, a), 2: m(.5, b), 3: m(.7, a, labels=8)}},
              'confirm': {('t', 'none', 'none'): {
                  1: m(.5, a), 2: m(.5, a), 3: m(.6, a), 4: m(.9, a)}}}
    out = fw.reproduction(groups)
    # seed 2 has the same area but a different curve: not reproduced
    assert out == dict(total=3, same_auc=2, same_initial=3, same_curve=2,
                       same_labels=2,
                       max_abs_auc_diff=pytest.approx(.1))
    assert 'informational' in capsys.readouterr().out
    assert fw.reproduction({'dense': {}}) is None


def test_fresh_packages_are_140_core_and_80_optional_without_overlap():
    core = [c for s in fw.FRESH_CORE for c in fw.suite_cells(s)]
    optional = [c for s in fw.FRESH_OPTIONAL for c in fw.suite_cells(s)]
    assert (len(core), len(optional)) == (140, 80)
    keys = [(c['task'], c['bonus'], c['arm'], c['seed'])
            for c in core + optional]
    assert len(set(keys)) == 220                       # no cell twice
    assert {fw.GROUP[s] for s in fw.FRESH_CORE + fw.FRESH_OPTIONAL} == {
        'fresh'}
    assert set(fw.FRESH_CORE + fw.FRESH_OPTIONAL) <= set(fw.DENSE_SUITES)
    # every optional arm pairs with core controls of the same seeds
    core_seeds = {(c['task'], c['bonus'], c['seed']) for c in core
                  if c['arm'] == 'none'}
    assert {(c['task'], c['bonus'], c['seed']) for c in optional} <= \
        core_seeds
    assert {c['arm'] for c in core} == {'none', 'rules_weak',
                                        'rules_mem_weak',
                                        'rules_mem_weak_noprog'}
    assert {c['arm'] for c in optional} == {'rules_weak_cc',
                                            'rules_mem_weak_cc',
                                            fw.XBANK_ARM}


def test_fresh_seeds_are_unused_by_every_other_suite_and_the_rule_bank():
    fresh = fw.FRESH_CORE + fw.FRESH_OPTIONAL
    mine = {c['seed'] for s in fresh for c in fw.suite_cells(s)}
    others = {c['seed'] for s in fw.SUITES if s not in fresh
              for c in fw.suite_cells(s)}
    assert len(mine) == 30 and not mine & others
    for suite in fresh:
        task = fw.SUITES[suite][0]
        seed0 = rb.SPECS[fw.BASE_SPEC[task]]['seed0']
        reps = {(c['seed'] - seed0) // 100 for c in fw.suite_cells(suite)}
        assert reps == set(range(80, 90))
        # the rule-bank study ran replicates 0-19 (valid-action KC 20-29)
        assert min(reps) >= 30


@pytest.mark.parametrize('suite, base', [('dk_fresh', 'dk_confirm'),
                                         ('mr_fresh', 'mr_confirm'),
                                         ('kc_fresh', 'kc_confirm'),
                                         ('dk_fresh_cc', 'dk_confirm'),
                                         ('mr_fresh_cc', 'mr_confirm'),
                                         ('kc_fresh_cc', 'kc_confirm'),
                                         ('mr_fresh_cross', 'mr_confirm')])
def test_fresh_cells_equal_their_historical_twins_but_seed_and_frames(
        suite, base):
    historical = {}
    for s in (base, 'kc_mem_confirm', 'kc_prog', 'dk_cc', 'mr_cc', 'kc_cc',
              'mr_cross'):
        for c in fw.suite_cells(s):
            if c['task'] == fw.SUITES[suite][0] and c['replicate'] == 30:
                historical[(c['bonus'], c['arm'])] = c['args']
    for c in fw.suite_cells(suite):
        if c['replicate'] != 80:
            continue
        twin = historical[(c['bonus'], c['arm'])]
        changed = {k for k in twin if twin[k] != c['args'][k]}
        assert changed <= {'experiment_id', 'seed', 'diagnostic_eval_frames'}
        assert {'experiment_id', 'seed'} <= changed
        assert c['args']['diagnostic_eval_frames'] == fw.DENSE_FRAMES


def test_content_control_arm_trains_end_to_end_with_dense_frames(
        tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc_cc', 'rules_mem_weak_cc', 30, 'count'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1,
                   diagnostic_eval_frames='0,1024')
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    rows = [json.loads(line) for line in
            (run / 'diagnostic_evaluations.jsonl').read_text().splitlines()]
    assert [r['requested_frames'] for r in rows] == [[0], [1024]]
    assert rows[0]['global_step'] == 0 and rows[0]['rng_isolated']


def test_crossover_arm_trains_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('mr_cross', fw.XBANK_ARM, 30, 'none'),
                   total_timesteps=2048, eval_interval=1, eval_episodes=1,
                   diagnostic_eval_frames='')
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['args']['rule_bank'].endswith('blind_strict.json')


@pytest.mark.parametrize('suite, arm, bonus', [
    ('kc_fresh', 'rules_mem_weak_noprog', 'none'),
    ('mr_fresh_cross', 'rules_weak_xbank', 'count')])
def test_dense_fresh_run_passes_the_workers_terminal_checker(
        tmp_path, monkeypatch, suite, arm, bonus):
    """The cluster worker validates every run with validate(); a run with
    dense evaluations must pass it unchanged, keep its regular evaluations
    in evaluations.jsonl and its extra ones in their own file."""
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args(suite, arm, 80, bonus),
                   total_timesteps=4096, eval_interval=2, eval_episodes=1,
                   diagnostic_eval_frames='0,1024')
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    metrics = fw.validate(run, dict(trainer='algos.ppo_distill',
                                    args=asdict(args)))
    regular = [json.loads(line) for line in
               (run / 'evaluations.jsonl').read_text().splitlines()]
    extra = [json.loads(line) for line in
             (run / 'diagnostic_evaluations.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in regular] == [2048, 4096]
    assert [r['requested_frames'] for r in extra] == [[0], [1024]]
    assert [c[0] for c in metrics['curve']] == [2048, 4096]
    assert 0.0 <= metrics['auc'] <= 1.0 and metrics['initial_sha256']


def test_incomplete_declared_families_are_provisional():
    title = 'addendum 15 primary, fresh seeds, rules - none'
    assert fw.provisional(title, 6) is None
    assert 'PROVISIONAL: 4 of 6' in fw.provisional(title, 4)
    # every declared contrast present but some with too few seed pairs
    cells = [(t, b) for t in ('doorkey_8x8', 'multiroom_n6',
                              'keycorridor_s3r3') for b in ('none', 'count')]
    full = [dict(task=t, bonus=b, n=10) for t, b in cells]
    assert fw.provisional(title, full) is None
    short = full[:4] + [dict(c, n=2) for c in full[4:]]
    assert '2 with fewer than 10 seed pairs' in fw.provisional(title, short)
    # six contrasts, but one cell twice: still provisional
    assert '1 repeated' in fw.provisional(title, full[:5] + full[4:5])


def test_rlingua_families_count_only_the_confirmed_arm_of_each_task():
    # Expected families: DoorKey absent, KeyCorridor with both banks.
    rows = [dict(task=t, bonus=b, contrast=f'{arm} - rlingua_view', n=10,
                 p=0.001, mean=0.5, ci95=[0.4, 0.6], positive=10)
            for t, arm in (('multiroom_n6', 'rules_weak'),
                           ('keycorridor_s3r3', 'rules_weak'),
                           ('keycorridor_s3r3', 'rules_mem_weak'))
            for b in ('none', 'count')]
    family = fw.family_members(rows, fw.RL_VIEW_FAMILY, confirmed_only=True)
    assert {(c['task'], c['contrast']) for c in family} == {
        ('multiroom_n6', 'rules_weak - rlingua_view'),
        ('keycorridor_s3r3', 'rules_mem_weak - rlingua_view')}
    title = [t for t in fw.DECLARED_FAMILY_SIZE
             if t.startswith('addendum 16 B')][0]
    assert 'PROVISIONAL: 4 of 6' in fw.provisional(title, family)
    # older families keep counting every named contrast
    assert len(fw.family_members(rows, fw.RL_VIEW_FAMILY)) == 6
    # older families keep their historical output unchanged
    assert fw.provisional('primary family, rules_weak - none', 1) is None


def test_every_declared_family_title_is_reported():
    import ast
    import inspect
    import textwrap
    # adjacent string literals are one constant in the syntax tree
    tree = ast.parse(textwrap.dedent(inspect.getsource(fw.report)))
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert set(fw.DECLARED_FAMILY_SIZE) <= strings


# ---------- addendum 16: the RLingua baseline

RL_SUITES = ('dk_rl', 'mr_rl', 'kc_rl', 'dk_rlfull', 'mr_rlfull',
             'kc_rlfull', 'dk16_rl_long', 'mr10_rl_long', 'kc_s4_rl_long')
RLEX_SUITES = ('dk_rlex', 'mr_rlex', 'kc_rlex')     # addendum 17


@pytest.mark.parametrize('suite', RL_SUITES + RLEX_SUITES)
def test_rlingua_cells_are_the_no_teacher_cell_plus_rlingua_settings(suite):
    for c in fw.suite_cells(suite):
        none = asdict(fw.arm_args(suite, 'none', c['replicate'], c['bonus']))
        changed = {k for k in none if none[k] != c['args'][k]}
        assert changed <= {'experiment_id'} | fw.RLINGUA_KEYS
        a = c['args']
        assert not a['guidance'] and a['execute_teacher_start'] == 0.0
        assert (a['rlingua_p0'], a['rlingua_decay'], a['rlingua_bc_coef']) \
            == (0.25, 0.999999, 1.0)                 # RLingua Table A-II
        assert a['rlingua_variant'] == fw.RLINGUA_ARMS[c['arm']]
        family = fw.RLINGUA_FAMILY_TASK[fw.TRANSFER.get(c['task'],
                                                        c['task'])]
        assert a['rlingua_controller'].endswith(
            f"{family}_{a['rlingua_variant']}.py")


def test_rlingua_controllers_match_their_receipts():
    import hashlib
    for family in ('dk', 'mr', 'kc'):
        for variant in ('full', 'view'):
            path = ROOT / fw.RLINGUA_DIR / f'{family}_{variant}.py'
            receipt = json.loads(path.with_name(
                f'{family}_{variant}_receipt.json').read_text(
                    encoding='utf-8'))
            digest = hashlib.sha256(path.read_bytes().replace(
                b'\r\n', b'\n')).hexdigest()
            assert receipt['controller_sha256'] == digest
            assert receipt['model'].startswith('gpt-5-mini')


@pytest.mark.parametrize('suite, arm, bonus', [
    ('kc_rlfull', 'rlingua_full', 'none'),
    ('mr_rl', 'rlingua_view', 'count'),
    ('kc_rlex', 'rlingua_view_ex', 'none')])
def test_rlingua_arm_trains_end_to_end_and_passes_the_checker(
        tmp_path, monkeypatch, suite, arm, bonus):
    torch.set_num_threads(1)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args(suite, arm, 30, bonus),
                   total_timesteps=4096, eval_interval=2, eval_episodes=1,
                   rlingua_controller=str(
                       ROOT / fw.arm_args(suite, arm, 30, bonus)
                       .rlingua_controller))
    ppo.train(args)
    run = next((tmp_path / 'results/runs').iterdir())
    summary = json.loads((run / 'run_summary.json').read_text())
    assert summary['status'] == 'completed'
    stats = summary['latest']['rlingua']
    assert stats['total_executed'] > 0 and stats['buffer'] > 0
    assert stats['bc_loss_mean'] is not None
    metrics = fw.validate(run, dict(trainer='algos.ppo_distill',
                                    args=asdict(args)))
    assert [c[0] for c in metrics['curve']] == [2048, 4096]


def test_rlingua_refuses_to_mix_with_rule_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    args = replace(fw.arm_args('kc_confirm', 'rules_weak', 30, 'none'),
                   total_timesteps=2048,
                   rlingua_controller=str(
                       ROOT / fw.RLINGUA_DIR / 'kc_full.py'))
    with pytest.raises(ValueError, match='RLingua'):
        ppo.train(args)


def test_rlingua_families_are_declared_and_reported():
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fw.report)))
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    titles = [t for t in fw.DECLARED_FAMILY_SIZE if 'addendum 16' in t]
    assert len(titles) == 5 and set(titles) <= strings
    assert sum('as published' in t for t in titles) == 1


def test_rlingua_full_state_runs_only_on_the_confirmation_seeds():
    for suite in RL_SUITES:
        arms = fw.ARMS_OF[suite]
        if suite.endswith('_rlfull'):
            assert arms == ('rlingua_full',)
            assert fw.SUITES[suite][2] == range(30, 40)
        else:
            assert arms == ('rlingua_view',)


# ---------- addendum 17: RLingua B with the example-informed controllers

@pytest.mark.parametrize('suite', RLEX_SUITES)
def test_rlex_cells_equal_the_rl_cells_except_the_controller(suite):
    rl = suite.replace('_rlex', '_rl')
    assert fw.SUITES[suite] == fw.SUITES[rl]
    assert fw.GROUP[suite] == fw.GROUP[rl] == 'confirm'
    assert fw.ARMS_OF[suite] == ('rlingua_view_ex',)
    old = {(c['bonus'], c['replicate']): c['args']
           for c in fw.suite_cells(rl)}
    cells = fw.suite_cells(suite)
    assert len(cells) == 20 and len(old) == 20
    for c in cells:
        a, b = c['args'], old[(c['bonus'], c['replicate'])]
        assert {k for k in a if a[k] != b[k]} == {
            'experiment_id', 'rlingua_controller',
            'rlingua_controller_sha256'}
        assert '/checks_20261005/examples/' in a['rlingua_controller']
        assert a['rlingua_variant'] == 'view'


def test_rlex_controllers_match_their_receipts_and_are_archived():
    import hashlib
    directory = fw.RLINGUA_ARM_DIR['rlingua_view_ex']
    for family in ('dk', 'mr', 'kc'):
        path = ROOT / directory / f'{family}_view.py'
        receipt = json.loads(path.with_name(
            f'{family}_view_receipt.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes().replace(
            b'\r\n', b'\n')).hexdigest()
        assert receipt['controller_sha256'] == digest
        assert f'{directory}{family}_view.py' in fw.REQUIRED


def test_rlex_family_counts_only_the_confirmed_arm_and_is_reported():
    import inspect
    rows = [dict(task=t, bonus=b, contrast=f'{arm} - rlingua_view_ex', n=10,
                 p=0.001, mean=0.5, ci95=[0.4, 0.6], positive=10)
            for t, arm in (('doorkey_8x8', 'rules_weak'),
                           ('multiroom_n6', 'rules_weak'),
                           ('keycorridor_s3r3', 'rules_weak'),
                           ('keycorridor_s3r3', 'rules_mem_weak'))
            for b in ('none', 'count')]
    family = fw.family_members(rows, fw.RL_VIEW_EX_FAMILY,
                               confirmed_only=True)
    assert len(family) == 6
    assert ('keycorridor_s3r3', 'rules_weak - rlingua_view_ex') not in {
        (c['task'], c['contrast']) for c in family}
    assert "('addendum 16', 'addendum 17,')" in inspect.getsource(fw.report)


def test_rlingua_traces_stop_before_controller_steps():
    from algos.distill_bonus import intrinsic_gae
    torch.manual_seed(0)
    steps, envs, gamma, lam = 6, 3, 0.99, 0.95
    rewards, values = torch.rand(steps, envs), torch.rand(steps, envs)
    dones = (torch.rand(steps, envs) < 0.2).float()
    next_value, next_done = torch.rand(1, envs), torch.zeros(envs)
    plain = ppo.extrinsic_gae(rewards, values, dones, next_value,
                              next_done, gamma, lam)
    # keep=None and keep=all-ones are the old GAE, bit for bit
    assert torch.equal(plain, ppo.extrinsic_gae(
        rewards, values, dones, next_value, next_done, gamma, lam,
        torch.ones(steps, envs)))
    keep = torch.ones(steps, envs)
    keep[3, 1] = 0.0                       # the controller acted at t = 3
    cut = ppo.extrinsic_gae(rewards, values, dones, next_value, next_done,
                            gamma, lam, keep)
    delta2 = (rewards[2, 1] + gamma * values[3, 1] * (1 - dones[3, 1])
              - values[2, 1])
    assert torch.isclose(cut[2, 1], delta2)          # bootstraps, no trace
    assert torch.equal(cut[3:], plain[3:])           # later steps unchanged
    assert torch.equal(cut[:, [0, 2]], plain[:, [0, 2]])
    adv, _ = intrinsic_gae(rewards, values, next_value[0], gamma, lam)
    adv_cut, _ = intrinsic_gae(rewards, values, next_value[0], gamma, lam,
                               keep=keep)
    assert torch.isclose(adv_cut[2, 1],
                         rewards[2, 1] + gamma * values[3, 1] - values[2, 1])
    assert torch.equal(adv_cut[:, [0, 2]], adv[:, [0, 2]])


def test_rlingua_controller_instances_are_per_env_and_per_episode(tmp_path):
    from envs.registry import build_env
    from teachers.minigrid.rlingua_controller import RLinguaController
    src = tmp_path / 'c.py'
    src.write_text(
        'def controller(obs, memory):\n'
        '    controller.calls = getattr(controller, "calls", 0) + 1\n'
        '    memory["seen"] = obs["last_action"]\n'
        '    return min(controller.calls, 5)\n', encoding='utf-8')
    env = build_env('doorkey_8x8', seed=0, obs_mode='symbolic')
    env.reset(seed=0)
    u = env.unwrapped
    rl = RLinguaController(src, 'view', num_envs=2)
    assert [rl.act(0, u) for _ in range(3)] == [1, 2, 3]
    assert rl.act(1, u) == 1               # env 1 has its own instance
    rl.executed(0, 4)                      # the student's action ran
    rl.act(0, u)
    assert rl.memories[0]['seen'] == 4     # the controller is told so
    rl.reset(0)                            # a new episode in env 0
    assert rl.act(0, u) == 1 and rl.memories[0]['seen'] is None
    assert rl.act(1, u) == 2               # env 1 untouched
    env.close()
