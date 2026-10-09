"""Four explanation formats: bank, checks, controls, schedule, trainer paths.

Cases are built from live DoorKey environments with the same fields and
strict-foil rule as the contrastive lesson panel, so the tests need no
synced data and make no API call.
"""

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import minigrid  # noqa: F401  (registers environments)
import numpy as np
import pytest
import torch

from algos import ppo_distill as ppo
from algos import ppo_lesson_formats as formats_trainer
from scripts import calibrate_explanation_formats as calibration
from scripts import collect_explanation_formats as collection
from scripts import explanation_formats as ef
from scripts import plan_explanation_formats as plan
from scripts.contrastive_lessons import geometry, reference, transition
from scripts.explanation_screen_panel import file_hash
from teachers.budget import BudgetError, CostLedger, PriceTable

N_CASES = 16


def phase_of(env):
    keys = [c for c in env.grid.grid if c is not None and c.type == 'key']
    door = next(c for c in env.grid.grid if c is not None and c.type == 'door')
    if env.carrying is None and keys:
        return 'seek_key'
    return 'unlock_door' if door.is_locked else 'reach_target'


def state_of(env):
    return dict(full_grid=env.grid.encode().tolist(),
                agent_pos=[int(v) for v in env.agent_pos],
                agent_dir=int(env.agent_dir),
                carrying=(['key', env.carrying.color] if env.carrying
                          else None),
                mission=env.mission,
                local_obs=env.gen_obs()['image'].tolist())


def make_panel(n=N_CASES):
    cases = []
    for seed in range(200):
        env = gym.make('MiniGrid-DoorKey-8x8-v0').unwrapped
        env.reset(seed=seed)
        for _ in range((seed * 7) % 25):
            ref = next((r for f in range(7)
                        if (r := reference(state_of(env), f))), None)
            if ref is None:
                break
            _obs, _r, term, _t, _i = env.step(ref['positive_action'])
            if term:
                break
        state = state_of(env)
        ref = next((reference(state, f) for f in (2, 5, 3, 4, 0, 1)
                    if reference(state, f)), None)
        if ref is None:
            continue
        i = len(cases)
        cases.append(dict(state=state, phase=phase_of(env),
                          episode_group=[f'seed{seed}', 0, 0], **ref,
                          case_id=f'lesson_{i:03d}',
                          split='audit' if i % 4 == 3 else 'train'))
        env.close()
        if len(cases) == n:
            break
    return dict(study='synthetic_panel', cases=cases)


@pytest.fixture(scope='module')
def panel():
    return make_panel()


def answer_for(case, unknown=()):
    truth = ef.true_consequences(case)
    target, predicate = ef.PHASE_SUBGOAL[case['phase']]
    plan = ef.plan_truth(case)
    answer = dict(
        plain_prose=f"plain {case['case_id']}",
        contrastive_endorsed=f"better {case['case_id']}",
        contrastive_foil=f"worse {case['case_id']}",
        subgoal_target=target, subgoal_relation='approach',
        subgoal_predicate=predicate,
        future_plan=f"then the {plan['next_target']} {case['case_id']}")
    for key, value in truth.items():
        answer[f'consequence_{key}'] = value
    for name, _ in ef.PLAN_FIELDS:
        answer[f'plan_{name}'] = plan[name]
    if 'plain' in unknown:
        answer['plain_prose'] = 'unknown'
    if 'subgoal' in unknown:
        answer['subgoal_target'] = 'unknown'
    if 'plan' in unknown:
        answer['plan_next_direction'] = 'unknown'
    return answer


def fake_embed(texts, dimension=16):
    out = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=dimension)
        out.append(v / np.linalg.norm(v))
    return out


def valid_replies(panel, unknown_cases=()):
    return {c['case_id']: dict(status='valid', answer=answer_for(
        c, unknown=('plain',) if c['case_id'] in unknown_cases else ()))
        for c in panel['cases']}


# ------------------------------------------------------------ request/parse

def hidden_case(panel):
    """A case whose next-subgoal object the student cannot see."""
    return next(c for c in panel['cases']
                if ef.plan_truth(c)['next_hidden'])


def object_line(case, kind):
    grid = np.asarray(case['state']['full_grid'])
    x, y = map(int, np.argwhere(grid[:, :, 0] == ef.OBJECT_INDEX[kind])[0])
    return f' {kind} at ({x}, {y})'


def test_full_state_is_the_default_access():
    assert ef.PRIMARY_ACCESS == 'full_state'
    assert formats_trainer.Args().format_access == 'full_state'


def test_full_state_request_shows_a_hidden_object_the_local_one_omits(
        panel):
    case = hidden_case(panel)
    kind = ef.plan_truth(case)['next_target']
    full = ef.request(case, 'full_state')['input'][0]['content']
    local = ef.request(case, 'local_only')['input'][0]['content']
    # The object is genuinely hidden: absent from the student's own cells.
    visible = {c[2] for c in ef.visible_cells(case)}
    assert kind not in visible
    assert object_line(case, kind) in full
    assert 'FULL MAP' in full and 'STUDENT VIEW MASK' in full
    assert '?' in full.split('STUDENT VIEW MASK')[1]
    assert 'need not stay' in full
    assert object_line(case, kind) not in local and 'FULL MAP' not in local
    assert json.dumps(ef.visible_cells(case)) in local


def test_both_access_conditions_keep_the_same_action_pair_and_schema(panel):
    case = panel['cases'][0]
    bodies = {a: ef.request(case, a) for a in ef.ACCESS}
    for body in bodies.values():
        text = body['input'][0]['content']
        assert (f"endorsed action {case['positive_action']} "
                f"({ef.ACTION_NAMES[case['positive_action']]})") in text
        assert (f"foil action {case['foil_action']} "
                f"({ef.ACTION_NAMES[case['foil_action']]})") in text
    schemas = [b['text']['format']['schema'] for b in bodies.values()]
    assert schemas[0] == schemas[1]
    assert not any(k == 'action' for k in schemas[0]['properties'])
    for key, spec in schemas[0]['properties'].items():
        if 'enum' in spec:
            assert 'unknown' in spec['enum'], key


def test_parse_marks_unknowns_and_rejects_schema_violations(panel):
    case = panel['cases'][0]
    parsed = ef.parse_reply(answer_for(case, unknown=('plain', 'subgoal',
                                                      'plan')))
    assert parsed['plain'] is None and parsed['subgoal'] is None
    assert parsed['plan'] is None
    assert parsed['contrastive'] and parsed['consequence']
    bad = answer_for(case)
    bad['consequence_endorsed_door'] = 'explodes'
    with pytest.raises(ValueError, match='enum'):
        ef.parse_reply(bad)
    missing = answer_for(case)
    missing.pop('future_plan')
    with pytest.raises(ValueError, match='fields'):
        ef.parse_reply(missing)


def test_native_consequences_agree_with_independent_exact_physics(panel):
    for case in panel['cases']:
        truth = ef.true_consequences(case)
        layout, start = geometry(case['state'])
        for role, action in zip(ef.ROLES, (case['positive_action'],
                                           case['foil_action'])):
            after = transition(layout, start, action)
            moved = after[:2] != start[:2]
            turn = (after[2] - start[2]) % 4
            expected_move = ('move_forward' if moved else
                             {1: 'turn_right', 3: 'turn_left'}.get(
                                 turn, 'no_move'))
            assert truth[f'{role}_movement'] == expected_move
            held_before, held_after = start[5], after[5]
            expected_inv = ('unchanged' if held_before == held_after else
                            'pick_up_key' if held_after else 'drop_key')
            assert truth[f'{role}_inventory'] == expected_inv


def test_scoring_checks_consequences_and_subgoals(panel):
    case = panel['cases'][0]
    good = ef.score_case(case, ef.parse_reply(answer_for(case)))
    assert all(good['consequence'].values())
    assert good['subgoal'] == dict(target=True, predicate=True)
    wrong = answer_for(case)
    wrong['consequence_endorsed_movement'] = next(
        m for m in ef.MOVEMENT if m not in (
            wrong['consequence_endorsed_movement'], 'unknown'))
    scored = ef.score_case(case, ef.parse_reply(wrong))
    assert scored['consequence']['endorsed_movement'] is False


# ------------------------------------------------------------------- freeze

def test_bank_uses_a_common_intersection_and_a_marginal_preserving_permutation(
        panel):
    excluded = {panel['cases'][0]['case_id']}
    bank = ef.freeze_bank(panel, valid_replies(panel, excluded), fake_embed)
    train_ids = {r['case_id'] for r in bank['train']}
    assert not train_ids & excluded
    assert bank['coverage']['train_excluded_by_intersection'] == 1
    assert all(r['split'] == 'audit' for r in bank['audit'])
    for row in bank['train']:
        assert row['permuted']['donor_id'] in train_ids
        donor = next(r for r in bank['train']
                     if r['case_id'] == row['permuted']['donor_id'])
        assert donor['episode'] != row['episode']
    for fmt in ef.FORMATS:
        aligned = sorted(ef.canonical(r['targets'][fmt])
                         for r in bank['train'])
        permuted = sorted(ef.canonical(r['permuted']['targets'][fmt])
                          for r in bank['train'])
        assert aligned == permuted
        assert bank['permuted_changed_fraction'][fmt] > 0
    assert bank['embedding_dimension'] == 16
    assert len(ef.audit_sheet(bank)) == len(bank['audit'])


# ------------------------------------------------------------- schedule

def test_default_schedule_reproduces_the_isolation_study():
    iterations = 10_000_000 // 1024
    isolation = [i for i in range(1, iterations + 1)
                 if i % 4 == 0 and i <= int(iterations * 0.75)]
    assert formats_trainer.lesson_schedule(iterations, 0, 4, 1830) == \
        isolation


def test_early_and_late_timing_are_matched():
    summary = plan.schedule_summary()
    assert summary['equal_length']
    assert summary['late'][0] > plan.ITERATIONS // 4
    assert summary['late'][1] < summary['teacher_free_from']
    with pytest.raises(ValueError, match='past the horizon'):
        formats_trainer.lesson_schedule(100, 90, 4, 10)


# ------------------------------------------------------------ real trainer

@pytest.fixture(scope='module')
def bank_file(panel, tmp_path_factory):
    bank = ef.freeze_bank(panel, valid_replies(panel), fake_embed)
    path = tmp_path_factory.mktemp('bank') / 'format_bank.json'
    path.write_text(json.dumps(bank))
    return path, file_hash(path)


def short_run(tmp_path, monkeypatch, bank_file, **changes):
    path, sha = bank_file
    values = dict(seed=5, total_timesteps=2048, num_steps=16,
                  update_epochs=1, eval_interval=0, eval_episodes=1,
                  format_bank=str(path), format_bank_sha256=sha,
                  lesson_batch=4, lesson_every=4, lesson_exposures=4,
                  experiment_id='formats_test')
    values.update(changes)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    torch.set_num_threads(1)
    ppo.train(formats_trainer.Args(**values),
              auxiliary=formats_trainer.FormatLessonAuxiliary())
    run = next((tmp_path / 'results/runs').iterdir())
    updates = [json.loads(l) for l in
               (run / 'lesson_updates.jsonl').read_text().splitlines()]
    finished = json.loads((run / 'lesson_finished.json').read_text())
    contract = json.loads((run / 'lesson_contract.json').read_text())
    return updates, finished, contract


@pytest.mark.parametrize('fmt', ef.FORMATS)
@pytest.mark.parametrize('targets', ['aligned', 'permuted'])
def test_every_format_trains_through_the_real_trainer(
        fmt, targets, tmp_path, monkeypatch, bank_file):
    updates, finished, contract = short_run(
        tmp_path, monkeypatch, bank_file, lesson_mode='explanation',
        lesson_format=fmt, lesson_targets=targets, lesson_scale=0.5)
    assert len(updates) == 4 and contract['schedule'] == [4, 8, 12, 16]
    assert all(u['shared_grad_norm'] > 0 for u in updates)
    assert finished['effective_action_exposures'] == 0
    assert finished['explanation_integral'] == pytest.approx(2.0)
    assert finished['initial_policy_sha256'] != \
        finished['final_policy_sha256']
    assert finished['audit_head_metrics']


def test_replay_stream_matches_across_modes_and_detached_equals_ppo(
        tmp_path, monkeypatch, bank_file):
    runs = {}
    for mode, extra in (('ppo', {}), ('detached', dict(lesson_scale=0.5)),
                        ('actions', dict(lesson_action_coef=1.0)),
                        ('combined', dict(lesson_scale=0.5,
                                          lesson_action_coef=1.0))):
        out = tmp_path / mode
        runs[mode] = short_run(out, monkeypatch, bank_file,
                               lesson_mode=mode, lesson_format='subgoal',
                               **extra)
    ids = {m: [u['ids'] for u in r[0]] for m, r in runs.items()}
    assert len({json.dumps(v) for v in ids.values()}) == 1
    final = {m: r[1]['final_policy_sha256'] for m, r in runs.items()}
    assert final['detached'] == final['ppo']
    assert final['actions'] != final['ppo']
    assert final['combined'] != final['actions']
    assert runs['actions'][1]['effective_action_exposures'] == 16
    assert runs['ppo'][1]['effective_action_exposures'] == 0


def test_mode_and_scale_consistency_is_enforced(tmp_path, monkeypatch,
                                                bank_file):
    with pytest.raises(ValueError, match='lesson_scale'):
        short_run(tmp_path, monkeypatch, bank_file,
                  lesson_mode='explanation', lesson_scale=0.0)
    with pytest.raises(ValueError, match='lesson_action_coef'):
        short_run(tmp_path / 'b', monkeypatch, bank_file,
                  lesson_mode='actions', lesson_action_coef=0.0)


def test_cached_categorical_lessons_serve_the_timing_arms(
        tmp_path, monkeypatch, panel):
    lessons = dict(train=[dict(case_id=c['case_id'],
                               observation=c['state']['local_obs'],
                               positive_action=c['positive_action'],
                               foil_action=c['foil_action'])
                          for c in panel['cases'] if c['split'] == 'train'],
                   audit=[])
    path = tmp_path / 'lessons.json'
    path.write_text(json.dumps(lessons))
    updates, finished, contract = short_run(
        tmp_path / 'run', monkeypatch, (path, file_hash(path)),
        lesson_mode='actions', lesson_action_coef=1.0, lesson_offset=4,
        lesson_exposures=2)
    assert contract['schedule'] == [8, 12]
    assert finished['action_integral'] == pytest.approx(2.0)


# ------------------------------------------------------------ calibration

def test_calibration_matches_reference_influence_at_the_real_init(
        tmp_path, monkeypatch, panel, bank_file):
    path, sha = bank_file
    bank = json.loads(path.read_text())
    lessons = dict(train=[dict(case_id=r['case_id'], positive=1, negative=2)
                          for r in bank['train']])
    lessons_path = tmp_path / 'lessons.json'
    lessons_path.write_text(json.dumps(lessons))
    size = len(bank['train'])
    result = calibration.calibrate(path, lessons_path, [5], size=size)
    norms = result['norms']['5']
    for fmt in ef.FORMATS:
        assert result['primary_scale'][fmt] * norms[fmt] == pytest.approx(
            norms['reference_codes'])
        assert result['sensitivity_scale'][fmt] == pytest.approx(
            3 * result['primary_scale'][fmt])
    _u, _f, contract = short_run(tmp_path / 'run', monkeypatch, bank_file,
                                 lesson_mode='ppo')
    assert contract['initial_policy_sha256'] == norms[
        'initial_policy_sha256']


# ------------------------------------------------------------- collection

class FakeClient:
    def __init__(self, panel, broken=()):
        self.by_prompt = {ef.prompt_text(c): c for c in panel['cases']}
        self.broken, self.calls = set(broken), []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **body):
        case = self.by_prompt[body['input'][0]['content']]
        self.calls.append(case['case_id'])
        text = ('not json' if case['case_id'] in self.broken
                else json.dumps(answer_for(case)))
        return SimpleNamespace(output_text=text, usage=SimpleNamespace(
            input_tokens=1000, output_tokens=500))


def test_collection_reserves_first_tries_once_and_settles(tmp_path, panel):
    prices = PriceTable.load(collection.PRICES)
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 50.0)
    bound = collection.price(panel, prices)
    assert bound['chat_bound_usd'] > 0
    assert collection.check_ledger(ledger, bound['total_bound_usd'])[
        'sufficient']
    broken = panel['cases'][1]['case_id']
    client = FakeClient(panel, broken=[broken])
    summary = collection.collect(panel, client, prices, ledger,
                                 tmp_path / 'out', workers=2)
    assert sorted(client.calls) == sorted(c['case_id']
                                          for c in panel['cases'])
    assert summary['failed'] == 1 and summary['valid'] == len(client.calls) - 1
    status = CostLedger(str(ledger)).status()
    assert not status['open_reservations']
    assert status['settled_usd'] == pytest.approx(summary['settled_usd'])
    rows = collection.replies_by_case(tmp_path / 'out' / 'raw_replies.jsonl')
    assert rows[broken]['status'] == 'failed' and rows[broken]['attempts'] == 1


def test_collection_refuses_an_uncovered_pool_before_any_call(tmp_path,
                                                              panel):
    prices = PriceTable.load(collection.PRICES)
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 0.01)
    client = FakeClient(panel)
    with pytest.raises(BudgetError):
        collection.collect(panel, client, prices, ledger, tmp_path / 'out')
    assert client.calls == []


# ---------------------------------------------------------- second teacher

def test_complementarity_bounds_are_consistent(panel):
    a = valid_replies(panel)
    b = {k: dict(v) for k, v in a.items()}
    flip = panel['cases'][0]
    wrong = answer_for(flip)
    wrong['subgoal_target'] = next(t for t in ef.SUBGOAL_TARGET
                                   if t not in (wrong['subgoal_target'],
                                                'unknown'))
    b[flip['case_id']] = dict(status='valid', answer=wrong)
    b[panel['cases'][2]['case_id']] = dict(status='failed')
    result = ef.complementarity(ef.checkable_units(panel, a),
                                ef.checkable_units(panel, b))
    assert result['oracle_chooser'] >= result['best_single']
    assert result['accuracy_a'] == 1.0 and result['only_b_right'] == 0.0
    assert result['coverage_b'] < 1.0
    both_right = result['oracle_chooser'] - result['only_a_right'] - \
        result['only_b_right']
    assert both_right + result['only_a_right'] + result['only_b_right'] + \
        result['both_wrong'] == pytest.approx(1.0)


# ---------------------------------------------------------------- planner

def test_menus_repeat_isolation_settings_and_resolve():
    scales = {f: 0.3 for f in ef.FORMATS}
    cells = plan.format_menu('bank.json', 'a' * 64, scales)
    assert len(cells) == 120
    conditions = {c['condition'] for c in cells}
    assert conditions == {'ppo', 'actions'} | {
        f'{f}_{t}' for f in ef.FORMATS for t in ('aligned', 'permuted')}
    assert {c['args']['format_access'] for c in cells} == {'full_state'}
    access = plan.access_menu('local.json', 'c' * 64, scales)
    assert len(access) == 50
    for cell in cells + access + plan.timing_menu('lessons.json', 'b' * 64):
        for key, value in plan.ISOLATION_SHARED.items():
            assert cell['args'][key] == value, key
        assert cell['args']['lesson_every'] == 4
    for cell in cells:
        if cell['condition'].endswith(('_aligned', '_permuted')):
            assert cell['args']['lesson_action_coef'] == 0.0
            assert cell['args']['lesson_scale'] == 0.3


def test_access_comparison_pairs_with_primary_cells_in_bank_only():
    scales = {f: 0.3 for f in ef.FORMATS}
    primary = {(c['seed'], c['bonus'], c['condition']): c['args']
               for c in plan.format_menu('bank.json', 'a' * 64, scales)}
    allowed = {'format_bank', 'format_bank_sha256', 'format_access',
               'experiment_id'}
    for cell in plan.access_menu('local.json', 'c' * 64, scales):
        partner = primary[(cell['seed'], cell['bonus'],
                           cell['condition'].replace('_local', ''))]
        differing = {k for k in partner if partner[k] != cell['args'][k]}
        assert differing <= allowed, differing
        assert cell['args']['format_access'] == 'local_only'


def interaction_cases():
    """States whose endorsed action picks up or unlocks, plus a foil drop."""
    found = {}
    for seed in range(60):
        env = gym.make('MiniGrid-DoorKey-8x8-v0').unwrapped
        env.reset(seed=seed)
        for _ in range(60):
            state = state_of(env)
            ref = next((r for f in range(7)
                        if (r := reference(state, f))), None)
            if ref is None:
                break
            best = ref['positive_action']
            if best in (3, 5) and best not in found:
                found[best] = dict(state=state, phase=phase_of(env),
                                   episode_group=[f's{seed}', 0, 0], **ref,
                                   case_id=f'x{best}', split='train')
            if env.carrying is not None and 'drop' not in found:
                drop = reference(state, 4)
                if drop:
                    found['drop'] = dict(state=state, phase=phase_of(env),
                                         episode_group=[f's{seed}', 1, 0],
                                         **drop, case_id='xdrop',
                                         split='train')
            _o, _r, term, _t, _i = env.step(best)
            if term:
                break
        env.close()
        if len(found) == 3:
            break
    return found


def test_native_consequences_cover_key_and_door_changes():
    found = interaction_cases()
    assert set(found) == {3, 5, 'drop'}
    assert ef.true_consequences(found[3])['endorsed_inventory'] == \
        'pick_up_key'
    assert ef.true_consequences(found[5])['endorsed_door'] == \
        'unlock_and_open'
    assert ef.true_consequences(found['drop'])['foil_inventory'] == \
        'drop_key'
    names = {0: 'open', 1: 'closed', 2: 'locked'}
    for case in found.values():
        truth = ef.true_consequences(case)
        layout, start = geometry(case['state'])
        for role, action in zip(ef.ROLES, (case['positive_action'],
                                           case['foil_action'])):
            after = transition(layout, start, action)
            before_door, after_door = names[start[6]], names[after[6]]
            expected = ('unchanged' if before_door == after_door else
                        'unlock_and_open' if before_door == 'locked'
                        else 'open' if after_door == 'open' else 'close')
            assert truth[f'{role}_door'] == expected
            held = ('unchanged' if start[5] == after[5] else
                    'pick_up_key' if after[5] else 'drop_key')
            assert truth[f'{role}_inventory'] == held



# --------------------------------------------- privileged teacher access

def test_plan_truth_matches_minigrid_view_coordinates(panel):
    """Independent egocentric check with MiniGrid's own view transform."""
    from scripts.explanation_screen_panel import restore
    checked = 0
    for case in panel['cases']:
        truth = ef.plan_truth(case)
        if truth['next_target'] == 'none':
            continue
        env = restore(case['state'])
        grid = np.asarray(case['state']['full_grid'])
        tx, ty = map(int, np.argwhere(
            grid[:, :, 0] == ef.OBJECT_INDEX[truth['next_target']])[0])
        vx, vy = env.get_view_coords(tx, ty)
        size = env.agent_view_size
        forward, right = (size - 1) - vy, vx - size // 2
        env.close()
        expected = (('ahead' if forward > 0 else 'behind')
                    if abs(forward) >= abs(right)
                    else ('right' if right > 0 else 'left'))
        assert truth['next_direction'] == expected
        checked += 1
    assert checked > 0


def test_request_and_bank_identities_differ_by_access(panel):
    ids = {a: ef.request_identity(panel, a) for a in ef.ACCESS}
    assert (ids['full_state']['request_sha256']
            != ids['local_only']['request_sha256'])
    assert {i['request_version'] for i in ids.values()} == {
        ef.REQUEST_VERSION}
    replies = valid_replies(panel)
    banks = {a: ef.freeze_bank(panel, replies, fake_embed, a, ids[a])
             for a in ef.ACCESS}
    assert banks['full_state']['study'] != banks['local_only']['study']
    assert banks['full_state']['information_access'] == 'full_state'
    assert (banks['local_only']['request_sha256']
            == ids['local_only']['request_sha256'])
    assert json.dumps(banks['full_state']) != json.dumps(banks['local_only'])
    with pytest.raises(ValueError, match='different access'):
        ef.freeze_bank(panel, replies, fake_embed, 'full_state',
                       ids['local_only'])


def test_future_subgoal_content_reaches_the_plan_target(panel):
    bank = ef.freeze_bank(panel, valid_replies(panel), fake_embed)
    rows = bank['train']
    hidden = [r for r in rows if r['checks']['plan_next_hidden']]
    assert hidden, 'the synthetic panel needs a hidden next subgoal'
    targets = formats_trainer.target_tensors(
        [r['targets'] for r in rows], 'plan', 'cpu')
    for name, values in ef.PLAN_FIELDS:
        for i, r in enumerate(rows):
            assert values[int(targets[name][i])] == r['targets']['plan'][name]
    case = next(c for c in panel['cases']
                if c['case_id'] == hidden[0]['case_id'])
    assert (hidden[0]['targets']['plan']['next_target']
            == ef.plan_truth(case)['next_target'])
    assert all(v for v in hidden[0]['checks']['plan'].values())
    assert bank['train_plan_next_hidden_fraction'] > 0
    assert 'future_plan' in ef.audit_sheet(bank)[0]


def test_paired_restriction_fixes_cases_and_donors_across_access(panel):
    full = valid_replies(panel)
    local = valid_replies(panel)
    dropped = panel['cases'][4]['case_id']
    local[dropped] = dict(status='valid', answer=answer_for(
        panel['cases'][4], unknown=('plan',)))
    both = ef.known_cases(panel, full) & ef.known_cases(panel, local)
    assert dropped not in both
    banks = {a: ef.freeze_bank(panel, r, fake_embed, a, restrict_to=both)
             for a, r in (('full_state', full), ('local_only', local))}
    ids = {a: [r['case_id'] for r in b['train']] for a, b in banks.items()}
    donors = {a: [r['permuted']['donor_id'] for r in b['train']]
              for a, b in banks.items()}
    assert ids['full_state'] == ids['local_only']
    assert donors['full_state'] == donors['local_only']
    coverage = banks['full_state']['coverage']
    assert (coverage.get('train_excluded_by_pairing', 0)
            + coverage.get('audit_excluded_by_pairing', 0)) == 1


def test_adapter_refuses_a_bank_of_the_other_access(tmp_path, monkeypatch,
                                                    bank_file):
    with pytest.raises(ValueError, match='information access differs'):
        short_run(tmp_path, monkeypatch, bank_file,
                  lesson_mode='explanation', lesson_format='plan',
                  lesson_scale=0.5, format_access='local_only')
