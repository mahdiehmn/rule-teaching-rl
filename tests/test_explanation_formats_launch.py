"""Explanation formats launcher: preparation, writers, gates, cells, chain.

Providers, git and sbatch are local fakes: no API call, no real ledger, no
archive of this repository and no Slurm submission. The real trainer runs
only at toy size, and the real panel is used only when it is available.
"""

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch

from algos import ppo_distill as ppo
from algos import ppo_lesson_formats as formats_trainer
from scripts import collect_explanation_formats as collection
from scripts import explanation_formats as ef
from scripts import run_explanation_formats_20260925 as run
from scripts.explanation_screen_panel import file_hash
from teachers.budget import BudgetError, CostLedger
from tests.test_explanation_formats import answer_for, fake_embed, make_panel

REPO = Path(__file__).resolve().parents[1]
REAL_SOURCE = (REPO.parent / 'vlm-rl-bench/results/vulcan_sync/data'
               '/explanation_lessons/contrastive_lessons_20260924_v1')
SMALL_GATES = dict(full_state=dict(train=8, audit=2),
                   local_only=dict(train=4, audit=1))


class FakeProvider:
    """Answers both writers like a careful teacher; 1,536-d embeddings.

    The local-only writer answers `unknown` for the plan whenever the next
    subgoal's object is outside the student's view.
    """

    def __init__(self, panel, fail=False):
        self.by_prompt = {ef.prompt_text(c, a): (c, a)
                          for c in panel['cases'] for a in ef.ACCESS}
        self.fail, self.calls, self.embedded = fail, [], []
        self.responses = SimpleNamespace(create=self.create)
        self.embeddings = SimpleNamespace(create=self.embed)

    def create(self, **body):
        case, access = self.by_prompt[body['input'][0]['content']]
        self.calls.append((access, case['case_id']))
        if self.fail:
            raise RuntimeError('provider unavailable')
        hidden = ef.plan_truth(case)['next_hidden']
        answer = answer_for(case, unknown=(
            ('plan',) if access == 'local_only' and hidden else ()))
        return SimpleNamespace(
            output_text=json.dumps(answer), status='completed',
            usage=SimpleNamespace(input_tokens=1200, output_tokens=600))

    def embed(self, model, input):
        self.embedded.append(len(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(map(float, v)))
                  for v in fake_embed(input, dimension=1536)],
            usage=SimpleNamespace(total_tokens=10 * len(input)))


@pytest.fixture(scope='module')
def panel():
    return make_panel()


def lessons_of(panel):
    """Cached categorical lessons for the same train cases."""
    rows = [dict(case_id=c['case_id'], observation=c['state']['local_obs'],
                 positive_action=c['positive_action'],
                 foil_action=c['foil_action'], positive=1, negative=2)
            for c in panel['cases'] if c['split'] == 'train']
    return dict(train=rows, audit=[])


def fake_git(monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as handle:
        for name in run.REQUIRED_SOURCES:
            handle.writestr(name, (REPO / name).read_text(encoding='utf-8')
                            if name == collection.PRICES
                            else 'Frozen synthetic source: ' + name)

    def git_output(command, **kwargs):
        assert command[0] == 'git'
        return 'a' * 40 if 'rev-parse' in command else archive.getvalue()

    monkeypatch.setattr(run.subprocess, 'check_output', git_output)
    monkeypatch.setattr(run.subprocess, 'run', lambda command, **kwargs:
                        SimpleNamespace(returncode=0))


@pytest.fixture
def world(tmp_path, monkeypatch, panel):
    """A synthetic source, repository, ledger and credential file."""
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'panel.json').write_text(json.dumps(panel))
    (source / 'lessons.json').write_text(json.dumps(lessons_of(panel)))
    monkeypatch.setattr(run, 'PANEL_SHA256', run.digest(source / 'panel.json'))
    monkeypatch.setattr(run, 'LESSONS_SHA256',
                        run.digest(source / 'lessons.json'))
    monkeypatch.setattr(run, 'GATES', SMALL_GATES)
    monkeypatch.setattr(run, 'CALIBRATION_SEEDS', [5])
    monkeypatch.setattr(run, 'CALIBRATION_SIZE', 4)
    # Real command-line parsing has its own test below.
    monkeypatch.setattr(run, 'check_commands', lambda cells: None)
    root = tmp_path / 'repo'
    (root / 'configs').mkdir(parents=True)
    (root / collection.PRICES).write_bytes((REPO / collection.PRICES)
                                           .read_bytes())
    ledger = tmp_path / 'ledger.json'
    CostLedger.initialize(str(ledger), 50.0)
    credential = tmp_path / 'credentials.env'
    credential.write_text('OPENAI_API_KEY=sk-test-not-a-key\n')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    fake_git(monkeypatch)
    return SimpleNamespace(source=source, root=root, ledger=ledger,
                           credential=credential, panel=panel)


def formats_batch(world, monkeypatch):
    batch = run.prepare_formats(world.root, world.source, world.ledger,
                                world.credential)
    monkeypatch.setattr(run, 'ROOT', batch / 'code')
    return batch


# ---------------------------------------------------------------- contracts

def test_timing_contract_is_matched_and_uses_only_cached_lessons():
    contract = run.timing_contract()
    cells = contract['cells']
    assert len(cells) == 30 and contract['api_calls'] == 0
    assert {c['args']['format_bank'] for c in cells} == {
        'corpora/lessons.json'}
    assert {c['args']['format_bank_sha256'] for c in cells} == {
        run.LESSONS_SHA256}
    arms = {c['condition']: c['args'] for c in cells}
    assert arms['actions_early']['lesson_offset'] == 0
    assert arms['actions_late']['lesson_offset'] == 2441
    assert {a['lesson_exposures'] for a in arms.values()} == {1220}
    assert contract['schedules']['equal_length'] is True
    run.check_commands(cells[:6] + run.placeholder_cells())


def test_wire_guard_allows_only_the_two_priced_endpoints():
    ok = httpx.Request('POST', collection.OPENAI_BASE + '/responses',
                       content=b'{}')
    assert collection.wire_guard(ok) is None
    collection.wire_guard(httpx.Request(
        'POST', collection.OPENAI_BASE + '/embeddings', content=b'{}'))
    for request in (
            httpx.Request('POST', collection.OPENAI_BASE + '/chat/completions',
                          content=b'{}'),
            httpx.Request('GET', collection.OPENAI_BASE + '/responses'),
            httpx.Request('POST', collection.OPENAI_BASE + '/responses',
                          content=b'x' * (collection.MAX_INPUT_TOKENS + 1))):
        with pytest.raises(BudgetError):
            collection.wire_guard(request)


def test_real_panel_resolves_to_the_reviewed_bounds_and_identities():
    if not (REAL_SOURCE / 'panel.json').is_file():
        pytest.skip('The real panel is synced locally only')
    assert run.digest(REAL_SOURCE / 'panel.json') == run.PANEL_SHA256
    assert run.digest(REAL_SOURCE / 'lessons.json') == run.LESSONS_SHA256
    prices = collection.PriceTable.load(str(REPO / collection.PRICES))
    contract = run.formats_contract(run.read(REAL_SOURCE / 'panel.json'),
                                    prices)
    ids = {a: i['request_sha256'] for a, i in contract['identities'].items()}
    assert ids['full_state'].startswith('caa16581')
    assert ids['local_only'].startswith('5fa5f7b9')
    assert contract['total_bound_usd'] == pytest.approx(4.8868, abs=1e-3)


# ------------------------------------------------------------- preparation

def test_formats_preparation_checks_funds_and_is_portable(world, monkeypatch):
    batch = formats_batch(world, monkeypatch)
    manifest = run.verify(batch)
    assert manifest['ledger'] == str(world.ledger.resolve())
    assert set(manifest['identities']) == set(ef.ACCESS)
    assert manifest['pairing'].startswith('none')
    assert not list((batch / 'cells').iterdir())
    assert not (batch / 'SUBMISSION_ATTEMPTED').exists()
    # Preparation reserves nothing: the pool is untouched.
    assert CostLedger(str(world.ledger)).status()['reserved_usd'] == 0
    assert 'sk-test' not in (batch / 'manifest.json').read_text()
    # A second preparation only verifies.
    assert run.prepare_formats(world.root, world.source, world.ledger,
                               world.credential) == batch


def test_formats_preparation_refuses_an_uncovered_pool(world):
    small = world.ledger.with_name('small.json')
    CostLedger.initialize(str(small), 0.05)
    with pytest.raises(BudgetError, match='Both writers'):
        run.prepare_formats(world.root, world.source, small, world.credential)
    assert not (world.root / 'results').exists()


@pytest.mark.parametrize('fault', ['source', 'manifest', 'panel'])
def test_verification_rejects_changed_sources_and_contracts(
        world, monkeypatch, fault):
    batch = formats_batch(world, monkeypatch)
    if fault == 'source':
        (batch / 'code' / run.RUNNER).write_text('Changed source.')
    elif fault == 'panel':
        (batch / 'source/panel.json').write_text('{}')
    else:
        manifest = run.read(batch / 'manifest.json')
        manifest['gates']['full_state']['train'] = 1
        run.write(batch / 'manifest.json', manifest)
        (batch / 'manifest.sha256').write_text(
            run.digest(batch / 'manifest.json'))
    with pytest.raises(ValueError):
        run.verify(batch)


# ----------------------------------------------------------------- writers

def test_writers_freeze_gate_calibrate_and_bind_cells(world, monkeypatch):
    batch = formats_batch(world, monkeypatch)
    provider = FakeProvider(world.panel)
    manifest = run.verify(batch)
    with pytest.raises(ValueError, match='No frozen full_state'):
        run.writer_stage(batch, 'local_only', client=provider)
    assert provider.calls == []

    full = run.writer_stage(batch, 'full_state', client=provider)
    assert full['gate']['passed'] and full['train_cases'] == 12
    assert len(provider.calls) == len(world.panel['cases'])
    status = CostLedger(str(world.ledger)).status()
    assert not status['open_reservations'] and not status['overspent_runs']
    settled = {e['run_id'] for e in run.read(world.ledger)['entries']}
    assert settled == {collection.run_id('full_state'),
                       collection.run_id('full_state') + '_embeddings'}
    calibration = run.read(batch / 'calibration.json')
    assert calibration['seeds'] == [5] and full['primary_scale'] == (
        calibration['primary_scale'])

    # Rerunning a sealed writer makes no call and no ledger entry.
    entries = len(run.read(world.ledger)['entries'])
    assert run.writer_stage(batch, 'full_state', client=provider) == full
    assert len(provider.calls) == len(world.panel['cases'])
    assert len(run.read(world.ledger)['entries']) == entries

    cells = run.cells_for(batch, manifest, 'formats')
    assert len(cells) == 120
    for cell in cells:
        args = cell['args']
        assert args['format_bank'] == 'banks/full_state/format_bank.json'
        assert args['format_bank_sha256'] == full['bank_sha256']
        if args['lesson_mode'] == 'explanation':
            assert args['lesson_scale'] == (
                calibration['primary_scale'][args['lesson_format']])
    with pytest.raises(ValueError, match='No frozen local_only'):
        run.cells_for(batch, manifest, 'access')

    local = run.writer_stage(batch, 'local_only', client=provider)
    assert local['gate']['passed']
    assert local['train_cases'] < full['train_cases']
    assert local['calibration_sha256'] == full['calibration_sha256']
    access = run.cells_for(batch, manifest, 'access')
    assert len(access) == 50
    assert {c['args']['format_access'] for c in access} == {'local_only'}
    assert {c['args']['format_bank_sha256'] for c in access} == {
        local['bank_sha256']}
    bank = run.read(batch / local['bank'])
    assert bank['information_access'] == 'local_only'
    assert not bank['paired_restriction']

    # A modified bank is refused at dispatch.
    path = batch / full['bank']
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='modified'):
        run.cells_for(batch, manifest, 'formats')


def test_failed_gate_withholds_cells_and_keeps_its_evidence(world,
                                                            monkeypatch):
    batch = formats_batch(world, monkeypatch)
    provider = FakeProvider(world.panel, fail=True)
    with pytest.raises(ValueError, match='failed its frozen gate'):
        run.writer_stage(batch, 'full_state', client=provider)
    failed = run.read(batch / 'banks/full_state/GATE_FAILED.json')
    assert failed['train_cases'] == 0 and not failed['gate']['passed']
    assert failed['collection']['failed'] == len(world.panel['cases'])
    assert not (batch / 'banks/full_state/FROZEN.json').exists()
    assert not (batch / 'calibration.json').exists()
    status = CostLedger(str(world.ledger)).status()
    assert not status['open_reservations']
    with pytest.raises(ValueError, match='No frozen full_state'):
        run.cells_for(batch, run.verify(batch), 'formats')
    # Nothing is retried: the sealed failure blocks a second attempt.
    calls = len(provider.calls)
    with pytest.raises(FileExistsError):
        run.writer_stage(batch, 'full_state', client=provider)
    assert len(provider.calls) == calls


def test_partial_collection_is_never_retried(world, monkeypatch):
    batch = formats_batch(world, monkeypatch)
    (batch / 'collect/full_state').mkdir()
    provider = FakeProvider(world.panel)
    with pytest.raises(FileExistsError, match='Partial'):
        run.writer_stage(batch, 'full_state', client=provider)
    assert provider.calls == []
    assert CostLedger(str(world.ledger)).status()['reserved_usd'] == 0


# ------------------------------------------------------- cells and chain

def test_cells_record_failures_and_refuse_duplicates(world, monkeypatch):
    batch = formats_batch(world, monkeypatch)
    provider = FakeProvider(world.panel)
    run.writer_stage(batch, 'full_state', client=provider)
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run.subprocess, 'run', fail)
    assert run.run_cell(batch, 'formats', 5) == 1
    assert 'algos.ppo_lesson_formats' in commands[0]
    outcome = run.read(batch / 'cells/formats/5/exit.json')
    assert outcome['artifact_status'] == 'process_failed'
    dispatch = run.read(batch / 'cells/formats/5/dispatch.json')
    assert Path(dispatch['resolved_args']['format_bank']).is_absolute()
    assert dispatch['suite'] == 'formats'
    with pytest.raises(FileExistsError):
        run.run_cell(batch, 'formats', 5)
    with pytest.raises(ValueError, match='range'):
        run.run_cell(batch, 'formats', 120)
    with pytest.raises(ValueError, match='No frozen local_only'):
        run.run_cell(batch, 'access', 0)
    assert len(commands) == 1


def test_timing_batch_prepares_and_dispatches_from_cached_lessons(
        world, monkeypatch):
    batch = run.prepare_timing(world.root, world.source)
    monkeypatch.setattr(run, 'ROOT', batch / 'code')
    manifest = run.verify(batch)
    assert manifest['artifact_hashes'] == {
        'corpora/lessons.json': run.LESSONS_SHA256}
    monkeypatch.setattr(run.subprocess, 'run', lambda command, **kwargs:
                        SimpleNamespace(returncode=3))
    assert run.run_cell(batch, 'timing', 29) == 1
    dispatch = run.read(batch / 'cells/timing/29/dispatch.json')
    assert dispatch['condition'] == 'actions_late'
    with pytest.raises(ValueError, match='Unknown suite'):
        run.cells_for(batch, manifest, 'formats')


def test_submission_chain_orders_writers_before_their_arrays(world,
                                                            monkeypatch):
    batch = formats_batch(world, monkeypatch)
    commands = []

    def sbatch(command, **kwargs):
        commands.append(command)
        return f'{1000 + len(commands)};cluster\n'

    monkeypatch.setattr(run.subprocess, 'check_output', sbatch)
    jobs = run.submit_formats(batch)
    assert jobs == dict(writer_full_state='1001', formats='1002',
                        writer_local_only='1003', access='1004')
    text = [' '.join(c) for c in commands]
    assert '--dependency' not in text[0] and '--mem=8G' in text[0]
    assert '--array=0-119' in text[1] and 'afterok:1001' in text[1]
    assert 'afterok:1001' in text[2] and text[2].endswith('writer local_only')
    assert '--array=0-49' in text[3] and 'afterok:1003' in text[3]
    assert all('%' not in c.split('--array=')[-1].split()[0]
               for c in text if '--array=' in c)
    assert all('--kill-on-invalid-dep=yes' in c for c in text[1:])
    with pytest.raises(FileExistsError):
        run.submit_formats(batch)
    assert len(commands) == 4


def test_launch_prepares_submits_once_and_reports_afterwards(
        world, monkeypatch, capsys):
    submitted = []

    def sbatch(command, **kwargs):
        if command[0] == 'git':
            return original(command, **kwargs)
        submitted.append(command)
        return '2001\n'

    original = run.subprocess.check_output
    monkeypatch.setattr(run.subprocess, 'check_output', sbatch)
    run.launch(world.root, ['timing'], world.source, world.ledger,
               world.credential)
    run.launch(world.root, ['timing'], world.source, world.ledger,
               world.credential)
    assert len(submitted) == 1 and '--array=0-29' in submitted[0]
    assert 'already submitted' in capsys.readouterr().out


# -------------------------------------------------- terminal validation

def trained(tmp_path, monkeypatch, bank, **changes):
    path = tmp_path / 'bank.json'
    path.write_text(json.dumps(bank))
    args = formats_trainer.Args(
        seed=5, total_timesteps=2048, num_steps=16, update_epochs=1,
        eval_interval=8, eval_episodes=1, format_bank=str(path),
        format_bank_sha256=file_hash(path), lesson_batch=4, lesson_every=4,
        lesson_exposures=4, experiment_id='launch_test', **changes)
    monkeypatch.setattr(ppo, '__file__', str(tmp_path / 'algos/x.py'))
    torch.set_num_threads(1)
    ppo.train(replace(args),
              auxiliary=formats_trainer.FormatLessonAuxiliary())
    return next((tmp_path / 'results/runs').iterdir()), args


@pytest.mark.parametrize('mode', ['explanation', 'actions'])
def test_validation_accepts_real_runs_and_rejects_tampering(
        tmp_path, monkeypatch, panel, mode):
    replies = {c['case_id']: dict(status='valid', answer=answer_for(c))
               for c in panel['cases']}
    bank = ef.freeze_bank(panel, replies, fake_embed)
    changes = (dict(lesson_mode='explanation', lesson_format='plan',
                    lesson_scale=0.5) if mode == 'explanation' else
               dict(lesson_mode='actions', lesson_action_coef=1.0))
    run_dir, args = trained(tmp_path, monkeypatch, bank, **changes)
    metrics = run.validate_completed(run_dir, args)
    assert metrics['lesson_updates'] == 4 and len(metrics['curve']) == 2
    assert metrics['auc_interval'] == [1024, 2048]
    assert 0 <= metrics['teacher_off_auc'] <= 1
    with pytest.raises(ValueError):
        run.validate_completed(run_dir, replace(args, lesson_exposures=3))
    with pytest.raises(ValueError):
        run.validate_completed(run_dir, replace(args, seed=6))
    curve = run_dir / 'evaluations.jsonl'
    lines = curve.read_text().splitlines(keepends=True)
    curve.write_text(''.join(lines[:-1]))
    with pytest.raises(ValueError, match='evaluation grid'):
        run.validate_completed(run_dir, args)
    curve.write_text(''.join(lines))
    updates = run_dir / 'lesson_updates.jsonl'
    rows = [json.loads(line) for line in updates.read_text().splitlines()]
    rows[0]['ids'] = list(reversed(rows[0]['ids']))
    updates.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='Replay IDs'):
        run.validate_completed(run_dir, args)
