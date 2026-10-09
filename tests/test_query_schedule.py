"""
Fixed query windows: the resolved list, and where queries actually land.

Two different claims are checked here, and conflating them is how a
schedule ends up specified but not followed.

The first is arithmetic: given a training shape, the window list is a
particular list of rollout indices. That is checked against the numbers
in the specification directly, because a schedule that quietly resolves
to something else would still run and still cost money.

The second is behavioural, and needs the real trainer: consultations
happen at those positions and nowhere else. `advising/schedule.py` can
be perfectly correct while the gate is wired somewhere that never
consults it, or consults it with the wrong index base. Only a run
settles that, so the position tests read the records a real run wrote.

The teacher is the offline bot and the embedder is a deterministic
fake: this is a schedule check, and a network call would make it slower
and less reliable while testing nothing extra about timing.
"""

import json

import pytest

from advising.schedule import QueryWindows
from algos import ppo_distill


# --- The resolved list ---------------------------------------------

def test_launch_shape_matches_the_specification():
    """
    The numbers in the spec, recomputed rather than restated.

    10M transitions in 1024-transition rollouts is 9765 complete
    rollouts, 7324 of them inside the first 75%, so the windows run
    from rollout 0 to rollout 7323 and offer 25 x 10 x 8 = 2000 slots.
    """

    w = QueryWindows(total_timesteps=10_000_000, batch_size=1024,
                     num_steps=128, num_envs=8)

    assert w.num_rollouts == 9765
    assert w.guided_rollouts == 7324
    assert len(w.rollouts) == 25
    assert w.rollouts[0] == 0
    assert w.rollouts[-1] == 7323
    assert w.scheduled_slots == 2000


def test_windows_are_distinct_and_ordered():
    """
    A duplicated index would halve the spend while the manifest still
    claimed the full schedule.
    """

    w = QueryWindows(total_timesteps=10_000_000, batch_size=1024,
                     num_steps=128, num_envs=8)
    assert len(set(w.rollouts)) == 25
    assert list(w.rollouts) == sorted(w.rollouts)


def test_the_manifest_carries_the_index_list():
    """
    The list is the thing an audit compares observed positions against,
    so it has to survive into the run summary, not just into memory.
    """

    w = QueryWindows(total_timesteps=10_000_000, batch_size=1024,
                     num_steps=128, num_envs=8)
    assert w.manifest()['rollout_indices'] == list(w.rollouts)


@pytest.mark.parametrize('kwargs, message', [
    (dict(window_steps=200), 'wider than'),
    (dict(num_windows=1), 'at least 2'),
    (dict(window_steps=0), 'must be >= 1'),
    (dict(guidance_fraction=0.0), 'must be in'),
    (dict(guidance_fraction=1.5), 'must be in'),
])
def test_impossible_configurations_are_refused(kwargs, message):
    """
    Refusing is the point. A silently truncated schedule still runs.
    """

    with pytest.raises(ValueError, match=message):
        QueryWindows(total_timesteps=10_000_000, batch_size=1024,
                     num_steps=128, num_envs=8, **kwargs)


def test_too_few_rollouts_for_the_windows_is_refused():
    """
    25 windows do not fit in a run with 3 guided rollouts.
    """

    with pytest.raises(ValueError, match='do not fit'):
        QueryWindows(total_timesteps=4096, batch_size=1024,
                     num_steps=128, num_envs=8)


# --- Through the real trainer ---------------------------------------

class _FakeEmbedder:
    """
    Deterministic, offline, and never asked for a real vector.
    """

    def __init__(self, *args, **kwargs):
        self.model = 'fake'
        self.dimension = 13

    @staticmethod
    def key_for(text):
        return str(abs(hash(text)) % 10_000_019)

    def embed(self, texts):
        import numpy as np
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            vector = rng.normal(size=13).astype('float32')
            out.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        return out

    def save(self):
        pass

    def stats(self, price_per_million=0.0):
        return {'embedding_model': 'fake', 'dimension': 13}


# Four windows in a 32-rollout run: floor(j * 23 / 3) for j in 0..3.
# 24 of the 32 rollouts lie inside the 75% guidance window, so the last
# window sits at rollout 23 and the teacher-off tail stays empty.
EXPECTED_ROLLOUTS = [0, 7, 15, 23]
WINDOW_STEPS = 2


def _windowed_args(explanation, tmp_path, **over):
    """
    A run small enough to execute, large enough to have real gaps.
    """

    settings = dict(
        task='keycorridor_s3r3', obs_mode='symbolic', recurrent=True,
        teacher='bot', guidance=True, bonus='count', dual_value=True,
        cuda=False, total_timesteps=1024, num_envs=4, num_steps=8,
        num_minibatches=1, update_epochs=1,
        eval_interval=99, eval_episodes=1, eval_sampled=False,
        gamma=0.999, seed=0,
        query_windows=len(EXPECTED_ROLLOUTS),
        query_window_steps=WINDOW_STEPS,
        explanation=explanation, embed_dim=13, lambda_aux=1.0,
    )
    settings.update(over)
    return ppo_distill.Args(**settings)


def _run(explanation, tmp_path, monkeypatch, **over):
    """
    Run the real trainer into an isolated results root, return its dir.
    """

    import torch

    monkeypatch.setattr(ppo_distill, '__file__',
                        str(tmp_path / 'algos' / 'ppo_distill.py'))
    monkeypatch.setattr(ppo_distill, 'EmbeddingProvider', _FakeEmbedder)
    torch.set_num_threads(1)
    ppo_distill.train(_windowed_args(explanation, tmp_path, **over))
    return next((tmp_path / 'results' / 'runs').iterdir())


def _records(run_dir):
    path = run_dir / 'explanations' / 'explanation_records.jsonl'
    assert path.exists(), 'no explanation records were persisted'
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def test_the_schedule_resolves_to_the_expected_rollouts():
    """
    Pin the small configuration's list, so the behavioural tests below
    are comparing against a known answer rather than against whatever
    the code happens to produce.
    """

    w = QueryWindows(total_timesteps=1024, batch_size=32, num_steps=8,
                     num_envs=4, num_windows=4, window_steps=WINDOW_STEPS)
    assert list(w.rollouts) == EXPECTED_ROLLOUTS
    assert w.scheduled_slots == 4 * WINDOW_STEPS * 4


@pytest.mark.parametrize('arm', ['none', 'correct', 'shuffled', 'detached'])
def test_queries_only_happen_inside_the_windows(arm, tmp_path, monkeypatch):
    """
    The behavioural claim: every recorded consultation is at a scheduled
    position. `none` is included because R1 pays for the same calls and
    its records are what make R1-vs-R2 auditable.
    """

    run = _run(arm, tmp_path, monkeypatch)
    rows = _records(run)
    assert rows, 'a windowed run produced no records at all'

    off_schedule = {r['rollout'] for r in rows} - set(EXPECTED_ROLLOUTS)
    assert not off_schedule, (
        f'consultations happened at unscheduled rollouts {off_schedule}; '
        f'the window gate is not controlling when the teacher is paid'
    )
    late = [r for r in rows if r['step'] >= WINDOW_STEPS]
    assert not late, (
        f'{len(late)} consultations fell past step {WINDOW_STEPS - 1} of '
        f'their rollout, outside the window'
    )


def test_no_query_reaches_the_teacher_off_tail(tmp_path, monkeypatch):
    """
    The last 25% of training must cost nothing. `distill_coef` already
    gates on it, but the windows must not be placed there in the first
    place, or the two guards would be relying on each other.
    """

    run = _run('correct', tmp_path, monkeypatch)
    rows = _records(run)
    # 32 rollouts; the tail begins at 0.75 * 32 = 24.
    assert max(r['rollout'] for r in rows) < 24


def test_the_summary_reports_the_resolved_schedule(tmp_path, monkeypatch):
    """
    The schedule and the positions used both reach the run summary, so
    an audit does not have to parse the record file to do the check.
    """

    run = _run('correct', tmp_path, monkeypatch)
    # Per-iteration diagnostics land under 'latest'; the final close
    # writes the last one, so this is what a finished run reports.
    latest = json.loads((run / 'run_summary.json').read_text())['latest']

    schedule = latest['query_schedule']
    assert schedule['rollout_indices'] == EXPECTED_ROLLOUTS
    assert schedule['scheduled_slots'] == 4 * WINDOW_STEPS * 4
    assert latest['query_rollouts_off_schedule'] == []
    assert latest['query_rollouts_observed']

    # Spend and control accounting, wired rather than merely available.
    explanation = latest['explanation']
    assert explanation['records_written'] > 0
    assert explanation['embedding']['embedding_model']
    assert 'unchanged_fraction' in explanation['shuffle_coverage']
    calls = [json.loads(line) for line in
             (run / 'consultations.jsonl').read_text().splitlines()]
    assert len(calls) == latest['consultations']['records']
    assert len({r['sample_id'] for r in calls}) == len(calls)
    assert {r['sample_id'] for r in _records(run)} <= {
        r['sample_id'] for r in calls}


def test_failure_rate_stops_real_trainer_after_persisting_calls(
        tmp_path, monkeypatch):
    from teachers.base import Advice, Cost
    from teachers.budget import BudgetError

    original = ppo_distill.make_teacher

    def failing(*args, **kwargs):
        teacher = original(*args, **kwargs)

        def recommend(state, context=None):
            return Advice(cost=Cost(tokens_out=3500, dollars=.007,
                                    metadata={'failed': True,
                                              'outcome': 'schema_failure'}))

        teacher.recommend = recommend
        return teacher

    monkeypatch.setattr(ppo_distill, 'make_teacher', failing)
    with pytest.raises(BudgetError, match='Consultation failures 4/4'):
        _run('none', tmp_path, monkeypatch,
             consultation_failure_limit=.2,
             consultation_failure_min_samples=4)
    run = next((tmp_path / 'results/runs').iterdir())
    rows = [json.loads(line) for line in
            (run / 'consultations.jsonl').read_text().splitlines()]
    assert len(rows) == 4
    assert sum(r['tokens_out'] for r in rows) == 14000
    assert all(r['delivered'] is False for r in rows)


def test_the_same_positions_are_scheduled_for_every_arm(
        tmp_path, monkeypatch):
    """
    R1-R4 must be matched on when they paid. Their trajectories diverge,
    so the states differ; the schedule may not.
    """

    schedules = {}
    for arm in ('none', 'correct', 'shuffled', 'detached'):
        run = _run(arm, tmp_path / arm, monkeypatch)
        latest = json.loads(
            (run / 'run_summary.json').read_text())['latest']
        schedules[arm] = latest['query_schedule']['rollout_indices']

    assert len(set(map(tuple, schedules.values()))) == 1, (
        f'arms were scheduled differently: {schedules}'
    )


def test_history_survives_the_gap_between_windows(tmp_path, monkeypatch):
    """
    The gate skips the consultation, not the bookkeeping.

    Between windows the student keeps acting, and those actions must
    still be waiting in the history at the next window -- otherwise the
    teacher's prompt describes a trajectory with hundreds of steps
    missing from it. Under the previous every-step schedule a history
    was never longer than one action, so a run where some consultation
    sees several is the discriminating evidence.
    """

    seen = []
    original = ppo_distill.make_teacher

    def recording(*args, **kwargs):
        teacher = original(*args, **kwargs)

        class Proxy:
            def __getattr__(self, name):
                return getattr(teacher, name)

            def recommend(self, state, context=None):
                seen.append(list((context or {}).get('executed_actions', [])))
                return teacher.recommend(state, context=context)

        return Proxy()

    monkeypatch.setattr(ppo_distill, 'make_teacher', recording)
    _run('correct', tmp_path, monkeypatch)

    assert seen, 'the teacher was never consulted'
    assert max(len(h) for h in seen) > 1, (
        'no consultation ever saw more than one executed action; the '
        'history is being cleared during the gap between windows'
    )


def test_a_schedule_reaching_past_the_cutoff_is_refused(tmp_path):
    """
    Windows past `distill_cutoff` would buy advice the loss has already
    switched off. Checked against the coefficient schedule itself, so
    the two cannot drift apart.
    """

    with pytest.raises(ValueError, match='teacher-off tail'):
        ppo_distill.train(_windowed_args(
            'none', tmp_path, query_window_fraction=1.0))


def test_windows_require_a_teacher(tmp_path):
    """
    R0 has nothing to schedule.
    """

    with pytest.raises(ValueError, match='needs --guidance'):
        ppo_distill.train(_windowed_args(
            'none', tmp_path, guidance=False))


# --- Abstain versus decline -----------------------------------------

def test_advisor_declines_are_not_counted_as_teacher_failures(
        tmp_path, monkeypatch):
    """
    An empty answer has two causes, and merging them libels the teacher.

    The dispatch loop runs for every environment at every step while the
    distillation coefficient is positive, whether or not the advisor is
    willing to consult. Once a query budget is spent the advisor
    declines every remaining dispatch, and those declines were being
    added to the same counter as genuine teacher abstentions. The
    five-arm smoke reported 32 consultations and 3032 "abstains" -- a
    number that reads as a 99% failure rate and is nothing of the kind.

    The BFS oracle always answers, so every empty result in this run is
    a decline and the abstain counter must stay at zero.
    """

    import torch

    monkeypatch.setattr(ppo_distill, '__file__',
                        str(tmp_path / 'algos' / 'ppo_distill.py'))
    torch.set_num_threads(1)
    ppo_distill.train(ppo_distill.Args(
        task='doorkey_8x8', obs_mode='historical', recurrent=True,
        teacher='oracle', guidance=True, cuda=False,
        total_timesteps=256, num_envs=4, num_steps=8,
        num_minibatches=1, update_epochs=1,
        eval_interval=99, eval_episodes=1, eval_sampled=False,
        seed=0, advisor='early', query_budget=4, advice_budget=4,
    ))
    run = next((tmp_path / 'results' / 'runs').iterdir())
    latest = json.loads((run / 'run_summary.json').read_text())['latest']

    assert latest['teacher_total_queries'] <= 4, 'the cap did not hold'
    assert latest['teacher_total_declined'] > latest[
        'teacher_total_queries'], (
        'the budget was never exhausted, so this run cannot show the '
        'difference between a decline and an abstention'
    )
    assert latest['teacher_total_abstains'] == 0, (
        f"the BFS oracle always answers, so "
        f"{latest['teacher_total_abstains']} abstentions means advisor "
        f"declines are still being counted as teacher failures"
    )
