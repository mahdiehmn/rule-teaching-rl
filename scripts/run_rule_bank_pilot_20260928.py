"""Learning pilot: does an LLM explanation make one consultation go further?

Question: can an explanation make one teacher consultation useful across
multiple situations, beyond simply replaying its action label?

Every guided arm uses the SAME 36 GPT-5-mini consultations (v3, frozen
before training); the scope-checked and shuffled arms also use the same
26 scope-check calls. Training makes no API calls: the frozen banks are
committed under research/rule_banks/ and read by the free `rule_bank`
teacher from the student's own 7x7 view (teachers/minigrid/rule_bank.py).

Arms, on DoorKey-8x8 plain PPO, with five fresh paired seeds
14,500,000 + 100r and a 5M horizon:

  none                    no teacher
  llm_action_replay       the LLM's action_now, reused only on identical
                          views (action-only control)
  llm_rules_direct        the LLM's rules, conditions only
  llm_rules_scoped        the LLM's rules with full v3 scope semantics
  llm_rules_scope_checked the rules after the LLM's scope check
  llm_rules_shuffled      the scoped rules' conditions with actions
                          permuted across rules (content control, same
                          scopes as the primary arm)

Guided arms share the reviewed rule-reference distillation settings
(coefficient 1 to .01, labelled normalisation, isolated advisor RNG).
Labels come densely wherever the bank advises, with no query cap:
consultations are what the bank cost, not training queries. Reuse adds
supervision, so delivered labels are logged per run and reported; the
shuffled arm matches the scoped structure with wrong content.

Frozen development rule, revised BEFORE any learning run from offline
development evidence only: the LLM's own scope check did not reduce
incorrect reuse (precision .656 raw vs .633 checked; it endorsed its rule
in 17 of 21 true counterexamples). The primary explanation arm is
therefore `llm_rules_scoped`. The explanation is useful beyond replay only
if `llm_rules_scoped - llm_action_replay` has a positive mean AUC with at
least 4/5 positive pairs, AND `llm_rules_scoped - llm_rules_shuffled` is
also positive with at least 4/5 positive pairs. The scope-checked arm is
secondary. Everything else is reported descriptively.

Blind-check extension (study `rule_bank_pilot_20260928_blind_v1`), added
while the pilot above was running and BEFORE any of its learning outcomes
was seen. The blind strict check was selected on development and then
confirmed once on the untouched panel (precision .983, 354 correct and 6
wrong). Two arms, on
the same seeds and settings, change only the bank:

  llm_rules_blind_strict  the rules kept by the blind strict check (the
                          same LLM, without seeing the rule, agreed with
                          the rule's action in every sampled situation)
  llm_rules_random_subset as many raw rules, chosen at random (a
                          different frozen subset per replicate): a
                          selection-size control without verification

The extension is a separate frozen batch. Its preparation refuses unless
the running pilot batch exists, every trainer-side source hashes the
same, and each cell equals that seed's `llm_rules_scoped` cell except for
the bank. Frozen extension rule: the verified explanation is useful
beyond replay if `llm_rules_blind_strict` beats BOTH `llm_action_replay`
and `llm_rules_shuffled` with a positive mean AUC and at least 4/5
positive pairs. Verification, not mere pruning, matters if
`llm_rules_blind_strict - llm_rules_random_subset` meets the same bar.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
from scipy.stats import t

from scripts.run_explanation_formats_20260925 import (
    derived_of, eval_steps, read_jsonl, runtime_identity)
from scripts.run_explanation_grid import trainer_argv
from scripts.run_plan_repair_20260927 import digest, inside, read, write

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'rule_bank_pilot_20260928_v1'
SOURCE = Path('results/conditional_rules_v3_20260928')
BANKS = Path('research/rule_banks/v3_20260928')
SEEDS = tuple(14_500_000 + 100 * r for r in range(5))
HORIZON = 5_000_000
ARMS = ('none', 'llm_action_replay', 'llm_rules_direct', 'llm_rules_scoped',
        'llm_rules_scope_checked', 'llm_rules_shuffled')
BANK_OF = {'llm_action_replay': 'replay', 'llm_rules_direct': 'direct',
           'llm_rules_scoped': 'scoped',
           'llm_rules_scope_checked': 'scope_checked',
           'llm_rules_shuffled': 'shuffled',
           'llm_rules_blind_strict': 'blind_strict',
           'llm_rules_random_subset': 'random_subset_r{r}'}
EXTENSION = 'rule_bank_pilot_20260928_blind_v1'
EXTENSION_ARMS = ('llm_rules_blind_strict', 'llm_rules_random_subset')
# Only the bank may differ between an extension cell and its seed's
# `llm_rules_scoped` cell.
BANK_KEYS = {'experiment_id', 'rule_bank', 'rule_bank_sha256'}
# The runner itself orchestrates cells; a training cell never imports it.
RUNNER = ('scripts/run_rule_bank_pilot_20260928.py',
          'scripts/launch_rule_bank_pilot.sh')
REQUIRED = (*RUNNER, 'teachers/minigrid/rule_bank.py',
            'scripts/conditional_rules_v3.py',
            'scripts/submit_rule_bank_pilot.sh',
            'tests/test_rule_bank_pilot.py',
            *(f'{BANKS.as_posix()}/{name}.json' for name in
              ('replay', 'direct', 'scoped', 'scope_checked', 'shuffled',
               'blind_strict',
               *(f'random_subset_r{r}' for r in range(len(SEEDS))))))


# ------------------------------------------------------------------ banks

def rule_json(rule):
    condition, action, exceptions = rule
    return dict(condition=condition, action=action,
                exceptions=[list(e) for e in exceptions])


def receipts_cost(path):
    ends = [json.loads(line) for line in Path(path).read_text().splitlines()
            if json.loads(line)['event'] == 'END']
    return dict(calls=len(ends), completed=sum(
        e['response_status'] == 'completed' for e in ends),
        dollars=sum(e.get('dollars') or 0 for e in ends))


def build_banks(source=SOURCE, out=BANKS):
    """Frozen banks from the collected v3 replies (no calls, no oracle)."""
    from scripts import conditional_rules_v3 as v3
    source, out = Path(source), Path(out)
    panels, parsed = v3.consult_rules(source)
    replies = v3.read_replies(source / 'refine_replies.jsonl')
    raw, checked, status = [], [], Counter()
    replay = {}
    for k, (state, now, rule, _) in enumerate(parsed):
        if now is not None:
            replay[state['image']] = now
        if rule is None:
            continue
        raw.append(rule)
        row = replies.get(f'refine_{k:03d}')
        if row is None:
            status['kept_unchecked'] += 1
            checked.append(rule)
        elif row['response_status'] != 'completed':
            status['check_incomplete_dropped'] += 1
        else:
            _, new = v3.parse(row['answer'], with_now=False)
            status['refined' if new else 'withdrawn'] += 1
            if new:
                checked.append(new)
    rng = np.random.default_rng(20260928)
    order = rng.permutation(len(raw))
    while len(raw) > 1 and np.any(order == np.arange(len(raw))):
        order = rng.permutation(len(raw))
    shuffled = [(raw[i][0], raw[j][1], raw[i][2])
                for i, j in enumerate(order)]
    cost = dict(consult=receipts_cost(source / 'consult_replies.raw.jsonl'),
                scope_check=receipts_cost(source /
                                          'refine_replies.raw.jsonl'))
    common = dict(study=STUDY, model='gpt-5-mini-2025-08-07',
                  source_replies_sha256=dict(
                      consult=digest(source / 'consult_replies.jsonl'),
                      refine=digest(source / 'refine_replies.jsonl')),
                  cost=cost, scope_check_status=dict(status))
    banks = dict(
        replay=dict(mode='replay', replay=replay, **common),
        direct=dict(mode='direct', rules=[rule_json(r) for r in raw],
                    **common),
        scoped=dict(mode='scoped', rules=[rule_json(r) for r in raw],
                    **common),
        scope_checked=dict(mode='scoped',
                           rules=[rule_json(r) for r in checked], **common),
        shuffled=dict(mode='scoped', rules=[rule_json(r) for r in shuffled],
                      **common))
    out.mkdir(parents=True, exist_ok=True)
    for name, bank in banks.items():
        path = out / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        path.write_text(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(rules_raw=len(raw), rules_checked=len(checked),
                          replay_views=len(replay), cost=cost,
                          scope_check_status=dict(status)), indent=1))


def build_extension_banks(source=SOURCE, out=BANKS):
    """Blind-strict bank and its size-matched random-selection controls."""
    from scripts import conditional_rules_v3 as v3
    source, out = Path(source), Path(out)
    panels, parsed = v3.consult_rules(source)
    raw = [rule for _, _, rule, _ in parsed if rule is not None]
    kept, status = v3.blind_filtered(source, parsed, panels, strict=True)
    consult = receipts_cost(source / 'consult_replies.raw.jsonl')
    common = dict(study=EXTENSION, model='gpt-5-mini-2025-08-07',
                  source_replies_sha256=dict(
                      consult=digest(source / 'consult_replies.jsonl'),
                      blind=digest(source / 'blind_replies.jsonl')))
    banks = dict(blind_strict=dict(
        mode='scoped', rules=[rule_json(r) for r in kept],
        cost=dict(consult=consult, blind_check=receipts_cost(
            source / 'blind_replies.raw.jsonl')),
        blind_check_status=status, **common))
    for r in range(len(SEEDS)):
        pick = sorted(np.random.default_rng([20260929, r]).choice(
            len(raw), len(kept), replace=False).tolist())
        banks[f'random_subset_r{r}'] = dict(
            mode='scoped', rules=[rule_json(raw[i]) for i in pick],
            raw_rule_indices=pick, cost=dict(consult=consult), **common)
    out.mkdir(parents=True, exist_ok=True)
    for name, bank in banks.items():
        path = out / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        path.write_text(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(rules_raw=len(raw), rules_blind_strict=len(kept),
                          blind_check_status=status,
                          cost=banks['blind_strict']['cost']), indent=1))


# ------------------------------------------------------------------ cells

def base_args():
    from algos.ppo_distill import Args
    from scripts.run_rule_reference_20260927 import menu
    cells = {c['arm']: c['args'] for c in menu()
             if c['task'] == 'doorkey_8x8' and c['bonus'] == 'none'
             and c['replicate'] == 0}
    return Args(**cells['none']), Args(**cells['random480'])


def cells(root=ROOT, study=STUDY):
    from algos.ppo_distill import Args
    from teachers.minigrid.rule_bank import file_sha256
    import tyro
    none, guided = base_args()
    out = []
    for r, seed in enumerate(SEEDS):
        for arm in (ARMS if study == STUDY else EXTENSION_ARMS):
            if arm == 'none':
                args = replace(none, seed=seed, total_timesteps=HORIZON,
                               experiment_id=f'{study}_{arm}')
            else:
                bank = f'{BANKS.as_posix()}/{BANK_OF[arm].format(r=r)}.json'
                args = replace(
                    guided, seed=seed, total_timesteps=HORIZON,
                    experiment_id=f'{study}_{arm}', teacher='rule_bank',
                    rule_bank=bank,
                    rule_bank_sha256=file_sha256(Path(root) / bank),
                    advisor='unlimited', query_budget=0, advice_budget=0,
                    uniform_queries=False, advisor_no_teacher_peek=False,
                    importance_source='none', audit_explanations=False)
            if asdict(tyro.cli(Args, args=trainer_argv(asdict(args)),
                               console_outputs=False)) != asdict(args):
                raise ValueError('Trainer command changed resolved settings')
            out.append(dict(index=len(out), arm=arm, seed=seed,
                            trainer='algos.ppo_distill', args=asdict(args)))
    return out


# ------------------------------------------------------------ preparation

def batch_dir(root, study=STUDY):
    return Path(root) / 'results/efficiency' / study


def outside_trainer(path):
    """Files that cannot change what a training cell executes."""
    return (path in RUNNER or path.endswith('.md')
            or path.startswith(('tests/', 'docs/', 'paper/'))
            or (path.startswith('research/')
                and not path.startswith('research/rule_banks/')))


def same_trainer(old, new):
    """Refuse an extension whose trainer-side sources differ from the pilot."""
    changed = sorted(p for p, h in old.items()
                     if not outside_trainer(p) and new.get(p) != h)
    if changed:
        raise ValueError(f'Trainer-side sources differ: {changed[:5]}')


def paired_cells(main, extension):
    """Each extension cell must equal its seed's scoped cell but the bank."""
    scoped = {c['seed']: c['args'] for c in main
              if c['arm'] == 'llm_rules_scoped'}
    for c in extension:
        other = scoped[c['seed']]
        if set(c['args']) != set(other) or {
                k for k in other if c['args'][k] != other[k]} - BANK_KEYS:
            raise ValueError(f"Extension cell {c['index']} differs from "
                             'the paired scoped cell beyond its bank')


def prepare(root, study=STUDY):
    root = Path(root).resolve()
    batch = batch_dir(root, study)
    if batch.exists():
        verify(batch)
        return batch
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root,
                   check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                   text=True).strip()
    data = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         head], cwd=root)
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        if not set(REQUIRED) <= set(zipped.namelist()):
            raise ValueError('Commit the complete packet (with banks) first')
        if study != STUDY:
            main = verify(batch_dir(root, STUDY))
            same_trainer(main['source_hashes'], {
                n: hashlib.sha256(zipped.read(n)).hexdigest()
                for n in zipped.namelist() if not n.endswith('/')})
            paired_cells(main['cells'], cells(root, study))
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    built = cells(batch / 'code', study)
    if study != STUDY:
        paired_cells(main['cells'], built)
    manifest = dict(study=study, head=head, runtime=runtime_identity(),
                    cells=built, api_calls_in_training=0,
                    source_hashes={
                        p.relative_to(batch / 'code').as_posix(): digest(p)
                        for p in (batch / 'code').rglob('*')
                        if p.is_file()})
    write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(digest(batch / 'manifest.json'))
    (batch / 'READY').write_text(head + '\n')
    verify(batch)
    return batch


def verify(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / 'manifest.json')
    if (digest(batch / 'manifest.json') !=
            (batch / 'manifest.sha256').read_text().strip()
            or manifest['study'] not in (STUDY, EXTENSION)
            or manifest['study'] != batch.name
            or (batch / 'READY').read_text().strip() != manifest['head']):
        raise ValueError('Frozen preparation identity differs')
    for name, expected in manifest['source_hashes'].items():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    return manifest


# ------------------------------------------------------------- validation

def find_run(runs_root, args):
    hits = [p.parent for p in Path(runs_root).rglob(
        f"*{args['experiment_id']}*__{args['seed']}__*/run_summary.json")
        if read(p)['args'].get('experiment_id') == args['experiment_id']]
    if len(hits) != 1:
        raise FileNotFoundError(f'{len(hits)} runs for {args["seed"]}')
    return hits[0]


def validate_run(run, args):
    run = Path(run)
    summary = read(run / 'run_summary.json')
    ns = argparse.Namespace(**args)
    derived = derived_of(ns)
    endpoint = derived['num_iterations'] * derived['batch_size']
    if summary.get('status') != 'completed' or \
            summary['global_step'] != endpoint:
        raise ValueError('Incomplete run')
    # The trainer fills in the batch geometry at start-up, and a paid run's
    # cost cap from its ledger reservation. Check those against what they
    # must be; every other setting must equal the frozen cell exactly.
    expected = dict(args, **derived)
    for key, value in expected.items():
        if key == 'max_cost_dollars':
            continue
        if summary['args'].get(key) != value:
            raise ValueError(f'Setting differs: {key}')
    steps = eval_steps(ns, derived)
    rows = read_jsonl(run / 'evaluations.jsonl')
    if len(steps) < 2 or [r['global_step'] for r in rows] != steps or any(
            r['teacher_on'] or r['episodes'] != ns.eval_episodes
            or r['seed_base'] != ns.seed + 50_000 for r in rows):
        raise ValueError('Teacher-off evaluation contract differs')
    y = np.array([r['success_rate'] for r in rows], dtype=float)
    auc = float(np.sum(np.diff(steps) * (y[:-1] + y[1:]) / 2)
                / (steps[-1] - steps[0]))
    advising = summary['latest'].get('advising', {})
    return dict(auc=auc, final=float(y[-1]),
                initial_sha256=(run / 'initial_policy.sha256')
                .read_text().strip(),
                labels_delivered=advising.get('num_delivered', 0),
                teacher_consulted=advising.get('num_asked', 0),
                api_dollars=summary['latest'].get('teacher_cost_dollars', 0))


def run_cell(batch, index):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    if ROOT.resolve() != (batch / 'code').resolve():
        raise ValueError('Worker must execute the archived source')
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Worker environment differs from preparation')
    record = manifest['cells'][index]
    args = record['args']
    directory = batch / 'cells' / str(index)
    directory.mkdir(exist_ok=False)
    write(directory / 'dispatch.json', dict(
        cell=record, manifest_sha256=digest(batch / 'manifest.json'),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
        slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID')))
    outcome = dict(returncode=None, artifact_status='failed', runs=[])
    try:
        runs_root = ROOT / 'results/runs'
        if runs_root.exists() and list(runs_root.glob(
                f"*{args['experiment_id']}*__{args['seed']}__*")):
            raise FileExistsError('An earlier attempt exists for this cell')
        process = subprocess.run([sys.executable, '-u', '-m',
                                  record['trainer'], *trainer_argv(args)],
                                 cwd=ROOT)
        outcome['returncode'] = process.returncode
        run = find_run(runs_root, args)
        outcome['runs'] = [run.relative_to(batch).as_posix()]
        if process.returncode == 0:
            outcome['metrics'] = validate_run(run, args)
            outcome['artifact_status'] = 'terminal_contract_validated'
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
    outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'exit.json', outcome)
    return int(outcome['artifact_status'] != 'terminal_contract_validated')


def submit(batch):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    submitted = batch / 'SUBMITTED_JOB'
    if submitted.exists():
        print(f'Already submitted: {submitted.read_text().strip()}')
        return
    n = len(manifest['cells'])
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write(f'One unthrottled array, {n} free cells.\n')
    name = 'rule_bank' if manifest['study'] == STUDY else 'rule_bank_blind'
    command = ['sbatch', '--parsable', f'--job-name={name}',
               f'--array=0-{n - 1}', '--time=3-00:00:00',
               f'--output={batch}/slurm/%x_%A_%a.out',
               str(batch / 'code/scripts/submit_rule_bank_pilot.sh'),
               str(batch)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f"{manifest['study']}: {job} ({n} cells)")


# ----------------------------------------------------------------- report

def paired(a, b, label):
    keys = sorted(set(a) & set(b))
    if any(a[k]['initial_sha256'] != b[k]['initial_sha256'] for k in keys):
        raise ValueError('Paired initial policies differ')
    delta = np.array([a[k]['auc'] - b[k]['auc'] for k in keys])
    n = len(delta)
    mean = float(delta.mean()) if n else None
    half = (float(t.ppf(.975, n - 1) * delta.std(ddof=1) / np.sqrt(n))
            if n > 1 else None)
    return dict(contrast=label, n=n, mean=mean,
                ci95=None if half is None else [mean - half, mean + half],
                positive=int((delta > 0).sum()))


def report(root, batches=None):
    batches = [Path(b) for b in batches or (
        batch_dir(root, STUDY), batch_dir(root, EXTENSION))
        if Path(b).exists()]
    by_arm = {}
    for batch in batches:
        for record in verify(batch)['cells']:
            try:
                run = find_run(batch / 'code/results/runs', record['args'])
                by_arm.setdefault(record['arm'], {})[record['seed']] = \
                    validate_run(run, record['args'])
            except (FileNotFoundError, ValueError, KeyError):
                continue
    arms = {arm: dict(n=len(rows), mean_auc=float(np.mean(
        [m['auc'] for m in rows.values()])), mean_labels=float(np.mean(
            [m['labels_delivered'] for m in rows.values()])))
        for arm, rows in by_arm.items()}
    pairs = [('llm_rules_scoped', 'llm_action_replay'),
             ('llm_rules_scoped', 'llm_rules_shuffled'),
             ('llm_rules_scope_checked', 'llm_rules_scoped'),
             ('llm_rules_scoped', 'llm_rules_direct'),
             ('llm_action_replay', 'none'),
             ('llm_rules_scoped', 'none'),
             ('llm_rules_blind_strict', 'llm_action_replay'),
             ('llm_rules_blind_strict', 'llm_rules_shuffled'),
             ('llm_rules_blind_strict', 'llm_rules_random_subset'),
             ('llm_rules_blind_strict', 'llm_rules_scoped'),
             ('llm_rules_random_subset', 'llm_rules_scoped'),
             ('llm_rules_blind_strict', 'none')]
    contrasts = {f'{a} - {b}': paired(by_arm.get(a, {}), by_arm.get(b, {}),
                                      f'{a} - {b}') for a, b in pairs}

    def passes(*labels):
        return all(contrasts[k]['n'] == 5 and contrasts[k]['mean'] > 0
                   and contrasts[k]['positive'] >= 4 for k in labels)
    result = dict(
        studies=[b.name for b in batches], arms=arms,
        contrasts=list(contrasts.values()),
        explanation_beyond_replay=passes(
            'llm_rules_scoped - llm_action_replay',
            'llm_rules_scoped - llm_rules_shuffled'),
        verified_explanation_beyond_replay=passes(
            'llm_rules_blind_strict - llm_action_replay',
            'llm_rules_blind_strict - llm_rules_shuffled'),
        verification_beyond_pruning=passes(
            'llm_rules_blind_strict - llm_rules_random_subset'),
        inference='development pilot; five seeds')
    print(json.dumps(result, indent=2))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('banks', 'check', 'prepare',
                                        'launch', 'report', 'run-cell'))
    cli.add_argument('--batch', type=Path, action='append',
                     help='repeat to report several batches together')
    cli.add_argument('--index', type=int)
    cli.add_argument('--extension', action='store_true',
                     help='act on the blind-check extension batch')
    args = cli.parse_args()
    study = EXTENSION if args.extension else STUDY
    if args.action == 'banks':
        (build_extension_banks if args.extension else build_banks)()
        return 0
    if args.action == 'run-cell':
        return run_cell(args.batch[0], args.index)
    if args.action == 'check':
        built = cells(study=study)
        print(f'PASS {study}: {len(built)} resolved commands; arms '
              f"{sorted({c['arm'] for c in built})}")
        return 0
    if args.action == 'report':
        report(ROOT, args.batch)
        return 0
    batch = prepare(ROOT, study)
    print(f'Prepared {batch}')
    if args.action == 'launch':
        submit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
