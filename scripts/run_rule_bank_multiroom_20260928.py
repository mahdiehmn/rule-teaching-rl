"""Learning study: frozen LLM rule banks on MultiRoom-N6 (second task).

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? The method is frozen from DoorKey; the
offline MultiRoom check is scripts/conditional_rules_multiroom.py, and
the predictions were written before any MultiRoom reply.

Every guided arm reads a bank written from the SAME 36 GPT-5-mini
consultations; the blind-strict bank also used the blind-check calls.
Training makes no API calls: the free `rule_bank` teacher reads only the
student's own 7x7 view (teachers/minigrid/rule_bank.py, observer
`multiroom_v1`).

Students: plain PPO and Count-PPO (`bonus` none / count) with the
reviewed rule-reference MultiRoom settings, five fresh paired seeds
14,600,000 + 100r, a 5M horizon. Arms, per student:

  none                    no teacher
  llm_action_replay       the LLM's action_now on identical views only
  llm_rules_scoped        all of the LLM's rules, v3 scope semantics
  llm_rules_shuffled      the same scopes with actions permuted
  llm_rules_blind_strict  the rules the blind strict check kept
  llm_rules_random_subset as many raw rules, chosen at random (a
                          different frozen subset per replicate)

Guided arms share the DoorKey pilot's distillation settings: dense labels
wherever the bank advises, no query cap, isolated advisor RNG.

Frozen rule, per student: the verified explanation is useful beyond
replay if `llm_rules_blind_strict` beats BOTH `llm_action_replay` and
`llm_rules_shuffled` with a positive mean teacher-off AUC and at least
4/5 positive pairs; verification, not mere pruning, matters if it also
beats `llm_rules_random_subset` by the same bar. Count-PPO is the primary
student: plain PPO scored zero with every rule-reference teacher at 480
labels, so a plain contrast in which every run scores zero is reported as
uninformative, not as a failure or a success.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np

from scripts.run_explanation_formats_20260925 import runtime_identity
from scripts.run_explanation_grid import trainer_argv
from scripts.run_plan_repair_20260927 import digest, inside, read, write
from scripts.run_rule_bank_pilot_20260928 import (find_run, paired,
                                                  validate_run)

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'rule_bank_multiroom_20260928_v1'
TASK = 'multiroom_n6'
BANKS = Path('research/rule_banks/multiroom_20260928')
SEEDS = tuple(14_600_000 + 100 * r for r in range(5))
HORIZON = 5_000_000
BONUSES = ('none', 'count')
ARMS = ('none', 'llm_action_replay', 'llm_rules_scoped',
        'llm_rules_shuffled', 'llm_rules_blind_strict',
        'llm_rules_random_subset')
BANK_OF = {'llm_action_replay': 'replay', 'llm_rules_scoped': 'scoped',
           'llm_rules_shuffled': 'shuffled',
           'llm_rules_blind_strict': 'blind_strict',
           'llm_rules_random_subset': 'random_subset_r{r}'}
REQUIRED = ('scripts/run_rule_bank_multiroom_20260928.py',
            'scripts/submit_rule_bank_multiroom.sh',
            'scripts/launch_rule_bank_multiroom.sh',
            'teachers/minigrid/rule_bank.py',
            'scripts/conditional_rules_multiroom.py',
            'tests/test_rule_bank_multiroom.py',
            *(f'{BANKS.as_posix()}/{name}.json' for name in
              ('replay', 'scoped', 'shuffled', 'blind_strict',
               *(f'random_subset_r{r}' for r in range(len(SEEDS))))))


# ------------------------------------------------------------------ cells

def base_args(bonus):
    from algos.ppo_distill import Args
    from scripts.run_rule_reference_20260927 import menu
    cells = {c['arm']: c['args'] for c in menu()
             if c['task'] == TASK and c['bonus'] == bonus
             and c['replicate'] == 0}
    return Args(**cells['none']), Args(**cells['random480'])


def cells(root=ROOT):
    from algos.ppo_distill import Args
    from teachers.minigrid.rule_bank import file_sha256
    import tyro
    out = []
    for bonus in BONUSES:
        none, guided = base_args(bonus)
        for r, seed in enumerate(SEEDS):
            for arm in ARMS:
                name = f'{STUDY}_{bonus}_{arm}'
                if arm == 'none':
                    args = replace(none, seed=seed, total_timesteps=HORIZON,
                                   experiment_id=name)
                else:
                    bank = (f'{BANKS.as_posix()}/'
                            f'{BANK_OF[arm].format(r=r)}.json')
                    args = replace(
                        guided, seed=seed, total_timesteps=HORIZON,
                        experiment_id=name, teacher='rule_bank',
                        rule_bank=bank,
                        rule_bank_sha256=file_sha256(Path(root) / bank),
                        advisor='unlimited', query_budget=0, advice_budget=0,
                        uniform_queries=False, advisor_no_teacher_peek=False,
                        importance_source='none', audit_explanations=False)
                if asdict(tyro.cli(Args, args=trainer_argv(asdict(args)),
                                   console_outputs=False)) != asdict(args):
                    raise ValueError('Trainer command changed settings')
                out.append(dict(index=len(out), bonus=bonus, arm=arm,
                                seed=seed, trainer='algos.ppo_distill',
                                args=asdict(args)))
    return out


# ------------------------------------------------------------ preparation

def batch_dir(root):
    return Path(root) / 'results/efficiency' / STUDY


def prepare(root):
    root = Path(root).resolve()
    batch = batch_dir(root)
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
        batch.mkdir(parents=True, exist_ok=False)
        zipped.extractall(batch / 'code')
    for name in ('cells', 'slurm'):
        (batch / name).mkdir()
    manifest = dict(study=STUDY, head=head, runtime=runtime_identity(),
                    cells=cells(batch / 'code'), api_calls_in_training=0,
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
            or manifest['study'] != STUDY
            or (batch / 'READY').read_text().strip() != manifest['head']):
        raise ValueError('Frozen preparation identity differs')
    for name, expected in manifest['source_hashes'].items():
        if digest(inside(batch / 'code', name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    return manifest


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
    command = ['sbatch', '--parsable', '--job-name=rule_bank_mr',
               f'--array=0-{n - 1}', '--time=3-00:00:00',
               f'--output={batch}/slurm/%x_%A_%a.out',
               str(batch / 'code/scripts/submit_rule_bank_multiroom.sh'),
               str(batch)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Unrecognized scheduler receipt; inspect attempt')
    write(batch / 'submission.json', dict(command=command, job_id=job))
    submitted.write_text(job + '\n')
    print(f'{STUDY}: {job} ({n} cells)')


# ----------------------------------------------------------------- report

def report(root, batch=None):
    batch = Path(batch or batch_dir(root))
    manifest = verify(batch)
    by = {}
    for record in manifest['cells']:
        try:
            run = find_run(batch / 'code/results/runs', record['args'])
            by.setdefault((record['bonus'], record['arm']), {})[
                record['seed']] = validate_run(run, record['args'])
        except (FileNotFoundError, ValueError, KeyError):
            continue
    result = dict(study=STUDY, students={})
    pairs = [('llm_rules_blind_strict', 'llm_action_replay'),
             ('llm_rules_blind_strict', 'llm_rules_shuffled'),
             ('llm_rules_blind_strict', 'llm_rules_random_subset'),
             ('llm_rules_blind_strict', 'llm_rules_scoped'),
             ('llm_rules_blind_strict', 'none'),
             ('llm_rules_scoped', 'llm_action_replay'),
             ('llm_action_replay', 'none')]
    for bonus in BONUSES:
        arms = {arm: dict(n=len(rows), mean_auc=float(np.mean(
            [m['auc'] for m in rows.values()])), mean_labels=float(np.mean(
                [m['labels_delivered'] for m in rows.values()])))
            for (b, arm), rows in by.items() if b == bonus}
        contrasts = {f'{a} - {c}': paired(by.get((bonus, a), {}),
                                          by.get((bonus, c), {}),
                                          f'{a} - {c}') for a, c in pairs}

        def passes(*labels):
            return all(contrasts[k]['n'] == 5 and contrasts[k]['mean'] > 0
                       and contrasts[k]['positive'] >= 4 for k in labels)
        result['students'][bonus] = dict(
            arms=arms, contrasts=list(contrasts.values()),
            verified_explanation_beyond_replay=passes(
                'llm_rules_blind_strict - llm_action_replay',
                'llm_rules_blind_strict - llm_rules_shuffled'),
            verification_beyond_pruning=passes(
                'llm_rules_blind_strict - llm_rules_random_subset'))
    result['inference'] = 'development study; five seeds per student'
    print(json.dumps(result, indent=2))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('check', 'prepare', 'launch',
                                        'report', 'run-cell'))
    cli.add_argument('--batch', type=Path)
    cli.add_argument('--index', type=int)
    args = cli.parse_args()
    if args.action == 'run-cell':
        return run_cell(args.batch, args.index)
    if args.action == 'check':
        built = cells()
        print(f'PASS {STUDY}: {len(built)} resolved commands; students '
              f"{sorted({c['bonus'] for c in built})}; arms "
              f"{sorted({c['arm'] for c in built})}")
        return 0
    if args.action == 'report':
        report(ROOT, args.batch)
        return 0
    batch = prepare(ROOT)
    print(f'Prepared {batch}')
    if args.action == 'launch':
        submit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
