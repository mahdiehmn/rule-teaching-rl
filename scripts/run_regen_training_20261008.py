"""Prepare and submit the regenerated-bank learning batches (user runs).

Protocol:
research/regen_banks_protocol_2026-10-08.md. One batch per task in the fix
wave's layout, results/fix_wave/fix_wave_20260929_v1/<mr|kc>_regen. Its
code is the fix wave's archived commit 875e5bb, checked file by file
against the fresh cohort's recorded source hashes, plus the regenerated
bank files and this study's builder and protocol. 100 cells per task:
5 regenerated banks x the fresh cohort's 10 replicates x 2 students. They
pair with the fresh cohort's completed no-advice and selected-bank cells
(reused, not rerun). The archived fix-wave worker runs every cell.

  python -m scripts.run_regen_training_20261008 --fresh <dir> [--submit \
      --account def-zaiane_cpu]

<dir> holds mr_fresh/ and kc_fresh/ with their manifest.json files.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'fix_wave_20260929_v1'
COMMIT = '875e5bb0578f8771227b8d964f73b96bcc453169'
TASKS = {
    'multiroom': dict(fresh='mr_fresh', suite='mr_regen', job='rg_mr',
                      banks=('scoped.json', 'blind_strict.json')),
    'keycorridor': dict(fresh='kc_fresh', suite='kc_regen', job='rg_kc',
                        banks=('self_checked_pooled_valid.json',)),
}
BANK_DIR = 'research/rule_banks/regen_20261008'
EXTRA = ('scripts/regen_batch_builder_20261008.py',
         'scripts/run_regen_training_20261008.py',
         'research/regen_banks_protocol_2026-10-08.md')
REPS = range(5)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def batch_dir(task):
    return ROOT / 'results/fix_wave' / STUDY / TASKS[task]['suite']


def prepare(task, fresh_dir):
    spec, batch = TASKS[task], batch_dir(task)
    if batch.exists():
        if not (batch / 'READY').is_file():
            raise SystemExit(f'Incomplete batch {batch}; inspect it')
        print('Prepared already:', batch)
        return batch
    fresh_path = Path(fresh_dir) / spec['fresh'] / 'manifest.json'
    fresh = json.loads(fresh_path.read_text(encoding='utf-8'))
    marker = fresh_path.with_name('manifest.sha256')
    if (fresh['commit'] != COMMIT or fresh['study'] != STUDY
            or sha256(fresh_path.read_bytes())
            != marker.read_text().strip()):
        raise SystemExit('Fresh-cohort manifest differs from its record')
    archive = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip',
         COMMIT], cwd=ROOT)
    batch.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.regen-', dir=batch.parent) as t:
        stage = Path(t) / 'batch'
        code = stage / 'code'
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            files = {n: sha256(z.read(n)) for n in z.namelist()
                     if not n.endswith('/')}
            if files != fresh['source_hashes']:
                raise SystemExit('Archived source differs from the fresh '
                                 'cohort source, file by file')
            z.extractall(code)
        for rep in REPS:
            for name in spec['banks']:
                rel = f'{BANK_DIR}/{task}/rep_{rep}/{name}'
                (code / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, code / rel)
        for rel in EXTRA:
            (code / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, code / rel)
        env = dict(os.environ, PYTHONPATH=str(code))
        subprocess.run([sys.executable, '-u', '-m',
                        'scripts.regen_batch_builder_20261008', str(stage),
                        str(fresh_path.resolve()), task, spec['suite']],
                       cwd=code, env=env, check=True)
        stage.rename(batch)
    print('Prepared:', batch)
    return batch


def submit(task, account):
    spec, batch = TASKS[task], batch_dir(task)
    if (batch / 'SUBMITTED_JOB').exists():
        print(spec['job'], 'already submitted:',
              (batch / 'SUBMITTED_JOB').read_text().strip())
        return
    cells = json.loads((batch / 'manifest.json').read_text())['cells']
    # Measured: 5M fresh-cohort runs need ~4-5 h and ~1.4 GB.
    command = ['sbatch', '--parsable', f'--job-name={spec["job"]}',
               f'--array=0-{len(cells) - 1}', '--time=1-00:00:00',
               '--mem=4G', f'--output={batch}/slurm/%x_%A_%a.out']
    if account:
        command.append(f'--account={account}')
    command += [str(batch / 'code/scripts/submit_fix_wave.sh'), str(batch)]
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as handle:
        handle.write(shlex.join(command) + '\n')
    job = subprocess.check_output(command, text=True).strip()
    (batch / 'SUBMITTED_JOB').write_text(job + '\n')
    print(f'{spec["job"]}: {job} ({len(cells)} cells)')


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--fresh', type=Path, required=True)
    cli.add_argument('--task', choices=('all', *TASKS), default='all')
    cli.add_argument('--submit', action='store_true')
    cli.add_argument('--account', default='')
    args = cli.parse_args(argv)
    tasks = list(TASKS) if args.task == 'all' else [args.task]
    for task in tasks:
        prepare(task, args.fresh)
    if args.submit:
        for task in tasks:
            submit(task, args.account)
    else:
        print('Prepared only; add --submit to queue. No jobs submitted.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
