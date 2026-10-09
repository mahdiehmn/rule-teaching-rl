"""Regenerate the MultiRoom and KeyCorridor rule banks five times each.

Extends the DoorKey regeneration study
(paper_optional_regeneration_2026-10-05, O4) to the other two MiniGrid
tasks. Each repetition repeats the task's frozen recipe end to end:

  * the SAME 36 consultation requests (copied byte for byte from the
    original run and checked against their pinned digest),
  * the recipe's own self-check / blind-check export (pool sampling RNG 7),
  * the recipe's own bank builder.

MultiRoom (conditional_rules_multiroom, Sept 28): scoped = all rules
(PPO's selected recipe), blind_strict (Count-PPO's). KeyCorridor
(conditional_rules_keycorridor_mem, Sept 29): self_checked_pooled_valid
(both students). Repetitions are never selected, repaired or replaced;
empty or duplicate banks are kept.

Output: results/regen_banks_20261008/<task>/rep_<k>/ with its replies and
banks/ (results/ is not tracked). Originals and selected banks are never
written. Paid calls happen only in the collect steps: GPT-5-mini, one
attempt per request, never retried; MultiRoom through the capped
collector, KeyCorridor through its recipe's own collect.

  python -m scripts.regenerate_banks_20261008 run --task multiroom --rep 0 \\
      --source <original results/conditional_rules_multiroom_20260928>
  python -m scripts.regenerate_banks_20261008 run --task keycorridor --rep 0 \\
      --source <original results/conditional_rules_keycorridor_mem_20260929> \\
      --v1-source <original results/conditional_rules_keycorridor_20260928>
  python -m scripts.regenerate_banks_20261008 summary
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('results/regen_banks_20261008')
REPS = range(5)
DOTENV = '.env'
TASKS = {
    'multiroom': dict(
        module='scripts.conditional_rules_multiroom',
        consult_sha256=('493fbc82121de01bf156eabe860450c8f3cb6d47'
                        '36ba7b9510d11bbca10dfc38'),
        selected=('scoped', 'blind_strict')),
    'keycorridor': dict(
        module='scripts.conditional_rules_keycorridor_mem',
        consult_sha256=('c0ef9bd2ab122750edfd2c29657d1872abf97141'
                        '918a4f8c8cdc857bdb85166c'),
        v1_panels_sha256=('8ff8aba779e7f5312e827e5c6b62b2fd0906b20b'
                          '6f2d88c51ee03fec7568f530'),
        selected=('self_checked_pooled_valid',)),
}
STAGES = ('consult', 'refine', 'blind')
STAGE_CAP_USD = 2.0          # MultiRoom collector cap per stage (~$0.7 bound)


def module(task):
    import importlib
    return importlib.import_module(TASKS[task]['module'])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rep_dir(task, rep):
    return OUT / task / f'rep_{rep}'


def prepare(task, rep, source, v1_source=None):
    """Copy the original consultation inputs; verify their pinned digest."""
    from scripts import conditional_rules_v3 as v3
    spec, out = TASKS[task], rep_dir(task, rep)
    if task == 'keycorridor':
        from scripts import conditional_rules_keycorridor as kc
        v1 = Path('results') / kc.STUDY / 'panels.json'
        if not v1.exists():
            if v1_source is None:
                raise SystemExit('KeyCorridor needs --v1-source once')
            v1.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(v1_source) / 'panels.json', v1)
        if sha256(v1) != spec['v1_panels_sha256']:
            raise SystemExit('KeyCorridor v1 panels differ from the recipe')
    if not out.exists():
        out.mkdir(parents=True)
        for name in ('panels.json', 'consult_requests.json',
                     'consult_requests_manifest.json'):
            shutil.copyfile(Path(source) / name, out / name)
    rows = json.loads((out / 'consult_requests.json').read_text())
    manifest = json.loads((out / 'consult_requests_manifest.json').read_text())
    if not (v3.digest(rows) == manifest['requests_sha256']
            == spec['consult_sha256']) or len(rows) != 36:
        raise SystemExit('Consultation requests differ from the original')
    return out


def complete(out, stage):
    replies = out / f'{stage}_replies.jsonl'
    requests = json.loads((out / f'{stage}_requests.json').read_text())
    return replies.exists() and len(
        replies.read_text().splitlines()) == len(requests)


def collect(task, out, stage, dotenv):
    if complete(out, stage):
        return
    if task == 'keycorridor':
        module(task).collect(out, Path(dotenv), stage)
        return
    # Verify TLS with the OS certificate store, as the KeyCorridor recipe's
    # own collect does; without it this Windows host fails every handshake.
    bootstrap = ('import runpy, truststore; truststore.inject_into_ssl(); '
                 "runpy.run_module('scripts.collect_prompt_reliability_20260927',"
                 " run_name='__main__', alter_sys=True)")
    subprocess.run([sys.executable, '-u', '-c', bootstrap,
                    '--requests', str(out / f'{stage}_requests.json'),
                    '--manifest', str(out / f'{stage}_requests_manifest.json'),
                    '--split', 'all', '--out', str(out / f'{stage}_replies'),
                    '--cap-usd', str(STAGE_CAP_USD), '--dotenv', dotenv,
                    '--workers', '6'], check=True)


def run(task, rep, source, v1_source, dotenv):
    out = prepare(task, rep, source, v1_source)
    mod = module(task)
    collect(task, out, 'consult', dotenv)
    if not (out / 'refine_requests.json').exists():
        mod.export_checks(out)
    for stage in ('refine', 'blind'):
        collect(task, out, stage, dotenv)
    banks = out / 'banks'
    if not banks.exists():
        mod.build_banks(out, banks)
    print(json.dumps(bank_summary(task, rep), indent=1))


def spent(out):
    """Dollars recorded in a repetition's replies (both reply formats)."""
    total = 0.0
    for stage in STAGES:
        raw = out / f'{stage}_replies.raw.jsonl'
        plain = out / f'{stage}_replies.jsonl'
        if raw.exists():
            for line in raw.read_text().splitlines():
                event = json.loads(line)
                if event.get('event') == 'END':
                    total += (event.get('dollars')
                              if event.get('dollars') is not None
                              else event.get('dollars_bound', 0.0))
        elif plain.exists():
            total += sum(json.loads(line).get('usd', 0.0)
                         for line in plain.read_text().splitlines())
    return round(total, 4)


def bank_summary(task, rep):
    out = rep_dir(task, rep)
    banks = out / 'banks'
    rules = {}
    for name in TASKS[task]['selected']:
        path = banks / f'{name}.json'
        rules[name] = (len(json.loads(path.read_text())['rules'])
                       if path.exists() else None)
    return dict(task=task, rep=rep, rules=rules, usd=spent(out),
                banks={name: sha256(banks / f'{name}.json')
                       for name in TASKS[task]['selected']
                       if (banks / f'{name}.json').exists()})


FROZEN = Path('research/rule_banks/regen_20261008')


def freeze(tasks=tuple(TASKS)):
    """Copy every repetition's selected banks into the tracked bank folder.

    Line endings become LF so the bytes (and the trainer's file digest) are
    the same here, in git and on the cluster. Refuses to change a frozen file.
    """
    index = dict(author='mahdiehmn', date='2026-10-08',
                 recipe_sources={t: s['module'] for t, s in TASKS.items()},
                 repetitions=[])
    for task in tasks:
        spec = TASKS[task]
        for rep in REPS:
            row = bank_summary(task, rep)
            if any(row['rules'][n] is None for n in spec['selected']):
                raise SystemExit(f'{task} rep {rep} has no banks yet')
            files = {}
            for name in spec['selected']:
                data = (rep_dir(task, rep) / 'banks' / f'{name}.json'
                        ).read_bytes().replace(b'\r\n', b'\n')
                target = FROZEN / task / f'rep_{rep}' / f'{name}.json'
                if target.exists() and target.read_bytes() != data:
                    raise SystemExit(f'{target} is frozen with other bytes')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                files[name] = hashlib.sha256(data).hexdigest()
            index['repetitions'].append(dict(
                task=task, rep=rep, rules=row['rules'], usd=row['usd'],
                bank_sha256=files))
    (FROZEN / 'index.json').write_bytes(
        (json.dumps(index, indent=1) + '\n').encode())
    print(json.dumps(index, indent=1))


def main(argv=None):
    os.chdir(ROOT)                  # recipes read results/ relative to here
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('run', 'summary', 'freeze'))
    cli.add_argument('--task', choices=tuple(TASKS))
    cli.add_argument('--rep', type=int, choices=tuple(REPS))
    cli.add_argument('--source', type=Path)
    cli.add_argument('--v1-source', type=Path)
    cli.add_argument('--dotenv', default=DOTENV)
    args = cli.parse_args(argv)
    if args.action == 'run':
        if args.task is None or args.rep is None or args.source is None:
            cli.error('run needs --task, --rep and --source')
        run(args.task, args.rep, args.source, args.v1_source, args.dotenv)
        return 0
    if args.action == 'freeze':
        freeze((args.task,) if args.task else tuple(TASKS))
        return 0
    rows = [bank_summary(t, r) for t in TASKS for r in REPS
            if rep_dir(t, r).exists()]
    for row in rows:
        print(json.dumps(row))
    print(f'total ${sum(r["usd"] for r in rows):.4f} over {len(rows)} reps')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
