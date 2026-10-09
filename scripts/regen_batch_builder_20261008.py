"""Build a regenerated-bank batch INSIDE the archived fix-wave source.

Run by scripts/run_regen_training_20261008
with the working directory and PYTHONPATH set to <batch>/code, i.e. the
fix wave's own archived commit 875e5bb, so cells are made by that commit's
`cell()` (argument round trip through its trainer) and the runtime and
digests are that commit's. Each cell copies one completed fresh-cohort cell
of the selected-bank arm and changes only the bank file, its digest and
the experiment id.

  python -m scripts.regen_batch_builder_20261008 <batch> <fresh manifest> \
      <task> <suite>
"""

import copy
import json
from pathlib import Path
import sys

from algos import ppo_distill as ppo
from scripts import run_fix_wave_20260929 as fw
from teachers.minigrid.rule_bank import file_sha256

SELECTED_ARM = {'multiroom': 'rules_weak', 'keycorridor': 'rules_mem_weak'}
BANK_DIR = 'research/rule_banks/regen_20261008'
REPS = range(5)
CHANGED = {'rule_bank', 'rule_bank_sha256', 'experiment_id'}


def main(batch, fresh_path, task, suite):
    batch, code = Path(batch).resolve(), Path.cwd().resolve()
    if code != (batch / 'code').resolve() or fw.ROOT.resolve() != code:
        raise SystemExit('Run inside <batch>/code with PYTHONPATH=<batch>/code')
    fresh = json.loads(Path(fresh_path).read_text(encoding='utf-8'))
    sources = [c for c in fresh['cells'] if c['arm'] == SELECTED_ARM[task]]
    if len(sources) != 20:
        raise SystemExit('Expected 10 replicates x 2 students of the bank arm')
    cells, banks = [], {}
    for source in sources:
        for rep in REPS:
            old = source['args']
            path = f'{BANK_DIR}/{task}/rep_{rep}/{Path(old["rule_bank"]).name}'
            if path not in banks:
                banks[path] = file_sha256(code / path)
            values = copy.deepcopy(old)
            values.update(rule_bank=path, rule_bank_sha256=banks[path],
                          experiment_id=(f'{fw.STUDY}_{suite}_'
                                         f'{source["bonus"]}_regen{rep}'))
            changed = {k for k in values if values[k] != old[k]}
            if changed - CHANGED or set(values) != set(old):
                raise SystemExit(f'Unexpected change: {sorted(changed)}')
            fw.cell(cells, suite, f'regen{rep}', ppo.Args(**values),
                    source['replicate'])
    manifest = dict(
        study=fw.STUDY, suite=suite, commit=fresh['commit'],
        runtime=fw.runtime_identity(), cells=cells,
        protocol='research/regen_banks_protocol_2026-10-08.md', api_calls=0,
        required=fresh['required'],
        source_hashes={p.relative_to(code).as_posix(): fw.digest(p)
                       for p in code.rglob('*') if p.is_file()},
        reused_controls=dict(suite=fresh['suite'],
                             arms=['none', SELECTED_ARM[task]],
                             manifest_sha256=fw.digest(Path(fresh_path))),
        fresh_runtime=fresh['runtime'], regenerated_banks=banks)
    fw.write(batch / 'manifest.json', manifest)
    (batch / 'manifest.sha256').write_text(fw.digest(batch / 'manifest.json'))
    for name in ('cells', 'slurm'):
        (batch / name).mkdir(exist_ok=True)
    (batch / 'READY').write_text(manifest['commit'] + '\n')
    fw.verify(batch, rebuild=False)   # the archived worker's own admission
    same = manifest['runtime'] == fresh['runtime']
    print(json.dumps(dict(suite=suite, cells=len(cells), banks=len(banks),
                          runtime_matches_fresh_cohort=same,
                          runtime=manifest['runtime']), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
