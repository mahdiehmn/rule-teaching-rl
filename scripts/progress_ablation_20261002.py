"""Progress-only ablation banks: the selected rules minus their progress clauses.

Protocol:
research/fix_wave_protocol_2026-09-29.md, addendum 10 (KeyCorridor).

The offline counterfactual (scripts/analyze_executable_teaching_20261002.py,
parsed_rules(strip_progress=True)) shows at the level of ADVICE that the
KeyCorridor bank's progress clauses are what suppress the repeated
key-pickup labels after unlocking (55/323 saved states -> 0 -> 55). This
writes the same counterfactual as a TRAINING bank, so the effect on LEARNING
can be measured: every condition and exception on a progress predicate
(`door_unlocked`, `done_*`) is removed; every other condition, exception,
action, the mode and the observer stay exactly as they are. Rules that
become identical are kept as they are (the matcher treats identical rules
as one recommendation), so label counts follow that counterfactual.

    python -m scripts.progress_ablation_20261002 banks
    python -m scripts.progress_ablation_20261002 check
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = 'research/rule_banks/progress_ablation_20261002'
BANKS = {
    'research/rule_banks/keycorridor_mem_20260929/'
    'self_checked_pooled_valid.json':
        f'{OUT_DIR}/keycorridor_mem_noprogress.json',
    # Crafter v3b (research/crafter_vulcan_protocol_2026-10-02.md): its
    # done_<achievement> conditions removed, everything else kept.
    'research/rule_banks/crafter_v3_20261001/v3b_preconditions.json':
        f'{OUT_DIR}/crafter_v3b_noprogress.json',
}


def is_progress(key):
    return key == 'door_unlocked' or key.startswith('done_')


def sha256(path):
    """Of the content with LF line ends (Windows checkouts and the cluster
    agree)."""
    return hashlib.sha256(Path(path).read_bytes().replace(
        b'\r\n', b'\n')).hexdigest()


def ablated(source, root=ROOT):
    bank = json.loads((Path(root) / source).read_text(encoding='utf-8'))
    removed = dict(conditions=0, exceptions=0)
    rules = []
    for rule in bank['rules']:
        condition = {k: v for k, v in rule['condition'].items()
                     if not is_progress(k)}
        exceptions = [e for e in rule['exceptions'] if not is_progress(e[0])]
        removed['conditions'] += len(rule['condition']) - len(condition)
        removed['exceptions'] += len(rule['exceptions']) - len(exceptions)
        rules.append(dict(rule, condition=condition, exceptions=exceptions))
    bank['rules'] = rules
    bank['progress_ablation_of'] = dict(
        bank=source, sha256=sha256(Path(root) / source), removed=removed)
    return bank


def write_banks(root=ROOT):
    for source, target in BANKS.items():
        path = Path(root) / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ablated(source, root), indent=1) + '\n',
                        encoding='utf-8')
        print('wrote', target, ablated(source, root)['progress_ablation_of'])


def check_banks(root=ROOT):
    """Each ablated bank: its source minus progress clauses, nothing else."""
    for source, target in BANKS.items():
        src = json.loads((Path(root) / source).read_text(encoding='utf-8'))
        out = json.loads((Path(root) / target).read_text(encoding='utf-8'))
        if out != ablated(source, root):
            raise ValueError(f'{target} is not the progress ablation of '
                             f'{source}')
        if any(is_progress(k) for r in out['rules'] for k in r['condition']) \
                or any(is_progress(e[0]) for r in out['rules']
                       for e in r['exceptions']):
            raise ValueError(f'{target} still has progress clauses')
        if len(out['rules']) != len(src['rules']) or \
                out.get('observer') != src.get('observer'):
            raise ValueError(f'{target}: rule count or observer changed')
        if not out['progress_ablation_of']['removed']['conditions'] + \
                out['progress_ablation_of']['removed']['exceptions']:
            raise ValueError(f'{target}: nothing was removed')
    print(f'PASS {len(BANKS)} progress-ablation bank(s)')
    return True


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('banks', 'check'))
    args = cli.parse_args()
    if args.action == 'banks':
        write_banks()
    check_banks()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
