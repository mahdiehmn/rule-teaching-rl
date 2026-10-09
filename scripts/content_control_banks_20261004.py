"""Content-control banks: the final banks with every action target deranged.

Protocol:
research/fix_wave_protocol_2026-09-29.md, addendum 12.

Question. Do the source-task gains come from WHAT the rules recommend, or
from receiving an auxiliary cross-entropy target on the same states? The
existing shuffled-rule control (rule-bank study) answers a looser version:
it used the raw pre-selection banks, the paper weight, and it permuted
actions BETWEEN rules, which changes which states conflict and abstain.

Construction. Every rule keeps its
conditions, exceptions, mode and observer; only its action is replaced by a
fixed derangement SIGMA of the six MiniGrid actions (no action maps to
itself). Because SIGMA is a bijection, two applicable rules conflict after
the change exactly when they conflicted before, so the executor labels the
same states, abstains on the same states, and its target is always
SIGMA(original target): identical coverage, wrong content. MiniGrid banks
have no runtime precondition filter (KeyCorridor's preconditions were
applied offline to the rule conditions, which stay unchanged), so this bank
transformation is equivalent to deranging the executor's output.

SIGMA = a -> (a + 3) mod 6 pairs every turn/forward with an object action
and vice versa: left->pickup, right->drop, forward->toggle, pickup->left,
drop->right, toggle->forward. Frozen before any run.

    python -m scripts.content_control_banks_20261004 banks
    python -m scripts.content_control_banks_20261004 check
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = 'research/rule_banks/content_control_20261004'
# The banks of the paper's confirmed arms (fix-wave RULE_ARM / MEM_BANK).
BANKS = {
    'research/rule_banks/v3_20260928/blind_strict.json':
        f'{OUT_DIR}/doorkey_blind_strict_deranged.json',
    'research/rule_banks/multiroom_20260928/scoped.json':
        f'{OUT_DIR}/multiroom_scoped_deranged.json',
    'research/rule_banks/multiroom_20260928/blind_strict.json':
        f'{OUT_DIR}/multiroom_blind_strict_deranged.json',
    'research/rule_banks/keycorridor_mem_20260929/'
    'self_checked_pooled_valid.json':
        f'{OUT_DIR}/keycorridor_mem_pooled_valid_deranged.json',
}
N_ACTIONS = 6                  # left, right, forward, pickup, drop, toggle
SIGMA = {a: (a + 3) % N_ACTIONS for a in range(N_ACTIONS)}


def sha256(path):
    """Of the content with LF line ends (Windows and the cluster agree)."""
    return hashlib.sha256(Path(path).read_bytes().replace(
        b'\r\n', b'\n')).hexdigest()


def deranged(source, root=ROOT):
    bank = json.loads((Path(root) / source).read_text(encoding='utf-8'))
    if bank.get('mode') != 'scoped':
        raise ValueError(f'{source}: content control expects a scoped bank')
    rules = []
    for rule in bank['rules']:
        action = int(rule['action'])
        if action not in SIGMA:
            raise ValueError(f'{source}: action {action} has no derangement')
        rules.append(dict(rule, action=SIGMA[action]))
    bank['rules'] = rules
    bank['content_control_of'] = dict(
        bank=source, sha256=sha256(Path(root) / source),
        sigma={str(k): v for k, v in SIGMA.items()})
    return bank


def write_banks(root=ROOT):
    for source, target in BANKS.items():
        path = Path(root) / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(deranged(source, root), indent=1) + '\n',
                        encoding='utf-8')
        print('wrote', target)


def check_banks(root=ROOT):
    """Each bank equals its source with only the actions mapped by SIGMA."""
    assert all(SIGMA[a] != a for a in SIGMA)
    assert sorted(SIGMA.values()) == list(range(N_ACTIONS))
    for source, target in BANKS.items():
        src = json.loads((Path(root) / source).read_text(encoding='utf-8'))
        out = json.loads((Path(root) / target).read_text(encoding='utf-8'))
        assert out['content_control_of']['sha256'] == sha256(
            Path(root) / source), target
        assert len(out['rules']) == len(src['rules']), target
        for a, b in zip(src['rules'], out['rules']):
            assert a['condition'] == b['condition'], target
            assert a['exceptions'] == b['exceptions'], target
            assert SIGMA[int(a['action'])] == int(b['action']), target
        for key in set(src) | set(out):
            if key not in ('rules', 'content_control_of'):
                assert src.get(key) == out.get(key), (target, key)
    return True


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('banks', 'check'))
    args = cli.parse_args()
    if args.action == 'banks':
        write_banks()
    print('check', check_banks())


if __name__ == '__main__':
    main()
