"""Valid-action rules: fire a rule only where its action has an effect.

Many wrong labels in KeyCorridor come from MiniGrid mechanics rather
than from what the student cannot see: toggling a locked door that is
visible but not in front, walking forward into a key, dropping the key
onto a doorway. The student can tell from its own front cell whether an
action can do anything, so each rule is restricted to the front cells (and
carried object) where its action has an effect:

  forward  front empty, open door or goal
  pickup   front key or ball, carrying nothing
  drop     front empty, carrying something
  toggle   front closed or locked door
  left/right unchanged; done never

A rule that already names a front cell or carried object keeps it (and is
dropped if that makes its action a no-op); otherwise it is split into one
copy per allowed value, so the bank format and the trainer are unchanged.
No LLM call is made. On DoorKey and MultiRoom the transform leaves the
advice unchanged; on KeyCorridor's frozen raw rules it raises confirmation
precision from .694 to .782 with 487 instead of 404 correct labels.

  build     write <banks>/scoped_valid.json from <banks>/scoped.json
  compare   precision and coverage of both banks on the KC panels
"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts import conditional_rules_keycorridor as kc
from scripts import conditional_rules_v3 as v3
from teachers.minigrid.rule_bank import load_bank

TRANSFORM = 'valid_action_front_v1'
FORWARD, PICKUP, DROP, TOGGLE = 2, 3, 4, 5
VALID_FRONT = {FORWARD: {'empty', 'door_open', 'goal'},
               PICKUP: {'key', 'ball'},
               DROP: {'empty'},
               TOGGLE: {'door_closed', 'door_locked'}}
NOTHING = 'nothing'


def restrict(rules, fields):
    """Each rule, limited to front cells where its action has an effect."""
    out = []
    for condition, action, exceptions in rules:
        if action in (0, 1):
            out.append((condition, action, exceptions))
            continue
        if action not in VALID_FRONT:
            continue
        carried = set(fields.get('carrying', ()))
        need = {PICKUP: {NOTHING}, DROP: carried - {NOTHING}}.get(action)
        allowed = VALID_FRONT[action] & set(fields['front'])
        front_opts = ([condition['front']] if 'front' in condition
                      else sorted(allowed))
        carry_opts = ([None] if need is None else
                      [condition['carrying']] if 'carrying' in condition
                      else sorted(need))
        for front in front_opts:
            if front not in allowed:
                continue
            for carry in carry_opts:
                if need is not None and carry not in need:
                    continue
                new = dict(condition, front=front)
                if carry is not None:
                    new['carrying'] = carry
                out.append((new, action, exceptions))
    return out


def build(banks=kc.BANKS, fields=kc.FIELDS):
    banks = Path(banks)
    source, target = banks / 'scoped.json', banks / 'scoped_valid.json'
    if target.exists():
        raise ValueError(f'{target} exists; banks are frozen')
    bank = json.loads(source.read_text(encoding='utf-8'))
    _, rules, _ = load_bank(source)
    valid = restrict(rules, fields)
    # hash the committed (LF) form, not a Windows checkout's CRLF bytes
    committed = source.read_bytes().replace(b'\r\n', b'\n')
    bank.update(rules=[dict(condition=c, action=a,
                            exceptions=[list(e) for e in x])
                       for c, a, x in valid],
                derived_from=dict(bank=source.name, sha256=hashlib.sha256(
                    committed).hexdigest(), transform=TRANSFORM,
                    script='scripts/valid_action_rules_20260928.py'))
    with open(target, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(f'{source.name}: {len(rules)} rules -> {target.name}: '
          f'{len(valid)} rules (no LLM calls)')


def compare(banks=kc.BANKS, out=kc.OUT):
    states = json.loads((Path(out) / 'panels.json').read_text())
    for name in ('scoped', 'scoped_valid'):
        _, rules, _ = load_bank(Path(banks) / f'{name}.json')
        for panel in ('develop', 'confirm'):
            c = v3.coverage(lambda s: v3.advise(rules, s['pred']),
                            states[panel])
            print(f"{name:13s} {panel:8s} rules {len(rules):3d}  advised "
                  f"{c['advised']:4d}/{c['states']}  correct {c['correct']:4d}"
                  f"  precision {c['precision']:.3f}")


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('build', 'compare'))
    {'build': build, 'compare': compare}[cli.parse_args().action]()


if __name__ == '__main__':
    main()
