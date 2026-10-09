"""Banks for the confirmation cohort and the consultation-budget arms.

Question: can an explanation make one
teacher consultation useful across multiple situations, beyond simply
replaying its action label? No LLM call; everything comes from the
frozen consultations and blind checks of each task.

For DoorKey (research/rule_banks/v3_20260928), MultiRoom and KeyCorridor
this writes:

  random_subset_r5 .. r19  size-matched random raw-rule subsets for the
                           15 fresh confirmation replicates, drawn exactly
                           as r0..r4 were (the script checks r0..r4 are
                           reproduced byte for byte in their rules);
  scoped_12                the raw rules of the FIRST 12 consultations
                           (four from each task stretch, in the frozen
                           consultation order);
  blind_strict_12          those rules after the same blind strict check.

The 12-consultation banks record their own LLM call counts: 12
consultations, plus the blind checks of their rules. They give the rule
method a second budget point, and need no new call because each rule's
check is independent of the others.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import conditional_rules_keycorridor as kc
from scripts import conditional_rules_multiroom as mr
from scripts import conditional_rules_v3 as v3
from scripts.run_rule_bank_pilot_20260928 import receipts_cost, rule_json

TASKS = {
    'doorkey_8x8': (v3, Path('results/conditional_rules_v3_20260928'),
                    Path('research/rule_banks/v3_20260928'), 'doorkey_v3'),
    'multiroom_n6': (mr, mr.OUT, mr.BANKS, 'multiroom_v1'),
    'keycorridor_s3r3': (kc, kc.OUT, kc.BANKS, 'keycorridor_v1'),
}
FIRST = 12
REPLICATES = range(20)


def subset(raw, size, r):
    return sorted(np.random.default_rng([20260929, r]).choice(
        len(raw), size, replace=False).tolist())


def write(path, bank):
    if path.exists():
        raise ValueError(f'{path} exists; banks are frozen')
    path.write_text(json.dumps(bank, indent=1, sort_keys=True) + '\n')


def build(task):
    module, out, banks, observer = TASKS[task]
    panels, parsed = module.consult_rules(out)
    raw = [rule for _, _, rule, _ in parsed if rule is not None]
    kept, _ = v3.blind_filtered(out, parsed, panels, strict=True)
    common = dict(task=task, observer=observer, model=v3.MODEL,
                  study='confirm_banks_20260928', source_replies_sha256={
                      stage: hashlib.sha256((out / f'{stage}_replies.jsonl')
                                            .read_bytes()).hexdigest()
                      for stage in ('consult', 'blind')})
    consult = receipts_cost(out / 'consult_replies.raw.jsonl')
    for r in REPLICATES:
        pick = subset(raw, len(kept), r)
        path = banks / f'random_subset_r{r}.json'
        if r < 5:                               # frozen: must reproduce
            frozen = json.loads(path.read_text())
            if frozen['rules'] != [rule_json(raw[i]) for i in pick]:
                raise ValueError(f'{path} is not reproduced')
            continue
        write(path, dict(mode='scoped', rules=[rule_json(raw[i])
                                               for i in pick],
                         raw_rule_indices=pick, cost=dict(consult=consult),
                         **common))
    first = [(s, now, rule if k < FIRST else None,
              st if k < FIRST else 'outside_budget')
             for k, (s, now, rule, st) in enumerate(parsed)]
    raw12 = [rule for _, _, rule, _ in first if rule is not None]
    kept12, status = v3.blind_filtered(out, first, panels, strict=True)
    plan = {p['case']: p for p in json.loads(
        (out / 'refine_plan.json').read_text())}
    checks = sum(1 for k, (_, _, rule, _) in enumerate(first)
                 if rule is not None and plan.get(k, {}).get('situations'))
    calls = dict(consultations=FIRST, blind_checks=checks)
    write(banks / 'scoped_12.json', dict(
        mode='scoped', rules=[rule_json(r) for r in raw12],
        llm_calls=dict(consultations=FIRST), first_consultations=FIRST,
        **common))
    write(banks / 'blind_strict_12.json', dict(
        mode='scoped', rules=[rule_json(r) for r in kept12],
        llm_calls=calls, blind_check_status=status,
        first_consultations=FIRST, **common))
    return dict(raw=len(raw), kept=len(kept), raw12=len(raw12),
                kept12=len(kept12), calls12=calls)


def main():
    for task in TASKS:
        print(task, build(task))


if __name__ == '__main__':
    main()
