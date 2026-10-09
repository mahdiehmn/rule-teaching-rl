"""KeyCorridor rule bank with the `door_unlocked` progress fact.

Reuses the pipeline of scripts/conditional_rules_keycorridor.py (the base
pipeline): the same 36 consultation states, task text, model and settings,
rule form, scope semantics and valid-action restriction. It differs in
three ways.

1. Predicates. The student predicates include

     door_unlocked   yes once the locked door has been unlocked in this
                     episode, else no; known even when no door is in view

   Only the agent can unlock that door, so the value is determined by the
   agent's own history. The executor reads it from the episode (no locked
   door remains), which gives the same value.

2. Action preconditions. The action list states MiniGrid's own
   preconditions: pickup does nothing while carrying, and drop needs an
   EMPTY floor cell in front (an open doorway is not one).

3. Pooled checks. The self-check and blind check run on up to five pool
   situations per rule. Identical rules are treated as one rule checked on
   all of their situations (pooled_self_check): an original rule stays
   unchanged only if every copy's check leaves it unchanged. The selected
   bank is also reported on a second, held-out map set.

  build           requests for the 36 consultation states (0 calls)
  collect         --stage consult | refine | blind, each request once
                  (GPT-5-mini, Standard tier)
  export-checks   self-check and blind-check requests, base sampling
  banks           research/rule_banks/keycorridor_mem_20260929/ (0 calls):
                  scoped, self_checked, blind_strict and their _pooled
                  forms, each also with the valid-action restriction
  evaluate        precision on the develop/confirm panels, and every bank
                  run AS A POLICY on two sets of fresh maps, against the
                  base pipeline's banks and against random useful actions

Result (evaluation.json): the training bank, self_checked_pooled_valid,
solves 55/200 fresh maps as a policy (48/200 on the held-out set); random
useful actions solve 2/200. On the confirmation panel it gives 587 correct
labels of 762 (.770).
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts import conditional_rules_keycorridor as kc
from scripts import conditional_rules_multiroom as mr
from scripts import conditional_rules_v3 as v3

STUDY = 'conditional_rules_keycorridor_mem_20260929'
OBSERVER = 'keycorridor_mem_v1'
PROMPT_VERSION = 'scoped_rule_kc_mem_v1'
OUT = Path('results') / STUDY
V1_OUT = Path('results') / kc.STUDY
BANKS = Path('research/rule_banks/keycorridor_mem_20260929')
PRICE = (0.25, 2.00)            # USD per 1M input / output tokens
POLICY_SEED0 = 31_000_000       # fresh maps, disjoint from the panels

MEMORY = ('- door_unlocked: yes once the student itself has unlocked the '
          'locked door in this episode (it remembers doing so), otherwise '
          'no. It is known even when no door is in view.\n')
ACTION_TEXT = (
    'ACTIONS: 0 turn_left, 1 turn_right, 2 forward, 3 pickup (the key or '
    'ball in front; does nothing while already carrying an object), 4 drop '
    '(the carried object onto the cell in front; works only if that cell is '
    'EMPTY floor: an open door is a doorway, not an empty cell, so nothing '
    'can be dropped onto it), 5 toggle (open/close the door in front; '
    'unlocking the locked door needs the key), 6 done (does nothing here). '
    'A wall, a closed or locked door, a key or a ball in front blocks '
    'forward.\n')
FIELDS = dict(kc.FIELDS, door_unlocked=('yes', 'no'))
_ANCHOR = '- carrying: nothing or key\n'
assert kc.VOCAB.count(_ANCHOR) == 1 and kc.VOCAB.endswith(kc.ACTION_TEXT)
VOCAB = (kc.VOCAB[:-len(kc.ACTION_TEXT)].replace(_ANCHOR, _ANCHOR + MEMORY)
         + ACTION_TEXT)


# ---------------------------------------------------------------- observe

def door_unlocked(u):
    """door_unlocked: no locked door remains in this episode."""
    return 'no' if any(
        (c := u.grid.get(x, y)) is not None and c.type == 'door'
        and c.is_locked for x in range(u.width)
        for y in range(u.height)) else 'yes'


def door_unlocked_from_facts(facts):
    return 'no' if any(f['kind'] == 'door' and f.get('state') == 'locked'
                       for f in facts) else 'yes'


def observe_kc_mem(img, u):
    """Base KeyCorridor predicates from the 7x7 view, plus door_unlocked."""
    return dict(kc.observe_kc(img), door_unlocked=door_unlocked(u))


# ---------------------------------------------------------------- prompts

def consult_prompt(state):
    """The base consultation prompt with the door_unlocked predicate."""
    return (
        f'PROMPT VERSION: {PROMPT_VERSION}\n' + kc.TASK_TEXT +
        ' You see the full state; the student sees only its 7x7 view and '
        'remembers whether it has unlocked the locked door.\n'
        f'FULL MAP (arrow = agent, D = door, k = key, o = ball):\n'
        f'{state["full_map"]}\n'
        f'OBJECT FACTS (teacher-visible): {json.dumps(state["facts"])}\n'
        f'{mr.POSE_TEXT}{state["pose"]}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB + v3.SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


def refine_prompt(rule, situations):
    """The base self-check prompt with the door_unlocked predicate."""
    return (
        'PROMPT VERSION: scope_check_kc_mem_v1\n' + kc.TASK_TEXT +
        f' Earlier you gave the student this reusable rule:\n'
        f'{v3.rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. Using the full state, judge for EACH '
        'situation whether the rule\'s action is the best action there, and '
        'give the best action. Then return a REFINED rule that keeps the '
        'rule\'s action wherever it is best but excludes situations where it '
        'is not (add conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + VOCAB + v3.SEMANTICS + mr.CONVENTIONS
        + mr._situations(situations, True))


def blind_prompt(situations):
    """The base blind check, with the exact action preconditions."""
    return (
        'PROMPT VERSION: blind_check_kc_mem_v1\n' + kc.TASK_TEXT +
        ' For EACH situation below, give the single best next action for the '
        'agent, using the full state.\n' + ACTION_TEXT + mr.CONVENTIONS
        + mr._situations(situations, False))


def rule_schema(with_now=True):
    cond = {k: {'type': 'string', 'enum': ['any', *v]}
            for k, v in FIELDS.items()}
    exception = dict(type='object', additionalProperties=False,
                     required=['field', 'value'], properties=dict(
                         field={'type': 'string', 'enum': list(FIELDS)},
                         value={'type': 'string', 'enum': sorted(
                             {v for vs in FIELDS.values() for v in vs})}))
    props = dict(
        abstain={'type': 'boolean'},
        condition=dict(type='object', additionalProperties=False,
                       required=list(FIELDS), properties=cond),
        action={'type': 'integer', 'enum': list(range(7))},
        exceptions=dict(type='array', maxItems=2, items=exception),
        rationale={'type': 'string', 'maxLength': 240})
    if with_now:
        props = dict(action_now={'type': 'integer', 'enum': list(range(7))},
                     **props)
    return dict(type='object', additionalProperties=False,
                required=list(props), properties=props)


def refine_schema():
    schema = rule_schema(False)
    schema['properties'] = dict(
        verdicts=v3.refine_schema()['properties']['verdicts'],
        **schema['properties'])
    schema['required'] = ['verdicts', *schema['required']]
    return schema


def parse(answer, with_now=True):
    now = int(answer['action_now']) if with_now else None
    if answer['abstain']:
        return now, None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in FIELDS[e['field']])
    return now, (condition, int(answer['action']), exceptions)


# ------------------------------------------------------------ pipeline

def panels_with_memory(v1_out=V1_OUT):
    """The base frozen panels; every state's predicates include door_unlocked."""
    panels = json.loads((Path(v1_out) / 'panels.json').read_text())
    for name in ('consult', 'pool', 'develop', 'confirm'):
        for s in panels[name]:
            s['pred'] = dict(s['pred'], door_unlocked=
                             door_unlocked_from_facts(s['facts']))
    return panels


def request(case_id, stage, prompt, schema, version):
    b = v3.body(prompt, schema, version)
    return dict(case_id=case_id, condition=f'kc_mem_{stage}', split=stage,
                model=kc.MODEL, request=b, request_sha256=v3.digest(b))


def export(out, rows, name):
    (out / f'{name}.json').write_text(json.dumps(rows, indent=1))
    (out / f'{name}_manifest.json').write_text(json.dumps(dict(
        study=STUDY, stage=name, cases=len(rows),
        requests_sha256=v3.digest(rows), model=kc.MODEL, api_calls=0,
        v1_panels_sha256=hashlib.sha256(
            (V1_OUT / 'panels.json').read_bytes()).hexdigest()), indent=1))
    if rows:
        (out / f'example_{name}_prompt.txt').write_text(
            rows[0]['request']['input'][0]['content'])


def build(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    panels = panels_with_memory()
    rows = [request(f'consult_{k:03d}', 'consult', consult_prompt(s),
                    rule_schema(True), PROMPT_VERSION)
            for k, s in enumerate(panels['consult'])]
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(panels))
    export(out, rows, 'consult_requests')
    after = sum(s['pred']['door_unlocked'] == 'yes'
                for s in panels['consult'])
    print(f'{len(rows)} consultation requests in {out}; {after} of them '
          'after unlocking')


def export_checks(out=OUT):
    """Self-check and blind-check requests on the SAME pool situations
    (the base procedure: up to 5 pool situations where the rule applies)."""
    out = Path(out)
    panels, parsed = consult_rules(out)
    rng = np.random.default_rng(7)
    refine, blind, plan = [], [], []
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        matches = [s for s in panels['pool']
                   if v3.executable(rule, s['pred'])]
        if not matches:
            plan.append(dict(case=k, situations=0))
            continue
        pick = [matches[i] for i in sorted(rng.choice(
            len(matches), min(kc.K_SITUATIONS, len(matches)),
            replace=False))]
        refine.append(request(f'refine_{k:03d}', 'refine',
                              refine_prompt(rule, pick), refine_schema(),
                              'scope_check_kc_mem_v1'))
        blind.append(request(f'blind_{k:03d}', 'blind', blind_prompt(pick),
                             v3.blind_schema(), 'blind_check_kc_mem_v1'))
        plan.append(dict(case=k, situations=len(pick),
                         pool_matches=len(matches),
                         natives=[s['native'] for s in pick]))
    (out / 'refine_plan.json').write_text(json.dumps(plan, indent=1))
    export(out, refine, 'refine_requests')
    export(out, blind, 'blind_requests')
    print(f'{len(blind)} rules checked twice (self-check shows the rule, '
          f"blind check does not); {sum(p['situations'] == 0 for p in plan)}"
          ' rules had no pool match')


def refined_rules(out, parsed):
    """The base self-check outcome: the refined rule, or none if withdrawn."""
    replies = v3.read_replies(Path(out) / 'refine_replies.jsonl')
    refined, status = [], Counter()
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is None:
            continue
        row = replies.get(f'refine_{k:03d}')
        if row is None:
            refined.append(rule)
            status['kept_unchecked'] += 1
        elif row['response_status'] != 'completed':
            status['check_incomplete_dropped'] += 1
        else:
            _, new = parse(row['answer'], with_now=False)
            status['refined' if new else 'withdrawn'] += 1
            if new:
                refined.append(new)
    return refined, dict(status)


def rule_key(rule):
    condition, action, exceptions = rule
    return (tuple(sorted(condition.items())), action,
            tuple(sorted(exceptions)))


def copies(parsed):
    """Consultation indices of each distinct raw rule."""
    groups = {}
    for k, (_, _, rule, _) in enumerate(parsed):
        if rule is not None:
            groups.setdefault(rule_key(rule), []).append(k)
    return groups


def pooled_self_check(out, parsed):
    """Identical rules share their checks (no further calls).

    Several consultations can return the SAME rule; each copy was checked
    on its own five situations, and a copy whose sample missed a phase can
    pass where another copy's check found it wrong (here the pickup-key
    rule: four copies, one check saw states after unlocking and added
    door_unlocked=no, three did not). A rule stands unchanged only if the
    check of every copy left it unchanged; otherwise it is replaced by
    each distinct refinement, and a withdrawal or failed check adds none.
    """
    replies = v3.read_replies(Path(out) / 'refine_replies.jsonl')
    pooled, status = [], Counter()
    for ident, cases in copies(parsed).items():
        raw = parsed[cases[0]][2]
        outcomes = []
        for k in cases:
            row = replies.get(f'refine_{k:03d}')
            if row is None:                   # no pool match: unchecked
                outcomes.append(raw)
            elif row['response_status'] != 'completed':
                outcomes.append(None)
            else:
                outcomes.append(parse(row['answer'], with_now=False)[1])
        changed = [o for o in outcomes if o is None or rule_key(o) != ident]
        status['copies_%d' % len(cases)] += 1
        if not changed:
            pooled.append(raw)
            status['kept'] += 1
            continue
        status['replaced'] += 1
        for o in changed:
            if o is not None and rule_key(o) not in map(rule_key, pooled):
                pooled.append(o)
    return pooled, dict(status)


def pooled_blind_strict(out, parsed):
    """The strict blind check with identical rules pooled: a rule is kept
    only if the blind answers agree with it in every situation of every
    copy's check."""
    replies = v3.read_replies(Path(out) / 'blind_replies.jsonl')
    plan = {p['case']: p for p in json.loads(
        (Path(out) / 'refine_plan.json').read_text())}

    def agrees(k, action):
        row = replies.get(f'blind_{k:03d}')
        if row is None or row['response_status'] != 'completed':
            return False
        n = plan[k]['situations']
        answers = {a['situation']: a['best_action']
                   for a in row['answer']['answers'] if 0 <= a['situation'] < n}
        return all(answers.get(i) == action for i in range(n))
    kept, status = [], Counter()
    for cases in copies(parsed).values():
        rule = parsed[cases[0]][2]
        ok = all(agrees(k, rule[1]) for k in cases)
        status['kept' if ok else 'dropped'] += 1
        if ok:
            kept.append(rule)
    return kept, dict(status)


def collect(out, credential_file, stage='consult'):
    """Send each request of a stage once; replies in v3's reply format."""
    try:
        import truststore              # verify TLS with the OS store
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import dotenv_values
    from openai import OpenAI
    key = (dotenv_values(credential_file, encoding='utf-8-sig',
                         interpolate=False).get('OPENAI_API_KEY') or '')
    if not key.strip():
        raise ValueError('OPENAI_API_KEY is missing')
    client = OpenAI(api_key=key.strip(), timeout=300, max_retries=0)
    out = Path(out)
    replies = out / f'{stage}_replies.jsonl'
    if replies.exists():
        raise ValueError(f'{replies} exists; replies are frozen')
    rows = json.loads((out / f'{stage}_requests.json').read_text())
    spent = 0.0
    with replies.open('w', encoding='utf-8') as handle:
        for row in rows:
            t0 = time.perf_counter()
            rec = dict(case_id=row['case_id'],
                       request_sha256=row['request_sha256'])
            try:
                r = client.responses.create(**row['request'])
                usd = (r.usage.input_tokens / 1e6 * PRICE[0]
                       + r.usage.output_tokens / 1e6 * PRICE[1])
                spent += usd
                rec.update(response_status=r.status,
                           service_tier=r.service_tier,
                           response_model=r.model,
                           tokens_in=r.usage.input_tokens,
                           tokens_out=r.usage.output_tokens,
                           usd=round(usd, 6))
                rec['answer'] = json.loads(r.output_text)
            except Exception as error:           # recorded, never retried
                rec.update(response_status='failed',
                           error=f'{type(error).__name__}: {error}'[:300])
            rec['seconds'] = round(time.perf_counter() - t0, 2)
            rec['at'] = datetime.now(timezone.utc).isoformat()
            handle.write(json.dumps(rec) + '\n')
            handle.flush()
            print(f"{row['case_id']}: {rec['response_status']}", flush=True)
    print(f'spent ${spent:.4f}')


def consult_rules(out=OUT):
    panels = json.loads((Path(out) / 'panels.json').read_text())
    replies = v3.read_replies(Path(out) / 'consult_replies.jsonl')
    parsed = []
    for k, state in enumerate(panels['consult']):
        row = replies.get(f'consult_{k:03d}')
        if not row or row['response_status'] != 'completed':
            parsed.append((state, None, None, 'missing_or_incomplete'))
            continue
        now, rule = parse(row['answer'])
        parsed.append((state, now, rule, 'rule' if rule else 'abstained'))
    return panels, parsed


def build_banks(out=OUT, banks=BANKS):
    """Base-pipeline banks from the collected replies (no calls): the raw rules,
    the self-checked rules and the strictly blind-checked rules, each with
    identical rules' checks pooled or not, and each also under the
    valid-action restriction."""
    from scripts.run_rule_bank_pilot_20260928 import rule_json
    from scripts.valid_action_rules_20260928 import TRANSFORM, restrict
    out, banks = Path(out), Path(banks)
    panels, parsed = consult_rules(out)
    raw = [r for _, _, r, _ in parsed if r]
    refined, refine_status = refined_rules(out, parsed)
    kept, blind_status = v3.blind_filtered(out, parsed, panels, strict=True)
    pooled, pooled_status = pooled_self_check(out, parsed)
    pooled_kept, pooled_blind_status = pooled_blind_strict(out, parsed)

    def spent(stage):
        rows = [json.loads(line) for line in
                (out / f'{stage}_replies.jsonl').read_text().splitlines()]
        return dict(calls=len(rows), dollars=round(sum(
            r.get('usd', 0) for r in rows), 6))
    stages = ('consult', 'refine', 'blind')
    common = dict(
        study=STUDY, task=kc.TASK, observer=OBSERVER, model=kc.MODEL,
        prompt_version=PROMPT_VERSION, added_predicates=['door_unlocked'],
        source_replies_sha256={
            stage: hashlib.sha256((out / f'{stage}_replies.jsonl')
                                  .read_bytes()).hexdigest()
            for stage in stages})
    sets = dict(scoped=(raw, ('consult',), {}),
                self_checked=(refined, ('consult', 'refine'),
                              dict(self_check_status=refine_status)),
                blind_strict=(kept, ('consult', 'blind'),
                              dict(blind_check_status=blind_status)),
                self_checked_pooled=(pooled, ('consult', 'refine'),
                                     dict(self_check_status=pooled_status)),
                blind_strict_pooled=(pooled_kept, ('consult', 'blind'),
                                     dict(blind_check_status=
                                          pooled_blind_status)))
    made = {}
    for name, (rules, paid, extra) in sets.items():
        cost = {stage: spent(stage) for stage in paid}
        made[name] = dict(mode='scoped', rules=[rule_json(r) for r in rules],
                          cost=cost, **extra, **common)
        made[f'{name}_valid'] = dict(
            mode='scoped', rules=[rule_json(r) for r in
                                  restrict(rules, FIELDS)],
            transform=TRANSFORM, cost=cost, **extra, **common)
    banks.mkdir(parents=True, exist_ok=True)
    for name, bank in made.items():
        path = banks / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(
        consult=dict(Counter(p[3] for p in parsed)),
        self_check=refine_status, blind_check=blind_status,
        self_check_pooled=pooled_status,
        blind_check_pooled=pooled_blind_status,
        rules={name: len(bank['rules']) for name, bank in made.items()}),
        indent=1))


# ------------------------------------------------------------ evaluation

def precision(rules, states):
    """Labelled states and the share whose label is optimal."""
    labelled = correct = 0
    for s in states:
        action, status = v3.advise(rules, s['pred'])
        if status == 'advised':
            labelled += 1
            correct += action in s['optimal']
    return dict(labelled=labelled, correct=correct, states=len(states),
                precision=correct / labelled if labelled else None)


def as_policy(rules, observe, stateful, n=200, seed0=POLICY_SEED0):
    """The rules act; where none fires, a random useful action."""
    wins, steps, labelled, total = 0, [], 0, 0
    unlocked, repicks, repick_episodes = 0, 0, 0
    for i in range(n):
        env = kc.make_env(seed0 + i)
        u, rng = env.unwrapped, np.random.default_rng(seed0 + i)
        again = 0
        for t in range(u.max_steps):
            img = u.gen_obs()['image']
            action, status = v3.advise(
                rules, observe(img, u) if stateful else observe(img))
            total += 1
            labelled += status == 'advised'
            if action is None:
                action = int(rng.choice(kc.WALK))
            was_open = door_unlocked(u) == 'yes'
            empty = u.carrying is None
            _, reward, done, trunc, _ = env.step(action)
            if (was_open and empty and u.carrying is not None
                    and u.carrying.type == 'key'):
                again += 1
            if done or trunc:
                if reward > 0:
                    wins += 1
                    steps.append(t + 1)
                break
        unlocked += door_unlocked(u) == 'yes'
        repicks += again
        repick_episodes += again > 0
        env.close()
    return dict(solved=wins, episodes=n, unlocked=unlocked,
                median_steps=float(np.median(steps)) if steps else None,
                labelled_steps=labelled / max(total, 1),
                key_repicked_after_unlock=repicks,
                repick_episodes=repick_episodes)


HELD_OUT_SEED0 = 33_000_000     # a second map set, used once the bank was
                                # chosen on the first (pooling was added
                                # after the first set's results)


def evaluate(banks=BANKS, v1_banks=kc.BANKS, n=200):
    from teachers.minigrid.rule_bank import load_bank
    panels = panels_with_memory()
    result = {'no rules (random useful actions)': dict(
        rules=0, policy=as_policy([], observe_kc_mem, True, n=n),
        policy_held_out=as_policy([], observe_kc_mem, True, n=n,
                                  seed0=HELD_OUT_SEED0))}
    cases = [(f'v1 {name}', Path(v1_banks) / f'{name}.json', kc.observe_kc,
              False) for name in ('scoped', 'scoped_valid', 'blind_strict')]
    cases += [(f'mem {name}{valid}', Path(banks) / f'{name}{valid}.json',
               observe_kc_mem, True)
              for name in ('scoped', 'self_checked', 'blind_strict',
                           'self_checked_pooled', 'blind_strict_pooled')
              for valid in ('', '_valid')]
    for label, path, observe, stateful in cases:
        if not path.exists():
            continue
        _mode, rules, _ = load_bank(path)
        entry = dict(rules=len(rules), policy=as_policy(
            rules, observe, stateful, n=n), policy_held_out=as_policy(
            rules, observe, stateful, n=n, seed0=HELD_OUT_SEED0))
        for name in ('develop', 'confirm'):
            entry[name] = precision(rules, panels[name])
        result[label] = entry
    for label, entry in result.items():
        p, h = entry['policy'], entry['policy_held_out']
        c = entry.get('confirm') or dict(correct=0, labelled=0,
                                         precision=None)
        print(f"{label:34s} {entry['rules']:3d} rules | confirm "
              f"{c['correct']:4d}/{c['labelled']:4d} = "
              f"{c['precision'] or 0:.3f} | as a policy solves "
              f"{p['solved']:3d}/{p['episodes']} (held-out maps "
              f"{h['solved']:3d}/{h['episodes']}), unlocks {p['unlocked']:3d}"
              f", key re-picked {p['key_repicked_after_unlock']}x")
    return result


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('build', 'collect', 'export-checks',
                                        'banks', 'evaluate'))
    cli.add_argument('--out', type=Path, default=OUT)
    cli.add_argument('--banks', type=Path, default=BANKS)
    cli.add_argument('--credential-file', type=Path)
    cli.add_argument('--stage', choices=('consult', 'refine', 'blind'),
                     default='consult')
    cli.add_argument('--episodes', type=int, default=200)
    args = cli.parse_args()
    if args.action == 'build':
        build(args.out)
    elif args.action == 'collect':
        collect(args.out, args.credential_file, args.stage)
    elif args.action == 'export-checks':
        export_checks(args.out)
    elif args.action == 'banks':
        build_banks(args.out, args.banks)
    else:
        result = evaluate(args.banks, n=args.episodes)
        (Path(args.out) / 'evaluation.json').write_text(
            json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
