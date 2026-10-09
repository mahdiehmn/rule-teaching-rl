"""Crafter rules v2: self-check and action preconditions on the Crafter bank.

Starts from the rules of the offline pilot
(scripts/crafter_rules_pilot_20260928.py), writes no new rules, and adds
two steps:

  self-check   the LLM sees each distinct rule (identical copies pooled
               first) with up to five fresh states where it fires, told
               where its action would change nothing, and narrows,
               replaces or withdraws it (one request per distinct rule,
               at most 36, $2 hard ceiling);
  preconditions  a rule fires only where its action can change something,
               decided from the student's own predicates with Crafter's
               own recipe table (crafter/data.yaml): `do` on a tree, water,
               grass, a creature or a ripe plant, or on stone, coal, iron
               or diamond with the pickaxe it needs; place_* with the
               materials and a valid cell in front; make_* with the
               materials and the table (and furnace) nearby; no `noop`;
               no move into a blocked cell the player already faces.

One bank is chosen on a development panel before the gate, which runs
once on fresh states and seeds with the pilot's frozen thresholds (GO: at
least +1.5 mean achievements over random, place_table >= 40%, wood pickaxe
>= 20%, coverage >= 20%, effective labels >= 70%). Only GO admits a
learning study.

  build-checks  check pool and self-check requests (0 calls)
  collect       send them once each (GPT-5-mini; ceiling enforced)
  banks         v2 banks with and without preconditions (0 calls)
  develop       every bank on the development panel; writes the choice
  gate          the chosen bank, once, on the untouched gate panel
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts import conditional_rules_v3 as v3
from scripts import crafter_rules_pilot_20260928 as v1

STUDY = 'crafter_rules_v2_20260930'
OUT = Path('results') / STUDY
BANKS = Path('research/rule_banks/crafter_v2_20260930')
PROMPT_VERSION = 'scope_check_crafter_v2'
# Fresh world seeds; v1 used 30.0M (consult), 30.1M (held out), 30.2M
# (rollouts). Check, development and gate never overlap.
CHECK_SEED0, N_CHECK = 30_300_000, 300
DEV_STATES_SEED0, N_DEV_STATES = 30_400_000, 200
DEV_ROLLOUT_SEED0, N_DEV_ROLLOUT = 30_450_000, 20
GATE_STATES_SEED0, N_GATE_STATES = 30_500_000, 600
GATE_ROLLOUT_SEED0, N_GATE_ROLLOUT = 30_600_000, 30
K_SITUATIONS, MAX_CANDIDATES = 5, 40
MAX_REQUESTS, CEILING = 36, 2.00
PRICE = (0.25, 2.00)            # USD per 1M input / output tokens
FILTER = 'crafter_preconditions_v1'
WALKABLE = ('grass', 'sand', 'path', 'lava')
DIR_OF = {1: 'left', 2: 'right', 3: 'up', 4: 'down'}


# --------------------------------------------------------- preconditions

def have(pred, item, need):
    """Whether an inventory bucket guarantees at least `need` of `item`."""
    bucket = pred[item]
    if item == 'wood':
        return bucket == '2+' if need >= 2 else bucket != '0'
    if item == 'stone':
        return bucket == '4+' if need >= 4 else bucket != '0'
    if item in v1.COUNTS:
        return bucket != '0' and need <= 1
    return pred.get(item) == 'yes'                     # tools


def applicable(action, pred):
    """Whether `action` can change anything, from the student's predicates
    and Crafter's recipe table alone."""
    from crafter import constants
    name, front = v1.ACTIONS[action], pred['front']
    if name == 'noop':
        return False
    if action in DIR_OF:
        return not (pred['facing'] == DIR_OF[action]
                    and front not in WALKABLE)
    if name == 'sleep':
        return True
    if name == 'do':
        if front in ('cow', 'zombie', 'skeleton', 'plant_ripe'):
            return True
        info = constants.collect.get(front)
        return info is not None and all(
            have(pred, tool, n) for tool, n in info['require'].items())
    if name.startswith('place_'):
        info = constants.place[name[len('place_'):]]
        return front in info['where'] and all(
            have(pred, item, n) for item, n in info['uses'].items())
    if name.startswith('make_'):
        info = constants.make[name[len('make_'):]]
        near = {'table': pred['near_table'], 'furnace': pred['near_furnace']}
        return all(near[n] == 'yes' for n in info['nearby']) and all(
            have(pred, item, n) for item, n in info['uses'].items())
    return True


def advise(rules, pred, preconditions):
    """v3 semantics; with preconditions, only rules whose action applies."""
    if preconditions:
        rules = [r for r in rules if applicable(r[1], pred)]
    return v3.advise(rules, pred)


# ------------------------------------------------------------- the rules

def rule_key(rule):
    condition, action, exceptions = rule
    return (tuple(sorted(condition.items())), action,
            tuple(sorted(exceptions)))


def distinct(rules):
    seen, out = set(), []
    for rule in rules:
        if rule_key(rule) not in seen:
            seen.add(rule_key(rule))
            out.append(rule)
    return out


def v1_rules():
    _panels, parsed = v1.consult_rules(v1.OUT)
    return [r for _, _, r, _ in parsed if r]


def situation_text(i, record, ineffective):
    return (f'SITUATION {i}:\nMAP:\n{record["full_map"]}\n'
            f'EXACT STATE: {json.dumps(record["facts"])}\n'
            f'Student observes: {json.dumps(record["pred"])}\n' +
            ('Simulator check: the rule\'s action changes nothing here (the '
             'same as noop).\n' if ineffective else ''))


def refine_prompt(rule, situations):
    return (
        f'PROMPT VERSION: {PROMPT_VERSION}\n' + v1.TASK_TEXT +
        f' Earlier you gave the student this reusable rule:\n'
        f'{v1.rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. A rule with no conditions applies in '
        'EVERY state. Using the full state, judge for EACH situation whether '
        'the rule\'s action is the best action there, and give the best '
        'action. Then return a REFINED rule that keeps the rule\'s action '
        'wherever it is best but excludes situations where it is not (add '
        'conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + v1.VOCAB + v3.SEMANTICS + '\n'.join(situations))


def refine_schema():
    schema = v1.rule_schema()
    props = dict(schema['properties'])
    props.pop('action_now')
    verdict = dict(type='object', additionalProperties=False,
                   required=['situation', 'rule_action_is_best',
                             'best_action'],
                   properties=dict(
                       situation={'type': 'integer'},
                       rule_action_is_best={'type': 'boolean'},
                       best_action={'type': 'integer', 'enum': list(
                           range(len(v1.ACTIONS)))}))
    props = dict(verdicts=dict(type='array', items=verdict), **props)
    return dict(type='object', additionalProperties=False,
                required=list(props), properties=props)


def parse_refined(answer):
    if answer['abstain']:
        return None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in v1.FIELDS[e['field']])
    return condition, int(answer['action']), exceptions


# ---------------------------------------------------------------- build

def build_checks(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory; panels are frozen')
    rules = distinct(v1_rules())
    pool = [v1.make_state(CHECK_SEED0 + k)[1] for k in range(N_CHECK)]
    rng = np.random.default_rng(7)
    rows, plan = [], []
    for k, rule in enumerate(rules):
        matches = [s for s in pool if v3.executable(rule, s['pred'])]
        if not matches:
            plan.append(dict(rule=k, situations=0))
            continue
        if len(matches) > MAX_CANDIDATES:
            matches = [matches[i] for i in sorted(rng.choice(
                len(matches), MAX_CANDIDATES, replace=False))]
        flagged = []
        for record in matches:
            env, again = v1.make_state(record['seed'])
            if again != record:
                raise ValueError(f"State {record['seed']} is not deterministic")
            flagged.append((record, not v1.effective(env, rule[1])))
        bad = [f for f in flagged if f[1]]
        good = [f for f in flagged if not f[1]]
        take_bad = min(len(bad), 3, K_SITUATIONS)
        pick = ([bad[i] for i in rng.permutation(len(bad))[:take_bad]] +
                [good[i] for i in rng.permutation(len(good))[
                    :K_SITUATIONS - take_bad]])
        if len(pick) < K_SITUATIONS:      # too few effective: more bad ones
            rest = [b for b in bad if b not in pick]
            pick += rest[:K_SITUATIONS - len(pick)]
        order = [pick[i] for i in rng.permutation(len(pick))]
        prompt = refine_prompt(rule, [situation_text(i, r, bad_)
                                      for i, (r, bad_) in enumerate(order)])
        body = v3.body(prompt, refine_schema(), PROMPT_VERSION)
        rows.append(dict(case_id=f'refine_{k:03d}', condition='crafter_v2',
                         split='refine', model=v1.MODEL, request=body,
                         request_sha256=v3.digest(body)))
        plan.append(dict(rule=k, situations=len(order), matches=len(matches),
                         seeds=[r['seed'] for r, _ in order],
                         ineffective=[bool(b) for _, b in order]))
    if len(rows) > MAX_REQUESTS:
        raise ValueError(f'{len(rows)} requests exceed the cap')
    out.mkdir(parents=True)
    (out / 'rules_v1_distinct.json').write_text(json.dumps(
        [v1.rule_text(r) for r in rules], indent=1))
    (out / 'check_plan.json').write_text(json.dumps(plan, indent=1))
    (out / 'refine_requests.json').write_text(json.dumps(rows, indent=1))
    (out / 'refine_requests_manifest.json').write_text(json.dumps(dict(
        study=STUDY, cases=len(rows), model=v1.MODEL, api_calls=0,
        requests_sha256=v3.digest(rows), ceiling_usd=CEILING), indent=1))
    (out / 'example_refine_prompt.txt').write_text(
        rows[0]['request']['input'][0]['content'])
    print(f'{len(v1_rules())} v1 rules, {len(rules)} distinct; '
          f'{len(rows)} self-check requests; '
          f"{sum(p['situations'] == 0 for p in plan)} rules never fire in "
          f'the check pool; ineffective situations shown: '
          f"{sum(sum(p.get('ineffective', [])) for p in plan)}")


def collect(out, credential_file):
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
    replies = out / 'refine_replies.jsonl'
    if replies.exists():
        raise ValueError(f'{replies} exists; replies are frozen')
    rows = json.loads((out / 'refine_requests.json').read_text())
    spent = 0.0
    with replies.open('w', encoding='utf-8') as handle:
        for row in rows:
            if spent >= CEILING:
                print('ceiling reached; stopping')
                break
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


def refined(out=OUT):
    """Each distinct v1 rule after its check: refined, kept or withdrawn."""
    rules = distinct(v1_rules())
    replies = v3.read_replies(Path(out) / 'refine_replies.jsonl')
    kept, status = [], Counter()
    for k, rule in enumerate(rules):
        row = replies.get(f'refine_{k:03d}')
        if row is None:
            kept.append(rule)
            status['unchecked_kept'] += 1
        elif row['response_status'] != 'completed':
            status['check_failed_dropped'] += 1
        else:
            new = parse_refined(row['answer'])
            status['refined' if new else 'withdrawn'] += 1
            if new:
                kept.append(new)
    return distinct(kept), dict(status)


def build_banks(out=OUT, banks=BANKS):
    from scripts.run_rule_bank_pilot_20260928 import rule_json
    out, banks = Path(out), Path(banks)
    raw = v1_rules()
    checked, status = refined(out)
    rows = [json.loads(line) for line in
            (out / 'refine_replies.jsonl').read_text().splitlines()]
    common = dict(study=STUDY, task='crafter', observer='crafter_v1',
                  model=v1.MODEL, source_replies_sha256=dict(
                      v1_consult=hashlib.sha256(
                          (v1.OUT / 'consult_replies.jsonl').read_bytes())
                      .hexdigest(),
                      refine=hashlib.sha256(
                          (out / 'refine_replies.jsonl').read_bytes())
                      .hexdigest()))
    refine_cost = dict(calls=len(rows), dollars=round(sum(
        r.get('usd', 0) for r in rows), 6))
    sets = dict(v1_raw=(raw, {}),
                v1_without_empty=([r for r in raw if r[0]], {}),
                self_checked=(checked, dict(refine=refine_cost,
                                            self_check_status=status)))
    banks.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, (rules, extra) in sets.items():
        for pre in (False, True):
            label = f'{name}_preconditions' if pre else name
            made[label] = dict(mode='scoped', rules=[rule_json(r)
                                                     for r in rules],
                               filter=FILTER if pre else None,
                               **extra, **common)
    for name, bank in made.items():
        path = banks / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(self_check=status, refine_cost=refine_cost,
                          rules={n: len(b['rules'])
                                 for n, b in made.items()}), indent=1))


# ------------------------------------------------------------ evaluation

def load(path):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    rules = [(dict(r['condition']), int(r['action']),
              tuple(tuple(e) for e in r['exceptions']))
             for r in bank['rules']]
    return rules, bank.get('filter') == FILTER


def labels(rules, pre, seeds):
    """Coverage and effective share on constructed states."""
    counts = Counter()
    for seed in seeds:
        env, record = v1.make_state(seed)
        action, status = advise(rules, record['pred'], pre)
        counts[status] += 1
        if action is not None:
            counts['effective' if v1.effective(env, action)
                   else 'no_effect'] += 1
    advised = counts['effective'] + counts['no_effect']
    return (advised / len(seeds), counts['effective'] / advised
            if advised else 0.0, dict(counts))


def rollout(seed, rules, pre):
    """v1's rollout: the rule action where one fires, else uniform random."""
    env = v1.make_env(seed)
    rng = np.random.default_rng([seed, 1])
    fired = steps = 0
    done = False
    while not done and steps < v1.ROLLOUT_CAP:
        action = None
        if rules is not None:
            action, _ = advise(rules, v1.observe(env), pre)
        if action is None:
            action = int(rng.integers(len(v1.ACTIONS)))
        else:
            fired += 1
        _, _, done, _ = env.step(action)
        steps += 1
    unlocked = sorted(k for k, v in env._player.achievements.items() if v)
    return dict(seed=seed, steps=steps, fired=fired,
                died=bool(env._player.health <= 0), achievements=unlocked)


def evaluate(rules, pre, state_seeds, rollout_seeds):
    coverage, effect, counts = labels(rules, pre, state_seeds)
    run = v1.summary([rollout(s, rules, pre) for s in rollout_seeds])
    return dict(coverage=coverage, effective=effect, counts=counts, run=run)


def develop(out=OUT, banks=BANKS):
    out, banks = Path(out), Path(banks)
    states = [DEV_STATES_SEED0 + k for k in range(N_DEV_STATES)]
    seeds = [DEV_ROLLOUT_SEED0 + k for k in range(N_DEV_ROLLOUT)]
    random_run = v1.summary([rollout(s, None, False) for s in seeds])
    result = dict(random=random_run, banks={})
    for path in sorted(banks.glob('*.json')):
        rules, pre = load(path)
        res = evaluate(rules, pre, states, seeds)
        res['decision_if_gate'] = v1.decide(res['coverage'], res['effective'],
                                            res['run'], random_run)
        result['banks'][path.stem] = res
        r = res['run']
        print(f"{path.stem:32s} rules {len(rules):2d} | coverage "
              f"{res['coverage']:.2f} effective {res['effective']:.2f} | "
              f"achievements {r['mean_achievements']:.2f} (random "
              f"{random_run['mean_achievements']:.2f}) table "
              f"{r['rates']['place_table']:.0f}% pickaxe "
              f"{r['rates']['make_wood_pickaxe']:.0f}% rule share "
              f"{r['rule_share']:.2f}")
    # The choice, before the gate: most achievements over the development
    # rollouts; ties by coverage.
    chosen = max(result['banks'], key=lambda n: (
        result['banks'][n]['run']['mean_achievements'],
        result['banks'][n]['coverage']))
    result['chosen'] = chosen
    (out / 'develop.json').write_text(json.dumps(result, indent=1))
    print(f'CHOSEN for the gate: {chosen}')


def gate(out=OUT, banks=BANKS):
    out = Path(out)
    if (out / 'gate.json').exists():
        raise ValueError('The gate runs once')
    chosen = json.loads((out / 'develop.json').read_text())['chosen']
    rules, pre = load(Path(banks) / f'{chosen}.json')
    states = [GATE_STATES_SEED0 + k for k in range(N_GATE_STATES)]
    seeds = [GATE_ROLLOUT_SEED0 + k for k in range(N_GATE_ROLLOUT)]
    random_run = v1.summary([rollout(s, None, False) for s in seeds])
    res = evaluate(rules, pre, states, seeds)
    decision = v1.decide(res['coverage'], res['effective'], res['run'],
                         random_run)
    (out / 'gate.json').write_text(json.dumps(dict(
        chosen=chosen, random=random_run, **res, decision=decision),
        indent=1))
    r = res['run']
    print(f"GATE {chosen}: {decision['verdict']} | achievements "
          f"{r['mean_achievements']:.2f} vs random "
          f"{random_run['mean_achievements']:.2f} (gain "
          f"{decision['achievement_gain']:+.2f}); table "
          f"{decision['place_table']:.0%}, wood pickaxe "
          f"{decision['make_wood_pickaxe']:.0%}; coverage "
          f"{decision['coverage']:.2f}, effective "
          f"{decision['effective_share']:.2f}")


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('build-checks', 'collect', 'banks',
                                        'develop', 'gate'))
    cli.add_argument('--out', type=Path, default=OUT)
    cli.add_argument('--banks', type=Path, default=BANKS)
    cli.add_argument('--credential-file', type=Path)
    args = cli.parse_args()
    if args.action == 'build-checks':
        build_checks(args.out)
    elif args.action == 'collect':
        collect(args.out, args.credential_file)
    elif args.action == 'banks':
        build_banks(args.out, args.banks)
    elif args.action == 'develop':
        develop(args.out, args.banks)
    else:
        gate(args.out, args.banks)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
