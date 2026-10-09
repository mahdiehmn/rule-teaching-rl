"""Crafter rules v3: rules with episodic progress facts (achievements).

One LLM, and the rule form, semantics and checks of v2. Compared with v2:

  predicates   v2's predicates plus done_<achievement> (yes once unlocked
               in this episode) for all 22 achievements, a function of
               the agent's own history; the executor reads it from the
               episode
  facts        the prompt states Crafter's exact recipes (crafter/data.yaml)
               and that an achievement counts once per episode
  states       consultation, check and development states are sampled from
               episodes of students learning the task (the exploratory
               study's trained no-teacher policies, and random play as a
               novice), not constructed by injecting inventory
  writing      36 consultations, then v2's self-check (identical rules
               pooled; situations where the action changes nothing are
               flagged), at most 36 + 36 requests, $3 ceiling

  build      visited-state panels and consultation requests (0 calls)
  collect    --stage consult | refine (GPT-5-mini, one attempt each)
  checks     self-check requests on the check pool (0 calls)
  banks      raw and self-checked banks, each with and without the
             preconditions filter (0 calls)
  develop    coverage, effective labels, conflicts and agreement with a
             trained student on the development panel, and rules-as-policy
             rollouts against random play
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
from scripts import crafter_rules_pilot_20260928 as pilot
from scripts import crafter_rules_v2_20260930 as c2

STUDY = 'crafter_rules_v3_20261001'
OUT = Path('results') / STUDY
BANKS = Path('research/rule_banks/crafter_v3_20261001')
OBSERVER = 'crafter_mem_v1'
CONSULT_VERSION, CHECK_VERSION = 'scoped_rule_crafter_v3', 'scope_check_crafter_v3'
EXPLORATORY = Path('results/crafter_learning_20260930')
POLICY_SEEDS = (1, 2, 3, 4)          # trained no-teacher students
# Fresh world seeds (v1/v2 used 30.0M-30.9M; training pools 40M; eval 41M)
CONSULT_SEED0, CHECK_SEED0, DEV_SEED0 = 43_000_000, 43_100_000, 43_200_000
N_CONSULT, N_CHECK, N_DEV = 36, 300, 200
ROLLOUT_SEED0, N_ROLLOUT = 43_300_000, 20
CEILING = 3.00
PRICE = (0.25, 2.00)


def achievements():
    from crafter import constants
    return tuple(constants.achievements)


MEMORY = {f'done_{a}': ('yes', 'no') for a in achievements()}
FIELDS = dict(pilot.FIELDS, **MEMORY)

TASK_TEXT = (pilot.TASK_TEXT
             .replace('place_furnace needs 4 stone and a table nearby',
                      'place_furnace needs 4 stone')
             + ' Each achievement counts once per episode: repeating it '
             '(for example placing a second table) earns nothing and spends '
             'the materials.')
assert 'and a table nearby; place_stone' not in TASK_TEXT
VOCAB = (pilot.VOCAB[:-len(pilot.ACTION_TEXT)] +
         '- done_<achievement>: yes once the student has unlocked that '
         'achievement in this episode (it remembers its own progress), '
         'otherwise no; known even when nothing related is in view. '
         'Achievements: ' + ', '.join(achievements()) + '\n'
         + pilot.ACTION_TEXT)


# --------------------------------------------------------------- observe

def observe_mem(env):
    """Base Crafter predicates plus the done_<achievement> progress facts."""
    unlocked = env._player.achievements
    return dict(pilot.observe(env), **{
        f'done_{a}': 'yes' if unlocked[a] > 0 else 'no'
        for a in achievements()})


def load(path):
    """(rules, preconditions, observe) for a v2 or v3 bank."""
    rules, pre = c2.load(path)
    observer = json.loads(Path(path).read_text(encoding='utf-8')).get(
        'observer')
    return rules, pre, observe_mem if observer == OBSERVER else pilot.observe


# ---------------------------------------------------- visited-state panels

def policy(seed):
    """A trained no-teacher student (seed 1-4) or random play (None)."""
    import torch
    from algos import ppo_crafter as pc
    if seed is None:
        return None
    net = pc.Net()
    net.load_state_dict(torch.load(
        EXPLORATORY / f'none_s{seed}' / 'final_model.pt'))
    net.eval()
    return net


def record(env, source, world, step):
    return dict(world=world, source=source, step=step,
                pred=observe_mem(env), full_map=pilot.teacher_map(env),
                facts=pilot.teacher_facts(env),
                effective=[a for a in range(1, len(pilot.ACTIONS))
                           if pilot.effective(env, a)])


EPISODES_PER_WORLD = 8      # a world costs ~1-6 s to generate, a copy ~1 ms
EPISODE_CAP = 600


def visited(seed0, n, rng):
    """n states from episodes of the trained students and of random play.

    Fresh worlds (seeds seed0+) are generated once, EPISODES_PER_WORLD
    episodes are played on copies of each (cycling through the sources),
    and each episode contributes one state drawn uniformly from its own
    steps (reservoir sampling), so no episode is wasted.
    """
    import copy
    import torch
    from algos import ppo_crafter as pc
    from envs import crafter_symbolic as cs
    torch.set_num_threads(1)       # many training runs share the CPU
    sources = [*POLICY_SEEDS, None]
    nets = {s: policy(s) for s in sources}
    gen = torch.Generator().manual_seed(int(seed0 % 2**31))
    worlds = cs.world_pool(seed0 + w for w in range(
        -(-n // EPISODES_PER_WORLD)))
    out = []
    for k in range(n):
        world, source = k // EPISODES_PER_WORLD, sources[k % len(sources)]
        env = cs.CrafterSymbolic([worlds[world]], seed=k)
        obs = env.reset()
        keep, keep_step = copy.deepcopy(env.env), 0
        for step in range(1, EPISODE_CAP):
            if source is None:
                action = int(rng.integers(len(pilot.ACTIONS)))
            else:
                with torch.no_grad():
                    logits, _ = nets[source](*pc.stack([obs]))
                action = int(torch.multinomial(torch.softmax(logits, -1), 1,
                                               generator=gen))
            obs, _, done, _ = env.step(action)
            if done:
                break
            if rng.random() < 1 / (step + 1):          # reservoir of one
                keep, keep_step = copy.deepcopy(env.env), step
        keep._player.sleeping = False
        out.append(record(keep, 'random' if source is None else
                          f'student_s{source}', seed0 + world, keep_step))
    return out


# --------------------------------------------------------------- prompts

def consult_prompt(state):
    return (
        f'PROMPT VERSION: {CONSULT_VERSION}\n' + TASK_TEXT +
        ' You see a 17x13 map and the exact state; the student sees only its '
        '9x7 view and inventory, and remembers which achievements it has '
        'unlocked in this episode.\n'
        'MAP (rows top to bottom; the player is the arrow at the centre, '
        'pointing where it faces; . grass, : sand, _ path, ~ water, # stone, '
        'T tree, ! lava, c coal, i iron, d diamond, t table, f furnace, C '
        'cow, Z zombie, S skeleton, * arrow, p plant, P ripe plant):\n'
        f'{state["full_map"]}\n'
        f'EXACT STATE (teacher-visible): {json.dumps(state["facts"])}\n'
        f'The student currently observes: {json.dumps(state["pred"])}\n'
        + VOCAB + v3.SEMANTICS +
        'Return: action_now = the best action in THIS situation; then ONE '
        'reusable rule (WHEN conditions, PREFER action, UNLESS exceptions) '
        'that the student can apply on its own elsewhere. Use "any" for '
        'fields the rule does not need. If no reliable rule exists, set '
        'abstain=true (action_now is still required).\n')


def refine_prompt(rule, situations):
    return (
        f'PROMPT VERSION: {CHECK_VERSION}\n' + TASK_TEXT +
        f' Earlier you gave the student this reusable rule:\n'
        f'{pilot.rule_text(rule)}\n'
        'The student met the situations below in its own experience; your '
        'rule applies in each of them. A rule with no conditions applies in '
        'EVERY state. Using the full state, judge for EACH situation whether '
        'the rule\'s action is the best action there, and give the best '
        'action. Then return a REFINED rule that keeps the rule\'s action '
        'wherever it is best but excludes situations where it is not (add '
        'conditions or exceptions, or change the action), or set '
        'abstain=true if no reliable rule over these predicates exists.\n'
        + VOCAB + v3.SEMANTICS + '\n'.join(situations))


def rule_schema(with_now):
    actions = {'type': 'integer', 'enum': list(range(len(pilot.ACTIONS)))}
    condition = {k: {'type': 'string', 'enum': ['any', *v]}
                 for k, v in FIELDS.items()}
    exception = dict(type='object', additionalProperties=False,
                     required=['field', 'value'], properties=dict(
                         field={'type': 'string', 'enum': list(FIELDS)},
                         value={'type': 'string', 'enum': sorted(
                             {v for vs in FIELDS.values() for v in vs})}))
    props = dict(abstain={'type': 'boolean'},
                 condition=dict(type='object', additionalProperties=False,
                                required=list(FIELDS), properties=condition),
                 action=actions,
                 exceptions=dict(type='array', maxItems=2, items=exception),
                 rationale={'type': 'string', 'maxLength': 240})
    if with_now:
        props = dict(action_now=actions, **props)
    else:
        verdict = dict(type='object', additionalProperties=False,
                       required=['situation', 'rule_action_is_best',
                                 'best_action'],
                       properties=dict(
                           situation={'type': 'integer'},
                           rule_action_is_best={'type': 'boolean'},
                           best_action=actions))
        props = dict(verdicts=dict(type='array', items=verdict), **props)
    return dict(type='object', additionalProperties=False,
                required=list(props), properties=props)


def parse(answer):
    if answer['abstain']:
        return None
    condition = {k: v for k, v in answer['condition'].items() if v != 'any'}
    exceptions = tuple((e['field'], e['value']) for e in answer['exceptions']
                       if e['value'] in FIELDS[e['field']])
    return condition, int(answer['action']), exceptions


def request(case_id, stage, prompt, schema, version):
    body = v3.body(prompt, schema, version)
    return dict(case_id=case_id, condition=f'crafter_v3_{stage}',
                split=stage, model=pilot.MODEL, request=body,
                request_sha256=v3.digest(body))


# -------------------------------------------------------------- pipeline

def build(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory; panels are frozen')
    rng = np.random.default_rng(11)
    candidates = visited(CONSULT_SEED0, 4 * N_CONSULT, rng)
    # Balance progress: equal numbers by achievements already unlocked.
    def level(s):
        n = len(s['facts']['achievements'])
        return 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 5 else 3
    by = {}
    for s in candidates:
        by.setdefault(level(s), []).append(s)
    consult = []
    while len(consult) < N_CONSULT:
        for lv in sorted(by):
            if by[lv] and len(consult) < N_CONSULT:
                consult.append(by[lv].pop(int(rng.integers(len(by[lv])))))
    check = visited(CHECK_SEED0, N_CHECK, rng)
    dev = visited(DEV_SEED0, N_DEV, rng)
    rows = [request(f'consult_{k:03d}', 'consult', consult_prompt(s),
                    rule_schema(True), CONSULT_VERSION)
            for k, s in enumerate(consult)]
    out.mkdir(parents=True)
    (out / 'panels.json').write_text(json.dumps(dict(
        consult=consult, check=check, dev=dev)))
    (out / 'consult_requests.json').write_text(json.dumps(rows, indent=1))
    (out / 'example_consult_prompt.txt').write_text(
        rows[0]['request']['input'][0]['content'])
    print(f'{len(rows)} consultation requests; consult progress levels '
          f"{dict(Counter(level(s) for s in consult))}; sources "
          f"{dict(Counter(s['source'] for s in consult))}; check {len(check)}"
          f', dev {len(dev)}')


def collect(out, credential_file, stage):
    try:
        import truststore
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
    spent = sum(json.loads(line).get('usd', 0) for path in out.glob(
        '*_replies.jsonl') for line in path.read_text().splitlines())
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
    print(f'spent so far ${spent:.4f}')


def consulted(out=OUT):
    replies = v3.read_replies(Path(out) / 'consult_replies.jsonl')
    rules, status = [], Counter()
    for k in range(N_CONSULT):
        row = replies.get(f'consult_{k:03d}')
        if not row or row['response_status'] != 'completed':
            status['missing'] += 1
            continue
        rule = parse(row['answer'])
        status['rule' if rule else 'abstained'] += 1
        if rule:
            rules.append(rule)
    return rules, dict(status)


V2_BANK = 'research/rule_banks/crafter_v2_20260930/self_checked.json'


def v2_rules():
    return c2.distinct(c2.load(V2_BANK)[0])


def checks(out=OUT, rules=None, stage='refine'):
    """Self-check requests: each distinct rule (the consulted ones, or v3b:
    the v2 bank's) on up to five check states where it fires, at most three
    flagged as changing nothing."""
    out = Path(out)
    rules = c2.distinct(consulted(out)[0]) if rules is None else rules
    pool = json.loads((out / 'panels.json').read_text())['check']
    rng = np.random.default_rng(7)
    rows, plan = [], []
    for k, rule in enumerate(rules):
        matches = [s for s in pool if v3.executable(rule, s['pred'])]
        if not matches:
            plan.append(dict(rule=k, situations=0))
            continue
        bad = [s for s in matches if rule[1] not in s['effective']]
        good = [s for s in matches if rule[1] in s['effective']]
        take_bad = min(len(bad), 3)
        pick = ([bad[i] for i in rng.permutation(len(bad))[:take_bad]] +
                [good[i] for i in rng.permutation(len(good))[
                    :5 - take_bad]])
        if len(pick) < 5:
            pick += [b for b in bad if b not in pick][:5 - len(pick)]
        order = [pick[i] for i in rng.permutation(len(pick))]
        text = [c2.situation_text(i, s, rule[1] not in s['effective'])
                for i, s in enumerate(order)]
        rows.append(request(f'{stage}_{k:03d}', stage,
                            refine_prompt(rule, text), rule_schema(False),
                            CHECK_VERSION))
        plan.append(dict(rule=k, situations=len(order), matches=len(matches),
                         ineffective=[rule[1] not in s['effective']
                                      for s in order]))
    suffix = '' if stage == 'refine' else '_' + stage
    (out / f'check_plan{suffix}.json').write_text(json.dumps(plan, indent=1))
    (out / f'{stage}_requests.json').write_text(json.dumps(rows, indent=1))
    print(f'{len(rules)} distinct rules; {len(rows)} self-check '
          f"requests; {sum(p['situations'] == 0 for p in plan)} never fire")


def self_checked(out=OUT, rules=None, stage='refine'):
    out = Path(out)
    rules = c2.distinct(consulted(out)[0]) if rules is None else rules
    replies = v3.read_replies(out / f'{stage}_replies.jsonl')
    kept, status = [], Counter()
    for k, rule in enumerate(rules):
        row = replies.get(f'{stage}_{k:03d}')
        if row is None:
            kept.append(rule)
            status['unchecked_kept'] += 1
        elif row['response_status'] != 'completed':
            status['check_failed_dropped'] += 1
        else:
            new = parse(row['answer'])
            status['refined' if new else 'withdrawn'] += 1
            if new:
                kept.append(new)
    return c2.distinct(kept), dict(status)


def build_banks(out=OUT, banks=BANKS):
    from scripts.run_rule_bank_pilot_20260928 import rule_json
    out, banks = Path(out), Path(banks)
    raw, consult_status = consulted(out)
    checked, check_status = self_checked(out)
    cost = {stage: round(sum(json.loads(line).get('usd', 0) for line in
                             (out / f'{stage}_replies.jsonl').read_text()
                             .splitlines()), 6)
            for stage in ('consult', 'refine')}
    common = dict(study=STUDY, task='crafter', observer=OBSERVER,
                  model=pilot.MODEL, cost_usd=cost, source_replies_sha256={
                      stage: hashlib.sha256((out / f'{stage}_replies.jsonl')
                                            .read_bytes()).hexdigest()
                      for stage in ('consult', 'refine')})
    banks.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, rules, extra in (
            ('raw', c2.distinct(raw), dict(consult_status=consult_status)),
            ('self_checked', checked, dict(self_check_status=check_status))):
        for pre in (False, True):
            label = f'{name}_preconditions' if pre else name
            made[label] = dict(mode='scoped', rules=[rule_json(r)
                                                     for r in rules],
                               filter=c2.FILTER if pre else None,
                               **extra, **common)
    for name, bank in made.items():
        path = banks / f'{name}.json'
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(consult=consult_status, self_check=check_status,
                          cost=cost, rules={n: len(b['rules'])
                                            for n, b in made.items()}),
                     indent=1))


def build_banks_v3b(out=OUT, banks=BANKS):
    """v3b: the v2 bank's rules after a self-check with the progress facts."""
    from scripts.run_rule_bank_pilot_20260928 import rule_json
    out, banks = Path(out), Path(banks)
    checked, status = self_checked(out, v2_rules(), 'refine_v2')
    cost = round(sum(json.loads(line).get('usd', 0) for line in
                     (out / 'refine_v2_replies.jsonl').read_text()
                     .splitlines()), 6)
    common = dict(study=STUDY, task='crafter', observer=OBSERVER,
                  model=pilot.MODEL, derived_from=V2_BANK, cost_usd=dict(
                      refine_v2=cost), self_check_status=status,
                  source_replies_sha256=dict(refine_v2=hashlib.sha256(
                      (out / 'refine_v2_replies.jsonl').read_bytes())
                      .hexdigest()))
    for pre in (False, True):
        path = banks / ('v3b_preconditions.json' if pre else 'v3b.json')
        if path.exists():
            raise ValueError(f'{path} exists; banks are frozen')
        bank = dict(mode='scoped', rules=[rule_json(r) for r in checked],
                    filter=c2.FILTER if pre else None, **common)
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(bank, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(self_check=status, cost=cost,
                          rules=len(checked)), indent=1))


# ------------------------------------------------------------ evaluation

def rollout(seed, rules, pre, observe):
    env = pilot.make_env(seed)
    rng = np.random.default_rng([seed, 1])
    fired = steps = 0
    done = False
    while not done and steps < pilot.ROLLOUT_CAP:
        action = None
        if rules is not None:
            action, _ = c2.advise(rules, observe(env), pre)
        if action is None:
            action = int(rng.integers(len(pilot.ACTIONS)))
        else:
            fired += 1
        _, _, done, _ = env.step(action)
        steps += 1
    unlocked = sorted(k for k, v in env._player.achievements.items() if v)
    return dict(seed=seed, steps=steps, fired=fired,
                died=bool(env._player.health <= 0), achievements=unlocked)


def develop(out=OUT, banks=BANKS, names=None, tag=''):
    out = Path(out)
    dev = json.loads((out / 'panels.json').read_text())['dev']
    seeds = [ROLLOUT_SEED0 + k for k in range(N_ROLLOUT)]
    random_run = pilot.summary([rollout(s, None, False, None) for s in seeds])
    paths = sorted(p for p in Path(banks).glob('*.json')
                   if names is None or p.stem in names) + [Path(V2_BANK)]
    result = dict(random=random_run, banks={})
    for path in paths:
        rules, pre, observe = load(path)
        counts = Counter()
        for s in dev:
            pred = s['pred'] if observe is observe_mem else {
                k: v for k, v in s['pred'].items() if k in pilot.FIELDS}
            action, status = c2.advise(rules, pred, pre)
            counts[status] += 1
            if action is not None:
                counts['effective' if action in s['effective']
                       else 'no_effect'] += 1
        advised = counts['effective'] + counts['no_effect']
        run = pilot.summary([rollout(s, rules, pre, observe) for s in seeds])
        name = f'{path.parent.name}/{path.stem}'
        result['banks'][name] = dict(
            rules=len(rules), coverage=advised / len(dev),
            effective=counts['effective'] / advised if advised else 0,
            conflict=counts['conflict'] / len(dev), run=run)
        r = result['banks'][name]
        print(f"{name:52s} rules {len(rules):2d} | coverage "
              f"{r['coverage']:.2f} effective {r['effective']:.2f} conflict "
              f"{r['conflict']:.2f} | achievements "
              f"{run['mean_achievements']:.2f} (random "
              f"{random_run['mean_achievements']:.2f}) table "
              f"{run['rates']['place_table']:.0f}% pickaxe "
              f"{run['rates']['make_wood_pickaxe']:.0f}%", flush=True)
    (out / f'develop{tag}.json').write_text(json.dumps(result, indent=1))
    # The frozen choice: the v3 bank with the most rollout achievements,
    # ties by coverage.
    v3_banks = {n: b for n, b in result['banks'].items()
                if n.startswith(Path(banks).name + '/')}
    chosen = max(v3_banks, key=lambda n: (
        v3_banks[n]['run']['mean_achievements'], v3_banks[n]['coverage']))
    path = Path(banks) / (chosen.split('/', 1)[1] + '.json')
    (out / f'choice{tag}.json').write_text(json.dumps(dict(
        bank=path.as_posix(), name=chosen, scores=v3_banks[chosen]['run'][
            'mean_achievements'], coverage=v3_banks[chosen]['coverage']),
        indent=1))
    print(f'CHOSEN v3 bank: {chosen}')


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('action', choices=('build', 'collect', 'checks',
                                        'banks', 'develop', 'checks-v3b',
                                        'banks-v3b', 'develop-v3b'))
    cli.add_argument('--out', type=Path, default=OUT)
    cli.add_argument('--banks', type=Path, default=BANKS)
    cli.add_argument('--credential-file', type=Path)
    cli.add_argument('--stage', choices=('consult', 'refine', 'refine_v2'),
                     default='consult')
    args = cli.parse_args()
    if args.action == 'build':
        build(args.out)
    elif args.action == 'collect':
        collect(args.out, args.credential_file, args.stage)
    elif args.action == 'checks':
        checks(args.out)
    elif args.action == 'banks':
        build_banks(args.out, args.banks)
    elif args.action == 'checks-v3b':
        checks(args.out, v2_rules(), 'refine_v2')
    elif args.action == 'banks-v3b':
        build_banks_v3b(args.out, args.banks)
    elif args.action == 'develop-v3b':
        develop(args.out, args.banks, names={'v3b', 'v3b_preconditions'},
                tag='_v3b')
    else:
        develop(args.out, args.banks, names={
            'raw', 'raw_preconditions', 'self_checked',
            'self_checked_preconditions'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
