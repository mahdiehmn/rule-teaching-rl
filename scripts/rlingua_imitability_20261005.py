"""Can a student that sees only its view imitate each teacher? (offline)

Mechanism measurement for addendum 16. On
the same visited states we compare three teachers: RLingua's full-state
controller, RLingua's student-view controller, and our frozen rule bank.
For each we report

  coverage      fraction of states that receive a target (a complete
                controller always gives one; our bank abstains);
  accuracy      fraction of targets in the shortest-path optimal set;
  ceiling       in-sample agreement of the best lookup from the student's
                current 7x7 view to that teacher's targets (group states by
                identical view, count each group's most frequent target).
                An optimistic bound: a view seen once agrees trivially.
  held_out_agreement
                the same lookup fitted on the even episodes and scored on
                the odd episodes' states whose view it has seen (the
                fraction of such states is reported). Our KeyCorridor bank
                also reads a progress fact, so its agreement with a
                view-only lookup need not be 1.

States come from 100 fresh layouts per task (seeds 38,000,000+, unused
elsewhere), visited by a behaviour that follows the full-state controller
half of the time and a uniformly random useful action otherwise, so that
both on-route and off-route states appear. No training, no API call.

  python -m scripts.rlingua_imitability_20261005
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.registry import build_env
from scripts import conditional_rules_keycorridor as kc
from scripts import conditional_rules_multiroom as mr
from scripts import conditional_rules_v3 as v3
from teachers.minigrid.rlingua_controller import RLinguaController
from teachers.minigrid.rule_bank import RuleBankTeacher

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / 'research/rlingua_controllers_20261005'
OUT = ROOT / 'docs/assets/rlingua_imitability_2026-10-05'
SEED = 38_000_000
EPISODES = 100
TASKS = {
    'dk': ('doorkey_8x8', 'research/rule_banks/v3_20260928/blind_strict.json',
           v3.v1.optimal),
    'mr': ('multiroom_n6', 'research/rule_banks/multiroom_20260928/scoped.json',
           mr.optimal),
    'kc': ('keycorridor_s3r3', 'research/rule_banks/keycorridor_mem_20260929/'
           'self_checked_pooled_valid.json', kc.optimal),
}


def view_key(u):
    return hashlib.sha1(u.gen_obs()['image'].tobytes()).hexdigest()


def collect(key):
    task, bank, optimal = TASKS[key]
    full = RLinguaController(CTRL / f'{key}_full.py', 'full')
    view = RLinguaController(CTRL / f'{key}_view.py', 'view')
    rules = RuleBankTeacher(ROOT / bank)
    rng = np.random.default_rng(SEED)
    rows = []
    for episode in range(EPISODES):
        seed = SEED + episode
        env = build_env(task, seed=seed, obs_mode='symbolic')
        env.reset(seed=seed)
        u = env.unwrapped
        full.reset(0)                          # fresh instances per episode
        view.reset(0)
        for _ in range(u.max_steps):
            a_full = full.act(0, u)
            a_view = view.act(0, u)            # every step: keeps memory
            a_rule, _ = rules._advise(u.gen_obs()['image'], u)
            rows.append(dict(episode=episode, view=view_key(u),
                             optimal=list(optimal(u)),
                             full=a_full, view_ctrl=a_view,
                             rule=None if a_rule is None else int(a_rule)))
            act = a_full if rng.random() < .5 else int(rng.integers(6))
            full.executed(0, act)              # what the env actually ran
            view.executed(0, act)
            _, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break
        env.close()
    return rows


def summarize(rows, teacher):
    labelled = [r for r in rows if r[teacher] is not None]
    groups = defaultdict(Counter)
    for r in labelled:
        groups[r['view']][r[teacher]] += 1
    best = sum(c.most_common(1)[0][1] for c in groups.values())
    repeated = [c for c in groups.values() if sum(c.values()) > 1]
    scored = [r for r in labelled if r['optimal']]
    # Held out: fit the view -> most frequent target lookup on the even
    # episodes, score it on the odd episodes' states whose view was seen.
    fit = defaultdict(Counter)
    for r in labelled:
        if r['episode'] % 2 == 0:
            fit[r['view']][r[teacher]] += 1
    test = [r for r in labelled
            if r['episode'] % 2 == 1 and r['view'] in fit]
    held_out = sum(fit[r['view']].most_common(1)[0][0] == r[teacher]
                   for r in test)
    return dict(
        states=len(rows), labelled=len(labelled),
        coverage=len(labelled) / max(1, len(rows)),
        accuracy=(sum(r[teacher] in r['optimal'] for r in scored)
                  / max(1, len(scored))),
        ceiling=best / max(1, len(labelled)),
        held_out_agreement=held_out / max(1, len(test)),
        held_out_states=len(test),
        held_out_seen_fraction=len(test) / max(1, sum(
            r['episode'] % 2 == 1 for r in labelled)),
        conflicting_views=(sum(len(c) > 1 for c in repeated)
                           / max(1, len(repeated))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for key in TASKS:
        rows = collect(key)
        result[key] = {t: summarize(rows, t)
                       for t in ('full', 'view_ctrl', 'rule')}
        for t, s in result[key].items():
            print(f"{key} {t:9s} coverage {s['coverage']:.2f} accuracy "
                  f"{s['accuracy']:.2f} in-sample {s['ceiling']:.2f} "
                  f"held-out {s['held_out_agreement']:.2f} "
                  f"(on {s['held_out_seen_fraction']:.2f} of held-out "
                  f"states) conflicting views {s['conflicting_views']:.2f} "
                  f"({s['labelled']}/{s['states']})")
    (OUT / 'imitability.json').write_text(json.dumps(dict(
        seed=SEED, episodes=EPISODES, result=result), indent=1) + '\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
