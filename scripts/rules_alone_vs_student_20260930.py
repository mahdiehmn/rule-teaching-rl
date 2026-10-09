"""Rules alone against the student they teach (and the rules on a larger map).

A reviewer-style
question: if the LLM's rules could solve a task by themselves, training a
neural student on them would add nothing. So this runs every rule bank the
paper teaches with AS A POLICY (the rules act; where none fires or rules
disagree, a uniformly random useful action), and sets its success beside
PPO alone and PPO taught by the same rules (final teacher-off success after
5M steps, fresh replicates 30-39). No training, no API.

It also runs the banks, unchanged, on larger maps of the same tasks
(DoorKey-16x16; KeyCorridor S4R3, S5R3, S6R3, the last a standard
hard-exploration benchmark with a 1080-step limit). The rules read only the
student's own 7x7 view (and the unlocking memory), so nothing in them
depends on the map's size.

    python -m scripts.rules_alone_vs_student_20260930 --data <vulcan_sync/data>
"""

import argparse
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from envs.registry import build_env  # noqa: E402
from scripts import conditional_rules_v3 as v3  # noqa: E402
from scripts.plot_rule_speed_20260929 import (BLUE, GREY, GRID, INK,  # noqa: E402
                                              INK2, ORANGE)
from scripts.plot_rule_speed_20260929 import load as load_curves  # noqa: E402
from teachers.minigrid.rule_bank import OBSERVERS, STATEFUL, load_bank  # noqa: E402

USEFUL = (0, 1, 2, 3, 4, 5)       # every action but `done`
SEED0 = 35_000_000                # fresh maps, disjoint from every panel
MEM = 'research/rule_banks/keycorridor_mem_20260929/self_checked_pooled_valid.json'
# The bank each student is taught with in the confirmation cohort (fix wave
# RULE_ARM; KeyCorridor: the memory bank of protocol addendum 3).
BANK = {('doorkey_8x8', 'none'): 'research/rule_banks/v3_20260928/blind_strict.json',
        ('doorkey_8x8', 'count'): 'research/rule_banks/v3_20260928/blind_strict.json',
        ('multiroom_n6', 'none'): 'research/rule_banks/multiroom_20260928/scoped.json',
        ('multiroom_n6', 'count'): 'research/rule_banks/multiroom_20260928/blind_strict.json',
        ('keycorridor_s3r3', 'none'): MEM, ('keycorridor_s3r3', 'count'): MEM}
TAUGHT = {'doorkey_8x8': 'rules_weak', 'multiroom_n6': 'rules_weak',
          'keycorridor_s3r3': 'rules_mem_weak'}
NAMES = {'doorkey_8x8': 'DoorKey-8x8', 'multiroom_n6': 'MultiRoom-N6',
         'keycorridor_s3r3': 'KeyCorridor-S3R3'}


def as_policy(task, bank_path, n, seed0=SEED0):
    """Success rate of the bank acting alone (random useful action in gaps)."""
    if bank_path is None:
        rules, observe, stateful = [], None, False
    else:
        bank = json.loads(Path(bank_path).read_text(encoding='utf-8'))
        name = bank.get('observer', 'doorkey_v3')
        module, fn = OBSERVERS[name]
        observe = getattr(importlib.import_module(module), fn)
        stateful = name in STATEFUL
        _mode, rules, _ = load_bank(bank_path)
    wins, labelled, total = 0, 0, 0
    for i in range(n):
        if task.startswith(('MiniGrid-', 'BabyAI-')):   # a raw Gymnasium id
            import gymnasium as gym
            import minigrid  # noqa: F401  (registers the ids)
            env = gym.make(task)
        else:
            env = build_env(task, seed=seed0 + i, obs_mode='symbolic')
        env.reset(seed=seed0 + i)
        u, rng = env.unwrapped, np.random.default_rng(seed0 + i)
        for _ in range(u.max_steps):
            action, status = (None, 'no_rule') if not rules else v3.advise(
                rules, observe(u.gen_obs()['image'], u) if stateful
                else observe(u.gen_obs()['image']))
            total += 1
            labelled += status == 'advised'
            if action is None:
                action = int(rng.choice(USEFUL))
            _, reward, done, trunc, _ = env.step(action)
            if done or trunc:
                wins += reward > 0
                break
        env.close()
    return dict(success=wins / n, episodes=n,
                labelled_steps=labelled / max(total, 1))


def finals(data):
    """Mean final teacher-off success per (task, bonus, arm), fresh seeds."""
    runs = load_curves(data, 'confirm')
    return {key: float(np.mean([y[-1] for _x, y in seeds.values()]))
            for key, seeds in runs.items()}


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True,
                             facecolor='white')
    width = .26
    for ax, (bonus, student) in zip(axes, (('none', 'plain PPO'),
                                           ('count', 'PPO + count bonus'))):
        sub = [r for r in rows if r['bonus'] == bonus]
        x = np.arange(len(sub))
        for k, (key, label, color) in enumerate((
                ('ppo_alone', 'PPO alone', GREY),
                ('rules_alone', 'the LLM rules alone', ORANGE),
                ('ppo_taught', 'PPO taught by the rules', BLUE))):
            vals = [r[key] for r in sub]
            bars = ax.bar(x + (k - 1) * width, vals, width, color=color,
                          label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + .015, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(x, [NAMES[r['task']] for r in sub], fontsize=9,
                      color=INK)
        ax.set_title(student, fontsize=10.5, color=INK, loc='left')
        ax.set_ylim(0, 1.12)
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_ylabel('success rate', color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle('Rules alone vs the student they teach: rules act where '
                 'they fire, random useful actions elsewhere; students: '
                 'final teacher-off success at 5M steps, fresh seeds 30-39',
                 y=.885, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--episodes', type=int, default=200)
    cli.add_argument('--large-episodes', type=int, default=100)
    cli.add_argument('--out', type=Path,
                     default=Path('docs/assets/rule_speed_2026-09-30'))
    args = cli.parse_args()
    final = finals(args.data)
    cache, rows = {}, []
    for (task, bonus), bank in BANK.items():
        if bank not in cache:
            cache[bank] = as_policy(task, bank, args.episodes)
        rows.append(dict(task=task, bonus=bonus, bank=bank,
                         ppo_alone=final[(task, bonus, 'none')],
                         rules_alone=cache[bank]['success'],
                         labelled_steps=cache[bank]['labelled_steps'],
                         ppo_taught=final[(task, bonus, TAUGHT[task])]))
    random_only = {task: as_policy(task, None, args.episodes)['success']
                   for task in NAMES}
    dk = BANK[('doorkey_8x8', 'none')]
    large = {f'{env_id} / {name}': as_policy(env_id, bank,
                                             args.large_episodes)
             for env_id, name, bank in (
                 ('MiniGrid-DoorKey-16x16-v0', 'DoorKey rules', dk),
                 ('MiniGrid-DoorKey-16x16-v0', 'random useful actions', None),
                 ('BabyAI-KeyCorridorS4R3-v0', 'memory rules', MEM),
                 ('BabyAI-KeyCorridorS5R3-v0', 'memory rules', MEM),
                 ('BabyAI-KeyCorridorS6R3-v0', 'memory rules', MEM),
                 ('BabyAI-KeyCorridorS6R3-v0', 'random useful actions',
                  None))}
    args.out.mkdir(parents=True, exist_ok=True)
    plot(rows, args.out / 'rules_alone_vs_student.png')
    (args.out / 'rules_alone_vs_student.json').write_text(json.dumps(dict(
        rows=rows, random_only=random_only, larger_maps_zero_shot=large),
        indent=1))
    for r in rows:
        print(f"{NAMES[r['task']]:17s} {r['bonus']:5s} PPO alone "
              f"{r['ppo_alone']:.2f} | rules alone {r['rules_alone']:.2f} "
              f"(rules label {r['labelled_steps']:.0%} of steps) | PPO taught "
              f"{r['ppo_taught']:.2f}")
    print('random useful actions alone:', {NAMES[t]: round(v, 3)
                                           for t, v in random_only.items()})
    print('Larger maps, banks unchanged (as a policy):')
    for name, res in large.items():
        print(f"   {name:52s} {res['success']:.2f} of {res['episodes']} "
              f"episodes; rules label {res['labelled_steps']:.0%} of steps")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
