"""MultiRoom-N10 bank-alone success: fills Table 2's two "n/r" cells.

Same procedure as the existing larger-map
rows of docs/assets/rule_speed_2026-09-30/rules_alone_vs_student.json
(scripts/rules_alone_vs_student_20260930.as_policy): the frozen bank acts
alone and a uniformly random useful action fills every state where it
abstains; fresh maps from seed 35,000,000; 100 episodes per bank. Both
MultiRoom banks are evaluated because the two students use different banks.
No training, no API call.

    python -m scripts.bank_alone_mr10_20261004
"""

import json
from pathlib import Path

from scripts.rules_alone_vs_student_20260930 import as_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/bank_alone_2026-10-04/multiroom_n10_bank_alone.json'
BANKS = {'plain PPO bank (scoped, 32 rules)':
         'research/rule_banks/multiroom_20260928/scoped.json',
         'Count-PPO bank (blind strict, 7 rules)':
         'research/rule_banks/multiroom_20260928/blind_strict.json',
         'random useful actions': None}
EPISODES = 100


def main():
    rows = {}
    for name, bank in BANKS.items():
        rows[name] = dict(as_policy('multiroom_n10',
                                    None if bank is None else ROOT / bank,
                                    EPISODES), bank=bank)
        print(f"{name:42s} success {rows[name]['success']:.3f} of "
              f"{EPISODES}; labelled {rows[name]['labelled_steps']:.0%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(task='multiroom_n10', seed0=35_000_000,
                                   episodes=EPISODES, rows=rows), indent=1)
                   + '\n', encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
