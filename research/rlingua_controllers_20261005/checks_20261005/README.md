# Controller-level checks for the RLingua comparison (no training)

Development
checks, not training inputs and not part of protocol addendum 16's
frozen suites. Controllers were written by GPT-5-mini; no researcher
edited them.

- `examples/` (Check B, `scripts/rlingua_examples_check_20261005.py`):
  RLingua's recipe for the student-view MiniGrid controllers, with one
  added message showing 36 example situations with the full map (the kind
  of information our rule writer saw). Success alone on the 100 standard
  layouts (seeds 37,000,000+), and with a random action at 25% of steps:
  DoorKey 1.00 (1.00), MultiRoom 0.00 (0.00), KeyCorridor 0.00 (0.02).
  Cost $0.155.
- `crafter/` (Check A, `scripts/rlingua_crafter_check_20261005.py`):
  RLingua's recipe for complete Crafter controllers, given the task text
  our Crafter rules were written from. Alone on 20 fresh worlds (seeds
  42,000,000+): full-world controller 10.25 achievements per episode
  (Crafter score 15.4); 9x7-view controller 2.0 (score 0.5; 2.95 with a
  random action at 25% of steps); random policy 2.25 (score 1.3).
  Cost $0.164.

Each `*_receipt.json` holds the full conversation, every call's tokens
and cost, every round's scores and the selected round. Related
illustrations: `scripts/rlingua_gap_examples_20261005.py` (identical
student views on which the full-state planner acts differently: 133 in
MultiRoom; a student-view controller's failed MultiRoom episode).
