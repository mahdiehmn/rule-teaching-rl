# RLingua controllers for the MiniGrid families (fix-wave addendum 16)

Generated with
`scripts/rlingua_controllers_20261005.py`; the
controller code itself was written by GPT-5-mini (reasoning effort medium)
through RLingua's prompting recipe (Chen et al., RA-L 2024, Appendix B.A),
with automatic feedback from test episodes. No researcher edited any
controller.

Files, one pair per task family (`dk` DoorKey, `mr` MultiRoom, `kc`
KeyCorridor) and information setting (`full`: the simulator state, RLingua
as published; `view`: what the student sees, i.e. its 7x7 view and the
action actually executed at the previous step, plus a per-episode memory
dict):

- `<family>_<setting>.py`: the frozen controller. Its sha256 (LF line
  endings) is pinned in the receipt and in each training cell's
  `rlingua_controller_sha256`.
- `<family>_<setting>_receipt.json`: the full conversation, every API call
  (model, tokens, cost), every feedback round with its success counts, the
  selected round, and, for `kc_full`, the speed-only feedback attempts
  (`speed_rounds`, `speed_baseline`; the reply text of the first, single
  attempt was not stored, only its statistics).
- `standalone_success.json`: each controller on 100 fresh layouts (seeds
  37,000,000+), alone, and for view controllers also with another policy
  acting at 25% of the steps; `<name>@<task>` rows are the larger maps.
  Not used for any selection.
- `superseded_20261005_view_not_execution_aware/`: the first view
  controllers and receipts. Their memory assumed their own outputs were
  executed, which is false under RLingua's mixing (found in review,
  2026-10-05); they were replaced, before any training run, by view
  controllers that receive the executed action and were selected on
  episodes in which another policy also acts.

The training harness (`teachers/minigrid/rlingua_controller.py`) gives each
env its own controller instance and recreates it at every episode start,
so nothing a controller keeps between calls is shared across envs or
episodes.

Generation cost: $0.27 for the first six controllers and the speed rounds,
$0.14 for the three execution-aware view controllers. Training use and
frozen predictions: `research/fix_wave_protocol_2026-09-29.md`, addendum 16.
These files are evidence about the controllers, not about trained students.
