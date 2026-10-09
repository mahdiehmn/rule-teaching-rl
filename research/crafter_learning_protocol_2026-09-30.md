# Exploratory Crafter learning study (frozen before any run)

Date: 2026-09-30. Code: envs/crafter_symbolic.py,
algos/ppo_crafter.py, scripts/run_crafter_learning_20260930.py.

## Deviation, stated first

The v2 offline gate (research/crafter_rules_v2_protocol_2026-09-30.md) was
**INCONCLUSIVE**: the chosen rule bank played better than random (+1.27
achievements per episode, place_table 37%, wood pickaxe 20%) but missed two
of the GO thresholds (+1.5 achievements, 40% tables). Under that contract no
learning study would follow before the paper deadline. One was run anyway,
as an **exploratory** study (2026-09-30). The paper must say
this, report the gate as it was, and not present this study as confirmatory.
The gate is a proxy: it runs the rules alone, while the claim concerns a
student that learns from them (MultiRoom's checked rules finished no
episode alone and still sped up learning).

## Interface (a declared variant of Crafter)

The student observes what the rule predicates are computed from: the 9x7
local view as one id per cell (19 kinds: materials, creatures, ripe plant,
edge) and a vector of the 16 inventory counts / 9, facing (one-hot),
daylight and sleep. Crafter's own 17 actions, rewards, achievements, death
and 10000-step limit (crafter 1.8.3), with the pilot's despawn fix. Scores
are not comparable to the pixel benchmark. Each run trains on a pool of 200
worlds generated from its seed (seeds 40,000,000 + 1000 x seed + i); every
episode starts from a copy of one. Evaluation uses 10 held-out worlds
(seeds 41,000,000+) that no training pool contains.

## Student and arms

Plain PPO, feed-forward (embedding and two convolutions over the view, an
MLP with the vector): 16 environments x 128 steps, learning rate 3e-4
annealed, gamma .99, GAE .95, 4 epochs, 4 minibatches, clip .2, entropy
.01, value .5, gradient norm .5; 1M steps; seeds 1-10. The arms differ
only in the teaching channel, as in MiniGrid: the student always acts by
its own policy, and where the frozen bank fires its action is a
cross-entropy target averaged over labelled steps, decaying linearly over
the first half of training and removed after 75%.

- `none`: no teacher.
- `rules_weak` (**primary**): the bank chosen by the v2 development rule,
  research/rule_banks/crafter_v2_20260930/self_checked.json (sha256 of
  its LF form 0c378a80191af395d332f810acacd969f40f5b7e300fcea294ba5b5f0f2a812c),
  weight .1 -> .001.
- `rules` (secondary): the same bank at the paper's weight, 1 -> .01.

Same seed: same training worlds, environment streams and initial network,
so the arms pair by seed. 30 runs on local CPUs; no LLM call anywhere.

## Measures and analysis

Every 50k steps: 10 teacher-free episodes on the held-out worlds, sampled
actions, capped at 3000 steps. Primary measure: mean achievements per
episode; AUC is its mean over the 20 evaluations. **Primary contrast:**
`rules_weak - none`, paired AUC difference with its 95% interval and
two-sided paired-t p. Secondary: `rules - none`, `rules_weak - rules`,
final achievements, Crafter score (geometric mean of success rates) and
per-achievement rates at 1M. Frozen prediction: `rules_weak - none` is
positive.

## Decision

Every run is reported; no seed is added or replaced and nothing is rerun
after seeing results. A positive primary interval is reported as
exploratory evidence that the same teaching channel speeds learning in a
second environment family; anything else is reported as it is.

## Result (2026-10-01 07:02 MDT; all 30 runs; docs/assets/crafter_2026-10-01/)

| Contrast | AUC difference [95% CI] | positive | p |
|---|---|---|---|
| **rules_weak - none (primary)** | **+0.38 [-0.00, +0.76]** | 7/10 | .051 |
| rules - none | -0.62 [-0.88, -0.37] | 0/10 | .0003 |
| rules_weak - rules | +1.00 [+0.80, +1.20] | 10/10 | 1.3e-6 |

Mean AUC (achievements per episode): none 5.57, rules_weak 5.95, rules
4.95. At 1M steps: achievements 6.25 / 6.75 / 6.87 and Crafter score 5.58 /
6.93 / 6.93 (none / rules_weak / rules). Rules labelled 31% of steps while
imitation was on. Runs took 2.3-2.4 h each with ten in parallel.

Reading, exploratory: the prediction (rules_weak above none) holds in sign
and its interval reaches zero (p .051). Full-weight imitation of this bank
slows early learning badly yet ends above no teacher; a tenth of the weight
keeps the late gain without the early cost. The pre-registered
confirmation (seeds 11-20) launched under its frozen condition.
