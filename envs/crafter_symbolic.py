"""A symbolic Crafter environment for the rule-teaching study.

Declared interface (a variant, not the
pixel benchmark): the student sees exactly what the rule predicates are
computed from (scripts/crafter_rules_pilot_20260928.observe), namely

  grid    the 9x7 local view (4 cells left and right, 3 up and down), one
          id per cell: its material, or the creature or plant on it, or
          `edge` outside the world (19 kinds, crafter_rules_pilot.FRONT)
  vector  the 16 inventory counts / 9, the facing direction (one-hot),
          daylight and whether the player is asleep

The game itself, its 17 actions, rewards, achievements, death and the
10000-step episode limit are Crafter's own (crafter 1.8.3), with the
pilot's despawn fix (despawning sorted by position, not memory address).

Generating a world costs about a second, a step about 0.4 ms, so each run
generates a fixed pool of worlds once and every episode starts from a
deep copy of one of them (about 1 ms). Pools are seeded, so two arms with
the same seed train on the same worlds.
"""

import copy

import numpy as np

from scripts import crafter_rules_pilot_20260928 as pilot

KINDS = pilot.FRONT
KIND_ID = {kind: i for i, kind in enumerate(KINDS)}
INVENTORY = ('health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone',
             'coal', 'iron', 'diamond', 'wood_pickaxe', 'stone_pickaxe',
             'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword')
FACINGS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GRID_SHAPE = (2 * pilot.HALF_H + 1, 2 * pilot.HALF_W + 1)      # (7, 9)
VECTOR_SIZE = len(INVENTORY) + len(FACINGS) + 2
N_ACTIONS = len(pilot.ACTIONS)


def symbolic(env):
    """The student's observation of a crafter.Env: (grid, vector)."""
    world, player = env._world, env._player
    px, py = (int(v) for v in player.pos)
    grid = np.empty(GRID_SHAPE, dtype=np.int64)
    for row, dy in enumerate(range(-pilot.HALF_H, pilot.HALF_H + 1)):
        for col, dx in enumerate(range(-pilot.HALF_W, pilot.HALF_W + 1)):
            grid[row, col] = KIND_ID[pilot.cell_kind(world, (px + dx,
                                                             py + dy))]
    facing = tuple(int(v) for v in player.facing)
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    vector[:len(INVENTORY)] = [player.inventory[k] / 9 for k in INVENTORY]
    vector[len(INVENTORY) + FACINGS.index(facing)] = 1
    vector[-2] = float(world.daylight)
    vector[-1] = float(player.sleeping)
    return grid, vector


def world_pool(seeds):
    """Freshly reset worlds, one per seed (about a second each)."""
    return [pilot.make_env(int(s)) for s in seeds]


class CrafterSymbolic:
    """One Crafter episode stream over a fixed pool of fresh worlds."""

    def __init__(self, pool, seed):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.env = None

    def reset(self):
        self.env = copy.deepcopy(self.pool[int(self.rng.integers(
            len(self.pool)))])
        return symbolic(self.env)

    def step(self, action):
        _, reward, done, info = self.env.step(int(action))
        return symbolic(self.env), float(reward), bool(done), info

    def achievements(self):
        return sorted(k for k, v in self.env._player.achievements.items()
                      if v > 0)
