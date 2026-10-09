import random
from collections import deque

# ACTIONS
NOOP = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
MOVE_UP = 3
MOVE_DOWN = 4
DO = 5
SLEEP = 6
PLACE_STONE = 7
PLACE_TABLE = 8
PLACE_FURNACE = 9
PLACE_PLANT = 10
MAKE_WOOD_PICK = 11
MAKE_STONE_PICK = 12
MAKE_IRON_PICK = 13
MAKE_WOOD_SWORD = 14
MAKE_STONE_SWORD = 15
MAKE_IRON_SWORD = 16

# player location in view
PR = 3
PC = 4

def controller(obs, memory) -> int:
    view = obs['view']  # 7 rows x 9 cols
    inv = obs['inventory']
    facing = obs['facing']
    daylight = obs.get('daylight', 1.0)
    sleeping = obs.get('sleeping', False)
    last_action = obs.get('last_action', None)

    # initialize memory
    if 'phase' not in memory:
        memory['phase'] = 1
        memory['wander_dir'] = None
        memory['seed'] = random.randint(0, 1000000)
        random.seed(memory['seed'])
        memory['last_seen'] = {}  # record last seen counts
        memory['idle'] = 0

    phase = memory['phase']

    # Helpers
    def in_bounds(r, c):
        return 0 <= r < 7 and 0 <= c < 9

    def cell(r, c):
        if not in_bounds(r, c):
            return 'edge'
        return view[r][c]

    def find_all(name):
        coords = []
        for r in range(7):
            for c in range(9):
                if view[r][c] == name:
                    coords.append((r, c))
        return coords

    def find_any(names):
        coords = []
        for r in range(7):
            for c in range(9):
                if view[r][c] in names:
                    coords.append((r, c))
        return coords

    def nearest(target_names):
        best = None
        bestd = 999
        for r in range(7):
            for c in range(9):
                if view[r][c] in target_names:
                    d = abs(r - PR) + abs(c - PC)
                    if d < bestd:
                        bestd = d
                        best = (r, c)
        return best

    def dir_to(dr, dc):
        # choose primary axis
        if abs(dr) >= abs(dc):
            if dr < 0:
                return 'up'
            elif dr > 0:
                return 'down'
        if dc < 0:
            return 'left'
        elif dc > 0:
            return 'right'
        return None

    def facing_action_for(dir_str):
        return {'left': MOVE_LEFT, 'right': MOVE_RIGHT, 'up': MOVE_UP, 'down': MOVE_DOWN}[dir_str]

    def move_valid_dir(dir_str):
        # move action turns player that way and steps only if that cell is grass, sand or path and free.
        if dir_str == 'left':
            nr, nc = PR, PC - 1
        elif dir_str == 'right':
            nr, nc = PR, PC + 1
        elif dir_str == 'up':
            nr, nc = PR - 1, PC
        elif dir_str == 'down':
            nr, nc = PR + 1, PC
        else:
            return False
        if not in_bounds(nr, nc):
            return False
        target = view[nr][nc]
        return target in ('grass', 'sand', 'path')

    def safe_move_towards(tr, tc):
        # pick primary direction to move/face towards target (may step if allowed)
        dr = tr - PR
        dc = tc - PC
        dir_pref = dir_to(dr, dc)
        if dir_pref is None:
            # already same cell (shouldn't happen)
            return NOOP
        # If move in preferred direction is valid (or at least turns), use it; otherwise try alternatives
        # We prefer non-stepping turns sometimes; but move action will at least change facing.
        if dir_pref and (dir_pref in ('left','right','up','down')):
            return facing_action_for(dir_pref)
        # fallback
        for d in ['up', 'left', 'right', 'down']:
            return facing_action_for(d)
        return NOOP

    def face_and_do_to(target_r, target_c):
        # determine direction to target cell from player
        dr = target_r - PR
        dc = target_c - PC
        # choose primary axis
        if abs(dc) > abs(dr):
            dir_str = 'left' if dc < 0 else 'right'
        else:
            dir_str = 'up' if dr < 0 else 'down'
        # If already facing correct direction, DO
        if facing == dir_str:
            return DO
        else:
            return facing_action_for(dir_str)

    def find_free_adjacent():
        # return direction string to a free place (grass/sand/path) adjacent to player
        for d, (nr, nc) in [('up', (PR - 1, PC)), ('left', (PR, PC - 1)), ('right', (PR, PC + 1)), ('down', (PR + 1, PC))]:
            if in_bounds(nr, nc) and view[nr][nc] in ('grass', 'sand', 'path'):
                return d
        return None

    def table_in_3x3():
        for r in range(PR - 1, PR + 2):
            for c in range(PC - 1, PC + 2):
                if in_bounds(r, c) and view[r][c] == 'table':
                    return (r, c)
        return None

    def furnace_in_3x3():
        for r in range(PR - 1, PR + 2):
            for c in range(PC - 1, PC + 2):
                if in_bounds(r, c) and view[r][c] == 'furnace':
                    return (r, c)
        return None

    # small safety: if on lava in front or around? avoid moving into lava by move_valid_dir checks
    # attempt to not repeat useless actions: if same action executed many times, do noop occasionally
    memory['idle'] = memory.get('idle', 0)
    if last_action is None:
        memory['idle'] = 0
    else:
        # if last actual action was noop increment idle
        if last_action == NOOP:
            memory['idle'] += 1
        else:
            memory['idle'] = 0

    # Phase progression checks (try to automatically progress if inventory shows completion)
    # Map earlier described phases to simple inventory/structure checks
    # Phase 2 done if we have at least 2 wood (for table)
    if phase == 1:
        # immediate move to collect wood -> we consider phase1 done and go to phase2 if we already have enough wood
        memory['phase'] = 2
        phase = 2

    # Advance through phases when prerequisites met
    if phase <= 3 and inv.get('wood', 0) >= 2 and phase < 3:
        memory['phase'] = 3
        phase = 3
    if phase <= 4:
        # if we have both wood tools (or made) go forward
        if inv.get('wood_pickaxe', 0) >= 1 and inv.get('wood_sword', 0) >= 1:
            memory['phase'] = 5
            phase = 5
    if phase == 5:
        # if we have stone and coal to make stone tools
        if inv.get('stone', 0) >= 1 and inv.get('coal', 0) >= 1:
            memory['phase'] = 7
            phase = 7
    if phase == 7:
        if inv.get('stone_pickaxe', 0) >= 1 and inv.get('stone_sword', 0) >= 1:
            memory['phase'] = 8
            phase = 8
    if phase == 8:
        if inv.get('iron', 0) >= 1:
            memory['phase'] = 10  # move to place furnace after collecting iron
            phase = 10
    if phase == 10:
        if furnace_in_3x3():
            memory['phase'] = 11
            phase = 11
    if phase == 11:
        if inv.get('iron_pickaxe', 0) >= 1 and inv.get('iron_sword', 0) >= 1:
            memory['phase'] = 12
            phase = 12
    if phase == 12:
        if inv.get('diamond', 0) >= 1:
            memory['phase'] = 13
            phase = 13
    if phase == 14:
        # after placing plant go to drinking
        memory['phase'] = 16
        phase = 16
    if phase == 16:
        # if drink present in inventory
        if inv.get('drink', 0) > 0:
            memory['phase'] = 17
            phase = 17
    if phase == 17:
        # if food positive
        if inv.get('food', 0) > 0:
            memory['phase'] = 18
            phase = 18
    if phase == 18:
        if inv.get('energy', 0) > 80 and not sleeping:
            memory['phase'] = 19
            phase = 19

    # Action selection per phase
    action = NOOP

    # Generic helpers inside phases
    def go_to_resource(names):
        tgt = nearest(names)
        if not tgt:
            # wander
            return wander()
        tr, tc = tgt
        # if adjacent and facing, try DO
        if abs(tr - PR) + abs(tc - PC) == 1:
            # face and do
            return face_and_do_to(tr, tc)
        # else move towards
        return safe_move_towards(tr, tc)

    def wander():
        # simple wandering: pick a random valid direction different from last if stuck
        dirs = ['up', 'left', 'right', 'down']
        # try prefer specific sequence to ensure determinism with seed
        random.shuffle(dirs)
        for d in dirs:
            if move_valid_dir(d):
                return facing_action_for(d)
        # if no walkable moves, try turning actions even if not stepable
        return facing_action_for(random.choice(['up', 'left', 'right', 'down']))

    # Phase behaviors
    if phase == 2:
        # collect wood until we have 2 wood
        if inv.get('wood', 0) >= 2:
            memory['phase'] = 3
            action = NOOP
        else:
            # find tree or any grass to try sapling
            if find_all('tree'):
                action = go_to_resource(['tree'])
            else:
                # wander to find trees
                action = wander()

    elif phase == 3:
        # place a table (requires 2 wood)
        if table_in_3x3():
            memory['phase'] = 4
            action = NOOP
        elif inv.get('wood', 0) >= 2:
            # find free adjacent to place table
            d = find_free_adjacent()
            if d is None:
                # move so free adjacent available: wander a bit
                action = wander()
            else:
                if facing == d:
                    action = PLACE_TABLE
                else:
                    action = facing_action_for(d)
        else:
            # fallback collect wood
            action = go_to_resource(['tree'])

    elif phase == 4:
        # make wood pickaxe and wood sword at table nearby
        t = table_in_3x3()
        if not t:
            # find table somewhere in view
            tbl = nearest(['table'])
            if tbl:
                tr, tc = tbl
                action = safe_move_towards(tr, tc)
            else:
                # move to create table location
                action = wander()
        else:
            # ensure we are near the table (3x3)
            if inv.get('wood_pickaxe', 0) == 0:
                action = MAKE_WOOD_PICK
            elif inv.get('wood_sword', 0) == 0:
                action = MAKE_WOOD_SWORD
            else:
                memory['phase'] = 5
                action = NOOP

    elif phase == 5:
        # explore to find stone and coal deposits; then collect them in phase6
        # If we already have any, move to collect them
        if inv.get('stone', 0) >= 1 and inv.get('coal', 0) >= 1:
            memory['phase'] = 7
            action = NOOP
        else:
            # move towards nearest stone or coal
            action = go_to_resource(['stone', 'coal', 'coal', 'stone'])

    elif phase == 6:
        # (Not directly used since we auto-jump to 7 when collected)
        action = wander()

    elif phase == 7:
        # make stone pickaxe and stone sword (need wood and stone, and table nearby)
        t = table_in_3x3()
        if not t:
            # move to table
            tbl = nearest(['table'])
            if tbl:
                tr, tc = tbl
                action = safe_move_towards(tr, tc)
            else:
                action = wander()
        else:
            if inv.get('stone_pickaxe', 0) == 0 and inv.get('wood', 0) >= 1 and inv.get('stone', 0) >= 1:
                action = MAKE_STONE_PICK
            elif inv.get('stone_sword', 0) == 0 and inv.get('wood', 0) >= 1 and inv.get('stone', 0) >= 1:
                action = MAKE_STONE_SWORD
            else:
                memory['phase'] = 8
                action = NOOP

    elif phase == 8:
        # explore to find iron
        if inv.get('iron', 0) >= 1:
            memory['phase'] = 10
            action = NOOP
        else:
            action = go_to_resource(['iron', 'coal', 'stone'])

    elif phase == 9:
        # fallback / collect iron - merged into phase8/10 chain
        action = wander()

    elif phase == 10:
        # place a furnace (needs 4 stone and a table nearby)
        if furnace_in_3x3():
            memory['phase'] = 11
            action = NOOP
        elif inv.get('stone', 0) >= 4 and table_in_3x3():
            # find a free adjacent to place furnace
            d = find_free_adjacent()
            if d is None:
                action = wander()
            else:
                if facing == d:
                    action = PLACE_FURNACE
                else:
                    action = facing_action_for(d)
        else:
            # if no table nearby, go to table
            tbl = nearest(['table'])
            if tbl:
                action = safe_move_towards(tbl[0], tbl[1])
            else:
                action = wander()

    elif phase == 11:
        # make iron pickaxe and iron sword (need table nearby and furnace)
        if not table_in_3x3() or not furnace_in_3x3():
            # move to nearest furnace or table
            f = nearest(['furnace'])
            if f:
                action = safe_move_towards(f[0], f[1])
            else:
                t = nearest(['table'])
                if t:
                    action = safe_move_towards(t[0], t[1])
                else:
                    action = wander()
        else:
            if inv.get('iron_pickaxe', 0) == 0 and inv.get('wood', 0) >= 1 and inv.get('coal', 0) >= 1 and inv.get('iron', 0) >= 1:
                action = MAKE_IRON_PICK
            elif inv.get('iron_sword', 0) == 0 and inv.get('wood', 0) >= 1 and inv.get('coal', 0) >= 1 and inv.get('iron', 0) >= 1:
                action = MAKE_IRON_SWORD
            else:
                memory['phase'] = 12
                action = NOOP

    elif phase == 12:
        # explore to find diamond, saplings and ripe plants
        if inv.get('diamond', 0) >= 1:
            memory['phase'] = 13
            action = NOOP
        else:
            # move toward diamond if seen, else look for plant_ripe or grass (for sapling attempts)
            if find_all('diamond'):
                action = go_to_resource(['diamond'])
            elif find_all('plant_ripe'):
                action = go_to_resource(['plant_ripe'])
            else:
                # try to do on grass occasionally to get saplings
                g = nearest(['grass'])
                if g:
                    # if adjacent attempt do to get sapling
                    if abs(g[0]-PR)+abs(g[1]-PC) == 1:
                        action = face_and_do_to(g[0], g[1])
                    else:
                        action = safe_move_towards(g[0], g[1])
                else:
                    action = wander()

    elif phase == 13:
        # collect diamond (requires iron pickaxe)
        if inv.get('diamond', 0) >= 1:
            memory['phase'] = 14
            action = NOOP
        else:
            if inv.get('iron_pickaxe', 0) >= 1 and find_all('diamond'):
                action = go_to_resource(['diamond'])
            else:
                # wander to find diamond
                action = wander()

    elif phase == 14:
        # collect saplings
        if inv.get('sapling', 0) >= 1:
            memory['phase'] = 15
            action = NOOP
        else:
            # try to find grass and do to get sapling (10% chance)
            g = nearest(['grass'])
            if g:
                if abs(g[0]-PR)+abs(g[1]-PC) == 1:
                    action = face_and_do_to(g[0], g[1])
                else:
                    action = safe_move_towards(g[0], g[1])
            else:
                action = wander()

    elif phase == 15:
        # place a plant (needs sapling and grass in front)
        if inv.get('sapling', 0) <= 0:
            memory['phase'] = 16
            action = NOOP
        else:
            # find adjacent grass to place on
            for d, (nr, nc) in [('up', (PR-1, PC)), ('left', (PR, PC-1)), ('right', (PR, PC+1)), ('down', (PR+1, PC))]:
                if in_bounds(nr, nc) and view[nr][nc] == 'grass':
                    if facing == d:
                        action = PLACE_PLANT
                    else:
                        action = facing_action_for(d)
                    break
            else:
                # move to find grass
                g = nearest(['grass'])
                if g:
                    action = safe_move_towards(g[0], g[1])
                else:
                    action = wander()

    elif phase == 16:
        # drink water
        if inv.get('drink', 0) > 0:
            memory['phase'] = 17
            action = NOOP
        else:
            if find_all('water'):
                action = go_to_resource(['water'])
            else:
                action = wander()

    elif phase == 17:
        # eat cows and ripe plants until food > 0
        if inv.get('food', 0) > 0:
            memory['phase'] = 18
            action = NOOP
        else:
            if find_all('cow') or find_all('plant_ripe'):
                action = go_to_resource(['cow', 'plant_ripe'])
            else:
                action = wander()

    elif phase == 18:
        # sleep to restore energy
        # try sleeping at night or when energy low
        if sleeping:
            # keep sleeping until energy restored (monitored via inventory)
            action = SLEEP
        else:
            # if safe (no zombies/skeletons adjacent) and energy low, sleep
            enemies = find_any(['zombie', 'skeleton'])
            if inv.get('energy', 0) < 50 and not enemies:
                action = SLEEP
            else:
                # if energy OK, move on
                memory['phase'] = 19
                action = NOOP

    elif phase == 19:
        # defeat zombies and skeletons
        enemies = find_any(['zombie', 'skeleton'])
        if enemies:
            # go to nearest enemy and attack
            action = go_to_resource(['zombie', 'skeleton'])
        else:
            # roam, still stay alert
            # also try to collect any remaining resources like cows or ripe plants
            if find_all('cow') or find_all('plant_ripe'):
                action = go_to_resource(['cow', 'plant_ripe'])
            else:
                action = wander()

    else:
        # default fallback
        action = wander()

    # safety: avoid issuing place actions where facing cell is not a valid placement
    if action in (PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, PLACE_STONE):
        # ensure facing cell is valid for placement
        if facing == 'left':
            fr, fc = PR, PC - 1
        elif facing == 'right':
            fr, fc = PR, PC + 1
        elif facing == 'up':
            fr, fc = PR - 1, PC
        else:
            fr, fc = PR + 1, PC
        if not in_bounds(fr, fc) or view[fr][fc] not in ('grass', 'sand', 'path'):
            # try to pick a free adjacent and face it instead
            d = find_free_adjacent()
            if d:
                action = facing_action_for(d)
            else:
                action = NOOP

    # safety: if action is a crafting action but no table nearby, try to move to table
    if action in (MAKE_WOOD_PICK, MAKE_STONE_PICK, MAKE_IRON_PICK, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD):
        if not table_in_3x3():
            tbl = nearest(['table'])
            if tbl:
                action = safe_move_towards(tbl[0], tbl[1])
            else:
                action = NOOP

    # safety: avoid trying to move into lava (move_valid_dir prevents stepping into lava)
    # final return
    return int(action)
