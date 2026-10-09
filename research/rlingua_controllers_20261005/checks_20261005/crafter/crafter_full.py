from collections import deque
import math
import random

# ACTION constants
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_DO = 5
ACTION_SLEEP = 6
ACTION_PLACE_STONE = 7
ACTION_PLACE_TABLE = 8
ACTION_PLACE_FURNACE = 9
ACTION_PLACE_PLANT = 10
ACTION_MAKE_WOOD_PICK = 11
ACTION_MAKE_STONE_PICK = 12
ACTION_MAKE_IRON_PICK = 13
ACTION_MAKE_WOOD_SWORD = 14
ACTION_MAKE_STONE_SWORD = 15
ACTION_MAKE_IRON_SWORD = 16

# Directions
DIRS = {
    1: (-1, 0),  # left
    2: (1, 0),   # right
    3: (0, -1),  # up
    4: (0, 1),   # down
}
FAC_STR_TO_DIR = {'left': 1, 'right': 2, 'up': 3, 'down': 4}
PASSABLE_TILES = {'grass', 'sand', 'path'}

# Tunables based on observed runs
MIN_WOOD_FOR_TABLE = 2
MIN_STONE_FOR_FURNACE = 4
STONE_BUFFER = 3        # keep extra stone to avoid stalls
COAL_TARGET = 2
IRON_TARGET = 2         # try to collect more iron to craft iron tools reliably
PREFERRED_FOOD = 2
PREFERRED_DRINK = 2
SAFE_SLEEP_ENERGY = 8   # lower threshold to avoid over-sleeping
NIGHT_DAYLIGHT = 0.35
RUN_AWAY_RADIUS = 4     # avoid monsters within this radius

def controller(state) -> int:
    width, height = state['size']
    game_map = state['map']
    objects = state['objects']
    player = state['player']
    px, py = player['pos']
    facing = player.get('facing', 'down')
    facing_dir = FAC_STR_TO_DIR.get(facing, 4)
    inv = player.get('inventory', {})
    daylight = state.get('daylight', 1.0)

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    # Occupancy and object lookup
    obj_pos_set = set()
    kind_positions = {}
    for o in objects:
        pos = o.get('pos')
        if pos is None:
            continue
        pos_t = tuple(pos)
        obj_pos_set.add(pos_t)
        kind_positions.setdefault(o['kind'], []).append(o)

    def tile_at(x, y):
        return game_map[y][x]

    def is_passable(x, y):
        if not in_bounds(x, y):
            return False
        if (x, y) in obj_pos_set:
            return False
        return tile_at(x, y) in PASSABLE_TILES

    def is_free_place_tile(x, y):
        if not in_bounds(x, y):
            return False
        if tile_at(x, y) not in PASSABLE_TILES:
            return False
        if (x, y) in obj_pos_set:
            return False
        return True

    def find_tiles_of_type(tile_type):
        res = []
        for y in range(height):
            row = game_map[y]
            for x in range(width):
                if row[x] == tile_type:
                    res.append((x, y))
        return res

    def find_tiles_of_types(tile_types):
        s = set(tile_types)
        res = []
        for y in range(height):
            row = game_map[y]
            for x in range(width):
                if row[x] in s:
                    res.append((x, y))
        return res

    def find_objects_of_kind(kind):
        return [o for o in objects if o['kind'] == kind]

    def find_map_item(name):
        for y in range(height):
            for x in range(width):
                if game_map[y][x] == name:
                    return (x, y)
        return None

    table_pos = find_map_item('table')
    furnace_pos = find_map_item('furnace')

    def nearby_map_item(name):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = px + dx, py + dy
                if in_bounds(nx, ny) and game_map[ny][nx] == name:
                    return True
        return False

    # BFS that finds a path to a position adjacent to any target such that the final facing points to target
    def bfs_to_face_targets(target_positions):
        if not target_positions:
            return None
        def adj_and_dir(x0, y0, t):
            tx, ty = t
            dx = tx - x0
            dy = ty - y0
            if abs(dx) + abs(dy) != 1:
                return None
            if dx == -1:
                return 1
            if dx == 1:
                return 2
            if dy == -1:
                return 3
            if dy == 1:
                return 4
            return None

        # already adjacent and facing?
        for t in target_positions:
            d = adj_and_dir(px, py, t)
            if d is not None and facing_dir == d:
                return ([], t, d)

        visited = set()
        q = deque()
        q.append((px, py, [], None))
        visited.add((px, py, None))
        while q:
            x, y, path, last_dir = q.popleft()
            for t in target_positions:
                d = adj_and_dir(x, y, t)
                if d is not None and last_dir == d:
                    return (path, t, d)
            for d, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                if not is_passable(nx, ny):
                    continue
                key = (nx, ny, d)
                if key in visited:
                    continue
                visited.add(key)
                q.append((nx, ny, path + [d], d))
        return None

    # BFS within Chebyshev radius
    def bfs_to_get_within(target_positions, radius=1):
        if not target_positions:
            return None
        def within(x, y):
            for tx, ty in target_positions:
                if abs(tx - x) <= radius and abs(ty - y) <= radius:
                    return True
            return False
        if within(px, py):
            return []
        visited = set()
        q = deque()
        q.append((px, py, []))
        visited.add((px, py))
        while q:
            x, y, path = q.popleft()
            if within(x, y):
                return path
            for d, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                if not is_passable(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                q.append((nx, ny, path + [d]))
        return None

    def dir_to_move_action(d):
        return d

    def first_move_for_path(path):
        if not path:
            return None
        return dir_to_move_action(path[0])

    def go_near_map_item(name):
        pos = find_map_item(name)
        if not pos:
            return None
        res = bfs_to_get_within([pos], radius=1)
        if res is None:
            return None
        if res:
            return first_move_for_path(res)
        return None

    # inventory counts
    wood = inv.get('wood', 0)
    stone = inv.get('stone', 0)
    coal = inv.get('coal', 0)
    iron = inv.get('iron', 0)
    diamond = inv.get('diamond', 0)
    sapling = inv.get('sapling', 0)
    food = inv.get('food', 0)
    drink = inv.get('drink', 0)
    energy = inv.get('energy', 0)
    health = inv.get('health', 0)
    w_pick = inv.get('wood_pickaxe', 0)
    s_pick = inv.get('stone_pickaxe', 0)
    i_pick = inv.get('iron_pickaxe', 0)
    w_sword = inv.get('wood_sword', 0)
    s_sword = inv.get('stone_sword', 0)
    i_sword = inv.get('iron_sword', 0)

    # Monsters helper
    monsters = [o for o in objects if o['kind'] in ('zombie', 'skeleton')]
    def nearest_monster_within(radius=999):
        best = None
        bestd = None
        for m in monsters:
            ox, oy = m['pos']
            d = abs(ox - px) + abs(oy - py)
            if d <= radius and (best is None or d < bestd):
                best = m
                bestd = d
        return best, bestd

    # Immediate defensive behavior: if monster very close, fight with sword if available, else retreat
    close_mon, close_d = nearest_monster_within(radius=3)
    if close_mon is not None and close_d <= 2:
        # if we have any sword, prefer to face and attack
        if i_sword + s_sword + w_sword > 0:
            res = bfs_to_face_targets([tuple(close_mon['pos'])])
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO
        # else retreat to base (table/furnace), else step away
        if table_pos:
            mv = go_near_map_item('table')
            if mv:
                return mv
        if furnace_pos:
            mv = go_near_map_item('furnace')
            if mv:
                return mv
        # step away from monster
        ox, oy = close_mon['pos']
        dx = px - ox
        dy = py - oy
        # prefer axis that increases distance most
        if abs(dx) >= abs(dy):
            if dx > 0 and is_passable(px + 1, py):
                return ACTION_RIGHT
            if dx < 0 and is_passable(px - 1, py):
                return ACTION_LEFT
        if dy > 0 and is_passable(px, py + 1):
            return ACTION_DOWN
        if dy < 0 and is_passable(px, py - 1):
            return ACTION_UP
        # fallback to attack to avoid being stuck
        res = bfs_to_face_targets([tuple(close_mon['pos'])])
        if res:
            path, tpos, d = res
            if path:
                return dir_to_move_action(path[0])
            else:
                return ACTION_DO
        return ACTION_NOOP

    # Proactive thresholds: gather drink/food if low
    if drink < PREFERRED_DRINK:
        water = find_tiles_of_type('water')
        if water:
            res = bfs_to_face_targets(water)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO

    if food < PREFERRED_FOOD:
        # prefer ripe plants then cows
        for o in objects:
            if o['kind'] == 'plant' and o.get('ripe'):
                res = bfs_to_face_targets([tuple(o['pos'])])
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        cows = find_objects_of_kind('cow')
        if cows:
            res = bfs_to_face_targets([tuple(c['pos']) for c in cows])
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO

    # Sleeping policy: avoid excessive sleeping observed earlier.
    # Only sleep if energy critically low and it's night and safe.
    if energy <= SAFE_SLEEP_ENERGY and not player.get('sleeping', False) and daylight <= NIGHT_DAYLIGHT:
        mon, md = nearest_monster_within(radius=RUN_AWAY_RADIUS)
        if mon is None:
            return ACTION_SLEEP
        # if not safe, move toward table/furnace before sleeping
        if table_pos:
            mv = go_near_map_item('table')
            if mv:
                return mv
        if furnace_pos:
            mv = go_near_map_item('furnace')
            if mv:
                return mv

    # STAGED BEHAVIOR (prioritise survival: get swords early, place furnace reliably, collect iron)
    # Phase 1: collect wood for table
    if wood < MIN_WOOD_FOR_TABLE:
        trees = find_tiles_of_type('tree')
        if trees:
            res = bfs_to_face_targets(trees)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO
        # explore a bit
        for d, (dx, dy) in DIRS.items():
            nx, ny = px + dx, py + dy
            if is_passable(nx, ny):
                return dir_to_move_action(d)
        return ACTION_NOOP

    # Phase 2: place table (base) early
    if table_pos is None:
        if wood >= MIN_WOOD_FOR_TABLE:
            free_tiles = [(x, y) for y in range(height) for x in range(width) if is_free_place_tile(x, y)]
            if free_tiles:
                res = bfs_to_face_targets(free_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_PLACE_TABLE
        else:
            # fallback gather wood
            trees = find_tiles_of_type('tree')
            if trees:
                res = bfs_to_face_targets(trees)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        return ACTION_NOOP

    # Phase 3: make wood pick & wood sword if missing
    if w_pick <= 0 or w_sword <= 0:
        if not nearby_map_item('table'):
            mv = go_near_map_item('table')
            if mv:
                return mv
        if w_pick <= 0:
            return ACTION_MAKE_WOOD_PICK
        if w_sword <= 0:
            return ACTION_MAKE_WOOD_SWORD

    # Phase 4: collect stone and coal but prioritise making stone sword ASAP for survival
    stone_needed = MIN_STONE_FOR_FURNACE + STONE_BUFFER
    if stone < 1:  # need at least stone to craft stone sword/pick
        # mine stone
        s_tiles = find_tiles_of_type('stone')
        if s_tiles:
            res = bfs_to_face_targets(s_tiles)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO
        # else wander
        for d, (dx, dy) in DIRS.items():
            nx, ny = px + dx, py + dy
            if is_passable(nx, ny):
                return dir_to_move_action(d)
        return ACTION_NOOP

    # If we don't have stone sword, craft it early (improves survival)
    if s_sword <= 0:
        if not nearby_map_item('table'):
            mv = go_near_map_item('table')
            if mv:
                return mv
        if wood >= 1 and stone >= 1:
            return ACTION_MAKE_STONE_SWORD
        # otherwise mine stone
        s_tiles = find_tiles_of_type('stone')
        if s_tiles:
            res = bfs_to_face_targets(s_tiles)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO

    # Ensure stone pick exists for iron mining
    if s_pick <= 0:
        if not nearby_map_item('table'):
            mv = go_near_map_item('table')
            if mv:
                return mv
        if wood >= 1 and stone >= 1:
            return ACTION_MAKE_STONE_PICK
        s_tiles = find_tiles_of_type('stone')
        if s_tiles:
            res = bfs_to_face_targets(s_tiles)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO

    # Keep collecting stone and coal until comfortable amounts for furnace and iron crafting
    if stone < stone_needed or coal < COAL_TARGET:
        # ensure wood pick exists for stone mining earlier; here s_pick is present
        if coal < COAL_TARGET:
            coal_tiles = find_tiles_of_type('coal')
            if coal_tiles:
                res = bfs_to_face_targets(coal_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        if stone < stone_needed:
            stone_tiles = find_tiles_of_type('stone')
            if stone_tiles:
                res = bfs_to_face_targets(stone_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        # explore
        for d, (dx, dy) in DIRS.items():
            nx, ny = px + dx, py + dy
            if is_passable(nx, ny):
                return dir_to_move_action(d)
        return ACTION_NOOP

    # Phase 5: place furnace promptly once enough stone and table exists
    if furnace_pos is None:
        if stone >= MIN_STONE_FOR_FURNACE and table_pos is not None:
            # go near table then place furnace on nearby free tile
            if not nearby_map_item('table'):
                mv = go_near_map_item('table')
                if mv:
                    return mv
            free_tiles = [(x, y) for y in range(height) for x in range(width) if is_free_place_tile(x, y)]
            if free_tiles:
                res = bfs_to_face_targets(free_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_PLACE_FURNACE
        # else keep mining stone if not enough
        if stone < MIN_STONE_FOR_FURNACE:
            stone_tiles = find_tiles_of_type('stone')
            if stone_tiles:
                res = bfs_to_face_targets(stone_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        return ACTION_NOOP

    # Phase 6: gather iron reliably (we aim for IRON_TARGET)
    if iron < IRON_TARGET or coal < COAL_TARGET:
        # ensure stone pick exists
        if s_pick <= 0:
            if not nearby_map_item('table'):
                mv = go_near_map_item('table')
                if mv:
                    return mv
            return ACTION_MAKE_STONE_PICK
        # mine iron if present
        if iron < IRON_TARGET:
            iron_tiles = find_tiles_of_type('iron')
            if iron_tiles:
                res = bfs_to_face_targets(iron_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        if coal < COAL_TARGET:
            coal_tiles = find_tiles_of_type('coal')
            if coal_tiles:
                res = bfs_to_face_targets(coal_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        # wander/search
        for d, (dx, dy) in DIRS.items():
            nx, ny = px + dx, py + dy
            if is_passable(nx, ny):
                return dir_to_move_action(d)
        return ACTION_NOOP

    # Phase 7: craft iron tools (iron pick then iron sword) when resources available and near table/furnace
    if i_pick <= 0 or i_sword <= 0:
        # ensure nearby table & furnace
        if not nearby_map_item('table'):
            mv = go_near_map_item('table')
            if mv:
                return mv
        if not nearby_map_item('furnace'):
            mv = go_near_map_item('furnace')
            if mv:
                return mv
        # craft iron sword first if combat threat exists or swords missing
        if i_sword <= 0 and wood >= 1 and coal >= 1 and iron >= 1:
            return ACTION_MAKE_IRON_SWORD
        if i_pick <= 0 and wood >= 1 and coal >= 1 and iron >= 1:
            return ACTION_MAKE_IRON_PICK
        # otherwise gather missing resource
        if iron < 1:
            iron_tiles = find_tiles_of_type('iron')
            if iron_tiles:
                res = bfs_to_face_targets(iron_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        if coal < 1:
            coal_tiles = find_tiles_of_type('coal')
            if coal_tiles:
                res = bfs_to_face_targets(coal_tiles)
                if res:
                    path, tpos, d = res
                    if path:
                        return dir_to_move_action(path[0])
                    else:
                        return ACTION_DO
        return ACTION_NOOP

    # Phase 8: diamond collection optional (requires iron pick)
    if diamond < 1 and i_pick > 0:
        diamond_tiles = find_tiles_of_type('diamond')
        if diamond_tiles:
            res = bfs_to_face_targets(diamond_tiles)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO

    # Saplings and plants: collect/plant if handy
    if sapling <= 0:
        grass_tiles = find_tiles_of_type('grass')
        if grass_tiles:
            res = bfs_to_face_targets(grass_tiles)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO
    else:
        grass_free = [(x, y) for y in range(height) for x in range(width) if game_map[y][x] == 'grass' and (x, y) not in obj_pos_set]
        if grass_free:
            res = bfs_to_face_targets(grass_free)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_PLACE_PLANT

    # Combat cleanup: if monsters exist, prefer to engage when armed, otherwise retreat + craft sword
    if monsters:
        if i_sword + s_sword + w_sword > 0:
            targets = [tuple(m['pos']) for m in monsters]
            res = bfs_to_face_targets(targets)
            if res:
                path, tpos, d = res
                if path:
                    return dir_to_move_action(path[0])
                else:
                    return ACTION_DO
        else:
            # retreat to table and craft stone sword if resources
            if table_pos:
                mv = go_near_map_item('table')
                if mv:
                    return mv
            if s_sword <= 0 and nearby_map_item('table') and wood >= 1 and stone >= 1:
                return ACTION_MAKE_STONE_SWORD
            # else step away from nearest monster
            mon, md = nearest_monster_within(radius=999)
            if mon:
                ox, oy = mon['pos']
                dx = px - ox
                dy = py - oy
                if abs(dx) >= abs(dy):
                    if dx > 0 and is_passable(px + 1, py):
                        return ACTION_RIGHT
                    if dx < 0 and is_passable(px - 1, py):
                        return ACTION_LEFT
                if dy > 0 and is_passable(px, py + 1):
                    return ACTION_DOWN
                if dy < 0 and is_passable(px, py - 1):
                    return ACTION_UP
            return ACTION_NOOP

    # Stay near base (table) to reduce wandering and improve survival
    if table_pos and nearby_map_item('table'):
        # craft any missing survival items rather than roam
        if s_sword <= 0 and wood >= 1 and stone >= 1:
            return ACTION_MAKE_STONE_SWORD
        if s_pick <= 0 and wood >= 1 and stone >= 1:
            return ACTION_MAKE_STONE_PICK
        if i_sword <= 0 and wood >= 1 and coal >= 1 and iron >= 1 and nearby_map_item('furnace'):
            return ACTION_MAKE_IRON_SWORD
        # if low drink/food, go fetch (handled above)
        return ACTION_NOOP

    # If base exists but not near, go to it for safety
    if table_pos:
        mv = go_near_map_item('table')
        if mv:
            return mv

    # Final: head to any remaining goals prioritized (drink, food, iron, coal, diamond)
    goals = []
    if drink < PREFERRED_DRINK:
        goals += find_tiles_of_type('water')
    if food < PREFERRED_FOOD:
        goals += [tuple(o['pos']) for o in objects if o['kind'] in ('cow',) or (o['kind'] == 'plant' and o.get('ripe'))]
    if iron < IRON_TARGET:
        goals += find_tiles_of_type('iron')
    if coal < COAL_TARGET:
        goals += find_tiles_of_type('coal')
    if diamond < 1:
        goals += find_tiles_of_type('diamond')
    if not goals:
        return ACTION_NOOP

    best_dir = None
    best_dist = None
    for d, (dx, dy) in DIRS.items():
        nx, ny = px + dx, py + dy
        if not is_passable(nx, ny):
            continue
        md = min((abs(gx - nx) + abs(gy - ny)) for gx, gy in goals)
        if best_dist is None or md < best_dist:
            best_dist = md
            best_dir = d
    if best_dir:
        return dir_to_move_action(best_dir)

    return ACTION_NOOP
