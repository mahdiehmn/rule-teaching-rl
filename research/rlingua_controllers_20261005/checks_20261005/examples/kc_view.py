from collections import deque
import random
import math

# Actions
A_TURN_LEFT = 0
A_TURN_RIGHT = 1
A_FORWARD = 2
A_PICKUP = 3
A_DROP = 4
A_TOGGLE = 5
A_DONE = 6

# Image geometry
W, H = 7, 7
AG_X, AG_Y = 3, 6  # agent always at (3,6) facing up in the provided view

# object codes in image cells: 0 unseen, 1 empty, 2 wall, 3 floor, 4 door, 5 key, 6 ball, 7 box, 8 goal, 9 lava, 10 agent
OBJ_UNSEEN = 0
OBJ_EMPTY = 1
OBJ_WALL = 2
OBJ_FLOOR = 3
OBJ_DOOR = 4
OBJ_KEY = 5
OBJ_BALL = 6
OBJ_BOX = 7
OBJ_GOAL = 8

def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H

def cell_raw(img, x, y):
    # returns tuple (obj, color, state) or None if coordinate out of bounds
    if not in_bounds(x, y):
        return None
    c = img[x][y]
    # Some environments give None for unseen; others use [0, ..]; treat either as unseen
    if c is None:
        return (OBJ_UNSEEN, 0, 0)
    if isinstance(c, (list, tuple)):
        # c[0] may be integer object code
        return tuple(c)
    return (OBJ_UNSEEN, 0, 0)

def is_unseen(img, x, y):
    c = cell_raw(img, x, y)
    if c is None:
        return True
    return c[0] == OBJ_UNSEEN

def is_traversable(img, x, y):
    c = cell_raw(img, x, y)
    if c is None:
        return False
    obj, col, st = c
    if obj in (OBJ_EMPTY, OBJ_FLOOR, OBJ_GOAL):
        return True
    if obj == OBJ_DOOR and st == 0:  # open door
        return True
    return False

def neighbors_4(pos):
    x, y = pos
    for nx, ny in ((x, y-1), (x+1, y), (x, y+1), (x-1, y)):
        if in_bounds(nx, ny):
            yield (nx, ny)

def find_all(img, predicate):
    res = []
    for x in range(W):
        for y in range(H):
            if is_unseen(img, x, y):
                continue
            c = cell_raw(img, x, y)
            if predicate(c):
                res.append((x, y))
    return res

def find_keys(img):
    return find_all(img, lambda c: c[0] == OBJ_KEY)

def find_balls(img):
    return find_all(img, lambda c: c[0] == OBJ_BALL)

def find_doors(img):
    return find_all(img, lambda c: c[0] == OBJ_DOOR)

def carrying_from_image(img):
    ac = cell_raw(img, AG_X, AG_Y)
    if ac is None:
        return None
    obj, col, st = ac
    if obj == OBJ_KEY:
        return ('key', col)
    if obj == OBJ_BALL:
        return ('ball', None)
    return None

def stands_for_target(target):
    tx, ty = target
    return (tx, ty + 1)

def bfs_to_targets(img, targets, allow_unseen_target=False):
    tgt = set(targets)
    start = (AG_X, AG_Y)
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur in tgt:
            # reconstruct path
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = prev[p]
            path.reverse()
            return path
        for nb in neighbors_4(cur):
            if nb in prev:
                continue
            nx, ny = nb
            # if nb is a target allow it if visible or allow_unseen_target True
            if nb in tgt:
                if not is_unseen(img, nx, ny) or allow_unseen_target:
                    prev[nb] = cur
                    q.append(nb)
            elif is_traversable(img, nx, ny):
                prev[nb] = cur
                q.append(nb)
    return None

def find_reachable_stand_for_targets(img, targets):
    best = None
    for t in targets:
        stand = stands_for_target(t)
        if not in_bounds(*stand):
            continue
        if is_unseen(img, stand[0], stand[1]):
            continue
        if not is_traversable(img, stand[0], stand[1]):
            continue
        path = bfs_to_targets(img, [stand])
        if path is not None:
            if best is None or len(path) < len(best[2]):
                best = (t, stand, path)
    return best  # (target, stand, path) or None

def find_any_empty_stand(img):
    empties = []
    for x in range(W):
        for y in range(H):
            if is_unseen(img, x, y):
                continue
            c = cell_raw(img, x, y)
            if c is None:
                continue
            if c[0] in (OBJ_EMPTY, OBJ_FLOOR):
                empties.append((x, y))
    for e in empties:
        s = stands_for_target(e)
        if not in_bounds(*s):
            continue
        if is_unseen(img, s[0], s[1]):
            continue
        if not is_traversable(img, s[0], s[1]):
            continue
        path = bfs_to_targets(img, [s])
        if path is not None:
            return (e, s, path)
    return None

def find_front_cell(img):
    fx, fy = AG_X, AG_Y - 1
    if not in_bounds(fx, fy):
        return None, None
    c = cell_raw(img, fx, fy)
    if c is None:
        return None, None
    return c, (fx, fy)

def find_explore_target(img):
    # return nearest traversable cell adjacent to unseen
    candidates = []
    for x in range(W):
        for y in range(H):
            if is_unseen(img, x, y):
                continue
            if not is_traversable(img, x, y):
                continue
            for nx, ny in neighbors_4((x, y)):
                if is_unseen(img, nx, ny):
                    candidates.append((x, y))
                    break
    best_path = None
    best_t = None
    for t in candidates:
        path = bfs_to_targets(img, [t])
        if path is not None:
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_t = t
    return best_t, best_path

def next_action_towards(img, next_cell):
    nx, ny = next_cell
    dx = nx - AG_X
    dy = ny - AG_Y
    # relative to current orientation (facing up): forward (0,-1), left (-1,0), right (1,0), back (0,1)
    if dx == 0 and dy == -1:
        front = cell_raw(img, AG_X, AG_Y - 1)
        # only forward when front is visible and traversable
        if front is None or front[0] == OBJ_UNSEEN:
            # prefer rotating to reveal instead of blind forward
            return random.choice([A_TURN_RIGHT, A_TURN_LEFT])
        if is_traversable(img, AG_X, AG_Y - 1):
            return A_FORWARD
        # blocked (wall/closed door/object): rotate to find alternate path
        return random.choice([A_TURN_RIGHT, A_TURN_LEFT])
    if dx == -1 and dy == 0:
        return A_TURN_LEFT
    if dx == 1 and dy == 0:
        return A_TURN_RIGHT
    if dx == 0 and dy == 1:
        return A_TURN_LEFT
    return random.choice([A_TURN_LEFT, A_TURN_RIGHT])

# Finite-state phases to reduce dithering
PHASE_EXPLORE = 'explore'
PHASE_GO_TO_KEY = 'get_key'
PHASE_WITH_KEY_GO_UNLOCK = 'with_key_go_unlock'
PHASE_WITH_KEY_DROP = 'with_key_drop_for_ball'
PHASE_GO_TO_BALL = 'get_ball'

def decide_action(obs, memory):
    img = obs['image']
    last_executed = obs.get('last_action', None)

    # initialize memory fields
    if memory is None:
        memory = {}
    memory.setdefault('t', 0)
    memory['t'] += 1
    t = memory['t']
    if 'phase' not in memory:
        memory['phase'] = PHASE_EXPLORE
    # avoid repeatedly toggling the same door
    last_toggled = memory.get('last_toggled', (None, -999))

    carried = carrying_from_image(img)
    keys = find_keys(img)
    balls = find_balls(img)
    doors = find_doors(img)
    front_cell, front_pos = find_front_cell(img)
    if front_cell is not None:
        front_obj, front_col, front_state = front_cell
    else:
        front_obj = front_col = front_state = None

    # If we have the ball already we are done (but the environment ends the episode upon pickup).
    # Replan every step based on observations:
    # Phase transitions:
    # If carrying a key -> either try to unlock door or drop key if ball visible
    if carried is not None and carried[0] == 'key':
        # If ball is visible and reachable we must drop key before pickup => move to drop phase
        if balls:
            memory['phase'] = PHASE_WITH_KEY_DROP
        else:
            memory['phase'] = PHASE_WITH_KEY_GO_UNLOCK

    # If not carrying key and key visible -> go pick key
    if (carried is None or carried[0] != 'key') and keys:
        memory['phase'] = PHASE_GO_TO_KEY

    # If not carrying key and no key visible but ball visible -> go pick ball
    if (carried is None or carried[0] != 'key') and (not keys) and balls:
        memory['phase'] = PHASE_GO_TO_BALL

    phase = memory['phase']

    # ----- Always safe checks -----
    # If front is a locked door and we carry matching key -> unlock
    if front_obj == OBJ_DOOR and front_state == 2 and carried is not None and carried[0] == 'key' and carried[1] == front_col:
        # avoid immediate repeated toggles: only toggle if we haven't toggled this exact door in the last 4 steps
        if last_toggled[0] != front_pos or (t - last_toggled[1]) >= 4:
            memory['last_toggled'] = (front_pos, t)
            return A_TOGGLE

    # If front is closed (unlocked) door -> open it (but cooldown to avoid toggling repeatedly)
    if front_obj == OBJ_DOOR and front_state == 1:
        if last_toggled[0] != front_pos or (t - last_toggled[1]) >= 4:
            memory['last_toggled'] = (front_pos, t)
            return A_TOGGLE

    # If ball directly in front and not carrying key -> pickup
    if front_obj == OBJ_BALL:
        if carried is None or carried[0] != 'key':
            return A_PICKUP
        # if carrying key, we must drop key first -> handled in phases below

    # ----- Phase behaviors -----
    # Phase: get key
    if phase == PHASE_GO_TO_KEY:
        # pick the nearest reachable key stand
        plan = find_reachable_stand_for_targets(img, keys)
        if plan is not None:
            keypos, stand, path = plan
            # if at stand, align and pick
            if (AG_X, AG_Y) == stand:
                # if key in front -> pick
                fx, fy = AG_X, AG_Y - 1
                f = cell_raw(img, fx, fy)
                if f is not None and f[0] == OBJ_KEY:
                    return A_PICKUP
                # else rotate to face the key
                dx = keypos[0] - AG_X
                if dx < 0:
                    return A_TURN_LEFT
                if dx > 0:
                    return A_TURN_RIGHT
                return A_TURN_LEFT
            # else follow path
            if len(path) >= 2:
                return next_action_towards(img, path[1])
        # If can't reach any visible key, explore
        explore_t, explore_path = find_explore_target(img)
        if explore_path is not None and len(explore_path) >= 2:
            return next_action_towards(img, explore_path[1])
        # else rotate to scan
        return A_TURN_RIGHT

    # Phase: with key, go unlock
    if phase == PHASE_WITH_KEY_GO_UNLOCK:
        mycol = carried[1]
        # find visible locked doors matching key
        matching_locked = [d for d in doors if cell_raw(img, d[0], d[1])[2] == 2 and cell_raw(img, d[0], d[1])[1] == mycol]
        if matching_locked:
            plan = find_reachable_stand_for_targets(img, matching_locked)
            if plan is not None:
                target, stand, path = plan
                if (AG_X, AG_Y) == stand:
                    # if front is the locked door it should have been toggled earlier; ensure alignment
                    fx, fy = AG_X, AG_Y - 1
                    f = cell_raw(img, fx, fy)
                    if f is not None and f[0] == OBJ_DOOR and f[2] == 2 and f[1] == mycol:
                        # cooldown checked earlier
                        if last_toggled[0] != (fx, fy) or (t - last_toggled[1]) >= 4:
                            memory['last_toggled'] = ((fx, fy), t)
                            return A_TOGGLE
                        # if recently toggled but still locked (maybe other policy intervened), rotate to re-evaluate
                        return A_TURN_RIGHT
                    # else rotate to face door
                    dx = target[0] - AG_X
                    if dx < 0:
                        return A_TURN_LEFT
                    if dx > 0:
                        return A_TURN_RIGHT
                    return A_TURN_LEFT
                # follow path
                if len(path) >= 2:
                    return next_action_towards(img, path[1])
        # No matching locked door visible: if any ball visible, must drop key before going to ball
        if balls:
            memory['phase'] = PHASE_WITH_KEY_DROP
            # fall through to drop behavior
        else:
            # explore to find doors
            explore_t, explore_path = find_explore_target(img)
            if explore_path is not None and len(explore_path) >= 2:
                return next_action_towards(img, explore_path[1])
            # otherwise rotate to scan
            return A_TURN_RIGHT

    # Phase: with key, drop for ball
    if phase == PHASE_WITH_KEY_DROP:
        # If front is empty and safe, drop now (prefer immediate drop to avoid carrying key into ball pickup)
        if front_obj is not None and front_obj in (OBJ_EMPTY, OBJ_FLOOR):
            return A_DROP
        # else try to find any empty stand to drop on (nearest)
        empty_choice = find_any_empty_stand(img)
        if empty_choice is not None:
            epos, stand, path = empty_choice
            if (AG_X, AG_Y) == stand:
                fx, fy = AG_X, AG_Y - 1
                f = cell_raw(img, fx, fy)
                if f is not None and f[0] in (OBJ_EMPTY, OBJ_FLOOR):
                    return A_DROP
                # rotate to align front with chosen empty
                dx = epos[0] - AG_X
                if dx < 0:
                    return A_TURN_LEFT
                if dx > 0:
                    return A_TURN_RIGHT
                return A_TURN_LEFT
            if len(path) >= 2:
                return next_action_towards(img, path[1])
        # if no empty found in view, explore to reveal empties or doors
        explore_t, explore_path = find_explore_target(img)
        if explore_path is not None and len(explore_path) >= 2:
            return next_action_towards(img, explore_path[1])
        # fallback: rotate to scan
        return A_TURN_RIGHT

    # Phase: go to ball (when not carrying key)
    if phase == PHASE_GO_TO_BALL:
        plan = find_reachable_stand_for_targets(img, balls)
        if plan is not None:
            ballpos, stand, path = plan
            if (AG_X, AG_Y) == stand:
                fx, fy = AG_X, AG_Y - 1
                f = cell_raw(img, fx, fy)
                if f is not None and f[0] == OBJ_BALL:
                    # ensure not carrying key (should be satisfied)
                    return A_PICKUP
                dx = ballpos[0] - AG_X
                if dx < 0:
                    return A_TURN_LEFT
                if dx > 0:
                    return A_TURN_RIGHT
                return A_TURN_LEFT
            if len(path) >= 2:
                return next_action_towards(img, path[1])
        # can't reach visible ball: explore
        explore_t, explore_path = find_explore_target(img)
        if explore_path is not None and len(explore_path) >= 2:
            return next_action_towards(img, explore_path[1])
        return A_TURN_RIGHT

    # Phase: explore baseline
    if phase == PHASE_EXPLORE:
        # if front is visible and traversable -> move forward to reveal new area
        if front_obj is not None and front_obj != OBJ_UNSEEN and is_traversable(img, AG_X, AG_Y - 1):
            return A_FORWARD
        # else go to nearest cell adjacent to unseen
        explore_t, explore_path = find_explore_target(img)
        if explore_path is not None:
            if (AG_X, AG_Y) == explore_t:
                # if front is safe and visible try forward
                if front_obj is not None and front_obj != OBJ_UNSEEN and is_traversable(img, AG_X, AG_Y - 1):
                    return A_FORWARD
                return A_TURN_RIGHT
            if len(explore_path) >= 2:
                return next_action_towards(img, explore_path[1])
        # as fallback rotate to scan
        return A_TURN_RIGHT

    # Catch-all fallback: rotate to gather information
    return A_TURN_RIGHT

def controller(obs, memory) -> int:
    # memory is a dict persisted across calls; initialize if missing
    if memory is None:
        memory = {}
    try:
        act = decide_action(obs, memory)
        # store last chosen for debugging
        memory['last_choice'] = act
        return int(act)
    except Exception:
        # safe fallback
        return A_TURN_RIGHT
