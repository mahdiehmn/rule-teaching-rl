from collections import deque

# ACTIONS
TURN_LEFT = 0
TURN_RIGHT = 1
FORWARD = 2
PICKUP = 3
DROP = 4
TOGGLE = 5
DONE = 6

def controller(obs, memory) -> int:
    """
    Controller for the key-door-goal task.
    Reactive per-step decision making. Memory is used only minimally.
    """
    image = obs['image']  # image[x][y] with x=0..6 left->right, y=0..6 far->near
    last_action = obs.get('last_action', None)
    # Agent position in view:
    ax, ay = 3, 6
    # helper accessors
    def cell(x, y):
        return image[x][y]  # [object, color, state]
    def in_bounds(x, y):
        return 0 <= x <= 6 and 0 <= y <= 6

    # decode what's carried: image[3][6][0] is object code of carried item (1 = nothing)
    carried_obj = cell(ax, ay)[0]
    carried_color = cell(ax, ay)[1]

    # find visible objects
    keys = []      # list of (x,y,color)
    doors = []     # list of (x,y,state,color)
    goals = []     # list of (x,y)
    for x in range(7):
        for y in range(7):
            c = image[x][y]
            if c[0] == 5:  # key
                keys.append((x, y, c[1]))
            elif c[0] == 4:  # door
                doors.append((x, y, c[2], c[1]))  # (x,y,state,color)
            elif c[0] == 8:  # goal
                goals.append((x, y))

    # passability check for walking (what forward can move onto)
    # Passable: empty(1), floor(3), goal(8), open door (4 state 0)
    def is_passable_cell(c):
        if c[0] == 0:
            return False
        if c[0] in (1, 3, 8):
            return True
        if c[0] == 4 and c[2] == 0:  # door open
            return True
        return False

    # blocking objects (will block forward)
    def is_blocking_cell(c):
        if c[0] == 0:
            return True
        # Walls, closed or locked door, key, ball, box block forward
        if c[0] == 2:  # wall
            return True
        if c[0] == 4 and c[2] != 0:  # closed or locked door
            return True
        if c[0] in (5, 6, 7):  # key, ball, box
            return True
        return False

    # BFS to nearest target node from agent pos on the passable graph
    def bfs_to_targets(target_nodes):
        # target_nodes: set of (x,y)
        start = (ax, ay)
        if start in target_nodes:
            return [start]
        q = deque()
        q.append(start)
        prev = {start: None}
        while q:
            node = q.popleft()
            if node in target_nodes:
                # reconstruct path
                path = []
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path  # list of (x,y)
            x, y = node
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) in prev:
                    continue
                # can step onto (nx,ny) if passable
                if is_passable_cell(cell(nx, ny)) or (nx, ny) in target_nodes:
                    # allow stepping onto target even if target currently blocked (we might be targeting an adjacent cell)
                    prev[(nx, ny)] = node
                    q.append((nx, ny))
        return None

    # Build candidate goal nodes for different tasks:
    #  - For reaching a key: we want to stand on a passable cell adjacent to the key (so we can face it and pickup)
    def key_adjacent_nodes(kx, ky):
        nodes = set()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = kx + dx, ky + dy
            if not in_bounds(nx, ny):
                continue
            if is_passable_cell(cell(nx, ny)):
                nodes.add((nx, ny))
        return nodes

    # For door: want to be adjacent (to face and toggle) OR stand such that door is in front to forward through when open.
    def door_adjacent_nodes(dx_door, dy_door):
        nodes = set()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = dx_door + dx, dy_door + dy
            if not in_bounds(nx, ny):
                continue
            if is_passable_cell(cell(nx, ny)):
                nodes.add((nx, ny))
        return nodes

    # Determine immediate action to turn to face an object at (ox,oy) relative to agent at (ax,ay)
    # Return an action intended to start rotating or to indicate already facing (None)
    def action_to_face_object(ox, oy):
        rx = ox - ax
        ry = oy - ay
        # object in front
        if rx == 0 and ry == -1:
            return None
        # left
        if rx == -1 and ry == 0:
            return TURN_LEFT
        # right
        if rx == 1 and ry == 0:
            return TURN_RIGHT
        # behind -> choose a single turn to begin (we pick left)
        if rx == 0 and ry == 1:
            return TURN_LEFT
        # diagonal shouldn't happen for adjacency; default to turning left
        return TURN_LEFT

    # Simple exploration fallback when goal/key/door not visible or unreachable:
    def exploration_action():
        front = cell(ax, ay - 1) if in_bounds(ax, ay - 1) else [2, 0, 0]
        if not is_blocking_cell(front):
            return FORWARD
        else:
            return TURN_RIGHT

    # Helper: if a target is directly in front, maybe act (pickup/toggle/forward)
    front = cell(ax, ay - 1) if in_bounds(ax, ay - 1) else [2, 0, 0]

    # --- Main logic phases ---
    # If not carrying a key: seek and pick up the key.
    if carried_obj != 5:
        # If there is a visible key:
        if keys:
            # choose nearest key by Manhattan (simple heuristic)
            keys_sorted = sorted(keys, key=lambda k: abs(k[0]-ax) + abs(k[1]-ay))
            kx, ky, kcol = keys_sorted[0]
            # if key is directly in front -> pickup
            if (kx, ky) == (ax, ay - 1):
                return PICKUP
            # if adjacent but not in front -> turn to face
            if abs(kx - ax) + abs(ky - ay) == 1:
                a = action_to_face_object(kx, ky)
                if a is None:
                    # already facing (should have been front) but not in front due to weirdness: fallback to pickup
                    return PICKUP
                return a
            # else plan path to a passable cell adjacent to the key
            targets = key_adjacent_nodes(kx, ky)
            if targets:
                path = bfs_to_targets(targets)
                if path and len(path) >= 2:
                    # decide immediate move towards path[1]
                    cur = path[0]
                    nxt = path[1]
                    dx = nxt[0] - cur[0]
                    dy = nxt[1] - cur[1]
                    # movement relative to current facing (forward is dy == -1)
                    if dx == 0 and dy == -1:
                        # forward but check block (should be passable by BFS)
                        if not is_blocking_cell(cell(ax, ay - 1)):
                            return FORWARD
                        else:
                            # if a closed/locked door directly in front, and we don't have key, we cannot pass. Fallback to explore
                            return exploration_action()
                    if dx == -1 and dy == 0:
                        return TURN_LEFT
                    if dx == 1 and dy == 0:
                        return TURN_RIGHT
                    if dx == 0 and dy == 1:
                        # need two turns; start with turn_left
                        return TURN_LEFT
                else:
                    # unreachable adjacent node (no path) -> exploration
                    return exploration_action()
            else:
                # no adjacent passable node for the key (key might be behind blocking object) -> exploration
                return exploration_action()
        else:
            # no key visible -> explore to find it
            return exploration_action()

    # --- Carried a key (carried_obj == 5) ---
    # First, if goal visible and reachable in current room (rare), go to goal.
    if goals:
        # Try to path to the goal cell directly (goal is passable)
        goal_nodes = set((gx, gy) for gx, gy in goals)
        path = bfs_to_targets(goal_nodes)
        if path:
            # if goal directly in front, forward into it (ends episode)
            if (ax, ay - 1) in goal_nodes:
                return FORWARD
            # if adjacent but not in front, face and then forward next time
            if len(path) == 1:
                # already on goal cell (should have ended), but just return DONE as fallback
                return DONE
            # else follow path toward goal
            cur = path[0]
            nxt = path[1]
            dx = nxt[0] - cur[0]
            dy = nxt[1] - cur[1]
            if dx == 0 and dy == -1:
                if not is_blocking_cell(cell(ax, ay - 1)):
                    return FORWARD
                else:
                    return exploration_action()
            if dx == -1 and dy == 0:
                return TURN_LEFT
            if dx == 1 and dy == 0:
                return TURN_RIGHT
            if dx == 0 and dy == 1:
                return TURN_LEFT

    # If we have the key, we should go to the door, open/unlock it and go through.
    if doors:
        # assume single door; pick the nearest one
        doors_sorted = sorted(doors, key=lambda d: abs(d[0]-ax) + abs(d[1]-ay))
        dx_door, dy_door, door_state, door_color = doors_sorted[0]
        # if door is directly in front
        if (dx_door, dy_door) == (ax, ay - 1):
            # if door is open -> step forward through
            if door_state == 0:
                return FORWARD
            # if closed or locked -> toggle (if locked need key of same color)
            if door_state in (1, 2):
                # if locked and key matches color, toggle will unlock/open
                if door_state == 2:
                    if carried_color == door_color:
                        return TOGGLE
                    else:
                        # carrying wrong key (unlikely) -> explore/drop (fallback)
                        # drop key to maybe get correct one; but we don't know where correct key is: just explore
                        return exploration_action()
                else:
                    # closed but not locked -> toggle to open
                    return TOGGLE
        # not directly in front: if we are adjacent to the door, turn to face it
        if abs(dx_door - ax) + abs(dy_door - ay) == 1:
            # adjacent: face it
            a = action_to_face_object(dx_door, dy_door)
            if a is None:
                # already facing but not caught above -> try toggle
                return TOGGLE
            return a
        # else compute path to a passable cell adjacent to the door
        targets = door_adjacent_nodes(dx_door, dy_door)
        if targets:
            path = bfs_to_targets(targets)
            if path and len(path) >= 2:
                cur = path[0]
                nxt = path[1]
                dx = nxt[0] - cur[0]
                dy = nxt[1] - cur[1]
                if dx == 0 and dy == -1:
                    if not is_blocking_cell(cell(ax, ay - 1)):
                        return FORWARD
                    else:
                        # if blocked by the door directly in front (but we are not adjacent), fallback
                        return exploration_action()
                if dx == -1 and dy == 0:
                    return TURN_LEFT
                if dx == 1 and dy == 0:
                    return TURN_RIGHT
                if dx == 0 and dy == 1:
                    return TURN_LEFT
            else:
                return exploration_action()
        else:
            return exploration_action()
    else:
        # door not visible: try to find it by exploring (walk until you see it)
        return exploration_action()
