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
    Staged reactive controller for the chain-of-rooms / doors task.

    - Uses only the 7x7 egocentric obs['image'].
    - Agent is at image[3][6], facing 'up' in the image.
    - Reactive, one-step decisions based on BFS on the visible grid.
    - No long-running plan assumptions (re-plans each call).
    """
    image = obs['image']  # image[x][y] with x = 0..6 left->right, y = 0..6 top->bottom
    last_action = obs.get('last_action', None)

    # helper lambdas to access a cell
    def cell(x, y):
        # returns (obj, color, state) or (0,0,0) if out of range
        if 0 <= x <= 6 and 0 <= y <= 6:
            return image[x][y]
        else:
            return (0, 0, 0)

    # object codes (from spec)
    OBJ_UNSEEN = 0
    OBJ_EMPTY = 1
    OBJ_WALL = 2
    OBJ_FLOOR = 3
    OBJ_DOOR = 4
    OBJ_KEY = 5
    OBJ_BALL = 6
    OBJ_BOX = 7
    OBJ_GOAL = 8
    # pickable objects set
    PICKABLE = {OBJ_KEY, OBJ_BALL, OBJ_BOX}

    # agent position in image coords
    ax, ay = 3, 6

    # passability test for movement (forward movement allowed into these)
    def is_passable_obj(obj, state):
        # floor, empty, goal are passable; an open door (OBJ_DOOR with state==0) is passable
        return (obj in (OBJ_EMPTY, OBJ_FLOOR, OBJ_GOAL)) or (obj == OBJ_DOOR and state == 0)

    # returns list of coords where object == obj_type
    def find_objs(obj_type):
        res = []
        for x in range(7):
            for y in range(7):
                o, c, s = cell(x, y)
                if o == obj_type:
                    res.append((x, y, c, s))
        return res

    # BFS to any of a set of target coordinates (targets is set of (x,y) coords)
    # passable function uses is_passable_obj
    def bfs_to_targets(targets):
        # returns path as list of (x,y) from start to target inclusive, or None
        start = (ax, ay)
        if start in targets:
            return [start]
        q = deque()
        q.append(start)
        prev = {start: None}
        while q:
            x, y = q.popleft()
            for nx, ny in ((x, y-1), (x-1, y), (x+1, y), (x, y+1)):
                if not (0 <= nx <= 6 and 0 <= ny <= 6):
                    continue
                if (nx, ny) in prev:
                    continue
                o, col, st = cell(nx, ny)
                if o == OBJ_UNSEEN:
                    continue
                # allow stepping onto goal or floor/empty or through open door
                if not is_passable_obj(o, st):
                    continue
                prev[(nx, ny)] = (x, y)
                if (nx, ny) in targets:
                    # reconstruct path
                    path = [(nx, ny)]
                    cur = (x, y)
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path.reverse()
                    return path
                q.append((nx, ny))
        return None

    # compute list of candidate adjacent positions to a target object (useful for keys: we want cell adjacent to key)
    def adjacent_positions(x, y):
        res = []
        for nx, ny in ((x, y-1), (x-1, y), (x+1, y), (x, y+1)):
            if 0 <= nx <= 6 and 0 <= ny <= 6:
                # don't include the agent's own cell
                if (nx, ny) != (ax, ay):
                    res.append((nx, ny))
        return res

    # Utility: find any path to be adjacent-facing the key such that the key will be in front when we are in that cell and facing up.
    # Because the agent always perceives itself as facing up in the image, to pick up an object at (kx,ky) the agent must be at
    # cell (kx, ky+1) (i.e., directly below the key) so that the key is in front (image[3][5]).
    # However, the agent can also position left/right and turn to face the key. For simplicity, we will consider ANY adjacent cell
    # as a valid final cell and the reactive step selection will perform turns and pickup when the object appears in front.
    def targets_adjacent_to(x, y):
        # return only adjacent cells that are passable (we plan to be on those cells)
        res = []
        for (nx, ny) in adjacent_positions(x, y):
            o, col, st = cell(nx, ny)
            if o == OBJ_UNSEEN:
                continue
            if is_passable_obj(o, st):
                res.append((nx, ny))
        return res

    # Determine next low-level action to move from agent cell to neighbor cell (one step)
    # next_cell must be adjacent to agent cell.
    def action_to_step(next_cell):
        nx, ny = next_cell
        dx = nx - ax
        dy = ny - ay
        # If next cell is directly ahead (dx=0, dy=-1)
        if dx == 0 and dy == -1:
            front_obj, front_col, front_st = cell(ax, ay-1)
            # if something pickable in front, pick it up
            if front_obj in PICKABLE:
                return PICKUP
            # otherwise try forward
            return FORWARD
        # If next cell is left relative to agent -> turn left (we will move later)
        if dx == -1 and dy == 0:
            return TURN_LEFT
        # right
        if dx == 1 and dy == 0:
            return TURN_RIGHT
        # up-left or up-right: prefer turning towards that side first
        if dx == -1 and dy == -1:
            return TURN_LEFT
        if dx == 1 and dy == -1:
            return TURN_RIGHT
        # down (behind): pick a turn (turn right)
        if dx == 0 and dy == 1:
            return TURN_RIGHT
        # down-left or down-right: choose turn to approach
        if dx == -1 and dy == 1:
            return TURN_LEFT
        if dx == 1 and dy == 1:
            return TURN_RIGHT
        # Fallback
        return TURN_RIGHT

    # Check immediate front
    front_o, front_col, front_st = cell(ax, ay-1)

    # 1) If goal is immediately in front -> go forward
    if front_o == OBJ_GOAL:
        return FORWARD

    # 2) If goal visible anywhere, try to plan to it (we can step into it)
    goals = find_objs(OBJ_GOAL)
    if goals:
        # target is the goal cell itself (we can step onto it)
        targets = set((gx, gy) for (gx, gy, _, _) in goals)
        path = bfs_to_targets(targets)
        if path and len(path) >= 2:
            return action_to_step(path[1])
        # If no path found (blocked by closed doors), fall through to door handling

    # 3) If there is a door immediately in front
    if front_o == OBJ_DOOR:
        # door states: 0 open, 1 closed, 2 locked
        if front_st == 0:
            # open door: just go through
            return FORWARD
        elif front_st == 1:
            # closed: toggle to open
            return TOGGLE
        else:  # locked (2)
            # check if carrying a key of the same color
            carried = cell(3, 6)  # carried object shown in agent cell image[3][6]
            carried_obj, carried_col, _ = carried
            # if carrying the matching key -> toggle
            if carried_obj == OBJ_KEY and carried_col == front_col:
                return TOGGLE
            # otherwise, try to find a key of matching color in view and go pick it
            keys = find_objs(OBJ_KEY)
            matching_keys = [(kx, ky, kc, ks) for (kx, ky, kc, ks) in keys if kc == front_col]
            if matching_keys:
                # pick the nearest reachable adjacent cell to a matching key
                best_path = None
                for (kx, ky, kc, ks) in matching_keys:
                    adj_targets = targets_adjacent_to(kx, ky)
                    if not adj_targets:
                        continue
                    path = bfs_to_targets(set(adj_targets))
                    if path:
                        if best_path is None or len(path) < len(best_path):
                            best_path = (path, (kx, ky))
                if best_path:
                    path, _ = best_path
                    if len(path) >= 2:
                        return action_to_step(path[1])
                    else:
                        # already at target adjacent cell; decide action based on whether key is in front
                        # If the key is in front, pickup; else turn to face it
                        # find which adjacent key is in front
                        for (kx, ky, kc, ks) in matching_keys:
                            if (kx, ky) == (ax, ay-1):
                                return PICKUP
                        # otherwise rotate to explore
                        return TURN_LEFT
            # No matching key visible: try exploring by turning (prefer right)
            return TURN_RIGHT

    # 4) If there is any key visible, attempt to go pick the nearest one
    keys_all = find_objs(OBJ_KEY)
    if keys_all:
        # prefer any key (not necessarily matching a locked door); pick nearest accessible adjacent cell
        best_path = None
        for (kx, ky, kc, ks) in keys_all:
            adj_targets = targets_adjacent_to(kx, ky)
            if not adj_targets:
                continue
            path = bfs_to_targets(set(adj_targets))
            if path:
                if best_path is None or len(path) < len(best_path):
                    best_path = (path, (kx, ky))
        if best_path:
            path, (kx, ky) = best_path
            if len(path) >= 2:
                return action_to_step(path[1])
            else:
                # We are already on an adjacent cell; if the key is in front, pickup
                if (kx, ky) == (ax, ay-1):
                    return PICKUP
                # else rotate toward it: determine where the key is relative to agent
                dx = kx - ax
                dy = ky - ay
                if dx < 0:
                    return TURN_LEFT
                elif dx > 0:
                    return TURN_RIGHT
                else:
                    return TURN_LEFT

    # 5) If there's a visible door somewhere ahead in the middle column (typical chain), try to go to it
    # We'll look for doors in column x=3 above the agent (y < ay)
    candidate_doors = []
    for y in range(0, ay):
        o, col, st = cell(3, y)
        if o == OBJ_DOOR:
            candidate_doors.append((3, y, col, st))
    # if any door visible in that column, target its front position (3, y+1) if possible
    if candidate_doors:
        # prefer nearest door (largest y)
        candidate_doors.sort(key=lambda d: d[1], reverse=True)
        for (dx_d, dy_d, dcol, dst) in candidate_doors:
            target_cell = (dx_d, dy_d + 1)
            tx, ty = target_cell
            # ensure within bounds
            if not (0 <= tx <= 6 and 0 <= ty <= 6):
                continue
            # ensure target cell is passable
            o, c, s = cell(tx, ty)
            if o == OBJ_UNSEEN:
                continue
            if not is_passable_obj(o, s):
                continue
            path = bfs_to_targets({target_cell})
            if path:
                if len(path) >= 2:
                    return action_to_step(path[1])
                else:
                    # already at the front cell; if door is in front, handle it next loop iteration
                    # but for now, attempt to toggle if it is closed or forward if open
                    if front_o == OBJ_DOOR:
                        if front_st == 0:
                            return FORWARD
                        else:
                            return TOGGLE
                    # else rotate to see door
                    return TURN_LEFT

    # 6) If an open door or open corridor is visible in general, do a local BFS to nearest open/passable cell to move forward/explore
    # Try to find any reachable visible passable cell that is closer to the top (explore forward/upwards)
    # Build set of passable cells
    passable_cells = set()
    for x in range(7):
        for y in range(7):
            o, c, s = cell(x, y)
            if o != OBJ_UNSEEN and is_passable_obj(o, s):
                passable_cells.add((x, y))
    # prefer cells with smaller y (more "forward"/farther ahead) to explore forward
    candidates = sorted(passable_cells, key=lambda t: (t[1], abs(t[0]-3)))
    # remove agent's own cell
    candidates = [c for c in candidates if c != (ax, ay)]
    for target in candidates:
        path = bfs_to_targets({target})
        if path and len(path) >= 2:
            return action_to_step(path[1])
    # 7) Default exploration: if forward is passable, go forward, else turn right
    if front_o != OBJ_UNSEEN and is_passable_obj(front_o, front_st):
        # if something pickable directly in front, pick up
        if front_o in PICKABLE:
            return PICKUP
        return FORWARD
    # blocked: rotate to try other view
    return TURN_RIGHT
