"""
Symbolic state extraction for teachers.

The agent trains on pixels, but a planning teacher (the BFS oracle)
needs the underlying discrete MiniGrid state to reason about. This
module reads that state straight out of a live MiniGrid environment's
internals, so the teacher can be queried alongside the image-based
agent without the agent ever seeing the symbolic form.

The extracted tuple matches the 11-dimensional layout the BFS oracle
expects:

    (agent_x, agent_y, agent_dir, has_key, door_open,
     key_x, key_y, door_x, door_y, goal_x, goal_y)

Objects that are not present right now (e.g. a key being carried, so
it is no longer on the grid) report position (0, 0); has_key /
door_open carry the status bits that disambiguate those cases.
"""

from minigrid.core.world_object import Door, Goal, Key, Wall


def extract_doorkey_state(unwrapped):
    """
    Build the 11-tuple symbolic state from a MiniGrid env.

    Parameters
    ----------
    unwrapped: minigrid.MiniGridEnv
        The fully unwrapped MiniGrid environment (i.e.
        some_wrapped_env.unwrapped), exposing agent_pos, agent_dir,
        carrying, grid, width, and height.

    Returns
    -------
    tuple of 11 int
        (agent_x, agent_y, agent_dir, has_key, door_open,
         key_x, key_y, door_x, door_y, goal_x, goal_y)
    """

    u = unwrapped

    # Agent pose as plain ints so the tuple is hashable and free of
    # lingering numpy scalar types.
    agent_x = int(u.agent_pos[0])
    agent_y = int(u.agent_pos[1])
    agent_dir = int(u.agent_dir)

    # has_key: the carrying slot holds a single object or None.
    has_key = 1 if isinstance(u.carrying, Key) else 0

    # Scan the grid once for the key, door, and goal, plus the door's
    # open/closed status. (0, 0) is the sentinel for "not on the grid
    # right now" -- only meaningful for the key once it is picked up.
    key_x, key_y = 0, 0
    door_x, door_y = 0, 0
    door_open = 0
    goal_x, goal_y = 0, 0

    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            if isinstance(cell, Key):
                key_x, key_y = x, y
            elif isinstance(cell, Door):
                door_x, door_y = x, y
                if cell.is_open:
                    door_open = 1
            elif isinstance(cell, Goal):
                goal_x, goal_y = x, y

    return (
        agent_x, agent_y, agent_dir,
        has_key, door_open,
        key_x, key_y,
        door_x, door_y,
        goal_x, goal_y,
    )


def extract_generic_state(unwrapped):
    """
    Build a topology-agnostic symbolic state snapshot.

    Unlike extract_doorkey_state above (a fixed 11-tuple tailored to
    DoorKey's exactly-one-key/one-door/one-goal layout), this scans
    the grid for every Key, Door, Ball, Box, and Goal present -- any
    number of them, in any arrangement -- so it also works on tasks
    whose room structure isn't known in advance, e.g. KeyCorridor's
    R-row grid of rooms with a key hidden in one of them.

    This is used only as a cache key and as the source of status
    text a teacher supplies alongside an image (e.g. what the agent
    is carrying, which a rendered frame cannot show); it is
    deliberately NOT meant to drive a hand-written navigation
    procedure the way the DoorKey prompts do; unlike DoorKey's fixed
    two-room split, a general room graph's shortest path can't be
    described as a short, reliable, general-purpose text recipe, so
    that reasoning is left to a vision-capable teacher looking at
    the actual rendered map instead of being spelled out here.

    Returns
    -------
    tuple
        (agent_x, agent_y, agent_dir, carrying_str, objects), where
        objects is a sorted tuple of (kind, color, x, y, door_state)
        for every non-wall, non-empty cell. door_state is 'open',
        'closed', or 'locked' for doors, '' otherwise. Sorting makes
        the tuple a stable, hashable cache key independent of scan
        order.
    """

    u = unwrapped

    agent_x = int(u.agent_pos[0])
    agent_y = int(u.agent_pos[1])
    agent_dir = int(u.agent_dir)

    carrying = ''
    if u.carrying is not None:
        carrying = f'{u.carrying.color} {u.carrying.type}'

    objects = []
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is None or isinstance(cell, Wall):
                continue
            if isinstance(cell, Door):
                if cell.is_open:
                    door_state = 'open'
                elif cell.is_locked:
                    door_state = 'locked'
                else:
                    door_state = 'closed'
            else:
                door_state = ''
            objects.append((cell.type, cell.color, x, y, door_state))
    objects.sort()

    return (agent_x, agent_y, agent_dir, carrying, tuple(objects))
