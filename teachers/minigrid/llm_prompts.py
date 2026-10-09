"""
Prompts and pricing for the MiniGrid LLM teacher.

Each prompt is a function `(env_id, state, mission) -> str`
that turns the env wrapper's 11-tuple state and the episode's
natural-language mission into a single string suitable to send
as the LLM's user message. The structured JSON output schema is
defined in `llm.py` next to where it is used.

The PRICING table is shared with the rest of the project's cost
tracking. Numbers are in dollars per 1M tokens, taken from
OpenAI's published pricing as of mid-2026. Update when prices
move.
"""


# OpenAI pricing in dollars per 1M tokens, keyed by model id.
# Input price first, then output price. Models not in this dict
# fall back to estimate_dollars returning 0 with a warning-style
# behaviour built into the teacher.
PRICING = {
    'gpt-4o-mini':    (0.15,  0.60),
    'gpt-4.1-mini':   (0.40,  1.60),
    'gpt-4o':         (2.50, 10.00),
    'gpt-4.1':        (2.00,  8.00),
    'gpt-5-mini':     (0.25,  2.00),
    'gpt-5':          (1.25, 10.00),
    'gpt-5.5':        (5.00, 30.00),
    # o-series dedicated reasoning models (distinct line from
    # gpt-4.1/gpt-5 -- these spend extra hidden "reasoning tokens"
    # before answering, billed as output tokens). Prices below are
    # from this project's last known figures and NOT independently
    # re-verified for this session -- confirm against OpenAI's
    # current pricing page before trusting cost totals computed
    # with these two at any real scale.
    'o3-mini':        (1.10,  4.40),
    'o4-mini':        (1.10,  4.40),
}


def estimate_dollars(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Convert token counts into a dollar estimate for a given
    model. Returns 0.0 if the model is not in PRICING.
    """

    if model not in PRICING:
        return 0.0
    p_in, p_out = PRICING[model]
    return (tokens_in / 1_000_000.0) * p_in + (
        tokens_out / 1_000_000.0
    ) * p_out


# Direction names indexed by MiniGrid's dir convention:
#   0 = east, 1 = south, 2 = west, 3 = north.
_DIR_NAMES = ('east', 'south', 'west', 'north')


def _format_default(env_id: str, state, mission: str) -> str:
    """
    Build the default MiniGrid prompt.

    Includes:
    - The mission string verbatim (BabyAI-style language goal).
    - Coordinate system explanation (x grows east, y grows
      south).
    - The agent's position, facing, and inventory.
    - Object positions (key, door, goal) in absolute grid
      coordinates.
    - A short description of the 5x5 / DoorKey topology: two
      rooms split by a wall with a single door.
    - The action list with the exact meaning of each id.

    The prompt is deliberately verbose because the LLM cannot
    infer the env from the state tuple alone; we tell it
    everything it needs to plan.
    """

    (
        agent_x,
        agent_y,
        agent_dir,
        has_key,
        door_open,
        key_x,
        key_y,
        door_x,
        door_y,
        goal_x,
        goal_y,
    ) = state

    facing = _DIR_NAMES[agent_dir]
    has_key_str = 'YES' if has_key else 'NO'
    door_state_str = 'OPEN' if door_open else 'CLOSED'

    # If the agent is holding the key, the key has no position
    # in the world. Make the prompt say so explicitly rather
    # than reporting a sentinel (0, 0) location, which would
    # confuse the LLM.
    if has_key:
        key_line = 'Key: you are carrying it.'
    else:
        key_line = f'Key: on the floor at ({key_x}, {key_y}).'

    return (
        'You are controlling an agent in a MiniGrid '
        f'environment ({env_id}). Mission: "{mission}"\n'
        '\n'
        'Coordinate system: x grows EAST (right), y grows '
        'SOUTH (down). Position (0, 0) is the top-left corner. '
        'A vertical wall splits the grid into a left and a '
        'right room, with one door in the wall.\n'
        '\n'
        'Current state:\n'
        f'- Agent position: ({agent_x}, {agent_y}), facing '
        f'{facing}.\n'
        f'- Agent holds the key: {has_key_str}.\n'
        f'- {key_line}\n'
        f'- Door: at ({door_x}, {door_y}), currently '
        f'{door_state_str}.\n'
        f'- Goal: at ({goal_x}, {goal_y}).\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place (counter-clockwise; east -> '
        'north -> west -> south).\n'
        '- 1: turn RIGHT in place (clockwise; east -> south -> '
        'west -> north).\n'
        '- 2: move FORWARD one cell in the direction you are '
        'facing. Blocked by walls and by a closed door.\n'
        '- 3: PICK UP the object in the cell directly in front '
        'of you. Use this on the key while empty-handed.\n'
        '- 4: DROP what you are carrying into the cell in '
        'front of you.\n'
        '- 5: TOGGLE the object in front of you. With the key '
        'in hand and the door directly in front, this opens '
        'the door.\n'
        '\n'
        'Plan: collect the key if you do not have it, position '
        'yourself adjacent to and facing the door, toggle the '
        'door open, then walk through to reach the goal.\n'
        '\n'
        'What ONE action should the agent take next?'
    )


def _format_v2(env_id: str, state, mission: str) -> str:
    """
    Build the v2 MiniGrid prompt: an explicit decision procedure.

    The v1 (default) prompt gives all the facts but lets the model
    plan freely, which makes it loop at the goal: it flip-flops
    between heading to the door (already done) and the goal, and
    never just steps forward when aligned. v2 fixes this by handing
    the model a deterministic procedure:

    1. Pick the current TARGET from the phase (key -> door -> goal),
       and explicitly say to IGNORE the key/door once the door is
       open.
    2. If the target is directly in front, act on it (pickup /
       toggle / forward).
    3. Otherwise turn toward the target.

    Same facts as the default prompt; only the instructions change,
    so it is a clean A/B against 'default'.
    """

    (
        agent_x,
        agent_y,
        agent_dir,
        has_key,
        door_open,
        key_x,
        key_y,
        door_x,
        door_y,
        goal_x,
        goal_y,
    ) = state

    facing = _DIR_NAMES[agent_dir]
    has_key_str = 'YES' if has_key else 'NO'
    door_state_str = 'OPEN' if door_open else 'CLOSED'

    if has_key:
        key_line = 'Key: you are carrying it.'
    else:
        key_line = f'Key: on the floor at ({key_x}, {key_y}).'

    return (
        'You are controlling an agent in a MiniGrid '
        f'environment ({env_id}). Mission: "{mission}"\n'
        '\n'
        'Coordinate system: x grows EAST (right), y grows '
        'SOUTH (down). Position (0, 0) is the top-left corner. '
        'A vertical wall splits the grid into a left and a '
        'right room, with one door in the wall.\n'
        '\n'
        'Current state:\n'
        f'- Agent position: ({agent_x}, {agent_y}), facing '
        f'{facing}.\n'
        f'- Agent holds the key: {has_key_str}.\n'
        f'- {key_line}\n'
        f'- Door: at ({door_x}, {door_y}), currently '
        f'{door_state_str}.\n'
        f'- Goal: at ({goal_x}, {goal_y}).\n'
        '\n'
        'Facing directions: 0=east(+x), 1=south(+y), 2=west(-x), '
        '3=north(-y). The cell directly in front of you is one '
        'step in the direction you face.\n'
        '\n'
        'Follow this procedure exactly:\n'
        '\n'
        'STEP 1 - Pick your TARGET (first matching rule):\n'
        '- If you do NOT hold the key: TARGET = the key.\n'
        '- Else if the door is CLOSED: TARGET = the door.\n'
        '- Else (you hold the key AND the door is OPEN): '
        'TARGET = the goal. From now on completely IGNORE the '
        'key and the door; only the goal matters.\n'
        '\n'
        'STEP 2 - Choose the action:\n'
        '- If the TARGET cell is directly in front of you:\n'
        '    * target is the key   -> action 3 (PICK UP)\n'
        '    * target is the door  -> action 5 (TOGGLE)\n'
        '    * target is the goal  -> action 2 (FORWARD)\n'
        '- Else if moving forward steps you onto an empty floor '
        'cell that is closer to the TARGET (same row or column, '
        'no wall/closed door between): action 2 (FORWARD).\n'
        '- Else TURN to face the TARGET: compare the target cell '
        'to your position, decide which direction you need to '
        'face, and turn ONE step toward it with action 0 (LEFT: '
        'east->north->west->south) or action 1 (RIGHT: '
        'east->south->west->north). Pick whichever reaches the '
        'needed facing faster.\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place.\n'
        '- 1: turn RIGHT in place.\n'
        '- 2: move FORWARD one cell you are facing (blocked by '
        'walls and closed doors).\n'
        '- 3: PICK UP the object directly in front.\n'
        '- 4: DROP the carried object in front.\n'
        '- 5: TOGGLE the object in front (opens the door when you '
        'hold the key and face it).\n'
        '\n'
        'What ONE action should the agent take next?'
    )


def _grid_size_from_env_id(env_id: str) -> int:
    """
    Infer the square DoorKey grid size from the env id, defaulting
    to 5 if no known size token is present.
    """

    for token in ('16x16', '8x8', '6x6', '5x5'):
        if token in env_id:
            return int(token.split('x')[0])
    return 5


def _describe_cell(x, y, size, state) -> str:
    """
    Describe what occupies grid cell (x, y) from the state tuple.

    Mirrors the BFS teacher's walkability logic so the LLM is told,
    rather than left to infer, where the walls and objects are: the
    outer ring and the dividing wall column (except the door cell)
    are walls; the door reports open/closed; key and goal report
    their tiles; everything else is floor.
    """

    (
        _ax, _ay, _ad, has_key, door_open,
        key_x, key_y, door_x, door_y, goal_x, goal_y,
    ) = state

    # Anything outside the grid, the border ring, or the dividing
    # wall column (other than the door opening) is a wall.
    if x < 0 or x >= size or y < 0 or y >= size:
        return 'wall (outside grid)'
    if x == 0 or x == size - 1 or y == 0 or y == size - 1:
        return 'wall'
    if x == door_x and y == door_y:
        return 'door (OPEN)' if door_open else 'door (CLOSED)'
    if x == door_x:
        return 'wall'
    if (not has_key) and x == key_x and y == key_y:
        return 'key'
    if x == goal_x and y == goal_y:
        return 'goal'
    return 'floor'


def _format_v3(env_id: str, state, mission: str) -> str:
    """
    Build the v3 MiniGrid prompt: v2 plus wall awareness and detours.

    v2 still loops when the path to the target must go AROUND a wall,
    because its navigation rule only allows stepping toward a target
    in the same row/column, and the model is never told where the
    walls are. v3 fixes both: it lists the contents of the four cells
    around the agent (so walls are explicit) and explicitly permits
    moving away from the target to get around a wall.
    """

    (
        agent_x,
        agent_y,
        agent_dir,
        has_key,
        door_open,
        key_x,
        key_y,
        door_x,
        door_y,
        goal_x,
        goal_y,
    ) = state

    facing = _DIR_NAMES[agent_dir]
    has_key_str = 'YES' if has_key else 'NO'
    door_state_str = 'OPEN' if door_open else 'CLOSED'
    size = _grid_size_from_env_id(env_id)

    if has_key:
        key_line = 'Key: you are carrying it.'
    else:
        key_line = f'Key: on the floor at ({key_x}, {key_y}).'

    # Describe the four cardinal neighbours so the model does not have
    # to infer wall positions. Order: north, east, south, west.
    neighbours = []
    for name, ddx, ddy in (
        ('North', 0, -1),
        ('East', 1, 0),
        ('South', 0, 1),
        ('West', -1, 0),
    ):
        nx, ny = agent_x + ddx, agent_y + ddy
        neighbours.append(
            f'  {name} ({nx}, {ny}): '
            f'{_describe_cell(nx, ny, size, state)}'
        )
    neighbour_block = '\n'.join(neighbours)

    return (
        'You are controlling an agent in a MiniGrid '
        f'environment ({env_id}). Mission: "{mission}"\n'
        '\n'
        'Coordinate system: x grows EAST (right), y grows '
        'SOUTH (down). Position (0, 0) is the top-left corner. '
        'A vertical wall splits the grid into a left and a '
        'right room, with one door in the wall.\n'
        '\n'
        'Current state:\n'
        f'- Agent position: ({agent_x}, {agent_y}), facing '
        f'{facing}.\n'
        f'- Agent holds the key: {has_key_str}.\n'
        f'- {key_line}\n'
        f'- Door: at ({door_x}, {door_y}), currently '
        f'{door_state_str}.\n'
        f'- Goal: at ({goal_x}, {goal_y}).\n'
        '\n'
        'Cells next to you (what is in each adjacent tile):\n'
        f'{neighbour_block}\n'
        '\n'
        'Facing directions: 0=east(+x), 1=south(+y), 2=west(-x), '
        '3=north(-y). The cell directly in front of you is the one '
        'in the direction you face.\n'
        '\n'
        'Follow this procedure exactly:\n'
        '\n'
        'STEP 1 - Pick your TARGET (first matching rule):\n'
        '- If you do NOT hold the key: TARGET = the key.\n'
        '- Else if the door is CLOSED: TARGET = the door.\n'
        '- Else (you hold the key AND the door is OPEN): '
        'TARGET = the goal. Completely IGNORE the key and door '
        'from now on; only the goal matters.\n'
        '\n'
        'STEP 2 - Choose the action to reach the TARGET:\n'
        '- If the TARGET cell is directly in front of you:\n'
        '    * target is the key   -> action 3 (PICK UP)\n'
        '    * target is the door  -> action 5 (TOGGLE)\n'
        '    * target is the goal  -> action 2 (FORWARD)\n'
        '- Otherwise you must WALK toward the target. You can only '
        'move FORWARD, and only into a tile that is floor, the goal, '
        'or an OPEN door (never into a wall or a closed door). To '
        'get next to the target you will often have to go AROUND a '
        'wall: it is correct to move to an open tile that is NOT '
        'directly toward the target if a wall blocks the direct '
        'route. Do not turn back and forth in place.\n'
        '- Decision: look at the adjacent cells listed above. If the '
        'cell in FRONT of you is open (floor / open door / goal) and '
        'moving there makes progress toward the target, take action '
        '2 (FORWARD). Otherwise TURN one step (action 0 LEFT or 1 '
        'RIGHT) toward an open cell that leads around the wall to '
        'the target.\n'
        '\n'
        'Actions:\n'
        '- 0: turn LEFT in place (east->north->west->south).\n'
        '- 1: turn RIGHT in place (east->south->west->north).\n'
        '- 2: move FORWARD one cell you face (blocked by walls and '
        'closed doors).\n'
        '- 3: PICK UP the object directly in front.\n'
        '- 4: DROP the carried object in front.\n'
        '- 5: TOGGLE the object in front (opens the door when you '
        'hold the key and face it).\n'
        '\n'
        'What ONE action should the agent take next?'
    )


# Public prompt registry, keyed by short prompt id. Add
# variants here for ablation studies (no-coordinate-system,
# no-action-list, etc.). 'default' is the original
# all-the-information prompt; 'v2' adds an explicit decision
# procedure to stop the goal/door flip-flop loop; 'v3' adds wall
# awareness (adjacent-cell contents) and explicit detours so the
# model can navigate around walls.
PROMPTS = {
    'default': _format_default,
    'v2': _format_v2,
    'v3': _format_v3,
}
