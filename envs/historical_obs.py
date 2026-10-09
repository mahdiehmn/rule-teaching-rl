"""
"Historical" (fog-of-war) partial observation wrapper.

Vanilla `RGBImgPartialObsWrapper` (used by obs_mode='partial') is
memoryless: every step it renders a fresh `agent_view_size x
agent_view_size` crop and throws away everything outside it, even
cells the agent stood on one step ago. That makes "how partial" the
observation is an accident of map size relative to the fixed 7x7
crop -- on a 5x5 map the crop covers the whole grid (not partial at
all), while on a 22x22 BabyAI map it covers under 10% of it. Runs on
different-sized tasks are then not actually comparable as
"partial-observability" results.

This wrapper instead reproduces the observation design from
LLM4Teach (Zhou et al., IJCAI 2024, "Large Language Model as a
Policy Teacher for Training Reinforcement Learning Agents",
https://github.com/ZJLAB-AMMI/LLM4Teach, see env/historicalobs.py):
the agent's observation is rendered at the size of the *full* map,
but every cell the agent has never had in its field of view is
rendered as a uniform grey "unseen" tile. A persistent per-episode
mask records every cell that has ever been visible, so once a cell
is seen it stays in the observation (rendered from memory) even
after the agent walks away -- only cells truly never visited stay
grey. Per-step visibility is still bounded to the same
`agent_view_size` window MiniGrid's own partial wrapper uses, so
"how much can the agent see right now" is unchanged; what changes is
that the agent also remembers what it saw earlier, and the image is
always map-sized so coverage reflects real exploration effort rather
than a fixed-crop-vs-map-size accident.

The raycasting math (_looking_vertical / _looking_horizontal /
_process_vis below) is a close port of LLM4Teach's FlexibleGrid
class. It exists as a separate implementation from MiniGrid's own
`Grid.process_vis` because the stock method assumes it is being
called on a small view-sized grid that has already been rotated so
the agent faces "up" (see `MiniGridEnv.gen_obs_grid`), which is fine
for a single memoryless crop but awkward here: this wrapper needs
the visibility mask in un-rotated, world-aligned coordinates so it
can be OR'd directly into the persistent full-map mask without an
extra rotate-then-unrotate step every frame.
"""

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect

# Same grey used by MiniGrid's own "grey" color palette, so unseen
# tiles look visually consistent with anything else grey-colored in
# the env (e.g. grey keys/doors), rather than introducing a new hue.
_UNSEEN_RGB = np.array([100, 100, 100])


class _Unseen(WorldObj):
    """
    A placeholder tile rendered for cells the agent has never seen.

    This only stands in for rendering -- it is never placed into the
    real (world-state) grid the environment simulates, only into a
    throwaway copy built for the observation image, so it cannot
    affect env dynamics (collisions, pickup, etc.).
    """

    def __init__(self):
        super().__init__('unseen', 'grey')

    def render(self, img):
        """
        Fill the whole tile with flat grey.
        """

        fill_coords(img, point_in_rect(0, 1, 0, 1), _UNSEEN_RGB)


def _looking_vertical(grid, mask, i, direction, out_of_bound):
    """
    Propagate visibility along column `i`, north-to-south then back.

    Starting from every cell already marked visible in `mask`,
    extend visibility one row further (and, unless this is the
    outermost column in the sweep, one column further too) as long
    as the current cell does not block sight (see
    `WorldObj.see_behind`). Two passes (south-going, then
    north-going) are needed because visibility can spread in either
    vertical direction from a single visible cell.
    """

    height = grid.height
    for j in range(0, height - 1):
        if not mask[i, j]:
            continue
        cell = grid.get(i, j)
        if cell and not cell.see_behind():
            continue
        mask[i, j + 1] = True
        if not out_of_bound:
            mask[i + direction, j] = True
            mask[i + direction, j + 1] = True

    for j in reversed(range(1, height)):
        if not mask[i, j]:
            continue
        cell = grid.get(i, j)
        if cell and not cell.see_behind():
            continue
        mask[i, j - 1] = True
        if not out_of_bound:
            mask[i + direction, j] = True
            mask[i + direction, j - 1] = True


def _looking_horizontal(grid, mask, j, direction, out_of_bound):
    """
    Propagate visibility along row `j`, west-to-east then back.

    Mirror image of `_looking_vertical`, used when the agent faces
    north or south so the sweep direction is row-by-row instead of
    column-by-column.
    """

    width = grid.width
    for i in range(0, width - 1):
        if not mask[i, j]:
            continue
        cell = grid.get(i, j)
        if cell and not cell.see_behind():
            continue
        mask[i + 1, j] = True
        if not out_of_bound:
            mask[i, j + direction] = True
            mask[i + 1, j + direction] = True

    for i in reversed(range(1, width)):
        if not mask[i, j]:
            continue
        cell = grid.get(i, j)
        if cell and not cell.see_behind():
            continue
        mask[i - 1, j] = True
        if not out_of_bound:
            mask[i, j + direction] = True
            mask[i - 1, j + direction] = True


def _process_vis(grid, agent_pos, agent_dir):
    """
    Return a boolean visibility mask for `grid`, in `grid`'s own
    (un-rotated) coordinates.

    `grid` is expected to be a size-`agent_view_size` window already
    positioned so `agent_pos` is the agent's location within it.
    Visibility spreads outward from the agent's cell, one row or
    column at a time, and stops spreading past any cell that blocks
    sight -- this is the same "can the agent see past this" rule
    MiniGrid's own partial-obs wrapper uses, just computed without
    first rotating the grid to a canonical "facing up" orientation.
    agent_dir follows MiniGrid's convention: 0=east, 1=south,
    2=west, 3=north.
    """

    mask = np.zeros((grid.width, grid.height), dtype=bool)
    mask[agent_pos[0], agent_pos[1]] = True

    if agent_dir == 0:
        for i in range(0, grid.width):
            out_of_bound = i == grid.width - 1
            _looking_vertical(grid, mask, i, 1, out_of_bound)
    elif agent_dir == 2:
        for i in reversed(range(0, grid.width)):
            out_of_bound = i == 0
            _looking_vertical(grid, mask, i, -1, out_of_bound)
    elif agent_dir == 1:
        for j in range(0, grid.height):
            out_of_bound = j == grid.height - 1
            _looking_horizontal(grid, mask, j, 1, out_of_bound)
    elif agent_dir == 3:
        for j in reversed(range(0, grid.height)):
            out_of_bound = j == 0
            _looking_horizontal(grid, mask, j, -1, out_of_bound)
    else:
        raise ValueError(f'invalid agent_dir {agent_dir!r}')

    return mask


class HistoricalObsWrapper(ObservationWrapper):
    """
    Render obs['image'] as a full-map RGB image with fog of war.

    Cells the agent has never had in view are grey; cells it has
    seen at any point this episode keep showing their last-known
    appearance, even after the agent moves away. Per-step visibility
    is computed the same way MiniGrid computes it for the
    memoryless partial wrapper (a raycast bounded by
    `agent_view_size`), so this wrapper changes what the agent
    remembers, not what it can see in a single glance.
    """

    def __init__(self, env, tile_size=8):
        """
        Wrap `env` and size the observation to the full map.

        `env` must ultimately wrap a `MiniGridEnv` (accessed via
        `self.unwrapped`), since this reads `.grid`, `.agent_pos`,
        `.agent_dir`, `.agent_view_size`, and `.see_through_walls`
        directly rather than going through another wrapper's API --
        there is no stock MiniGrid wrapper that does this, so it is
        implemented against the base env's own attributes.
        """

        super().__init__(env)

        self._tile_size = tile_size
        self._width = self.unwrapped.width
        self._height = self.unwrapped.height

        # Boolean, cumulative "have we ever seen this cell" mask.
        # Reset to all-False at the start of every episode in
        # reset() below, then only ever grows (via |=) during the
        # episode -- that growth is the "historical" part.
        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )

        # Persistent RGB image buffer, updated incrementally rather
        # than rebuilt from scratch every step -- see observation()
        # for why that is both correct and much faster on large
        # maps. None means "needs a full rebuild", which is only
        # ever true right after reset().
        self._image = None

        # World-coordinate position of the agent tile that was last
        # drawn WITH the direction-triangle overlay, so observation()
        # can erase that overlay if the agent has since moved to a
        # cell outside the window this step would otherwise redraw.
        self._prev_agent_pos = None

        # The observation image is now the whole map at `tile_size`
        # pixels per cell, not the small agent_view_size crop, so
        # replace just the 'image' entry of the observation space
        # (mission/direction, if present, pass through unchanged).
        new_image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self._height * tile_size,
                self._width * tile_size,
                3,
            ),
            dtype='uint8',
        )
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, 'image': new_image_space}
        )

    def reset(self, **kwargs):
        """
        Clear the seen-cell memory at the start of each episode.

        Fog of war must not leak across episodes -- otherwise later
        episodes on the same map (e.g. repeated seeds) would start
        with parts of the map already "remembered" for free. The
        actual mask/image update for the first observation happens
        inside observation(), which the parent
        ObservationWrapper.reset() calls after the underlying env
        has reset.
        """

        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )
        self._image = None
        self._prev_agent_pos = None
        return super().reset(**kwargs)

    def _draw_tile(self, i, j, cell, agent_dir):
        """
        Render one cell's tile into the persistent image buffer.

        `Grid.render_tile` caches its output keyed by (object,
        agent_dir, highlight, tile_size), so redrawing the same
        kind of tile repeatedly -- the common case, since most
        cells are empty floor -- is a cache lookup and a small
        array copy, not a fresh render.
        """

        tile_img = Grid.render_tile(
            cell, agent_dir=agent_dir, highlight=False,
            tile_size=self._tile_size,
        )
        ts = self._tile_size
        self._image[j * ts:(j + 1) * ts, i * ts:(i + 1) * ts, :] = (
            tile_img
        )

    def observation(self, obs):
        """
        Update the seen-cell mask and the fog-of-war image buffer.

        Only the agent's current view window (at most
        agent_view_size^2 cells) is ever redrawn here, never the
        whole map. That is safe because a cell's true appearance
        can only change as a result of the agent's own actions, and
        the agent can only affect cells at or adjacent to its
        current position -- always inside its current window -- so
        any cell outside today's window cannot have changed since
        the step it was last drawn, and does not need to be
        touched again. This is what keeps this mode's per-step cost
        bounded by the view size rather than the map size, which
        matters a lot on a large map like gotoseq's 22x22 (484
        cells) versus the 49-cell window.
        """

        u = self.unwrapped

        # The same view window MiniGrid's own partial-obs wrapper
        # would use: a fixed agent_view_size x agent_view_size
        # square positioned in front of the agent. It is not
        # clipped to the map here -- get_view_exts() can return
        # coordinates outside [0, width) x [0, height) when the
        # agent is near an edge, and Grid.slice() below pads any
        # out-of-map cells with impassable, opaque Wall
        # placeholders, which is the correct "you can't see past
        # the map edge" behavior.
        agent_view_size = u.agent_view_size
        topX, topY, botX, botY = u.get_view_exts()
        window = u.grid.slice(
            topX, topY, agent_view_size, agent_view_size
        )

        if u.see_through_walls:
            vis = np.ones(
                (agent_view_size, agent_view_size), dtype=bool
            )
        else:
            agent_local = (u.agent_pos[0] - topX, u.agent_pos[1] - topY)
            vis = _process_vis(window, agent_local, u.agent_dir)

        # Only the overlap between the view window and the real map
        # can be drawn or marked seen -- the Wall-padded part of the
        # window (when the agent is near an edge) is not real map
        # area.
        map_topX, map_topY = max(0, topX), max(0, topY)
        map_botX = min(botX, self._width)
        map_botY = min(botY, self._height)

        if self._image is None:
            # First observation of a new episode: initialize the
            # whole buffer to the flat 'unseen' tile. Built once and
            # tiled across the buffer with numpy rather than looping
            # cell by cell -- a full rebuild only ever happens right
            # after reset(), when nothing has been drawn yet.
            #
            # The explicit uint8 cast matters: Grid.render_tile()
            # anti-aliases via downsample(), which averages with
            # numpy's .mean() and so always returns float64, even
            # for uint8 input. MiniGrid's own Grid.render() only
            # ends up uint8 because it assigns each tile into a
            # uint8-typed accumulator buffer, which truncates on the
            # way in; np.tile() on the raw float output has no such
            # buffer to truncate against, so without this cast the
            # persistent buffer would silently drift to float64 and
            # diverge in value (not just dtype) from every other
            # obs_mode in this registry.
            unseen_tile = Grid.render_tile(
                _Unseen(), tile_size=self._tile_size
            )
            self._image = np.tile(
                unseen_tile, (self._height, self._width, 1)
            ).astype(np.uint8)

        agent_pos = tuple(u.agent_pos)
        if (
            self._prev_agent_pos is not None
            and self._prev_agent_pos != agent_pos
        ):
            # The agent moved. Its previous cell was drawn WITH the
            # direction-triangle overlay last step; if that cell
            # does not fall inside today's window, the loop below
            # will never touch it again, leaving a stale "ghost
            # agent" triangle sitting on a cell the agent has left.
            # Redraw it once, without the overlay, from the world
            # grid's current (still accurate, per the docstring
            # above) content.
            pi, pj = self._prev_agent_pos
            self._draw_tile(pi, pj, u.grid.get(pi, pj), None)

        if map_botX > map_topX and map_botY > map_topY:
            for i in range(map_topX, map_botX):
                for j in range(map_topY, map_botY):
                    if not vis[i - topX, j - topY]:
                        continue
                    self._seen_mask[i, j] = True
                    is_agent_cell = (i, j) == agent_pos
                    self._draw_tile(
                        i, j, u.grid.get(i, j),
                        u.agent_dir if is_agent_cell else None,
                    )

        self._prev_agent_pos = agent_pos

        # Return a copy, not a reference to the persistent buffer:
        # this same array is mutated in place on the next step, so
        # anything holding onto this observation (e.g. a rollout
        # buffer) must not see it silently change afterward.
        return {**obs, 'image': self._image.copy()}


class SymbolicHistoricalObsWrapper(ObservationWrapper):
    """
    The fog-of-war observation above, as an integer grid encoding
    rather than a rendered RGB image.

    Why this exists
    ---------------
    `HistoricalObsWrapper` (RGB) and the `symbolic` mode (MiniGrid's
    native integer encoding of the agent's 7x7 crop) differ in TWO
    ways at once: representation (pixels vs category indices) AND
    memory (an accumulated map vs a bare egocentric view). Swapping
    one for the other therefore cannot attribute a result to either.

    This wrapper is the missing cell of that 2x2: full-map extent
    with the same persistent per-episode visibility mask, encoded as
    integers. Comparing it against `historical` holds memory,
    extent and partial-observability fixed and varies ONLY the
    representation, which is the clean version of the "is perception
    the bottleneck?" experiment.

    Semantics match the RGB wrapper exactly, including the part that
    is easy to get wrong: a cell keeps its LAST-KNOWN contents once
    seen, rather than being refreshed to its current true contents.
    A door the agent saw closed and has since walked away from stays
    encoded as closed until the agent looks at it again. Cells never
    seen this episode stay at MiniGrid's 'unseen' encoding (0, 0, 0).
    """

    def __init__(self, env):
        """
        Wrap `env` and size the observation to the full map.

        Like the RGB wrapper, this reads the base MiniGridEnv's own
        attributes (`.grid`, `.agent_pos`, `.agent_dir`,
        `.agent_view_size`, `.see_through_walls`) directly, since no
        stock MiniGrid wrapper provides fog-of-war memory.
        """

        super().__init__(env)

        self._width = self.unwrapped.width
        self._height = self.unwrapped.height

        # Cumulative "have we ever seen this cell" mask, and the
        # persistent encoding it gates. Both are reset per episode.
        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )
        # All-zero is MiniGrid's 'unseen' encoding (object index 0),
        # so an untouched buffer already means "nothing known yet".
        self._encoding = np.zeros(
            (self._width, self._height, 3), dtype='uint8'
        )

        # The observation is the whole map's encoding rather than
        # the agent_view_size crop, so replace just the 'image'
        # entry and let mission/direction pass through unchanged.
        new_image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._width, self._height, 3),
            dtype='uint8',
        )
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, 'image': new_image_space}
        )

    def reset(self, **kwargs):
        """
        Clear the fog-of-war memory at the start of each episode, so
        it can never leak across episodes that share a layout.
        """

        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )
        self._encoding = np.zeros(
            (self._width, self._height, 3), dtype='uint8'
        )
        return super().reset(**kwargs)

    def observation(self, obs):
        """
        Fold this step's visible cells into the persistent encoding
        and return it with the agent's own tile written in.
        """

        u = self.unwrapped

        # The same view window MiniGrid's memoryless partial wrapper
        # uses. get_view_exts() can run off the map near an edge;
        # Grid.slice() pads those cells with opaque Wall
        # placeholders, which correctly blocks sight past the edge.
        agent_view_size = u.agent_view_size
        top_x, top_y, bot_x, bot_y = u.get_view_exts()
        window = u.grid.slice(
            top_x, top_y, agent_view_size, agent_view_size
        )

        if u.see_through_walls:
            vis = np.ones(
                (agent_view_size, agent_view_size), dtype=bool
            )
        else:
            agent_local = (
                u.agent_pos[0] - top_x, u.agent_pos[1] - top_y
            )
            vis = _process_vis(window, agent_local, u.agent_dir)

        # Encode the window once. Cells outside `vis` come back as
        # the 'unseen' triple, which is why only the visible ones
        # are copied through below.
        window_encoding = window.encode(vis)

        # Intersect the (possibly off-map) window with the real map,
        # in world coordinates and again in window-local ones.
        x0, y0 = max(0, top_x), max(0, top_y)
        x1, y1 = min(bot_x, self._width), min(bot_y, self._height)
        if x1 > x0 and y1 > y0:
            lx0, ly0 = x0 - top_x, y0 - top_y
            lx1, ly1 = lx0 + (x1 - x0), ly0 + (y1 - y0)
            sub_vis = vis[lx0:lx1, ly0:ly1]
            sub_encoding = window_encoding[lx0:lx1, ly0:ly1]
            # Slicing gives a view, so this boolean-mask assignment
            # writes straight through into the persistent buffer --
            # updating exactly the currently-visible cells and
            # leaving every other cell's last-known contents alone.
            target = self._encoding[x0:x1, y0:y1]
            target[sub_vis] = sub_encoding[sub_vis]
            self._seen_mask[x0:x1, y0:y1] |= sub_vis

        # The agent is drawn into a COPY rather than into the stored
        # buffer, so its old position never has to be erased: the
        # buffer holds the world, the copy holds world-plus-agent.
        image = self._encoding.copy()
        image[u.agent_pos[0], u.agent_pos[1]] = (
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            u.agent_dir,
        )
        return {**obs, 'image': image}
