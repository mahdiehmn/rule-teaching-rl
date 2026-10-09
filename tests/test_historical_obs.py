"""
Tests for the 'historical' (fog-of-war) observation mode.

Two layers are tested: the raycasting helper in isolation (does it
respect walls and doors the way a real MiniGrid room would), and the
wrapper end to end through the registry (does the persistent mask
actually grow across steps and reset between episodes).
"""

import numpy as np
import pytest

pytest.importorskip('minigrid')

import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Wall

from envs.historical_obs import HistoricalObsWrapper, _Unseen, _process_vis


class _ReferenceHistoricalObsWrapper(gym.ObservationWrapper):
    """
    Deliberately unoptimized fog-of-war renderer, kept only as a
    correctness oracle for HistoricalObsWrapper in the differential
    test below.

    This is the original rebuild-the-whole-map-every-step
    implementation: obviously correct (no incremental state to get
    wrong), but O(map area) per step instead of O(view window).
    HistoricalObsWrapper was rewritten to only redraw the current
    view window for speed on large maps; this class exists so that
    optimization can be checked against a version simple enough to
    trust by inspection, rather than trusting the fast version to
    grade its own homework.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)
        self._tile_size = tile_size
        self._width = self.unwrapped.width
        self._height = self.unwrapped.height
        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )

    def reset(self, **kwargs):
        self._seen_mask = np.zeros(
            (self._width, self._height), dtype=bool
        )
        return super().reset(**kwargs)

    def observation(self, obs):
        u = self.unwrapped
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
            agent_local = (
                u.agent_pos[0] - topX, u.agent_pos[1] - topY
            )
            vis = _process_vis(window, agent_local, u.agent_dir)

        map_topX, map_topY = max(0, topX), max(0, topY)
        map_botX = min(botX, self._width)
        map_botY = min(botY, self._height)
        if map_botX > map_topX and map_botY > map_topY:
            self._seen_mask[
                map_topX:map_botX, map_topY:map_botY
            ] |= vis[
                map_topX - topX:map_botX - topX,
                map_topY - topY:map_botY - topY,
            ]

        render_grid = Grid(self._width, self._height)
        for i in range(self._width):
            for j in range(self._height):
                cell = (
                    u.grid.get(i, j) if self._seen_mask[i, j]
                    else _Unseen()
                )
                render_grid.set(i, j, cell)

        image = render_grid.render(
            self._tile_size, u.agent_pos, u.agent_dir
        )
        return {**obs, 'image': image}


def test_full_width_wall_blocks_vision_beyond_it():
    """
    A wall spanning the whole grid hides everything past it.

    This is the real shape of a MiniGrid room boundary (as opposed
    to a single free-standing blocking cell, which MiniGrid's own
    coarse visibility model can be diagonally 'peeked' around --
    that quirk is inherited from MiniGrid's own stock
    Grid.process_vis and is not something this wrapper should, or
    safely can, special-case away without diverging from how
    obs_mode='partial' already behaves).
    """

    grid = Grid(5, 5)
    for x in range(5):
        for y in range(5):
            grid.set(x, y, None)
    for x in range(5):
        grid.set(x, 2, Wall())

    # Agent at the bottom-center cell, facing north (dir=3), so the
    # wall sits directly ahead of it.
    vis = _process_vis(grid, (2, 4), 3)

    assert vis[2, 3], 'the row just in front of the agent is visible'
    assert not vis[0, 1] and not vis[2, 1] and not vis[4, 1], (
        'nothing beyond a full-width wall should be visible'
    )
    assert not vis[2, 0]


def test_open_door_lets_vision_through():
    """
    A gap in an otherwise solid wall reveals what is past the gap.
    """

    grid = Grid(5, 5)
    for x in range(5):
        for y in range(5):
            grid.set(x, y, None)
    for x in range(5):
        grid.set(x, 2, Wall())
    door = Door('blue')
    door.is_open = True
    grid.set(2, 2, door)

    vis = _process_vis(grid, (2, 4), 3)

    assert vis[2, 1], 'an open door should create a real sightline'


def test_closed_door_does_not_open_the_wall():
    """
    A closed (or locked) door still blocks sight, unlike an open one.
    """

    grid = Grid(5, 5)
    for x in range(5):
        for y in range(5):
            grid.set(x, y, None)
    for x in range(5):
        grid.set(x, 2, Wall())
    grid.set(2, 2, Door('blue', is_locked=True))

    vis = _process_vis(grid, (2, 4), 3)

    assert not vis[2, 1], 'a closed door should not reveal what is past it'


def test_historical_obs_image_is_full_map_sized():
    """
    'historical' mode renders the whole map, not an agent-view crop.
    """

    from envs.registry import build_env

    env = build_env('doorkey_8x8', seed=0, obs_mode='historical')
    try:
        obs, _ = env.reset(seed=0)

        # doorkey_8x8's map is 8x8 cells; the auto-scaled tile size
        # should land the image near the usual 56px baseline, same
        # as the 'full' mode auto-scaling test in test_registry.py.
        assert obs.shape == (56, 56, 3)
    finally:
        env.close()


def test_historical_mask_grows_and_resets():
    """
    The seen-cell mask accumulates across steps within an episode,
    and is cleared again at the start of the next episode.

    This exercises the wrapper through the exact chain
    algos/ppo.py uses (KeepMissionWrapper -> HistoricalObsWrapper ->
    ImgObsWrapper via build_env), reaching into the wrapper instance
    directly to inspect _seen_mask rather than trying to infer
    coverage from pixel colors (MiniGrid's Wall objects render in
    the same flat grey used for unseen cells, so a pixel-based check
    would conflate 'wall' with 'unseen').
    """

    from envs.historical_obs import HistoricalObsWrapper
    from envs.registry import build_env

    env = build_env('doorkey_8x8', seed=0, obs_mode='historical')
    try:
        # Walk the wrapper stack to find the HistoricalObsWrapper
        # instance so its internal mask can be inspected directly.
        wrapper = env.env
        while not isinstance(wrapper, HistoricalObsWrapper):
            wrapper = wrapper.env

        env.reset(seed=0)
        mask_after_reset = wrapper._seen_mask.sum()
        assert mask_after_reset > 0, (
            'the first observation should already mark the '
            "agent's immediate surroundings as seen"
        )

        # Take a few forward/turn actions and confirm the mask can
        # only grow, never shrink, as more of the map is explored.
        running_total = mask_after_reset
        for _ in range(15):
            action = env.action_space.sample()
            env.step(action)
            new_total = wrapper._seen_mask.sum()
            assert new_total >= running_total, (
                'the historical mask must be monotonically '
                'non-decreasing within an episode'
            )
            running_total = new_total

        # A fresh episode must not inherit the previous episode's
        # explored cells.
        env.reset(seed=1)
        assert wrapper._seen_mask.sum() <= mask_after_reset + (
            wrapper._width * wrapper._height
        ), 'sanity bound: mask must not exceed the map size'
        # More precisely: right after reset, the mask should match
        # what a *fresh* first observation looks like, not carry
        # over the fully-explored state from the previous episode.
        assert wrapper._seen_mask.sum() < running_total, (
            'reset() must clear fog-of-war memory between episodes'
        )
    finally:
        env.close()


def test_incremental_render_matches_reference_implementation():
    """
    HistoricalObsWrapper's incremental (view-window-only) renderer
    must produce pixel-identical images to the deliberately
    unoptimized whole-map-every-step reference above, across a
    scripted action sequence that walks forward, turns around and
    re-walks the same ground (exercising the 'agent leaves its old
    tile behind' cleanup path), explores sideways, and resets
    partway through (exercising the persistent-buffer rebuild path).

    A silent mismatch here would mean the agent trains on a subtly
    wrong observation -- nothing else in the test suite or the
    training loop itself would ever catch that, which is why this
    gets a dedicated differential test rather than just a smoke run.
    """

    import minigrid  # noqa: F401

    from envs.wrappers import KeepMissionWrapper

    def _build(obs_wrapper_cls):
        env = gym.make('MiniGrid-DoorKey-8x8-v0', agent_view_size=7)
        env = KeepMissionWrapper(env)
        return obs_wrapper_cls(env, tile_size=8)

    fast_env = _build(HistoricalObsWrapper)
    slow_env = _build(_ReferenceHistoricalObsWrapper)

    # left=0, right=1, forward=2 in MiniGrid's Actions enum.
    actions = (
        [2, 2, 2, 2]        # walk forward
        + [1, 1]            # turn around (two right turns)
        + [2, 2, 2, 2]      # walk back over the same ground
        + [0, 2, 2]         # turn, then explore sideways
    )

    try:
        for seed in (7, 11):
            fast_obs, _ = fast_env.reset(seed=seed)
            slow_obs, _ = slow_env.reset(seed=seed)
            assert np.array_equal(
                fast_obs['image'], slow_obs['image']
            ), f'mismatch on the first observation after seed={seed}'

            for step_index, action in enumerate(actions):
                fast_obs, _, ft, ftr, _ = fast_env.step(action)
                slow_obs, _, st, str_, _ = slow_env.step(action)
                assert np.array_equal(
                    fast_obs['image'], slow_obs['image']
                ), f'image mismatch at seed={seed} step={step_index}'
                assert ft == st and ftr == str_
    finally:
        fast_env.close()
        slow_env.close()


def _known_cells(obs):
    """
    Count cells whose object channel is not MiniGrid's 'unseen' (0).
    """

    return int((obs[:, :, 0] != 0).sum())


def test_symbolic_historical_is_full_map_sized_integers():
    """
    The symbolic fog mode must return the map-sized INTEGER grid,
    not a render: same extent as 'historical', category indices
    instead of pixels.

    This is the mode that makes "is perception the bottleneck?"
    answerable, because it differs from 'historical' in exactly one
    thing. If it silently returned pixels, or the agent's egocentric
    crop, that comparison would quietly become a two-variable one.
    """

    from envs.registry import build_env

    env = build_env(
        'keycorridor_s6r3', seed=0, obs_mode='symbolic_historical'
    )
    try:
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        # Map-sized, three channels of small category indices.
        assert obs.shape == (u.width, u.height, 3)
        assert obs.dtype == np.uint8
        # MiniGrid's largest object index is 10; anything near 255
        # would mean pixels leaked through.
        assert obs.max() <= 10
    finally:
        env.close()


def test_symbolic_historical_accumulates_and_stays_partial():
    """
    Fog must grow as the agent explores but must NOT reveal the
    whole map, which is what separates this mode from the oracle
    'symbolic_full'.
    """

    from envs.registry import build_env

    env = build_env(
        'keycorridor_s6r3', seed=0, obs_mode='symbolic_historical'
    )
    try:
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        start_known = _known_cells(obs)
        total = u.width * u.height

        for _ in range(40):
            obs, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break

        end_known = _known_cells(obs)
        # Memory accumulates...
        assert end_known > start_known
        # ...but the map is not handed over wholesale.
        assert end_known < total
    finally:
        env.close()


def test_symbolic_historical_resets_between_episodes():
    """
    Fog must not leak across episodes, or a repeated layout would
    start with part of the map remembered for free.
    """

    from envs.registry import build_env

    env = build_env(
        'keycorridor_s6r3', seed=0, obs_mode='symbolic_historical'
    )
    try:
        obs, _ = env.reset(seed=0)
        first_known = _known_cells(obs)
        for _ in range(30):
            obs, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
        assert _known_cells(obs) > first_known

        obs, _ = env.reset(seed=0)
        assert _known_cells(obs) == first_known
    finally:
        env.close()


def test_symbolic_historical_reveals_less_than_symbolic_full():
    """
    The fog mode must know strictly less than the oracle full-map
    mode on the same layout at reset -- the direct check that it is
    genuinely partial rather than FullyObsWrapper by another name.
    """

    from envs.registry import build_env

    fog = build_env(
        'keycorridor_s6r3', seed=0, obs_mode='symbolic_historical'
    )
    full = build_env(
        'keycorridor_s6r3', seed=0, obs_mode='symbolic_full'
    )
    try:
        fog_obs, _ = fog.reset(seed=0)
        full_obs, _ = full.reset(seed=0)
        assert fog_obs.shape == full_obs.shape
        assert _known_cells(fog_obs) < _known_cells(full_obs)
    finally:
        fog.close()
        full.close()
