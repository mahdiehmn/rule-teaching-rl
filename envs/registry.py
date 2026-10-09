"""
Task registry for the benchmark.

Maps short, stable friendly ids (used on the command line and in
config files) to the underlying Gymnasium/MiniGrid environment ids,
and builds each environment with a consistent image observation
pipeline so every algorithm sees the same observation.

Adding a task is a one-line edit to TASKS. The friendly id is what
appears in run names, configs, and result files, so keep it short
and filesystem-safe.
"""

import gymnasium as gym

# Importing minigrid registers all MiniGrid and BabyAI envs with
# Gymnasium. The import looks unused but is load-bearing --
# gym.make('BabyAI-...') / gym.make('MiniGrid-...') fails without it.
import minigrid  # noqa: F401
from minigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)

from envs.historical_obs import (
    HistoricalObsWrapper,
    SymbolicHistoricalObsWrapper,
)
from envs.mission_vocab import MAX_MISSION_LEN, MissionVocab
from envs.wrappers import KeepMissionWrapper, MissionTokenWrapper

# A named development restriction; existing benchmark registrations stay intact.
if 'VLM-GoToSeqS5R2Sequence-v0' not in gym.registry:
    gym.register(
        'VLM-GoToSeqS5R2Sequence-v0',
        entry_point='minigrid.envs.babyai.goto:GoToSeq',
        kwargs=dict(room_size=5, num_rows=2, num_cols=2,
                    num_dists=4, instr_kinds=('seq',)),
    )

# Friendly id -> Gymnasium id. See README for why each task is here.
#
# Tasks are grouped by what makes them hard. Axis A tasks are hard at
# *exploration* but solvable from the image alone, so they test
# whether a teacher beats plain curiosity (RND). Axis B tasks are
# language-conditioned: the per-episode mission decides the goal, so a
# vision-only agent cannot fully solve them -- they test whether a
# VLM/LLM teacher unlocks tasks curiosity cannot. Add a task by adding
# one line here; nothing else needs to change.
TASKS = {
    # --- Smoke tests (trivial; validate the loop) ---
    # Reach a goal in an empty 3x3 interior. Solvable in a few steps.
    # NOTE: at agent_view_size=7, a 5x5 map is entirely inside a
    # single partial-obs crop -- the agent sees the whole grid from
    # step one. Treat results here as "does the loop work", never as
    # evidence about partial-observability behavior.
    'empty5x5': 'MiniGrid-Empty-5x5-v0',

    # --- DoorKey (find key, unlock door, reach goal) ---
    # Single key / single door / single goal, procedurally placed.
    # These are the tasks the BFS oracle teacher can solve, used to
    # develop and validate teacher-guided training. 5x5 is quick;
    # 8x8 is slow enough for plain PPO that teacher help is visible.
    # Same caveat as empty5x5: doorkey_5x5's whole map fits inside
    # one 7x7 crop, so it is not a genuine partial-observability
    # data point -- use doorkey_8x8 (or obs_mode='historical', which
    # stays meaningfully partial regardless of map size) for that.
    'doorkey_5x5': 'MiniGrid-DoorKey-5x5-v0',
    'doorkey_8x8': 'MiniGrid-DoorKey-8x8-v0',
    # Four times the area of 8x8 and a 2560-step limit: the transfer test
    # of the 8x8 rule bank, unchanged (fix-wave protocol addendum 4).
    'doorkey_16x16': 'MiniGrid-DoorKey-16x16-v0',

    # --- Axis A: hard exploration, no language ---
    # Long-horizon navigation through six connected rooms.
    'multiroom_n6': 'MiniGrid-MultiRoom-N6-v0',
    # More rooms on the same 25x25 grid (MiniGrid registers only up to
    # N6; these ids are registered below): the larger-environment test
    # of the N6 rule banks, unchanged (fix-wave protocol addendum 9).
    'multiroom_n8': 'MiniGrid-MultiRoom-N8-v0',
    'multiroom_n10': 'MiniGrid-MultiRoom-N10-v0',
    # Find a hidden key behind doors, then reach the goal. A standard
    # hard-exploration benchmark where vanilla PPO is near zero.
    'keycorridor_s6r3': 'MiniGrid-KeyCorridorS6R3-v0',
    # The SAME KeyCorridor layout/dynamics, but via the BabyAI
    # registration, which adds the `instrs` machinery the free
    # BabyAIBot teacher plans with (verified: the bot solves S6R3
    # 10/10, mean 105 steps vs the 1080 limit). Use this id for
    # teacher-guided KeyCorridor runs so the teacher is free and
    # cluster-safe; the MiniGrid-registered id above is kept
    # unchanged for comparability with the Stage-0 runs already in
    # flight (and for the paid vlm_general VLM-teacher study). The
    # mission ('pick up the ball', exactly one ball in the level) is
    # non-disambiguating, so this stays an Axis-A task: the student
    # is the pixels-only agent, never the mission-conditioned one.
    'keycorridor_s6r3_babyai': 'BabyAI-KeyCorridorS6R3-v0',
    # Small, fast member of the same KeyCorridor family (7x7 map,
    # 270-step limit vs S6R3's 16x16/1080): same key-hidden-in-a-
    # room, locked-door, ball-behind-it structure, ~4x shorter
    # episodes. For rapid iteration before spending budget on S6R3.
    # BabyAI-registered for the same free-bot-teacher reason.
    'keycorridor_s3r3': 'BabyAI-KeyCorridorS3R3-v0',
    # One size up (4x4 rooms, 480-step limit): the transfer test of the
    # S3R3 memory bank, unchanged (fix-wave protocol addendum 5).
    'keycorridor_s4r3': 'BabyAI-KeyCorridorS4R3-v0',
    # Two more sizes for the larger-map tests of the S3R3 memory bank
    # (fix-wave protocol addendum 11); S6R3 is keycorridor_s6r3_babyai.
    'keycorridor_s5r3': 'BabyAI-KeyCorridorS5R3-v0',
    # The hardest exploration task in the suite: nested locked,
    # blocked rooms. Vanilla deep RL gets essentially no reward.
    'obstructedmaze_full': 'MiniGrid-ObstructedMaze-Full-v1',

    # --- Axis B: language-conditioned (BabyAI) ---
    # Simplest varying-mission task: 'go to the <color> <object>' in
    # one room with distractors. Small, good for developing the
    # teacher against a changing instruction.
    'gotolocal': 'BabyAI-GoToLocal-v0',
    # Compositional, sequential instructions ('go to X then Y'). The
    # headline language task.
    'gotoseq': 'BabyAI-GoToSeq-v0',
    # Small, fast member of the same GoToSeq family (9x9 map,
    # 100-step limit vs the full task's 22x22/2304): the same
    # compositional 'go to X then Y' missions, ~23x shorter
    # episodes, so evaluation is cheap and complete episodes (the
    # learning signal) come ~20x more often per frame. For rapid
    # iteration before spending budget on the full gotoseq.
    'gotoseq_s5r2': 'BabyAI-GoToSeqS5R2-v0',
    'gotoseq_s5r2_sequence': 'VLM-GoToSeqS5R2Sequence-v0',
    # Relational instruction ('put the X next to the Y') -- requires
    # carrying and placing, not just navigating.
    'putnextlocal': 'BabyAI-PutNextLocal-v0',
    # The hardest BabyAI level: long, nested, compositional missions.
    'bosslevel': 'BabyAI-BossLevel-v0',
}

# MiniGrid registers MultiRoom up to N6; the larger variants are the same
# MultiRoomEnv with more rooms (step limit 20 per room, as MiniGrid sets).
for _rooms in (8, 10):
    _id = f'MiniGrid-MultiRoom-N{_rooms}-v0'
    if _id not in gym.registry:
        gym.register(id=_id, entry_point='minigrid.envs:MultiRoomEnv',
                     kwargs={'minNumRooms': _rooms, 'maxNumRooms': _rooms})


# The tasks whose mission VARIES between episodes, and therefore the
# tasks that cannot be represented without reading the instruction.
#
# This is not the same as "registered under BabyAI-": keycorridor_s3r3
# and keycorridor_s6r3_babyai use BabyAI ids purely so the free
# offline bot teacher is available, but their mission is the constant
# 'pick up the ball', which carries no information. Only the levels
# below change what is being asked from one episode to the next, so
# only these need obs_mission=True -- and on these, an agent without
# the instruction is not merely handicapped, it is being asked an
# ambiguous question.
BABYAI_LANGUAGE_TASKS = (
    'gotolocal',
    'gotoseq',
    'gotoseq_s5r2',
    'gotoseq_s5r2_sequence',
    'putnextlocal',
    'bosslevel',
)


def _auto_tile_size(num_cells, target_size):
    """
    Pick a MiniGrid render tile size near a target image dimension.

    MiniGrid RGB wrappers render one grid cell as tile_size pixels.
    Choosing round(target / cells) keeps observations close to the
    old 7x7 * 8 = 56 pixel baseline while allowing larger views or
    full maps. The tile size cannot go below one pixel.
    """

    return max(1, int(round(target_size / max(1, num_cells))))


def _normalize_obs_mode(obs_mode):
    """
    Normalize observation-mode aliases used on the command line.
    """

    normalized = obs_mode.lower().replace('-', '_')
    aliases = {
        'partial': 'partial',
        'partial_rgb': 'partial',
        'egocentric': 'partial',
        'agent': 'partial',
        'full': 'full',
        'fully_obs': 'full',
        'full_obs': 'full',
        'full_rgb': 'full',
        # Full-map RGB render with a persistent per-episode
        # visibility mask (LLM4Teach-style fog of war): unlike
        # 'full', unvisited cells are greyed out rather than
        # revealed outright, so this stays a partial-observability
        # mode. See envs/historical_obs.py for why this exists.
        'historical': 'historical',
        'history': 'historical',
        'fog_of_war': 'historical',
        'memory': 'historical',
        # The native MiniGrid grid encoding rather than a render:
        # a (view, view, 3) integer array of (object, color, state)
        # per visible cell. This is the representation the MiniGrid
        # exploration literature (RIDE, AMIGo, NovelD, BeBold, E3B)
        # actually trains on, and BabyAI 1.1 measured it as up to 3x
        # more sample-efficient than pixels. Every reported result in
        # this repo so far uses 'historical' RGB, so comparing the
        # two isolates how much of a failure is perception rather
        # than exploration or memory.
        'symbolic': 'symbolic',
        'symbolic_partial': 'symbolic',
        'grid': 'symbolic',
        # The same integer encoding over the WHOLE map (an oracle,
        # non-partial view). Only useful as an upper bound that
        # removes partial observability entirely.
        'symbolic_full': 'symbolic_full',
        'grid_full': 'symbolic_full',
        # The integer encoding of the full map WITH the same
        # per-episode fog of war 'historical' uses. This is the
        # symbolic twin of 'historical': same extent, same memory,
        # same partial observability, differing ONLY in whether the
        # agent reads pixels or category indices. It is the mode to
        # use when asking "is perception the bottleneck?", because
        # 'symbolic' alone also removes the accumulated map and so
        # changes two things at once.
        'symbolic_historical': 'symbolic_historical',
        'grid_historical': 'symbolic_historical',
        'symbolic_fog': 'symbolic_historical',
    }
    if normalized not in aliases:
        valid = ', '.join(sorted(aliases))
        raise ValueError(
            f'unknown obs_mode {obs_mode!r}; valid aliases: {valid}'
        )
    return aliases[normalized]


def build_env(
    task_id,
    seed=0,
    render_mode=None,
    obs_mode='partial',
    agent_view_size=7,
    obs_tile_size=0,
    obs_target_size=56,
    obs_mission=False,
    mission_vocab=None,
    mission_max_len=None,
):
    """
    Build a single wrapped environment for the given friendly task id.

    The environment is wrapped so that the agent observation is a
    plain RGB image, while the per-episode mission string is preserved
    in the info dict for the teacher.

    Parameters
    ----------
    task_id: str
        A key of TASKS (e.g. 'gotoseq'). Raises KeyError with the
        list of valid ids if unknown.
    seed: int
        Seed applied to the env and its action space on first reset.
    render_mode: str or None
        Passed through to gym.make. Use 'human' for live viewing,
        'rgb_array' to capture frames, None for headless training.
    obs_mode: str
        'partial' for the memoryless egocentric agent view, 'full'
        for the (oracle, non-partial) full-map RGB render, or
        'historical' for a full-map render with fog of war -- see
        envs/historical_obs.py for why 'historical' is the mode
        that keeps "partial" meaningful across differently-sized
        tasks. 'symbolic' / 'symbolic_full' return MiniGrid's
        native integer grid encoding instead of pixels, which is
        the representation the MiniGrid exploration literature
        trains on; note the array is small-valued integers, not
        0-255 pixels, so a network reading it must not divide by
        255 (see algos/ppo_intrinsic.py's SymbolicEncoder).
    agent_view_size: int
        Odd MiniGrid view size for partial observations. Bounds
        per-step visibility in both 'partial' and 'historical'
        modes; ignored by 'full', which reveals everything.
    obs_tile_size: int
        Pixel size of each grid cell in the RGB observation. Use 0 to
        auto-scale near obs_target_size pixels on the larger side.
    obs_target_size: int
        Target image size used when obs_tile_size is 0.
    obs_mission: bool
        False (default) keeps every existing algorithm's behavior
        byte-identical: the observation is a bare image array
        (ImgObsWrapper). True is for BabyAI (language) tasks, whose
        mission is often the only thing that disambiguates the
        goal -- the observation becomes a Dict of {image,
        mission_ids, mission_len} via MissionTokenWrapper instead.
        See docs/ for why pixels alone cannot represent these
        tasks.
    mission_vocab: envs.mission_vocab.MissionVocab or None
        Only used when obs_mission=True. Pass an already-loaded
        vocabulary so multiple envs in a vectorized run share one
        instance instead of each re-reading the vocab file; None
        loads envs.mission_vocab.MissionVocab.load()'s default
        frozen file.
    mission_max_len: int or None
        Only used when obs_mission=True. None uses
        envs.mission_vocab.MAX_MISSION_LEN.

    Returns
    -------
    gymnasium.Env
        The fully wrapped environment.
    """

    # Fail loudly with the valid options if the id is unknown, rather
    # than letting gym.make raise an opaque registration error.
    if task_id not in TASKS:
        valid = ', '.join(sorted(TASKS))
        raise KeyError(
            f'unknown task id {task_id!r}; valid ids: {valid}'
        )

    obs_mode = _normalize_obs_mode(obs_mode)

    env = gym.make(
        TASKS[task_id],
        render_mode=render_mode,
        agent_view_size=agent_view_size,
    )

    # Capture the mission into info before the image-only wrapper
    # strips the rest of the observation away.
    env = KeepMissionWrapper(env)

    if obs_tile_size < 0:
        raise ValueError('obs_tile_size must be >= 0')
    if obs_target_size <= 0:
        raise ValueError('obs_target_size must be > 0')

    # Render the egocentric crop, the full map, or the full map
    # with fog of war, then reduce the observation to just that
    # image. 'full' and 'historical' both render at the map's full
    # size, so they share the same cells-to-pixels auto-scaling;
    # 'partial' scales off the (typically much smaller) view size
    # instead.
    if obs_mode == 'symbolic':
        # MiniGrid's own observation is ALREADY the symbolic grid
        # encoding, so the partial symbolic mode is simply the
        # absence of any RGB wrapper: obs['image'] stays the
        # (agent_view_size, agent_view_size, 3) integer array. No
        # tile size applies because nothing is rendered.
        pass
    elif obs_mode == 'symbolic_full':
        # FullyObsWrapper swaps obs['image'] for the same integer
        # encoding over the whole map instead of the agent's crop.
        env = FullyObsWrapper(env)
    elif obs_mode == 'symbolic_historical':
        # The integer twin of HistoricalObsWrapper: full-map extent
        # with a persistent per-episode visibility mask, so unseen
        # cells stay at MiniGrid's 'unseen' encoding.
        env = SymbolicHistoricalObsWrapper(env)
    elif obs_mode == 'partial':
        tile_size = obs_tile_size or _auto_tile_size(
            agent_view_size, obs_target_size
        )
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
    else:
        max_cells = max(env.unwrapped.width, env.unwrapped.height)
        tile_size = obs_tile_size or _auto_tile_size(
            max_cells, obs_target_size
        )
        if obs_mode == 'historical':
            env = HistoricalObsWrapper(env, tile_size=tile_size)
        else:
            env = RGBImgObsWrapper(env, tile_size=tile_size)

    # ImgObsWrapper (the default) reduces the observation to a bare
    # image array, matching every existing algorithm's Agent, which
    # takes only pixels. MissionTokenWrapper is the BabyAI-task
    # alternative: it keeps the image but replaces the raw mission
    # string with a fixed-length token array the network can
    # actually condition on, instead of dropping it.
    if obs_mission:
        if mission_vocab is None:
            mission_vocab = MissionVocab.load()
        env = MissionTokenWrapper(
            env,
            vocab=mission_vocab,
            max_len=mission_max_len or MAX_MISSION_LEN,
        )
    else:
        env = ImgObsWrapper(env)

    # Seed the env and its action space once so a run is reproducible.
    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def make_thunk(
    task_id,
    seed,
    idx,
    render_mode=None,
    obs_mode='partial',
    agent_view_size=7,
    obs_tile_size=0,
    obs_target_size=56,
    obs_mission=False,
    mission_vocab=None,
    mission_max_len=None,
):
    """
    Return a zero-argument callable that builds one environment.

    cleanrl-style vectorized training expects a list of such thunks
    to hand to gymnasium.vector.SyncVectorEnv. Each parallel env gets
    a distinct seed derived from the base seed and its index so the
    copies are decorrelated.

    Parameters
    ----------
    task_id: str
        Friendly task id, a key of TASKS.
    seed: int
        Base seed for the run.
    idx: int
        Index of this env within the vector; offsets the seed.
    render_mode: str or None
        Only env 0 typically renders; pass None for the rest.
    obs_mode, agent_view_size, obs_tile_size, obs_target_size
        Forwarded to build_env so all vectorized workers use the same
        observation pipeline.
    obs_mission, mission_vocab, mission_max_len
        Forwarded to build_env. Callers building many thunks for one
        vectorized run (BabyAI training/eval) should load one
        MissionVocab and pass the same instance to every thunk,
        rather than letting each env reload the vocab file.

    Returns
    -------
    callable
        A function of no arguments returning a wrapped env.
    """

    def thunk():
        return build_env(
            task_id,
            seed=seed + idx,
            render_mode=render_mode,
            obs_mode=obs_mode,
            agent_view_size=agent_view_size,
            obs_tile_size=obs_tile_size,
            obs_target_size=obs_target_size,
            obs_mission=obs_mission,
            mission_vocab=mission_vocab,
            mission_max_len=mission_max_len,
        )

    return thunk
