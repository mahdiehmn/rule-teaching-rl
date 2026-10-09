"""Observation-only history labels."""

import random

import gymnasium as gym
import numpy as np
import pytest
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

from algos.history_targets import (
    COLORS,
    IGNORE_INDEX,
    KNOWLEDGE_CLASSES,
    KNOWLEDGE_FIELDS,
    MEMORY_CLASSES,
    MEMORY_FIELDS,
    HistoryTargets,
)


def _view(*objects):
    """
    Encode actual MiniGrid objects as a small symbolic observation.
    """

    grid = Grid(7, 7)
    for index, obj in enumerate(objects):
        grid.set(index, 2, obj)
    return grid.encode()


def _update(tracker, image, start=False, inactive=False):
    """
    Apply one observation to a single stream.
    """

    return tracker.update(image[None], [start], [inactive])


def _door(color='red', state=2):
    """
    Build a native door with a specified encoded state.
    """

    return Door(color, is_open=state == 0, is_locked=state == 2)


class _HistoryRoom(MiniGridEnv):
    """
    Native observation/physics fixture with visible and hidden objects.
    """

    def __init__(self, state=2, facing=0, hidden='yellow', max_steps=40):
        self.test_state = state
        self.test_facing = facing
        self.test_hidden = hidden
        super().__init__(
            mission_space=MissionSpace(
                mission_func=lambda: 'remember what you have observed'
            ),
            width=15,
            height=15,
            max_steps=max_steps,
            agent_view_size=7,
            see_through_walls=False,
        )

    def _gen_grid(self, width, height):
        """
        Put the red door ahead and the blue ball behind the initial view.
        """

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.agent_pos = (7, 7)
        self.agent_dir = self.test_facing
        self.test_door = _door(state=self.test_state)
        self.put_obj(self.test_door, 8, 7)
        self.put_obj(Key('green'), 8, 6)
        self.put_obj(Ball('blue'), 6, 8)
        self.put_obj(_door(self.test_hidden), 2, 2)
        self.put_obj(Goal(), 13, 13)
        self.mission = 'remember what you have observed'


def _turn_away(env, tracker):
    """
    Turn through two real steps and collect their native observations.
    """

    for _ in range(2):
        observation, _, terminated, truncated, _ = env.step(
            env.actions.right
        )
        assert not (terminated or truncated)
        result = _update(tracker, observation['image'])
    return observation['image'], result


def test_vocabulary_and_coverage_are_explicit():
    assert COLORS == tuple(sorted(COLOR_TO_IDX, key=COLOR_TO_IDX.get))
    assert MEMORY_CLASSES == ('open', 'closed', 'locked')
    assert KNOWLEDGE_CLASSES == (
        'never_observed', 'previously_observed', 'visible'
    )
    assert len(MEMORY_FIELDS) == 6
    assert len(KNOWLEDGE_FIELDS) == 18
    assert KNOWLEDGE_FIELDS[:6] == tuple(f'key_{c}' for c in COLORS)
    assert KNOWLEDGE_FIELDS[6:12] == tuple(f'door_{c}' for c in COLORS)
    assert KNOWLEDGE_FIELDS[12:] == tuple(f'ball_{c}' for c in COLORS)
    result = _update(HistoryTargets(1), _view(Goal()), start=True)
    assert result['memory'].dtype == np.int64
    assert result['knowledge'].dtype == np.int64
    assert result['memory'].shape == (1, 6)
    assert result['knowledge'].shape == (1, 18)
    assert (result['memory'] == IGNORE_INDEX).all()
    assert (result['knowledge'] == 0).all()
    assert result['memory_count'].tolist() == [0]
    assert result['knowledge_counts'].tolist() == [[18, 0, 0]]
    assert result['knowledge_class_mask'].dtype == bool
    np.testing.assert_array_equal(
        result['knowledge_class_mask'].sum(axis=2),
        result['knowledge_mask'],
    )


def test_real_same_current_view_different_past_door_states():
    rooms = [_HistoryRoom(state=2), _HistoryRoom(state=0)]
    trackers = [HistoryTargets(1), HistoryTargets(1)]
    try:
        end = []
        for room, tracker in zip(rooms, trackers):
            observation, _ = room.reset(seed=4)
            visible = _update(tracker, observation['image'], start=True)
            assert visible['knowledge'][0, 6] == 2
            assert visible['memory'][0, 0] == IGNORE_INDEX
            end.append(_turn_away(room, tracker))
        np.testing.assert_array_equal(end[0][0], end[1][0])
        assert end[0][1]['memory'][0, 0] == 2
        assert end[1][1]['memory'][0, 0] == 0
        assert end[0][1]['knowledge'][0, 6] == 1
    finally:
        for room in rooms:
            room.close()


def test_real_same_current_view_distinguishes_unobserved_past():
    seen_env = _HistoryRoom(facing=0)
    unseen_env = _HistoryRoom(facing=2)
    try:
        seen_tracker = HistoryTargets(1)
        observation, _ = seen_env.reset(seed=8)
        _update(seen_tracker, observation['image'], start=True)
        current, seen = _turn_away(seen_env, seen_tracker)
        observation, _ = unseen_env.reset(seed=8)
        unseen = _update(
            HistoryTargets(1), observation['image'], start=True
        )
        np.testing.assert_array_equal(current, observation['image'])
        assert seen['knowledge'][0, 6] == 1
        assert unseen['knowledge'][0, 6] == 0
        assert seen['memory'][0, 0] == 2
        assert unseen['memory'][0, 0] == IGNORE_INDEX
    finally:
        seen_env.close()
        unseen_env.close()


def test_hidden_map_and_mission_do_not_supply_missing_information():
    rooms = [_HistoryRoom(hidden='yellow'), _HistoryRoom(hidden='purple')]
    trackers = [HistoryTargets(1), HistoryTargets(1)]
    try:
        observations = []
        for index, (room, tracker) in enumerate(zip(rooms, trackers)):
            observation, _ = room.reset(seed=3)
            room.mission = f'unavailable mission {index}'
            observation = room.gen_obs()
            observations.append(observation)
            _update(tracker, observation['image'], start=True)
        assert observations[0]['mission'] != observations[1]['mission']
        np.testing.assert_array_equal(
            observations[0]['image'], observations[1]['image']
        )
        assert rooms[0].grid.encode().tolist() != (
            rooms[1].grid.encode().tolist()
        )
        end = [
            _turn_away(room, tracker)
            for room, tracker in zip(rooms, trackers)
        ]
        for key in end[0][1]:
            np.testing.assert_array_equal(end[0][1][key], end[1][1][key])
        for color in ('yellow', 'purple'):
            field = KNOWLEDGE_FIELDS.index(f'door_{color}')
            assert end[0][1]['knowledge'][0, field] == 0
    finally:
        for room in rooms:
            room.close()


def test_memory_is_last_witnessed_state_not_hidden_current_truth():
    room = _HistoryRoom(state=2)
    try:
        tracker = HistoryTargets(1)
        observation, _ = room.reset(seed=11)
        _update(tracker, observation['image'], start=True)
        previous, _ = _turn_away(room, tracker)
        room.test_door.is_locked = False
        room.test_door.is_open = True
        changed = room.gen_obs()['image']
        np.testing.assert_array_equal(previous, changed)
        remembered = _update(tracker, changed)
        assert room.test_door.encode()[2] == 0
        assert remembered['memory'][0, 0] == 2
    finally:
        room.close()


def test_disappearance_and_carried_object_pixels_count_as_history():
    tracker = HistoryTargets(1)
    image = _view(Key('yellow'), _door('red'), Ball('blue'))
    # MiniGrid writes a carried object to the ego cell of this view.
    image[3, 6] = Key('purple').encode()
    visible = _update(tracker, image, start=True)
    fields = [
        KNOWLEDGE_FIELDS.index(name)
        for name in ('key_yellow', 'door_red', 'ball_blue', 'key_purple')
    ]
    assert (visible['knowledge'][0, fields] == 2).all()
    past = _update(tracker, _view())
    assert (past['knowledge'][0, fields] == 1).all()
    assert past['knowledge_counts'].tolist() == [[14, 4, 0]]
    assert past['memory_count'].tolist() == [1]
    assert past['memory'][0, 0] == 2
    np.testing.assert_array_equal(
        past['knowledge_class_mask'],
        past['knowledge'][..., None] == np.arange(3),
    )


def test_conflicting_latest_witness_erases_an_older_unambiguous_state():
    tracker = HistoryTargets(1)
    _update(tracker, _view(_door(state=2)), start=True)
    assert _update(tracker, _view())['memory'][0, 0] == 2
    conflict = _update(tracker, _view(_door(state=0), _door(state=1)))
    assert conflict['memory_visible_mask'][0, 0]
    assert conflict['memory_ambiguous_mask'][0, 0]
    assert conflict['knowledge'][0, 6] == 2
    hidden = _update(tracker, _view())
    assert hidden['memory'][0, 0] == IGNORE_INDEX
    assert not hidden['memory_mask'][0, 0]
    assert hidden['memory_ambiguous_mask'][0, 0]
    assert not hidden['memory_unseen_mask'][0, 0]
    assert hidden['knowledge'][0, 6] == 1

    # Equal-state duplicates do not require an object identity claim.
    _update(tracker, _view(_door(state=1), _door(state=1)))
    restored = _update(tracker, _view())
    assert restored['memory'][0, 0] == 1
    assert not restored['memory_ambiguous_mask'][0, 0]


def test_latest_witness_updates_each_color_independently():
    tracker = HistoryTargets(1)
    _update(
        tracker, _view(_door('red', 2), _door('blue', 0)), start=True
    )
    changed = _update(tracker, _view(_door('red', 1)))
    assert changed['memory'][0, 0] == IGNORE_INDEX
    assert changed['memory'][0, 2] == 0
    past = _update(tracker, _view())
    assert past['memory'][0, [0, 2]].tolist() == [1, 0]


def test_episode_reset_precedes_first_observation_and_is_slot_local():
    tracker = HistoryTargets(2)
    initial = np.stack([_view(_door('red')), _view(_door('blue', 0))])
    tracker.update(initial, [True, True])
    restarted = np.stack([_view(Key('green')), _view()])
    result = tracker.update(restarted, [True, False])
    assert (result['memory'][0] == IGNORE_INDEX).all()
    assert result['knowledge'][0, 6] == 0
    assert result['knowledge'][0, 1] == 2
    assert result['memory'][1, 2] == 0
    assert result['knowledge'][1, 8] == 1


def test_inactive_rows_emit_no_labels_and_do_not_consume_images():
    tracker = HistoryTargets(2)
    images = np.stack([_view(_door()), _view(Key('blue'))])
    tracker.update(images, [True, True])
    image = np.stack([_view(_door(state=0)), _view(Ball('purple'))])
    result = tracker.update(image, [False, False], [True, False])
    assert (result['memory'][0] == IGNORE_INDEX).all()
    assert (result['knowledge'][0] == IGNORE_INDEX).all()
    for key, value in result.items():
        if key.endswith('mask') or key.endswith('count'):
            assert not value[0].any()
    assert result['knowledge_counts'][0].tolist() == [0, 0, 0]
    active = tracker.update(np.stack([_view(), _view()]), [False, False])
    assert active['memory'][0, 0] == 2
    assert active['knowledge'][1, 2] == 1
    assert active['knowledge'][1, 15] == 1


def test_inactive_start_clears_history_even_with_ignored_invalid_pixels():
    tracker = HistoryTargets(1)
    _update(tracker, _view(_door()), start=True)
    ignored = np.full((1, 7, 7, 3), np.nan)
    tracker.update(ignored, [True], [True])
    fresh = _update(tracker, _view())
    assert (fresh['memory'] == IGNORE_INDEX).all()
    assert (fresh['knowledge'] == 0).all()


def test_next_step_autoreset_terminal_frame_is_not_a_training_target():
    envs = gym.vector.SyncVectorEnv([lambda: _HistoryRoom(max_steps=1)])
    try:
        assert envs.autoreset_mode == gym.vector.AutoresetMode.NEXT_STEP
        tracker = HistoryTargets(1)
        observations, _ = envs.reset(seed=7)
        first = tracker.update(observations['image'], [True])
        blue_ball = KNOWLEDGE_FIELDS.index('ball_blue')
        assert first['knowledge'][0, blue_ball] == 0

        # This actual terminal view reveals the previously unseen ball.
        observations, _, terminated, truncated, _ = envs.step([1])
        inactive = terminated | truncated
        assert inactive.tolist() == [True]
        assert np.any(
            (observations['image'][0, ..., 0] == OBJECT_TO_IDX['ball'])
            & (observations['image'][0, ..., 1] == COLOR_TO_IDX['blue'])
        )
        terminal = tracker.update(observations['image'], [False], inactive)
        assert (terminal['knowledge'] == IGNORE_INDEX).all()
        assert not terminal['memory_mask'].any()

        # The ignored action resets the environment. Reset history only
        # now, before consuming the genuine first view of the episode.
        observations, _, terminated, truncated, _ = envs.step([2])
        assert not (terminated | truncated).any()
        fresh = tracker.update(observations['image'], inactive)
        assert fresh['knowledge'][0, blue_ball] == 0
        assert not fresh['memory_mask'].any()
    finally:
        envs.close()


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5, '2'])
def test_invalid_stream_count_fails(invalid):
    with pytest.raises(ValueError, match='positive integer'):
        HistoryTargets(invalid)


@pytest.mark.parametrize(
    'bad',
    [
        np.zeros((7, 7, 3), dtype=np.uint8),
        np.zeros((1, 8, 8, 3), dtype=np.uint8),
        np.zeros((2, 7, 7, 3), dtype=np.uint8),
        np.zeros((1, 7, 7, 3), dtype=bool),
        np.zeros((1, 7, 7, 3), dtype=complex),
    ],
)
def test_invalid_image_shape_or_dtype_fails(bad):
    with pytest.raises(ValueError):
        HistoryTargets(1).update(bad, [True])


@pytest.mark.parametrize(
    'pixel',
    [(-1, 0, 0), (11, 0, 0), (4, 6, 0), (4, 0, 3), (10, 0, 4),
     (1, 0, 1), (4, 0, 0.5), (4, np.nan, 0), (4, 0, np.inf)],
)
def test_invalid_symbolic_codes_fail_without_changing_history(pixel):
    tracker = HistoryTargets(1)
    _update(tracker, _view(_door()), start=True)
    bad = _view().astype(np.float32)
    bad[0, 0] = pixel
    with pytest.raises(ValueError):
        _update(tracker, bad, start=True)
    assert _update(tracker, _view())['memory'][0, 0] == 2


@pytest.mark.parametrize('name', ['episode_start', 'inactive'])
@pytest.mark.parametrize('bad', [True, [2], [np.nan], ['false'], [0, 1]])
def test_invalid_flags_fail(name, bad):
    flags = dict(episode_start=[True], inactive=[False])
    flags[name] = bad
    with pytest.raises(ValueError):
        HistoryTargets(1).update(_view()[None], **flags)


def test_integral_floats_match_integer_inputs_without_mutation():
    image = _view(_door('blue', 1), Key('red'))
    original = image.copy()
    results = []
    for dtype in (np.uint8, np.int64, np.float32, np.float64):
        tracker = HistoryTargets(1)
        tracker.update(image[None].astype(dtype), np.ones(1))
        results.append(_update(tracker, _view().astype(dtype)))
    for other in results[1:]:
        for key in other:
            np.testing.assert_array_equal(results[0][key], other[key])
    np.testing.assert_array_equal(image, original)


def test_outputs_do_not_alias_state_and_tracker_consumes_no_rng():
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    tracker = HistoryTargets(1)
    visible = _update(tracker, _view(_door()), start=True)
    past = _update(tracker, _view())
    saved = {key: value.copy() for key, value in past.items()}
    for value in visible.values():
        value[:] = 0
    again = _update(tracker, _view())
    for key in saved:
        np.testing.assert_array_equal(saved[key], again[key])
    for value in past.values():
        value[:] = 0
    assert _update(tracker, _view())['memory'][0, 0] == 2
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()
