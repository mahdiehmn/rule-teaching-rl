"""
Observation-history supervision from the student's symbolic inputs.

Targets use only the student's native 7x7x3 symbolic images and explicit
episode boundaries. They describe observation history, not the current
hidden world, unique objects, positions, mission requirements, or goals.
This is a fixed reference vocabulary, not an LLM-selected explanation.
"""

import numpy as np
from minigrid.core.constants import (
    COLOR_TO_IDX,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)


IGNORE_INDEX = -100
COLORS = tuple(sorted(COLOR_TO_IDX, key=COLOR_TO_IDX.get))
MEMORY_CLASSES = ('open', 'closed', 'locked')
KNOWLEDGE_CLASSES = (
    'never_observed',
    'previously_observed',
    'visible',
)
KNOWLEDGE_OBJECTS = ('key', 'door', 'ball')
MEMORY_FIELDS = tuple(f'door_{color}_last_state' for color in COLORS)
KNOWLEDGE_FIELDS = tuple(
    f'{kind}_{color}' for kind in KNOWLEDGE_OBJECTS for color in COLORS
)

# A changed native vocabulary must not silently change the head shape.
if len(COLORS) != 6:
    raise RuntimeError('History targets require six MiniGrid colors')

_DOOR_CODES = tuple(STATE_TO_IDX[name] for name in MEMORY_CLASSES)
_COLOR_CODES = tuple(COLOR_TO_IDX[name] for name in COLORS)
_DOOR_FIELDS = slice(len(COLORS), 2 * len(COLORS))


def _flags(values, name, num_envs):
    """
    Validate binary vector flags without silently coercing other values.
    """

    flags = np.asarray(values)
    if flags.shape != (num_envs,) or flags.dtype.kind not in 'buif':
        raise ValueError(f'{name} must be a numeric binary [N] array')
    if not np.isin(flags, (0, 1)).all():
        raise ValueError(f'{name} must contain only zero or one')
    return flags.astype(bool, copy=True)


def _images(values, active, num_envs):
    """
    Check native symbolic input; inactive row contents are not consumed.
    """

    images = np.asarray(values)
    if images.shape != (num_envs, 7, 7, 3):
        raise ValueError('images must have shape [N,7,7,3]')
    if images.dtype.kind not in 'uif':
        raise ValueError('images must contain numeric symbolic codes')

    # The native learner stores images as float32 tensors. Accept exact
    # integer-valued floats, but never silently round or truncate them.
    consumed = images[active]
    if not np.isfinite(consumed).all():
        raise ValueError('Active images must contain finite symbolic codes')
    if images.dtype.kind == 'f' and not np.equal(
        consumed, np.floor(consumed)
    ).all():
        raise ValueError('Active images must contain integer symbolic codes')

    objects, colors, states = (
        consumed[..., channel] for channel in range(3)
    )
    if not np.isin(objects, tuple(OBJECT_TO_IDX.values())).all():
        raise ValueError('Active images contain invalid object codes')
    if not np.isin(colors, _COLOR_CODES).all():
        raise ValueError('Active images contain invalid color codes')

    # Native doors encode state; an agent marker may encode direction.
    # Every other native object has zero in its state channel.
    doors = objects == OBJECT_TO_IDX['door']
    agents = objects == OBJECT_TO_IDX['agent']
    if not np.isin(states[doors], _DOOR_CODES).all():
        raise ValueError('Active images contain invalid door state codes')
    if not np.isin(states[agents], (0, 1, 2, 3)).all():
        raise ValueError('Active images contain invalid agent directions')
    if np.any(states[~(doors | agents)] != 0):
        raise ValueError('Other symbolic objects must have state zero')
    return images


class HistoryTargets:
    """
    Track the past of N independent streams without storing their images.

    Memory has six fields in MiniGrid color order. Its target is the state
    at the most recent witnessing of that color's doors, if all doors of
    that color in that image agreed. A later conflicting witnessing erases
    the older state. The target is masked while any such door is visible,
    until first observed, or while the latest witnessing is ambiguous.
    There is no claim that the remembered state still holds offscreen.

    Knowledge has 18 fields: (key, door, ball) times the six colors. It is
    0 before any matching object was observed, 1 when observed previously
    but absent from this image, and 2 when currently visible. A carried
    object encoded in the image counts as visible. These labels cannot
    establish physical absence, unique object identity, or task relevance.
    """

    def __init__(self, num_envs):
        if (
            isinstance(num_envs, (bool, np.bool_))
            or not isinstance(num_envs, (int, np.integer))
            or num_envs < 1
        ):
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self._seen = np.zeros(
            (self.num_envs, len(KNOWLEDGE_FIELDS)), dtype=bool
        )
        self._last_door_state = np.full(
            (self.num_envs, len(MEMORY_FIELDS)),
            IGNORE_INDEX,
            dtype=np.int64,
        )
        self._last_door_ambiguous = np.zeros_like(
            self._last_door_state, dtype=bool
        )

    def update(self, images, episode_start, inactive=None):
        """
        Consume one native observation per active slot and return targets.

        images is numeric [N,7,7,3], with integer symbolic codes. Exactly
        integral floats are accepted. episode_start and optional inactive
        are binary [N] arrays. Set episode_start for the first observation
        of each episode, including the initial reset. Reset happens before
        consuming that observation, also when that slot is inactive.

        Set inactive on NEXT_STEP terminal observations whose next action
        is ignored by autoreset. They generate no labels and do not update
        history. On the following genuine reset observation, clear inactive
        and set episode_start. This class cannot infer missing boundaries.

        All outputs are fresh numpy arrays. memory [N,6] and knowledge
        [N,18] are int64 with IGNORE_INDEX for invalid entries. Their
        matching *_mask arrays are boolean. knowledge_class_mask [N,18,3]
        is diagnostic one-hot ground truth, never a logit exclusion mask.
        memory_count [N] and knowledge_counts [N,3] count valid fields per
        row. Memory diagnostic masks distinguish visible, never observed,
        and latest-witness ambiguity; these are false on inactive rows.
        The visible and ambiguous diagnostics may overlap.
        """

        starts = _flags(episode_start, 'episode_start', self.num_envs)
        skipped = (
            np.zeros(self.num_envs, dtype=bool)
            if inactive is None
            else _flags(inactive, 'inactive', self.num_envs)
        )
        active = ~skipped
        images = _images(images, active, self.num_envs)

        # Validate the whole call before changing state. A malformed
        # active row cannot partially reset or advance another stream.
        self._seen[starts] = False
        self._last_door_state[starts] = IGNORE_INDEX
        self._last_door_ambiguous[starts] = False

        visible = np.zeros_like(self._seen)
        objects, colors, states = (
            images[..., channel] for channel in range(3)
        )
        for kind_index, kind in enumerate(KNOWLEDGE_OBJECTS):
            kind_matches = objects == OBJECT_TO_IDX[kind]
            for color_index, color_code in enumerate(_COLOR_CODES):
                field = kind_index * len(COLORS) + color_index
                matches = kind_matches & (colors == color_code)
                visible[:, field] = matches.any(axis=(1, 2)) & active

        # State is associated with the latest image containing this
        # color, not with a globally identified or positioned door.
        door_visible = visible[:, _DOOR_FIELDS]
        for color_index, color_code in enumerate(_COLOR_CODES):
            matches = (
                (objects == OBJECT_TO_IDX['door'])
                & (colors == color_code)
            )
            witnessed = np.stack(
                [
                    (matches & (states == code)).any(axis=(1, 2))
                    for code in _DOOR_CODES
                ],
                axis=-1,
            )
            ambiguous = witnessed.sum(axis=1) > 1
            last_state = np.where(
                ambiguous, IGNORE_INDEX, witnessed.argmax(axis=1)
            )
            update_rows = door_visible[:, color_index]
            self._last_door_state[update_rows, color_index] = (
                last_state[update_rows]
            )
            self._last_door_ambiguous[update_rows, color_index] = (
                ambiguous[update_rows]
            )
        self._seen |= visible

        # Mask all present doors so memory supervision requires past
        # observation information. Unknown is never a fabricated state.
        memory_mask = (
            active[:, None]
            & ~door_visible
            & (self._last_door_state != IGNORE_INDEX)
        )
        memory = np.where(
            memory_mask, self._last_door_state, IGNORE_INDEX
        )
        knowledge_mask = np.broadcast_to(
            active[:, None], self._seen.shape
        ).copy()
        knowledge = np.where(visible, 2, self._seen.astype(np.int64))
        knowledge[~knowledge_mask] = IGNORE_INDEX
        knowledge_class_mask = knowledge[..., None] == np.arange(3)

        return dict(
            memory=memory,
            knowledge=knowledge,
            memory_mask=memory_mask,
            knowledge_mask=knowledge_mask,
            knowledge_class_mask=knowledge_class_mask,
            memory_count=memory_mask.sum(axis=1, dtype=np.int64),
            knowledge_counts=knowledge_class_mask.sum(
                axis=1, dtype=np.int64
            ),
            memory_visible_mask=door_visible.copy(),
            memory_unseen_mask=(
                ~self._seen[:, _DOOR_FIELDS] & active[:, None]
            ),
            memory_ambiguous_mask=(
                self._last_door_ambiguous & active[:, None]
            ),
        )
