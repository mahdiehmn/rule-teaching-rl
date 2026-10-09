"""RLingua-style complete controllers for MiniGrid (baseline port).

RLingua (Chen et al., RA-L 2024) asks an
LLM to write a COMPLETE rule-based controller once, then uses it during RL
(its Algorithm A-1). This module is the single place where a frozen,
LLM-written controller is loaded and given its inputs, so the generation
script (scripts/rlingua_controllers_20261005.py) and the trainer
(algos/ppo_distill.py, rlingua_* arguments) use exactly the same interface.

Two interfaces, matching the two information settings we compare:

  full   controller(state) -> action. `state` is the complete MiniGrid
         state, as RLingua's controllers read the full simulator state:
         the whole grid, the agent's position and direction, what it
         carries, the mission.
  view   controller(obs, memory) -> action. `obs` is exactly the student's
         observation (its 7x7x3 egocentric symbolic image) plus
         `last_action`, the action the env actually ran at the previous
         step of the episode (None at its first step): under RLingua's
         mixing the student acts at most steps, so a stateful controller
         must not assume its own outputs ran. `memory` is a dict the
         controller may use freely, emptied at each episode start (the
         student's recurrent policy also has memory).

Each env gets its own controller instance (a fresh namespace), recreated
at every episode start, so whatever a controller keeps between calls
(globals, function attributes, `memory`) belongs to one env and one
episode. A controller that raises or returns an invalid action is counted
and replaced by turn_left for that step (logged, never hidden).
"""

import builtins
import hashlib
import math
from pathlib import Path

ACTIONS = ('turn_left', 'turn_right', 'forward', 'pickup', 'drop',
           'toggle', 'done')
SAFE_MODULES = {'collections', 'heapq', 'math', 'random', 'itertools',
                'functools'}
SAFE_BUILTINS = (
    'abs', 'all', 'any', 'bool', 'dict', 'divmod', 'enumerate', 'filter',
    'float', 'frozenset', 'int', 'isinstance', 'iter', 'len', 'list', 'map',
    'max', 'min', 'next', 'range', 'reversed', 'round', 'set', 'sorted',
    'str', 'sum', 'tuple', 'zip', 'Exception', 'ValueError', 'KeyError',
    'IndexError', 'TypeError', 'StopIteration', 'print', 'hasattr',
    'getattr', 'object', 'staticmethod', 'classmethod', 'property', 'super',
    'type', 'NotImplementedError', 'RuntimeError', 'ZeroDivisionError')


def _safe_import(name, *args, **kwargs):
    if name.split('.')[0] not in SAFE_MODULES:
        raise ImportError(f'controller may not import {name}')
    return builtins.__import__(name, *args, **kwargs)


def compile_controller(source):
    return compile(source, '<rlingua_controller>', 'exec')


def instantiate(code):
    """A fresh `controller` function from compiled source, in its own
    restricted namespace (so no state is shared between instances)."""
    allowed = {k: getattr(builtins, k) for k in SAFE_BUILTINS
               if hasattr(builtins, k)}
    allowed['__import__'] = _safe_import
    namespace = {'__builtins__': allowed, '__name__': 'rlingua_controller',
                 'math': math}
    exec(code, namespace)
    fn = namespace.get('controller')
    if not callable(fn):
        raise ValueError('source defines no controller function')
    return fn


def load_controller(source):
    """Compile controller source in a restricted namespace; return the
    `controller` function."""
    return instantiate(compile_controller(source))


def full_state(u):
    """The complete state of a MiniGrid env (RLingua's information)."""
    grid = []
    for y in range(u.height):
        row = []
        for x in range(u.width):
            c = u.grid.get(x, y)
            if c is None:
                row.append(None)
                continue
            cell = {'type': c.type, 'color': c.color}
            if c.type == 'door':
                cell['is_open'] = bool(c.is_open)
                cell['is_locked'] = bool(c.is_locked)
            row.append(cell)
        grid.append(row)
    carrying = (None if u.carrying is None else
                {'type': u.carrying.type, 'color': u.carrying.color})
    return {'width': int(u.width), 'height': int(u.height), 'grid': grid,
            'agent_pos': (int(u.agent_pos[0]), int(u.agent_pos[1])),
            'agent_dir': int(u.agent_dir), 'carrying': carrying,
            'mission': str(getattr(u, 'mission', ''))}


def view_obs(u, last_action=None):
    """Exactly the student's observation (the 7x7x3 symbolic image) and
    the action the env actually ran at the previous step."""
    image = u.gen_obs()['image']
    return {'image': [[[int(v) for v in image[i][j]] for j in range(7)]
                      for i in range(7)],
            'last_action': None if last_action is None else int(last_action)}


class RLinguaController:
    """A frozen controller file run on one or more envs."""

    def __init__(self, path, variant, num_envs=1, sha256=None):
        if variant not in ('full', 'view'):
            raise ValueError("variant must be 'full' or 'view'")
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
        if sha256 and sha256 != digest:
            raise ValueError('controller file differs from its recorded hash')
        self.sha256 = digest
        self.variant = variant
        self.code = compile_controller(data.decode('utf-8'))
        self.fns = [instantiate(self.code) for _ in range(num_envs)]
        self.memories = [dict() for _ in range(num_envs)]
        self.last_actions = [None] * num_envs
        self.errors = 0
        self.invalid = 0
        self.calls = 0
        # A full-state controller is a function of the state, so it is
        # called only when it acts; a view controller keeps memory and
        # must see every step of its episode.
        self.every_step = variant == 'view'

    def reset(self, i):
        """A new episode in env i: a fresh instance and empty memory."""
        self.fns[i] = instantiate(self.code)
        self.memories[i] = {}
        self.last_actions[i] = None

    def executed(self, i, action):
        """Record the action env i actually ran (controller's or not)."""
        self.last_actions[i] = int(action)

    def act(self, i, u):
        """The controller's action for env i (unwrapped env u)."""
        self.calls += 1
        try:
            if self.variant == 'full':
                action = self.fns[i](full_state(u))
            else:
                action = self.fns[i](view_obs(u, self.last_actions[i]),
                                     self.memories[i])
        except Exception:                      # counted, never hidden
            self.errors += 1
            return 0
        try:
            action = int(action)
        except (TypeError, ValueError):
            self.invalid += 1
            return 0
        if not 0 <= action < len(ACTIONS):
            self.invalid += 1
            return 0
        return action
