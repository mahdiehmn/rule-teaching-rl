"""Rule subgoal teachers and a message-conditioned shaping potential.

Protocol:
research/potential_channel_protocol_2026-09-27.md.

A teacher MESSAGE claims the student's current stage (which object to
reach next). The student is rewarded through a potential over the full
state and the message:

    Phi(s, m) = -scale * (stages_after(m) + d(agent, target(m)) / dmax)
                / total_stages,          Phi(s, None) = 0,

and F = gamma * Phi(s', m') - Phi(s, m), with Phi = 0 at every episode
end. Phi is anchored on the true reset state, before the first policy
action, so each episode's discounted shaping is exactly -Phi(s_0, m_0).
That is a policy-independent constant for a given initial state and
message, whatever messages follow. So a wrong, aliased, missing or
withdrawn message cannot change which policy is optimal. It does NOT
promise faster or safer finite-budget learning, which is an empirical
question (dynamic potential-based shaping; Wiewiora 2003; Devlin and
Kudenko 2012).

A message expires when ITS OWN claimed subgoal completes (the claimed key
is carried, the claimed door unlocked, the claimed room left), never on
the true stage. So a wrong message does not learn true progress timing.
If the claimed subgoal was already complete when stated, the message
persists to the episode end.

Distances are BFS path lengths on the full grid (privileged). Doors
count as passable and other objects block, except the target. Messages
are drawn from a private RNG and never touch global NumPy or torch
streams.
"""

from collections import deque

import numpy as np

STAGE_NAMES = {
    'doorkey_8x8': ('key', 'door', 'goal'),
    'keycorridor_s3r3': ('key', 'door', 'ball'),
}
SUPPORTED = ('doorkey_8x8', 'multiroom_n6', 'keycorridor_s3r3')


def _cells(u, kind):
    return [(x, y) for x in range(u.width) for y in range(u.height)
            if (c := u.grid.get(x, y)) is not None and c.type == kind]


def _locked_doors(u):
    return [(x, y) for (x, y) in _cells(u, 'door')
            if u.grid.get(x, y).is_locked]


class TaskStages:
    """Per-episode stage structure: true stage and a target per stage."""

    def __init__(self, task):
        if task not in SUPPORTED:
            raise ValueError(f'No subgoal teacher for {task}')
        self.task = task
        self.locked_door = None      # remembered after it is unlocked

    def start_episode(self, u):
        if self.task != 'multiroom_n6':
            locked = _locked_doors(u)
            self.locked_door = locked[0] if locked else None

    def total(self, u):
        return len(u.rooms) if self.task == 'multiroom_n6' else 3

    def true_stage(self, u):
        if self.task == 'multiroom_n6':
            pos = tuple(int(v) for v in u.agent_pos)
            for k, room in enumerate(u.rooms):
                (tx, ty), (w, h) = room.top, room.size
                if tx < pos[0] < tx + w - 1 and ty < pos[1] < ty + h - 1:
                    return k
            for k, room in enumerate(u.rooms):
                if room.exitDoorPos is not None and \
                        tuple(room.exitDoorPos) == pos:
                    return k
            return 0
        if not _locked_doors(u):
            return 2
        carrying = u.carrying is not None and u.carrying.type == 'key'
        return 1 if carrying else 0

    def completed(self, u, stage):
        """Has the claimed stage's own completion predicate become true?"""
        if self.task == 'multiroom_n6':
            return stage < len(u.rooms) - 1 and self.true_stage(u) > stage
        if stage == 0:
            return u.carrying is not None and u.carrying.type == 'key'
        if stage == 1:
            return self.locked_door is not None and                 not u.grid.get(*self.locked_door).is_locked
        return False                # the goal/ball ends the episode

    def target(self, u, stage):
        agent = tuple(int(v) for v in u.agent_pos)
        if self.task == 'multiroom_n6':
            room = u.rooms[stage]
            pos = room.exitDoorPos if room.exitDoorPos is not None \
                else u.goal_pos
            return tuple(int(v) for v in pos)
        kind = STAGE_NAMES[self.task][stage]
        if kind == 'door':
            return self.locked_door or agent
        found = _cells(u, kind)
        return found[0] if found else agent     # a carried key: here


PLAN_TARGETS = ('key', 'locked_door', 'next_room_door', 'ball', 'goal',
                'box')
PLAN_DONE = ('carrying_key', 'door_unlocked', 'entered_next_room',
             'task_complete')
PLAN_REPEAT = ('once', 'each_room_before_the_goal_room')
# The hand-coded stages of TaskStages, written as plans. PlanStages with
# these reproduces TaskStages exactly (tests/test_llm_stage_plan.py).
HUMAN_PLAN = {
    'doorkey_8x8': (('key', 'carrying_key', 'once'),
                    ('locked_door', 'door_unlocked', 'once'),
                    ('goal', 'task_complete', 'once')),
    'keycorridor_s3r3': (('key', 'carrying_key', 'once'),
                         ('locked_door', 'door_unlocked', 'once'),
                         ('ball', 'task_complete', 'once')),
    'multiroom_n6': (('next_room_door', 'entered_next_room',
                      'each_room_before_the_goal_room'),
                     ('goal', 'task_complete', 'once')),
}


def plan_sha256(plan):
    """Digest of a plan's content, independent of how a file is written."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps([list(s) for s in plan],
                                     separators=(',', ':')).encode()
                          ).hexdigest()


def load_plan(path, task, sha256):
    """The task's plan from a frozen plan file, checked against its digest."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    plan = tuple(tuple(s) for s in data['tasks'][task]['plan'])
    if plan_sha256(plan) != sha256:
        raise ValueError('Subgoal plan content differs from its digest')
    return plan


def _interior(top, size, pos):
    (tx, ty), (w, h) = top, size
    return tx < pos[0] < tx + w - 1 and ty < pos[1] < ty + h - 1


def _room_index(u):
    """MultiRoom: the room the agent is in (its exit doorway counts)."""
    pos = tuple(int(v) for v in u.agent_pos)
    for k, room in enumerate(u.rooms):
        if _interior(room.top, room.size, pos):
            return k
    for k, room in enumerate(u.rooms):
        if room.exitDoorPos is not None and tuple(room.exitDoorPos) == pos:
            return k
    return 0


class PlanStages:
    """TaskStages driven by a written subgoal plan, e.g. the LLM teacher's.

    Same interface as TaskStages (start_episode, total, true_stage,
    completed, target), so PotentialTeacher and the potential are
    unchanged. With HUMAN_PLAN it reproduces TaskStages exactly.

    Fixed semantics, for any plan:
      target  key             the key's cell (the agent's, once carried)
              locked_door     the door locked at episode start
              next_room_door  MultiRoom room k: its exit door; elsewhere
                              the door locked at episode start (the door
                              toward the goal or the ball)
              ball/goal/box   that object's cell (the agent's if carried)
      done    carrying_key    the agent holds a key
              door_unlocked   a door locked at episode start is unlocked;
                              a door never locked (MultiRoom) is open
              entered_next_room  MultiRoom room k: the agent is in a later
                              room; DoorKey: past the splitting wall;
                              KeyCorridor: in the ball room's doorway or
                              inside it (S3 rooms have one interior cell,
                              which the ball occupies)
              task_complete   never mid-episode (success ends it)
      repeat  each_room_before_the_goal_room  expands, as one block with
                              the adjacent repeated steps, once per room
                              before the goal room: MultiRoom only. The
                              other tasks have no room chain toward the
                              goal, so there it means once.
    The true stage is one past the LAST step whose completion holds, so a
    later completion implies the earlier ones.
    """

    def __init__(self, task, plan):
        if task not in SUPPORTED:
            raise ValueError(f'No subgoal teacher for {task}')
        for step in plan:
            if (len(step) != 3 or step[0] not in PLAN_TARGETS
                    or step[1] not in PLAN_DONE
                    or step[2] not in PLAN_REPEAT):
                raise ValueError(f'Plan step outside the vocabulary: {step}')
        self.task, self.plan = task, tuple(tuple(s) for s in plan)
        self.locked_door, self.locked_at_start = None, frozenset()
        self.ball_room, self.steps = None, ()

    def start_episode(self, u):
        locked = _locked_doors(u)
        self.locked_at_start = frozenset(locked)
        self.locked_door = (locked[0] if locked and
                            self.task != 'multiroom_n6' else None)
        self.ball_room = None
        if self.task == 'keycorridor_s3r3':
            balls = _cells(u, 'ball')
            if balls:
                room = u.room_from_pos(*balls[0])
                self.ball_room = (tuple(room.top), tuple(room.size))
        self.steps = self._expand(u)

    def _expand(self, u):
        rooms = (len(u.rooms) - 1 if self.task == 'multiroom_n6' else None)
        out, i = [], 0
        while i < len(self.plan):
            if (self.plan[i][2] == 'each_room_before_the_goal_room'
                    and rooms is not None):
                j = i
                while j < len(self.plan) and self.plan[j][2] == \
                        'each_room_before_the_goal_room':
                    j += 1
                for k in range(rooms):
                    out.extend((t, d, k) for t, d, _ in self.plan[i:j])
                i = j
            else:
                out.append((self.plan[i][0], self.plan[i][1], None))
                i += 1
        return tuple(out)

    def total(self, u):
        return len(self.steps)

    def _door_of(self, u, target, k):
        if target == 'locked_door':
            return self.locked_door
        if target != 'next_room_door':
            return None
        if self.task == 'multiroom_n6':
            room = u.rooms[k if k is not None else _room_index(u)]
            return (tuple(int(v) for v in room.exitDoorPos)
                    if room.exitDoorPos is not None else None)
        return self.locked_door

    def _done(self, u, step):
        target, done, k = step
        if done == 'carrying_key':
            return u.carrying is not None and u.carrying.type == 'key'
        if done == 'door_unlocked':
            pos = self._door_of(u, target, k)
            door = u.grid.get(*pos) if pos is not None else None
            if door is None or door.type != 'door':
                return False
            return (not door.is_locked if pos in self.locked_at_start
                    else bool(door.is_open))
        if done == 'entered_next_room':
            pos = tuple(int(v) for v in u.agent_pos)
            if self.task == 'multiroom_n6':
                return _room_index(u) > (k if k is not None else 0)
            if self.task == 'doorkey_8x8':
                return (self.locked_door is not None
                        and pos[0] > self.locked_door[0])
            # KeyCorridor-S3R3 rooms are 3x3 with walls: the ball fills the
            # ball room's only interior cell, so the agent can only ever
            # stand in its doorway (and picks the ball up from there).
            # Standing in the doorway therefore counts as having entered.
            return (pos == self.locked_door or
                    (self.ball_room is not None
                     and _interior(*self.ball_room, pos)))
        return False                    # task_complete ends the episode

    def true_stage(self, u):
        for s in range(len(self.steps) - 1, -1, -1):
            if self._done(u, self.steps[s]):
                return min(s + 1, len(self.steps) - 1)
        return 0

    def completed(self, u, stage):
        return self._done(u, self.steps[stage])

    def target(self, u, stage):
        target, _done, k = self.steps[stage]
        agent = tuple(int(v) for v in u.agent_pos)
        if target in ('locked_door', 'next_room_door'):
            pos = self._door_of(u, target, k)
            return pos if pos is not None else agent
        found = _cells(u, target)
        return found[0] if found else agent


def bfs_map(u, target):
    """Path length from every cell to `target` (walls and objects block)."""
    width, height = u.width, u.height
    dist = np.full((width, height), -1, dtype=np.int32)
    dist[target] = 0
    queue = deque([target])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < width and 0 <= ny < height) or \
                    dist[nx, ny] >= 0:
                continue
            cell = u.grid.get(nx, ny)
            if cell is not None and cell.type not in ('door', 'goal'):
                continue            # walls, keys, balls, boxes block
            dist[nx, ny] = dist[x, y] + 1
            queue.append((nx, ny))
    return dist


class PotentialTeacher:
    """Budgeted, possibly wrong subgoal messages and the shaping signal.

    `queries < 0` means unlimited. A message is requested when an episode
    starts (from its true reset state) and when the current message's own
    claimed subgoal completes, while budget remains. With probability
    `wrong_rate` it claims a uniformly chosen stage other than the true
    one at the moment of the request. Call `begin_all` after every
    vectorised reset, before the first action; `shape` refuses otherwise.
    """

    def __init__(self, task, num_envs, gamma, scale=1.0, queries=-1,
                 wrong_rate=0.0, seed=0, log_limit=2000, truncation='zero',
                 trace_episodes=0, plan=None):
        if (not 0 <= wrong_rate <= 1 or scale <= 0
                or truncation not in ('zero', 'keep')):
            raise ValueError('Invalid potential-teacher settings')
        # 'zero': Phi = 0 at every episode end (exact invariance).
        # 'keep': Phi = 0 at task termination but kept at a time-limit
        # cut, as if the episode continued; the optimal policy is kept
        # whenever success pays more than any failure (Phi <= 0).
        self.truncation = truncation
        self.task, self.gamma, self.scale = task, gamma, scale
        self.budget, self.wrong_rate = queries, wrong_rate
        self.rng = np.random.default_rng(seed)
        # A written plan (e.g. the LLM teacher's) drives the stages; None
        # keeps the hand-coded TaskStages and every earlier run exactly.
        self.plan = tuple(tuple(s) for s in plan) if plan else None
        self.stages = [PlanStages(task, self.plan) if self.plan
                       else TaskStages(task) for _ in range(num_envs)]
        self.message = [None] * num_envs
        self.phi = [None] * num_envs
        self.cache = [dict() for _ in range(num_envs)]
        self.episode = [0] * num_envs
        self.used = self.wrong = 0
        self.potential_evaluations = self.message_steps = 0
        self.expired = self.withdrawn = 0
        self.persistence = []                 # steps each message lived
        self.first_step = self.last_step = None
        self.shaping_sum = self.shaping_abs = 0.0
        self.log, self.log_limit = [], log_limit
        self.trace_episodes = trace_episodes
        self.trace, self._open = [], [None] * num_envs

    # ----------------------------------------------------------- messages

    def _query(self, i, u, step):
        if self.budget >= 0 and self.used >= self.budget:
            return None
        stages = self.stages[i]
        true = stages.true_stage(u)
        claimed = true
        if self.wrong_rate and self.rng.random() < self.wrong_rate:
            others = [s for s in range(stages.total(u)) if s != true]
            claimed = int(self.rng.choice(others))
            self.wrong += 1
        self.used += 1
        if self.first_step is None:
            self.first_step = step
        self.last_step = step
        if len(self.log) < self.log_limit:
            self.log.append(dict(step=step, env=i, episode=self.episode[i],
                                 true=true, claimed=claimed))
        return dict(claimed=claimed, true=true, issued=step, steps=0,
                    done_at_issue=stages.completed(u, claimed))

    def _retire(self, i):
        message = self.message[i]
        if message is not None:
            self.persistence.append(message['steps'])
        self.message[i] = None

    def potential(self, i, u):
        message = self.message[i]
        if message is None:
            return 0.0
        self.potential_evaluations += 1
        stages = self.stages[i]
        claimed, total = message['claimed'], stages.total(u)
        target = stages.target(u, claimed)
        key = (self.episode[i], target, u.carrying is None)
        if key not in self.cache[i]:
            self.cache[i] = {key: bfs_map(u, target)}
        pos = tuple(int(v) for v in u.agent_pos)
        d = int(self.cache[i][key][pos])
        if d < 0:                      # unreachable: fall back to Manhattan
            d = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        dmax = u.width + u.height
        return -self.scale * ((total - 1 - claimed) + d / dmax) / total

    # ------------------------------------------------------------ episodes

    def begin(self, i, u, step):
        """Anchor a new episode on its true reset state (before acting)."""
        self._retire(i)
        self.episode[i] += 1
        self.stages[i].start_episode(u)
        self.message[i] = self._query(i, u, step)
        self.phi[i] = self.potential(i, u)
        if self.episode[i] <= self.trace_episodes:
            self._open[i] = dict(env=i, episode=self.episode[i],
                                 phi0=self.phi[i], rewards=[])

    def begin_all(self, unwrapped, step):
        for i, u in enumerate(unwrapped):
            self.begin(i, u, step)

    def shape(self, unwrapped, reset_only, done, step):
        """Shaping for one vector step; zero on autoreset ticks."""
        out = np.zeros(len(unwrapped), dtype=np.float32)
        for i, u in enumerate(unwrapped):
            if self.phi[i] is None:
                raise RuntimeError('begin_all must run at the reset state')
            if reset_only[i]:
                self.begin(i, u, step)      # NEXT_STEP autoreset tick
                continue
            truncated = bool(u.step_count >= u.max_steps)
            if done[i] and not (self.truncation == 'keep' and truncated):
                new = 0.0
            else:
                message = self.message[i]
                if message is not None:
                    message['steps'] += 1
                    self.message_steps += 1
                    if (not message['done_at_issue'] and
                            self.stages[i].completed(u, message['claimed'])):
                        self.expired += 1
                        self._retire(i)
                        self.message[i] = self._query(i, u, step)
                        if self.message[i] is None:
                            self.withdrawn += 1
                new = self.potential(i, u)
            out[i] = self.gamma * new - self.phi[i]
            self.phi[i] = new
            if self._open[i] is not None:
                self._open[i]['rewards'].append(float(out[i]))
                if done[i]:
                    self._open[i].update(
                        end_phi=new, truncated=truncated,
                        terminated=bool(done[i]) and not truncated)
                    self.trace.append(self._open[i])
                    self._open[i] = None
        self.shaping_sum += float(out.sum())
        self.shaping_abs += float(np.abs(out).sum())
        return out

    def summary(self):
        lived = self.persistence
        return dict(task=self.task, budget=self.budget,
                    plan=[list(s) for s in self.plan] if self.plan else None,
                    truncation=self.truncation,
                    wrong_rate=self.wrong_rate, scale=self.scale,
                    gamma=self.gamma, queries_used=self.used,
                    wrong_messages=self.wrong,
                    expired_on_claimed_completion=self.expired,
                    withdrawn_at_budget=self.withdrawn,
                    potential_evaluations=self.potential_evaluations,
                    message_steps=self.message_steps,
                    mean_message_persistence=(float(np.mean(lived))
                                              if lived else None),
                    first_query_step=self.first_step,
                    last_query_step=self.last_step,
                    shaping_sum=self.shaping_sum,
                    shaping_abs_sum=self.shaping_abs,
                    logged_queries=len(self.log))
