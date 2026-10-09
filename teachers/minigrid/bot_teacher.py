"""
BabyAI Bot as a teacher.

Wraps `minigrid.utils.baby_ai_bot.BabyAIBot` -- the hand-coded
expert from the BabyAI paper, shipped inside the minigrid package --
behind this project's BaseTeacher/Advice contract. The bot parses
the level's language mission (`env.instrs`), plans via an explicit
subgoal stack, and can solve every original BabyAI level, including
GoToSeq (verified 10/10 episodes in this repo's env stack).

Why this teacher matters here:

- It is free, offline, and deterministic, so the distillation
  mechanism can be developed and run on cluster compute nodes (no
  internet) against the *headline language task*, instead of only
  against DoorKey via the BFS oracle.
- It was explicitly designed for the advising role: the bot's own
  docstring says it can "advise a suboptimal agent, e.g. play the
  role of an oracle in algorithms like DAGGER", by being told which
  action the student actually took (`replan(action_taken)`) so its
  internal plan tracks the student's real trajectory rather than
  the trajectory the bot would have taken.
- Its subgoal stack doubles as a structured explanation of *why* an
  action is recommended (e.g. 'GoNextToSubgoal: (3, 5)'), which is
  the free supervision signal Stage 2's reasoning head will train
  on -- no language model needed.

Statefulness contract
---------------------
Unlike the stateless BFS oracle, the bot carries per-episode state
(its map of what it has seen and its plan stack), so the caller
must:

- pass `context={'new_episode': True}` on the first query of every
  episode, so a fresh bot is built for the new mission/layout;
- pass the student's previously executed action in
  `context={'last_action': a}` on every later query, so the bot's
  plan stays synchronized with what actually happened.

The training loop in algos/ppo_distill.py keeps one instance of
this teacher per parallel environment for exactly this reason.

Failure handling: the bot occasionally cannot replan from a state a
bad student dragged it into (measured ~1 in 750 queries under a
uniformly random student). Any exception from the bot is mapped to
an *abstention* Advice (action=None) rather than a crash, and the
internal bot is dropped so the next query rebuilds a fresh plan
from the current state.
"""

import time

from minigrid.utils.baby_ai_bot import BabyAIBot

from teachers.base import Advice, BaseTeacher, Cost


class BabyAIBotTeacher(BaseTeacher):
    """
    Per-environment BabyAI Bot teacher.

    `state` for recommend() is the *unwrapped* BabyAI environment
    itself (the bot plans directly on env internals, exactly like
    the BFS oracle reads the symbolic grid -- the student never
    sees any of this, it still trains on pixels).
    """

    def __init__(self, env_id: str = 'BabyAI-GoToSeq-v0', seed: int = 0):
        """
        Create the teacher. The bot itself is built lazily per
        episode inside recommend(), because it must parse each new
        episode's mission and layout.
        """

        super().__init__(teacher_id=f'bot:{env_id}', seed=seed)
        self._bot = None

    def recommend(self, state, context: dict | None = None) -> Advice:
        """
        Return the bot's recommended action for the env's current
        state, or an abstention if the bot cannot plan.

        Parameters
        ----------
        state: minigrid env (unwrapped)
            A BabyAI level exposing `.instrs`; the bot reads the
            grid, agent pose, and mission from it directly.
        context: dict or None
            'new_episode': bool -- rebuild the bot (new mission).
            'last_action': int or None -- the action the student
            actually executed since the previous query, fed to
            `replan` so the bot's plan tracks reality.
        """

        context = context or {}
        t0 = time.perf_counter()

        # Rebuild the bot at episode boundaries; also rebuild if a
        # previous failure dropped it (mid-episode rebuild is safe:
        # the bot plans from the env's current state, it does not
        # need the episode's history).
        last_action = context.get('last_action')
        if context.get('new_episode') or self._bot is None:
            try:
                self._bot = BabyAIBot(state)
            except Exception:
                self._bot = None
                return self._abstain(t0)
            # A fresh bot has not seen any prior action; replan()
            # must be called without one.
            last_action = None

        bfs_before = self._bot.bfs_step_counter
        try:
            action = self._bot.replan(last_action)
        except Exception:
            # The bot lost the plot (e.g. the student carried an
            # object somewhere the plan cannot recover from).
            # Abstain now; the next query rebuilds from scratch.
            self._bot = None
            return self._abstain(t0)

        # The top of the subgoal stack is the bot's own reason for
        # this action -- a faithful, structured explanation that
        # Stage 2 uses as supervision.
        if self._bot.stack:
            explanation = str(self._bot.stack[-1])
        else:
            explanation = 'mission complete'

        return Advice(
            action=int(action),
            confidence=1.0,
            explanation=explanation,
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(
                wall_time_s=time.perf_counter() - t0,
                # BFS steps the bot performed for this query -- the
                # same "effort" unit the DoorKey oracle reports.
                compute_units=(
                    self._bot.bfs_step_counter - bfs_before
                ),
            ),
        )

    def _abstain(self, t0):
        """
        Build the abstention Advice returned on any bot failure.
        """

        return Advice(
            action=None,
            confidence=0.0,
            explanation='bot failed to plan; abstaining',
            teacher_id=self.teacher_id,
            call_id=self._next_call_id(),
            cost=Cost(wall_time_s=time.perf_counter() - t0),
        )
