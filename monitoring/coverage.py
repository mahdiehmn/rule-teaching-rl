"""
Privileged environment diagnostics, used only for training logs.

These measurements never become policy inputs, teacher targets or rewards.
Position coverage is episodic because layouts change between episodes.
"""


class TrainingCoverage:
    """
    Track positions and key/locked-door milestones in MiniGrid.
    """

    def __init__(self, envs):
        self.envs = envs
        self.current = [self._fresh(env) for env in envs]
        self.completed = 0
        self.key_episodes = 0
        self.unlock_episodes = 0
        self.position_total = 0

    @staticmethod
    def _fresh(env):
        """
        Capture the new layout's initially locked doors.
        """

        env = env.unwrapped
        locked = []
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if (obj is not None and obj.type == 'door'
                        and obj.is_locked):
                    locked.append((x, y))
        return {
            'positions': {tuple(env.agent_pos)}, 'locked': locked,
            'key': False, 'unlock': False,
        }

    def step(self, reset_steps, terminals):
        """
        Observe a transition, skipping NEXT_STEP autoreset actions.
        """

        for i, wrapped in enumerate(self.envs):
            if reset_steps[i]:
                self.current[i] = self._fresh(wrapped)
                continue
            env = wrapped.unwrapped
            state = self.current[i]
            state['positions'].add(tuple(env.agent_pos))
            if env.carrying is not None and env.carrying.type == 'key':
                state['key'] = True
            for pos in state['locked']:
                door = env.grid.get(*pos)
                if door is not None and not door.is_locked:
                    state['unlock'] = True
            if terminals[i]:
                self.completed += 1
                self.key_episodes += int(state['key'])
                self.unlock_episodes += int(state['unlock'])
                self.position_total += len(state['positions'])

    def stats(self):
        """
        Return cumulative completed-episode measurements.
        """

        denominator = max(1, self.completed)
        return {
            'completed_episodes': self.completed,
            'key_episode_fraction': self.key_episodes / denominator,
            'unlock_episode_fraction': self.unlock_episodes / denominator,
            'mean_unique_positions': self.position_total / denominator,
        }
