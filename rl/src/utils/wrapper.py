import numpy as np

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper

from til_environment.types import RewardNames, Player
from til_environment.gridworld import NUM_ITERS


class CustomDictWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    """This wrapper flattens the Dict observation space."""

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
    ):
        super().__init__(env)

        self.action_history = {}

    def reset(self, *args, **kwargs):
        self.action_history = {}

        return super().reset(*args, **kwargs)

    def step(self, action: ActionType):
        """
        Takes in an action for the current agent (specified by agent_selection),
        only updating internal environment state when all actions have been received,
        with custom rewards.
        """

        agent = self.agent_selection

        if agent not in self.action_history:
            self.action_history[agent] = []

        self.action_history[agent].append(action)

        if self.agent_selector.is_last():
            for agent in self.agents:
                if len(self.action_history[agent]) > 2:
                    last_actions = self.action_history[agent][-2:]

                    repeat_action = last_actions[0] == last_actions[1]
                    repeat_turns = (
                        repeat_action
                        and last_actions[0] in [2, 3]
                    )

                    if repeat_turns:
                        self.rewards[agent] += self.rewards_dict.get(
                            RewardNames.STATIONARY_PENALTY, 0
                        )

        super().step(action)

    def state(self):
        _state = np.copy(super().state())

        # add players
        for _agent, loc in self.agent_locations.items():
            _state[loc] += (
                np.uint8(Player.SCOUT.power)
                if _agent == self.scout
                else np.uint8(Player.GUARD.power)
            )

        return _state
