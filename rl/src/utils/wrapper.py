import random

import numpy as np

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper

from til_environment.types import RewardNames, Player
from til_environment.gridworld import NUM_ITERS

from grid.map import Map
from grid.pathfinder import Pathfinder, PathfinderConfig
from grid.utils import Point, Direction


class CustomRewardsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
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

class MapWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
    ):
        super().__init__(env)

        self.maps = {}
        self.pathfinders = {}
        self.active_guards = []

    def set_num_active_guards(self, n: int):
        self.num_active_guards = n

    def init_active_guards(self):
        guards = [a for a in self.agents if a != self.scout]
        random.shuffle(guards)
        guards = guards[:self.num_active_guards]

        self.active_guards = guards

    def init_maps(self):
        self.maps.clear()
        self.pathfinders.clear()

        for guard in self.active_guards:
            self.maps[guard] = Map()
            self.maps[guard].create_trajectory_tree(Point(0, 0))
            self.pathfinders[guard] = Pathfinder(
                self.maps[guard],
                PathfinderConfig(
                    use_viewcone=False,
                    use_path_density=False
                )
            )

        self.maps[self.scout] = Map()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        self.init_active_guards()
        self.init_maps()

class ScoutWrapper(MapWrapper):
    def iter_guard(self, guard):
        observation, reward, termination, truncation, info = self.agent_last(guard)

        if guard in self.pathfinders:
            self.maps[guard](observation)

            location = observation.get('location')
            direction = observation.get('direction')

            action = int(self.pathfinders[guard].get_optimal_action(
                Point(location[0], location[1]),
                Direction(direction)
            ))
        else:
            action = self.action_space(guard).sample()

        super().step(action)


    def step(self, action: ActionType):
        while self.agent_selection != self.scout:
            self.iter_guard(self.agent_selection)

        assert self.agent_selection == self.scout

        scout = self.agent_selection

        if scout in self.maps:
            observation = self.observe(scout)

            self.maps[scout](observation)

        super().step(action)

    def agent_last(self, agent):
        observation = self.observe(agent)
        reward = self.rewards[agent]
        termination = self.terminations[agent]
        truncation = self.truncations[agent]
        info = self.get_info(agent)

        return observation, reward, termination, truncation, info
