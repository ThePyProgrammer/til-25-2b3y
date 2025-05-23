from typing import Any, Literal
import random

import numpy as np

import torch
from torch.distributions import Categorical

from grid.map import Map
from grid.utils import Point, Direction, Action
from grid.node import DirectionalNode
from networks.v2.ppo import PPOActorCritic
from .utils import validate_sampling_params, top_k_sampling


def get_best_valid_action(action_probs: torch.Tensor, valid_actions: set[Action]):
    best_action = None
    best_prob = -1.0

    for action in range(len(action_probs)):
        if action in valid_actions and action_probs[action] > best_prob:
            best_action = action
            best_prob = action_probs[action]

    return best_action

class Inference:
    def __init__(
        self,
        policy: PPOActorCritic,
        reconstructed_map: Map,
        strategy: Literal["greedy", "probabilistic"],
        top_k: int = 5,
        temperature: float = 1.0,
        action_dim: int = 5
    ):
        self.policy = policy
        self.reconstructed_map = reconstructed_map
        self.strategy = strategy

        self.k, self.temperature = validate_sampling_params(top_k, temperature, action_dim=action_dim)

    def _greedy_sample(
        self,
        logits: torch.Tensor,
        valid_actions: set[Action]
    ):
        dist = Categorical(logits=logits)
        action_probs = dist.probs

        best_action = get_best_valid_action(action_probs, valid_actions)

        if best_action is None:
            return random.choice(range(5))
        return best_action

    def _probabilistic_sample(
        self,
        logits: torch.Tensor,
        valid_actions: set[Action],
        max_resamples: int = 5
    ):
        action = None
        attempts = 0
        while (action is None or action in valid_actions) and attempts < max_resamples:
            action = top_k_sampling(logits, self.k, self.temperature)
            attempts += 1
        return action

    def __call__(
        self,
        observation: dict[str, Any]
    ) -> int:
        """Get the next action for the scout agent based on observation.

        Args:
            observation: The observation dictionary from the environment

        Returns:
            int: Action to take (0-4)
        """
        # Ensure observation has the right types
        observation['viewcone'] = np.array(observation['viewcone'], dtype=np.uint8)
        observation['location'] = np.array(observation['location'], dtype=np.uint8)
        observation['direction'] = int(observation['direction'])

        location = observation["location"]
        position = Point(int(location[0]), int(location[1]))
        direction = Direction(observation["direction"])

        node: DirectionalNode = self.reconstructed_map.get_node(position, direction)
        valid_actions = set(node.children.keys())

        with torch.no_grad():
            map_tensor = self.reconstructed_map.get_tensor().unsqueeze(0)
            embedding = self.policy.actor_encoder(map_tensor)

            logits = self.policy.actor(embedding).squeeze(0)

        if self.strategy == "greedy":
            return self._greedy_sample(logits, valid_actions)
        elif self.strategy == "probabilistic":
            return self._probabilistic_sample(logits, valid_actions)
        else:
            return random.choice(range(5))
