"""Manages the RL model."""

import random
from typing import Any

import torch
import numpy as np

from grid.map import Map
from grid.utils import Point
from grid.map import Direction
from grid.pathfinder import Pathfinder, PathfinderConfig

from networks.v2.utils import initialize_model
from networks.v2.ppo import ValueNetworkConfig, DiscretePolicyConfig
from networks.v2.encoder import MapEncoderConfig, TemporalMapEncoderConfig
from agent.inference import Inference


CHANNELS, MAP_SIZE, ACTION_DIM = 10, 31, 5


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RLManager:
    def __init__(self, seed: int = 42):
        """
        Initialize RL Manager.

        Args:
            k: Number of top actions to consider for sampling
            temperature: Temperature for softmax scaling
            use_top_k: Whether to use top-k sampling or deterministic selection
            seed: Random seed for reproducibility
        """
        # encoder_config = MapEncoderConfig(
        #     kernel_sizes=[7, 3, 3, 3],
        #     output_dim=32
        # )

        # actor_config = DiscretePolicyConfig(
        #     input_dim=32,
        #     action_dim=5,
        #     hidden_dims=[32, 32]
        # )

        # critic_config = ValueNetworkConfig(
        #     input_dim=32,
        #     hidden_dims=[32, 32]
        # )

        encoder_config = TemporalMapEncoderConfig(
            map_size=16,
            channels=10,
            output_dim=64,
            conv3d_channels=[32, 48, 48, 64],
            conv3d_kernel_sizes=[(3, 7, 7), (3, 3, 3), (3, 3, 3), (4, 1, 1)],
            conv3d_strides=[(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
            conv3d_paddings=[(1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 0)],
            conv_layers=[64],
            kernel_sizes=[3],
            strides=[1],
            paddings=[0],
            use_batch_norm=True,
            dropout_rate=0.1,
            use_layer_norm=True,
            use_center_only=True,
        )

        actor_config = DiscretePolicyConfig(
            input_dim=64, action_dim=ACTION_DIM, hidden_dims=[64, 64]
        )

        critic_config = ValueNetworkConfig(input_dim=64, hidden_dims=[64, 64])

        self.scout_policy = initialize_model(
            encoder_config, actor_config, critic_config
        )

        checkpoint = torch.load("./models/scout.pt", map_location="cpu")
        self.scout_policy.load_state_dict(checkpoint["model_state_dict"])
        self.scout_policy.eval()

        self.role = None  # 'scout' or 'guard'
        self.recon_map = Map()
        self.pathfinder = Pathfinder(
            self.recon_map,
            PathfinderConfig(
                use_viewcone=True,
            ),
        )

        self.scout = Inference(
            self.scout_policy,
            self.recon_map,
            strategy="greedy",
            n_frames=4,
            top_k=5,
            temperature=0.5,
            action_dim=ACTION_DIM,
        )

        self.initialized = False
        self.last_step = -1

        # Set random seeds
        self.seed = seed
        set_random_seeds(self.seed)

    def _reset_episode(self):
        """Reset manager state for new episode."""
        self.initialized = False
        self.recon_map = Map()

    def _initialize_role(self, observation: dict[str, Any]):
        """Initialize agent role and setup."""
        self.role = "scout" if observation.get("scout", 0) == 1 else "guard"

        # For guard, initialize the map
        if self.role == "guard":
            self.recon_map.create_trajectory_tree(Point(0, 0))

        self.initialized = True
        print(f"Initialized as {'SCOUT' if self.role == 'scout' else 'GUARD'}")

    def _preprocess_observation(self, observation: dict[str, Any]):
        """Preprocess observation data types."""
        observation["viewcone"] = np.array(observation["viewcone"], dtype=np.uint8)
        observation["location"] = np.array(observation["location"], dtype=np.uint8)
        observation["direction"] = int(observation["direction"])

    def rl(self, observation: dict[str, Any]) -> int:
        """Gets the next action for the agent, based on the observation.
        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.
        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        # Reset if starting a new episode (step went back to 0)
        current_step = observation.get("step", 0)
        if isinstance(current_step, int):
            if current_step == 0 and self.last_step > 0:
                self._reset_episode()
            self.last_step = current_step

        self._preprocess_observation(observation)

        # Determine role on first call
        if not self.initialized:
            self._initialize_role(observation)

        self.recon_map(observation)
        # Different logic for scout and guard
        if self.role == "scout":
            return self.scout(observation)
        else:
            location = observation["location"]
            direction = observation["direction"]

            action = self.pathfinder.get_optimal_action(
                Point(location[0], location[1]), Direction(direction), tree_index=0
            )

            return int(action)
