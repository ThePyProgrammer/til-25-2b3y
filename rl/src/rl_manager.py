"""Manages the RL model."""
# import sys
# import os
# import pathlib
import random
from typing import Any, Optional

import torch
import torch.nn.functional as F
import numpy as np

# Add paths for imports
# sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
# sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from grid.map import Map
from grid.utils import Point
from grid.map import Direction
from grid.pathfinder import Pathfinder, PathfinderConfig

from networks.ppo import PPOActorCritic
from agent.inference import Inference


CHANNELS, MAP_SIZE, ACTION_DIM = 12, 31, 5



def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class RLManager:
    def __init__(
        self,
        seed: int = 42
    ):
        """
        Initialize RL Manager.

        Args:
            k: Number of top actions to consider for sampling
            temperature: Temperature for softmax scaling
            use_top_k: Whether to use top-k sampling or deterministic selection
            seed: Random seed for reproducibility
        """
        self.scout_policy = PPOActorCritic(
            action_dim=ACTION_DIM,
            map_size=MAP_SIZE,
            channels=CHANNELS,
            encoder_type="small",
            shared_encoder=False,
            embedding_dim=32,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            encoder_kwargs={
                "use_center_only": True
            }
        )

        checkpoint = torch.load("./models/scout.pt", map_location='cpu')
        self.scout_policy.load_state_dict(checkpoint['model_state_dict'])
        self.scout_policy.eval()

        self.role = None  # 'scout' or 'guard'
        self.recon_map = Map()
        self.pathfinder = Pathfinder(
            self.recon_map,
            PathfinderConfig(
                use_viewcone = False,
            )
        )
                
        self.scout = Inference(
            self.scout_policy, 
            self.recon_map,
            strategy="greedy",
            top_k=5,
            temperature=0.5,
            action_dim=ACTION_DIM
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
        self.role = 'scout' if observation.get('scout', 0) == 1 else 'guard'

        # For guard, initialize the map
        if self.role == 'guard':
            self.recon_map.create_trajectory_tree(Point(0, 0))

        self.initialized = True
        print(f"Initialized as {'SCOUT' if self.role == 'scout' else 'GUARD'}")

    def _preprocess_observation(self, observation: dict[str, Any]):
        """Preprocess observation data types."""
        observation['viewcone'] = np.array(observation['viewcone'], dtype=np.uint8)
        observation['location'] = np.array(observation['location'], dtype=np.uint8)
        observation['direction'] = int(observation['direction'])

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
        current_step = observation.get('step', 0)
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
        if self.role == 'scout':
            return self.scout(observation)
        else:
            location = observation['location']
            direction = observation['direction']
            
            action = self.pathfinder.get_optimal_action(
                Point(location[0], location[1]),
                Direction(direction),
                tree_index=0
            )
            
            return int(action)
