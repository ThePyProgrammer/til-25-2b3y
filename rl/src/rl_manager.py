"""Manages the RL model."""

# import sys
# import os
# import pathlib
import random
from typing import Any

import numpy as np

# Add paths for imports
# sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
# sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from grid.map import Map
from grid.utils import Point
from grid.map import Direction
from grid.pathfinder import Pathfinder, PathfinderConfig


class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.role = None  # 'scout' or 'guard'
        self.recon_map = Map()

        self.pathfinder = Pathfinder(
            self.recon_map,
            PathfinderConfig(
                use_viewcone = False,
            )
        )

        self.initialized = False
        self.last_step = -1

        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)

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
                self.initialized = False
                self.recon_map = Map()

            self.last_step = current_step

        observation['viewcone'] = np.array(observation['viewcone'], dtype=np.uint8)
        observation['location'] = np.array(observation['location'], dtype=np.uint8)
        observation['direction'] = int(observation['direction'])

        # Determine role on first call
        if not self.initialized:
            self.role = 'scout' if observation.get('scout', 0) == 1 else 'guard'
            # For guard, initialize the map
            if self.role == 'guard':
                self.recon_map.create_trajectory_tree(Point(0, 0))
            self.initialized = True
            # Log initialization
            print(f"Initialized as {'SCOUT' if self.role == 'scout' else 'GUARD'}")

        # Different logic for scout and guard
        if self.role == 'scout':
            location = observation['location']
            direction = observation['direction']

            action = random.choice([0, 1, 2, 3])

            return action
        else:
            # Guard uses the map for navigation
            try:
                location = observation['location']
                direction = observation['direction']

                self.recon_map(observation)

                action = self.pathfinder.get_optimal_action(
                    Point(location[0], location[1]),
                    Direction(direction),
                    tree_index=0
                )
                return int(action)
            except Exception as e:
                print(f"Error in guard logic: {e}")
                # Fallback to random action if there's an error
                return random.choice([0, 1, 2, 3])
