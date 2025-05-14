"""Manages the RL model."""

# import sys
# import os
# import pathlib
import random
import numpy as np

# Add paths for imports
# sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
# sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from grid.map import Map
from grid.utils import Point
from grid.map import Direction


class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.role = None  # 'scout' or 'guard'
        self.recon_map = Map()
        self.initialized = False
        self.last_step = -1
        # Seed for reproducibility
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)

    def rl(self, observation: dict[str, int | list[int]]) -> int:
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
            # Scout uses a semi-intelligent exploration strategy
            # Prefer unexplored directions when available
            location = observation.get('location', [0, 0])
            direction = observation.get('direction', 0)
            visible = observation.get('visible', [])

            # Default to random action if we can't make a better decision
            action = random.choice([0, 1, 2, 3])

            # If we have visibility information, try to move to unexplored areas
            if visible and isinstance(visible, list):
                # Try to avoid walls and prefer unexplored territory
                # This is a simple strategy that can be further enhanced
                action = self._get_scout_action(location, direction, visible)

            return action
        else:
            # Guard uses the map for navigation
            try:
                location = observation['location']
                direction = observation['direction']

                if not isinstance(location, list):
                    location = [0, 0]

                # Update the map with the current observation
                self.recon_map(observation)

                # Get optimal action from the map
                return int(self.recon_map.get_optimal_action(
                    Point(location[0], location[1]),
                    Direction(direction), 0))
            except Exception as e:
                print(f"Error in guard logic: {e}")
                # Fallback to random action if there's an error
                return random.choice([0, 1, 2, 3])

    def _get_scout_action(self, location, direction, visible):
        """Helper method to determine scout action based on visibility."""
        # Direction mappings: 0=right, 1=down, 2=left, 3=up
        # Simple strategy: prefer directions with fewer visible walls
        # Could be enhanced with more sophisticated exploration algorithms

        # Count obstacles in each direction
        forward_obstacles = 0
        left_obstacles = 0
        right_obstacles = 0

        # Process visibility data (implementation depends on exact format)
        # This is a placeholder - should be adapted to match the actual visibility format

        # For now, just return a random action
        return random.choice([0, 1, 2, 3])
