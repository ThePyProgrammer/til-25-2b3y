import numpy as np
from enum import IntEnum


class Direction(IntEnum):
    """Direction enum with values matching the environment."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class Wall:
    """Wall bit positions."""
    RIGHT = 4  # Bit 4, value 16
    BOTTOM = 5  # Bit 5, value 32
    LEFT = 6   # Bit 6, value 64
    TOP = 7    # Bit 7, value 128


def rotate_wall_bits(tile_value, direction):
    """
    Rotates the wall bits based on the agent's direction to maintain global orientation.

    Args:
        tile_value (int): The observed tile value
        direction (int): Direction the agent is facing (0-3)

    Returns:
        int: Tile value with rotated wall bits
    """
    # Extract different parts of the tile value
    item_bits = tile_value & 0b11  # Last 2 bits (tile type)
    agent_bits = tile_value & 0b1100  # Bits 2-3 (agent type)
    wall_bits = tile_value & 0b11110000  # Bits 4-7 (walls)

    if direction == Direction.RIGHT:
        # No rotation needed for RIGHT direction (0)
        return tile_value

    # Extract individual wall bits
    right_wall = 1 if wall_bits & (1 << Wall.RIGHT) else 0
    bottom_wall = 1 if wall_bits & (1 << Wall.BOTTOM) else 0
    left_wall = 1 if wall_bits & (1 << Wall.LEFT) else 0
    top_wall = 1 if wall_bits & (1 << Wall.TOP) else 0

    # Initialize rotated walls
    rotated_right = right_wall
    rotated_bottom = bottom_wall
    rotated_left = left_wall
    rotated_top = top_wall

    if direction == Direction.DOWN:
        # Rotate clockwise once
        rotated_right = top_wall
        rotated_bottom = right_wall
        rotated_left = bottom_wall
        rotated_top = left_wall
    elif direction == Direction.LEFT:
        # Rotate clockwise twice
        rotated_right = left_wall
        rotated_bottom = top_wall
        rotated_left = right_wall
        rotated_top = bottom_wall
    elif direction == Direction.UP:
        # Rotate clockwise three times
        rotated_right = bottom_wall
        rotated_bottom = left_wall
        rotated_left = top_wall
        rotated_top = right_wall

    # Combine the rotated wall bits with the original agent and item bits
    rotated_wall_bits = (
        (rotated_right << Wall.RIGHT) |
        (rotated_bottom << Wall.BOTTOM) |
        (rotated_left << Wall.LEFT) |
        (rotated_top << Wall.TOP)
    )

    return item_bits | agent_bits | rotated_wall_bits


def view_to_world(agent_loc, agent_dir, view_coord):
    """
    Maps viewcone coordinate to world coordinate

    Args:
        agent_loc (np.ndarray): Agent's location in world coordinates
        agent_dir (int): Direction the agent is facing (0-3)
        view_coord (np.ndarray): Viewcone coordinate relative to agent

    Returns:
        np.ndarray: World coordinate
    """
    agent_dir = Direction(agent_dir)

    if agent_dir == Direction.RIGHT:
        return agent_loc + view_coord
    elif agent_dir == Direction.DOWN:
        return agent_loc - np.array((view_coord[1], -view_coord[0]))
    elif agent_dir == Direction.LEFT:
        return agent_loc - view_coord
    else:  # Direction.UP
        return agent_loc + np.array((view_coord[1], -view_coord[0]))


class Map:
    # Tile type constants
    EMPTY = 1
    RECON = 2
    MISSION = 3

    def __init__(self):
        """Initialize an empty map of the environment."""
        self.size = 16  # Assume a 16x16 environment
        self.map = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        self.last_updated = np.zeros((self.size, self.size), dtype=np.int32)  # Timestamp for last update
        self.step_counter = 0  # Count of steps/updates to use as timestamp

    def __call__(self, observation):
        """Update the map with a new observation."""
        return self.update(observation)

    def update(self, observation):
        """Update the map with a new observation.

        Args:
            observation: Dict containing 'viewcone', 'direction', 'location', etc.

        Returns:
            Updated map
        """
        viewcone = observation['viewcone']
        direction = observation['direction']
        agent_loc = observation['location']

        # Increment step counter
        self.step_counter += 1

        # Viewcone is 7x5 with agent at (2, 2)
        for i in range(viewcone.shape[0]):
            for j in range(viewcone.shape[1]):
                tile_value = viewcone[i, j]

                # Skip tiles with no vision (value 0 or last 2 bits are 0)
                if tile_value == 0 or (tile_value & 0b11) == 0:
                    continue

                # Skip empty tiles (last 2 bits = 1)
                if (tile_value & 0b11) == self.EMPTY:
                    continue

                # Convert viewcone coordinates to world coordinates
                # Agent is at position (2, 2) in the viewcone
                view_coord = np.array([i - 2, j - 2])  # Offset from agent position in viewcone
                world_coord = view_to_world(agent_loc, direction, view_coord)

                # Convert coordinates to integers to use as array indices
                x, y = int(world_coord[0]), int(world_coord[1])

                # Check if coordinates are within bounds
                if (0 <= x < self.size and 0 <= y < self.size and tile_value != 0):
                    # Rotate the wall bits to maintain global orientation
                    rotated_tile_value = rotate_wall_bits(tile_value, direction)
                    
                    # Clear agent bits when processing the agent's own position in viewcone
                    if i == 2 and j == 2:
                        # Clear bits 2-3 (agent bits) but keep all other bits
                        rotated_tile_value = rotated_tile_value & ~0b1100

                    # Update map with tile information
                    self.map[x, y] = rotated_tile_value
                    self.visited[x, y] = True
                    self.last_updated[x, y] = self.step_counter  # Record when this cell was updated

        return self.map

    def get_walls(self):
        """
        Extract wall information from the map.

        Returns:
            np.ndarray: Array of shape (size, size, 4) containing wall information.
                        The last dimension corresponds to [right, bottom, left, top] walls.
        """
        walls = np.zeros((self.size, self.size, 4), dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if self.visited[y, x]:
                    tile_value = self.map[y, x]
                    # Extract wall bits: [right, bottom, left, top]
                    walls[y, x, 0] = (tile_value & (1 << Wall.RIGHT)) > 0
                    walls[y, x, 1] = (tile_value & (1 << Wall.BOTTOM)) > 0
                    walls[y, x, 2] = (tile_value & (1 << Wall.LEFT)) > 0
                    walls[y, x, 3] = (tile_value & (1 << Wall.TOP)) > 0

        return walls

    def get_tile_type(self):
        """
        Extract tile type information (empty, recon, mission) from the map.

        Returns:
            np.ndarray: Array of shape (size, size) containing tile type information.
        """
        tile_types = np.zeros((self.size, self.size), dtype=np.uint8)

        for y in range(self.size):
            for x in range(self.size):
                if self.visited[y, x]:
                    # Extract the last 2 bits for tile type
                    tile_types[y, x] = self.map[y, x] & 0b11

        return tile_types

    def get_agents(self):
        """
        Extract agent information (scouts and guards) from the map.

        Returns:
            tuple: Two arrays of shape (size, size) for scouts and guards.
        """
        scouts = np.zeros((self.size, self.size), dtype=bool)
        guards = np.zeros((self.size, self.size), dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if self.visited[y, x]:
                    tile_value = self.map[y, x]
                    # Bit 2 for scouts
                    scouts[y, x] = (tile_value & (1 << 2)) > 0
                    # Bit 3 for guards
                    guards[y, x] = (tile_value & (1 << 3)) > 0

        return scouts, guards

    def get_visited(self):
        """
        Return the visited locations.

        Returns:
            np.ndarray: Boolean array of shape (size, size) for visited locations.
        """
        return self.visited

    @property
    def time_since_update(self):
        """
        Calculate how many steps have passed since each cell was last updated.

        Returns:
            np.ndarray: Array of shape (size, size) with the number of steps since last update.
                       Cells that have never been updated will have a large value.
        """
        # For cells that have never been updated (last_updated is 0),
        # we'll use a large value to indicate "not updated"
        result = np.ones((self.size, self.size), dtype=np.int32) * (self.step_counter + 1)

        # For visited cells, calculate the actual time difference
        mask = self.last_updated > 0
        result[mask] = self.step_counter - self.last_updated[mask]

        return result
