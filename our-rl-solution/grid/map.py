import numpy as np

from .utils import (
    Direction,
    Wall,
    Action,
    Point,
    rotate_wall_bits,
    view_to_world
)
from .tree import NodeRegistry, DirectionalNode, TrajectoryTree


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

        # Initialize node registry and populate nodes
        self.registry = NodeRegistry(self.size)
        self._populate_nodes()

    def _populate_nodes(self):
        """Populate all possible nodes in the grid (assuming no walls)."""
        # Create nodes for all positions and directions
        for y in range(self.size):
            for x in range(self.size):
                for direction in Direction:
                    self.registry.get_or_create_node(Point(x, y), direction)

    def get_node(self, coord: Point, direction: Direction) -> DirectionalNode:
        """
        Get a node from the registry.

        Args:
            x, y: Coordinates of the position
            direction: Direction the agent is facing

        Returns:
            DirectionalNode: The node at the specified position and direction
        """
        return self.registry.get_or_create_node(coord, direction)

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

        # Track which cells have wall updates
        updated_cells = []

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

                    # Check if wall information has changed
                    old_value = self.map[x, y]
                    if old_value != rotated_tile_value:
                        # Wall configuration might have changed
                        updated_cells.append((x, y, rotated_tile_value))

                    # Update map with tile information
                    self.map[x, y] = rotated_tile_value
                    self.visited[x, y] = True
                    self.last_updated[x, y] = self.step_counter  # Record when this cell was updated

        # Update node connections for cells with updated wall information
        for x, y, tile_value in updated_cells:
            self._update_node_connections(x, y, tile_value)

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

    def _update_node_connections(self, x, y, tile_value):
        """
        Update node connections based on wall information.

        Args:
            x, y: Coordinates of the updated cell
            tile_value: Updated tile value containing wall information
        """
        # Extract wall information as a dictionary for easier access
        walls = {
            'right': (tile_value & (1 << Wall.RIGHT)) > 0,
            'bottom': (tile_value & (1 << Wall.BOTTOM)) > 0,
            'left': (tile_value & (1 << Wall.LEFT)) > 0,
            'top': (tile_value & (1 << Wall.TOP)) > 0
        }

        position = Point(x, y)

        # Define a mapping from direction + action to the wall that would block it
        # Format: {direction: {action: wall_location}}
        blocking_walls = {
            Direction.RIGHT: {
                Action.FORWARD: 'right',
                Action.BACKWARD: 'left',
                Action.LEFT: 'top',
                Action.RIGHT: 'bottom'
            },
            Direction.DOWN: {
                Action.FORWARD: 'bottom',
                Action.BACKWARD: 'top',
                Action.LEFT: 'right',
                Action.RIGHT: 'left'
            },
            Direction.LEFT: {
                Action.FORWARD: 'left',
                Action.BACKWARD: 'right',
                Action.LEFT: 'bottom',
                Action.RIGHT: 'top'
            },
            Direction.UP: {
                Action.FORWARD: 'top',
                Action.BACKWARD: 'bottom',
                Action.LEFT: 'left',
                Action.RIGHT: 'right'
            }
        }

        # Update each directional node at this position
        for direction in Direction:
            node = self.registry.get_or_create_node(position, direction)

            # Determine which actions are blocked by walls
            invalid_actions = []

            # Check each action
            for action in [Action.FORWARD, Action.BACKWARD, Action.LEFT, Action.RIGHT]:
                # Get the wall location that would block this action from this direction
                wall_location = blocking_walls[direction][action]

                # If there's a wall in that location and the action exists in node children
                if walls[wall_location] and action in node.children:
                    invalid_actions.append(action)

            # Remove invalid actions from node children
            for action in invalid_actions:
                if action in node.children:
                    del node.children[action]

    def get_trajectory_tree(self, position, direction=None):
        """
        Create a trajectory tree starting from a specific position and direction.

        Args:
            position: Tuple (x, y) or Point representing the starting position
            direction: Optional starting direction (if None, will create trees for all directions)

        Returns:
            TrajectoryTree: A trajectory tree populated with valid paths
        """
        # Convert tuple to Point if necessary
        if isinstance(position, tuple):
            position = Point(position[0], position[1])

        # Create a trajectory tree with the current map's registry
        tree = TrajectoryTree(position, direction, self.size)
        tree.registry = self.registry  # Use the map's registry with wall information

        return tree
