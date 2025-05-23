from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch

from .utils import (
    Direction,
    Action,
    Point,
    rotate_wall_bits,
    view_to_world,
    Tile,
    TileContent
)
from .node import NodeRegistry, DirectionalNode
from .trajectory import TrajectoryTree


def get_init_map_array(size):
    map = np.zeros((size, size), dtype=np.uint8)

    # Add walls along the edges
    # Left edge: Add left walls to all tiles in leftmost column
    map[0:size, 0] |= (1 << 6)  # Left wall (bit 6)
    # Right edge: Add right walls to all tiles in rightmost column
    map[0:size, size-1] |= (1 << 4)  # Right wall (bit 4)
    # Top edge: Add top walls to all tiles in top row
    map[0, 0:size] |= (1 << 7)  # Top wall (bit 7)
    # Bottom edge: Add bottom walls to all tiles in bottom row
    map[size-1, 0:size] |= (1 << 5)  # Bottom wall (bit 5)

    return map

class Map:
    EMPTY = TileContent.EMPTY
    RECON = TileContent.RECON
    MISSION = TileContent.MISSION

    def __init__(self, use_viewcone=False):
        """
        Initialize an empty map of the environment.

        Args:
            use_viewcone: Whether to use viewcone for action selection (default: True)
        """
        self.size = 16  # Assume a 16x16 environment
        self.map = get_init_map_array(self.size)
        self.viewed = np.zeros((self.size, self.size), dtype=bool)
        self.last_updated = np.zeros((self.size, self.size), dtype=np.int32)  # Timestamp for last update
        self.recently_updated = np.zeros((self.size, self.size), dtype=np.uint8)
        self.step_counter = 0  # Count of steps/updates to use as timestamp
        self.use_viewcone = use_viewcone  # Feature flag for viewcone-based action selection

        self.registry = NodeRegistry(self.size)
        self._populate_nodes()

        self.trees: list[TrajectoryTree] = []

    def _populate_nodes(self):
        """Populate all possible nodes in the grid (assuming no walls)."""
        # Create nodes for all positions and directions
        for y in range(self.size):
            for x in range(self.size):
                for direction in Direction:
                    self.registry.get_or_create_node(Point(x, y), direction)

    def get_node(self, position: Point, direction: Direction) -> DirectionalNode:
        """
        Get a node from the registry.

        Args:
            x, y: Coordinates of the position
            direction: Direction the agent is facing

        Returns:
            DirectionalNode: The node at the specified position and direction
        """
        return self.registry.get_or_create_node(position, direction)

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
        self.direction = observation['direction']
        self.agent_loc: NDArray = observation['location']

        # Increment step counter
        self.step_counter += 1
        self.recently_updated = np.zeros((self.size, self.size), dtype=np.uint8)

        updated_cells = []
        observed_cells = []

        # Viewcone is 7x5 with agent at (2, 2)
        for i in range(viewcone.shape[0]):
            for j in range(viewcone.shape[1]):
                tile_value = viewcone[i, j]

                # Create a Tile instance for easy property access
                tile = Tile(tile_value)

                # Convert viewcone coordinates to world coordinates
                # Agent is at position (2, 2) in the viewcone
                view_coord = np.array([i - 2, j - 2])  # Offset from agent position in viewcone
                world_coord = view_to_world(self.agent_loc, self.direction, view_coord)

                # Convert coordinates to integers to use as array indices
                x, y = int(world_coord[0]), int(world_coord[1])

                # Check if coordinates are within bounds
                if (0 <= x < self.size and 0 <= y < self.size):
                    # Create a Tile instance and rotate its walls to maintain global orientation
                    rotated_tile_value = rotate_wall_bits(tile_value, self.direction)

                    # Clear agent bits when processing the agent's own position in viewcone
                    if i == 2 and j == 2:
                        # Clear bits 2-3 (agent bits) but keep all other bits
                        rotated_tile_value = rotated_tile_value & 0b11110011  # Use explicit mask instead of ~0b1100

                    info = (Point(x, y), Tile(rotated_tile_value))
                    observed_cells.append(info)

                    # Skip tiles with no vision
                    if tile.is_visible or tile.has_scout or tile.has_guard:
                        # Check if wall information has changed
                        old_value = self.map[y, x]
                        if old_value != rotated_tile_value:
                            # Wall configuration might have changed
                            updated_cells.append(info)

                        existing_walls = old_value & 0b11110000  # Extract wall bits (4-7)
                        new_non_walls = rotated_tile_value & 0b00001111  # Extract non-wall bits (0-3)
                        new_walls = rotated_tile_value & 0b11110000  # Extract new wall bits

                        # Update map with tile information
                        self.map[y, x] = new_non_walls | (existing_walls | new_walls)
                        self.viewed[y, x] = True
                    self.last_updated[y, x] = self.step_counter  # Record when this cell was updated

        # Update node connections for cells with updated wall information
        for position, tile in updated_cells:
            self._update_node_connections(position, tile)

        for tree in self.trees:
            # tree.prune(observed_cells, before_step=True)
            tree.step()
            tree.prune(observed_cells)

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
                if self.viewed[y, x]:
                    # Use Tile utility to extract wall information
                    tile = Tile(self.map[y, x])
                    walls[y, x, 0] = tile.has_right_wall
                    walls[y, x, 1] = tile.has_bottom_wall
                    walls[y, x, 2] = tile.has_left_wall
                    walls[y, x, 3] = tile.has_top_wall

        return walls

    def get_tiles(self) -> list[list[Tile]]:
        """
        Extract tile type information (empty, recon, mission) from the map.

        Returns:
            np.ndarray: Array of shape (size, size) containing tile type information.
        """
        return map_to_tiles(self.map)

    def get_tile_type(self):
        """
        Extract tile type information (empty, recon, mission) from the map.

        Returns:
            np.ndarray: Array of shape (size, size) containing tile type information.
        """
        tile_types: list[list[Optional[TileContent]]] = [[None for _ in range(16)] for _ in range(16)]

        for x in range(self.size):
            for y in range(self.size):
                if self.viewed[y, x]:
                    # Use Tile utility to extract tile type
                    tile = Tile(self.map[y, x])
                    tile_types[x][y] = tile.tile_content

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
                if self.viewed[y, x]:
                    # Use Tile utility to extract agent information
                    tile = Tile(self.map[y, x])
                    scouts[y, x] = tile.has_scout
                    guards[y, x] = tile.has_guard

        return scouts, guards

    def get_guards(self):
        """
        Extract guard information from the map.

        Returns:
            tuple: Two arrays of shape (size, size) for guards.
        """
        guards = np.zeros((self.size, self.size), dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if self.viewed[y, x]:
                    # Use Tile utility to extract agent information
                    tile = Tile(self.map[y, x])
                    guards[y, x] = tile.has_guard

        return guards

    def get_visited(self):
        """
        Return the visited locations.

        Returns:
            np.ndarray: Boolean array of shape (size, size) for visited locations.
        """
        return self.viewed

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
        result = np.ones((self.size, self.size), dtype=np.int32) * (self.step_counter)

        # For visited cells, calculate the actual time difference
        mask = self.last_updated > 0
        result[mask] = result[mask] - self.last_updated[mask]

        return result

    def _update_node_connections(self, position: Point, tile: Tile):
        """
        Update node connections based on wall information.

        Args:
            position: Coordinates of the updated cell
            tile_value: Updated tile value containing wall information
        """
        # Extract wall information from the Tile object
        walls = {
            'right': tile.has_right_wall,
            'bottom': tile.has_bottom_wall,
            'left': tile.has_left_wall,
            'top': tile.has_top_wall
        }

        # Define a mapping from direction + action to the wall that would block it
        # Format: {direction: {action: wall_location}}
        blocking_walls = {
            Direction.RIGHT: {
                Action.FORWARD: 'right',
                Action.BACKWARD: 'left',
            },
            Direction.DOWN: {
                Action.FORWARD: 'bottom',
                Action.BACKWARD: 'top',
            },
            Direction.LEFT: {
                Action.FORWARD: 'left',
                Action.BACKWARD: 'right',
            },
            Direction.UP: {
                Action.FORWARD: 'top',
                Action.BACKWARD: 'bottom',
            }
        }

        # Update each directional node at this position
        for direction in Direction:
            node = self.registry.get_or_create_node(position, direction)

            # Determine which actions are blocked by walls
            invalid_actions = []

            # Check each action
            for action in [Action.FORWARD, Action.BACKWARD]:
                # Get the wall location that would block this action from this direction
                wall_location = blocking_walls[direction][action]

                # If there's a wall in that location and the action exists in node children
                if walls[wall_location] and action in node.children:
                    invalid_actions.append(action)

            # Remove invalid actions from node children
            for action in invalid_actions:
                if action in node.children:
                    invalid_child = node.children[action]
                    del node.children[action]

                    for tree in self.trees:
                        tree.check_wall_trajectories(invalid_child)

        # Define wall relationships (opposites and position offsets)
        wall_relationships = {
            'right': ('left', Point(1, 0)),   # If right wall here, there's a left wall at x+1,y
            'left': ('right', Point(-1, 0)),  # If left wall here, there's a right wall at x-1,y
            'bottom': ('top', Point(0, 1)),   # If bottom wall here, there's a top wall at x,y+1
            'top': ('bottom', Point(0, -1))   # If top wall here, there's a bottom wall at x,y-1
        }

        wall_bit_map = {
            'right': 4,   # bit 4
            'bottom': 5,  # bit 5
            'left': 6,    # bit 6
            'top': 7      # bit 7
        }

        # Update adjacent cells for walls that exist in the current cell
        for wall_dir, has_wall in walls.items():
            if has_wall and wall_dir in wall_relationships:
                # Get opposite wall direction and position offset
                opposite_wall, offset = wall_relationships[wall_dir]
                adjacent_pos = Point(position.x + offset.x, position.y + offset.y)

                # Only process if adjacent position is within grid bounds
                if 0 <= adjacent_pos.x < self.size and 0 <= adjacent_pos.y < self.size:
                    # Set the appropriate wall bit in the adjacent cell
                    self.map[adjacent_pos.y, adjacent_pos.x] |= (1 << wall_bit_map[opposite_wall])

                    # For each direction at the adjacent cell
                    for adj_direction in Direction:
                        adj_node = self.registry.get_or_create_node(adjacent_pos, adj_direction)

                        # Find and remove actions that would be blocked by this wall
                        for adj_action in list(adj_node.children.keys()):
                            # Check if this action would be blocked by the opposite wall
                            if blocking_walls[adj_direction].get(adj_action) == opposite_wall:
                                # Remove the blocked action
                                if adj_action in adj_node.children:
                                    del adj_node.children[adj_action]

    def create_trajectory_tree(self, position, direction=None, parallel=False):
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
        tree = TrajectoryTree(self, position, direction, self.size, registry=self.registry)

        self.trees.append(tree)

        return tree

    def get_tensor(self) -> torch.Tensor:
        """
        Returns a tensor representation of the map with multiple channels.
        The map is rotated so the agent is always facing right (direction 0).

        Args:
            observation: Optional dictionary containing agent's current state.
                         Should include 'location' and 'direction' keys.

        Returns:
            torch.Tensor: A tensor of shape (12, 31, 31) containing:
                - Channel 0: vision (0/1)
                - Channel 1: empty tiles (0/1)
                - Channel 2: recon tiles (0/1)
                - Channel 3: mission tiles (0/1)
                - Channel 4: scout (0/1)
                - Channel 5: guard (0/1)
                - Channel 6: top_wall (0/1)
                - Channel 7: bottom_wall (0/1)
                - Channel 8: left_wall (0/1)
                - Channel 9: right_wall (0/1)
                - Channel 10: time_since_updated (0-1)
                - Channel 11: step (0-1) normalised from 0-100 to 0-1
        """

        return tiles_to_tensor(
            self.get_tiles(),
            self.agent_loc,
            self.direction,
            self.size,
            self.time_since_update,
            self.step_counter
        )

def map_to_tiles(map: NDArray):
    return [[Tile(map[y, x]) for y in range(16)] for x in range(16)]

def to_centered_coords(x: int, y: int, agent_loc, center: int):
    # Calculate offsets to place agent at center
    x_offset = center - agent_loc[0]
    y_offset = center - agent_loc[1]

    # Apply offset to place at center of tensor
    new_x = x + x_offset
    new_y = y + y_offset

    return new_x, new_y

def tiles_to_tensor(
    tiles: list[list[Tile]],
    location: NDArray | tuple[int],
    direction: int | Direction, # Get current direction (0:right, 1:down, 2:left, 3:up)
    size: int,
    time_since_update: NDArray,
    step_num: int
):
    # Output tensor size
    tensor_size = 16 * 2 - 1
    center = tensor_size // 2  # Center position is 15 for a 31x31 tensor

    # Initialize tensor with zeros (12 channels)
    tensor = torch.zeros((12, tensor_size, tensor_size), dtype=torch.float32)

    for y in range(size):
        for x in range(size):
            tile = tiles[x][y]

            new_x, new_y = to_centered_coords(x, y, location, center)

            # Channels 0-3: no_vision, empty, recon, mission tiles
            if tile.is_visible:
                tensor[0, new_y, new_x] = 1.0

            if tile.is_empty:
                tensor[1, new_y, new_x] = 1.0
            elif tile.is_recon:
                tensor[2, new_y, new_x] = 1.0
            elif tile.is_mission:
                tensor[3, new_y, new_x] = 1.0

            # Channels 4-5: scout and guard
            if tile.has_scout:
                tensor[4, new_y, new_x] = 1.0
            elif tile.has_guard:
                tensor[5, new_y, new_x] = 1.0

            # Channel 6-9: walls (top, bottom, left, right)
            top_bit = 1 if tile.has_top_wall else 0
            right_bit = 1 if tile.has_right_wall else 0
            bottom_bit = 1 if tile.has_bottom_wall else 0
            left_bit = 1 if tile.has_left_wall else 0

            for turn in range(direction):
                left_bit, bottom_bit, right_bit, top_bit = top_bit, left_bit, bottom_bit, right_bit

            tensor[6, new_y, new_x] = top_bit
            tensor[7, new_y, new_x] = bottom_bit
            tensor[8, new_y, new_x] = left_bit
            tensor[9, new_y, new_x] = right_bit

            # Channel 10: last_updated (recently updated cells)
            tensor[10, new_y, new_x] = time_since_update[y, x] / 100.0

    # Channel 11: step
    normalized_step = step_num / 100.0
    tensor[11].fill_(normalized_step)

    # Rotate the entire tensor based on the agent's direction
    if direction == 1:  # down - rotate 90° counter-clockwise
        tensor = torch.rot90(tensor, k=1, dims=[1, 2])
    elif direction == 2:  # left - rotate 180°
        tensor = torch.rot90(tensor, k=2, dims=[1, 2])
    elif direction == 3:  # up - rotate 90° clockwise
        tensor = torch.rot90(tensor, k=3, dims=[1, 2])
    # direction 0 (right) - no rotation needed

    import matplotlib.pyplot as plt

    tensor[9][15][15] = 0.5

    plt.imshow(tensor[9])
    plt.show()

    return tensor
