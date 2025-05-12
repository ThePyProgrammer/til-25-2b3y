from dataclasses import dataclass
from typing import Optional

import numpy as np
from enum import IntEnum
from .map import Map, Direction

class Action(IntEnum):
    """Action enum with values matching the environment."""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

def get_hash(x, y, direction):
    return hash(f"{x}-{y}-{direction}")

class DirectionalNode:
    """A node in the path tree that tracks both position and direction."""

    def __init__(self, coord: Point, direction: Direction):
        self.coord = coord
        self.direction = direction
        self.children: dict[Action, 'DirectionalNode'] = {}

    def _get_next_state(self, action: Action):
        """
        Calculate the next position and direction based on an action.

        Args:
            action: The action to take (FORWARD, BACKWARD, LEFT, RIGHT, STAY)

        Returns:
            tuple: (next_coord, next_direction)
        """
        # Movement vectors for each direction: (dx, dy)
        movement_vectors = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1)
        }

        # Direction changes for turns
        direction_changes = {
            Action.LEFT: {  # Left turns
                Direction.RIGHT: Direction.UP,
                Direction.DOWN: Direction.RIGHT,
                Direction.LEFT: Direction.DOWN,
                Direction.UP: Direction.LEFT
            },
            Action.RIGHT: {  # Right turns
                Direction.RIGHT: Direction.DOWN,
                Direction.DOWN: Direction.LEFT,
                Direction.LEFT: Direction.UP,
                Direction.UP: Direction.RIGHT
            }
        }

        # Start with current position and direction
        next_coord = Point(self.coord.x, self.coord.y)
        next_direction = self.direction

        if action == Action.STAY:
            # No change in position or direction
            pass

        elif action == Action.FORWARD:
            # Move forward in current direction
            dx, dy = movement_vectors[self.direction]
            next_coord.x += dx
            next_coord.y += dy

        elif action == Action.BACKWARD:
            # Move backward (opposite of current direction)
            dx, dy = movement_vectors[self.direction]
            next_coord.x -= dx
            next_coord.y -= dy

        elif action in (Action.LEFT, Action.RIGHT):
            # Turn and move in the new direction
            next_direction = direction_changes[action][self.direction]
            dx, dy = movement_vectors[next_direction]
            next_coord.x += dx
            next_coord.y += dy

        return next_coord, next_direction

    def populate_possible_children(self, node_lookup_table: 'DirectionalNodeTable'):
        """
        Populate the children dictionary with nodes that can be reached from this node.
        Does NOT take into account walls so those actions will have to be pruned.

        Args:
            node_lookup_table: Table containing all possible nodes in the environment
        """
        self.children = {}

        for action in Action:
            # Calculate next position and direction
            next_coord, next_direction = self._get_next_state(action)

            # Apply boundary constraints
            next_coord.x = max(0, min(next_coord.x, node_lookup_table.size - 1))
            next_coord.y = max(0, min(next_coord.y, node_lookup_table.size - 1))

            # if the move is effectively STAY, then don't add it.
            if (
                next_coord.x == self.coord.x
                and next_coord.y == self.coord.y
                and next_direction == self.direction
            ):
                continue

            # Look up the next node in the table
            node_hash = get_hash(next_coord.x, next_coord.y, next_direction)
            next_node = node_lookup_table.nodes.get(node_hash)

            if next_node:
                self.children[action] = next_node

    def __str__(self):
        dir_names = {
            Direction.RIGHT: "→",
            Direction.DOWN: "↓",
            Direction.LEFT: "←",
            Direction.UP: "↑"
        }
        return f"{self.coord} {dir_names[self.direction]})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return get_hash(self.coord.x, self.coord.y, self.direction)

    def is_same_state(self, coord, direction):
        """Check if this node has the same state (position and direction)"""
        return self.coord == coord and self.direction == direction

class DirectionalNodeTable:
    def __init__(self, size: int = 16):
        self.size = size
        self.nodes: dict[int, DirectionalNode] = {}

        # Create nodes for all positions and directions
        for i in range(self.size):
            for j in range(self.size):
                for d in Direction:
                    node = DirectionalNode(Point(i, j), d)
                    self.nodes[hash(node)] = node

    def __getitem__(self, key):
        """Get a node by coord and direction tuple."""
        if isinstance(key, tuple) and len(key) == 2:
            coord, direction = key
            return self.get_node(coord, direction)
        raise KeyError("Expected a tuple of (coord, direction)")

    def get_node(self, coord: Point, direction: Direction):
        """Get a node by separate coord and direction arguments."""
        return self.nodes[get_hash(coord.x, coord.y, direction)]

class Trajectory:
    def __init__(self, root_node):
        self.root: DirectionalNode = root_node
        self.route: list[Action] = []

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: list['Trajectory'] = []

    def copy(self):
        """Create a copy of this trajectory with the same root but a new route list."""
        new_trajectory = Trajectory(self.root)
        new_trajectory.route = self.route.copy()
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        self._inherits_to.append(new_trajectory)
        return new_trajectory

    def update(self, action):
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)

        # Check if this action is valid
        current_node = self.get_last_node()
        if current_node and action not in current_node.children:
            self.mark_as_invalid()

    def mark_as_invalid(self, invalid_action_idx: Optional[int] = None):
        """
        Mark this trajectory as invalid and propagate to inheriting trajectories.

        Args:
            invalid_action_idx: The index of the action in the route that is invalid.
                                If None, the entire trajectory is marked invalid.
        """
        # avoid propagating if already done.
        if self.invalid:
            return

        # Store the index of the invalid action
        if invalid_action_idx is not None:
            if invalid_action_idx < len(self.route):
                self.invalid = True
                self.invalid_action_idx = invalid_action_idx

                # If this has a parent trajectory, check if the parent should be invalidated
                if self._inherited_from is not None:
                    self._inherited_from.mark_as_invalid(invalid_action_idx)
            else:
                return
        else:
            self.invalid = True

        # Propagate to all trajectories that inherit from this one
        for trajectory in self._inherits_to:
            trajectory.mark_as_invalid(invalid_action_idx)

    def get_last_node(self):
        """
        Get the last node in the trajectory.

        Returns:
            DirectionalNode: The last node in the trajectory, or None if invalid.
        """
        if self.invalid:
            return None

        current_node = self.root
        for i, action in enumerate(self.route):
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                self.mark_as_invalid(i)  # Mark invalid at the specific action index
                return None
        return current_node

    def get_new_trajectories(self) -> list['Trajectory']:
        """
        Generate all possible new trajectories from the current state.

        Returns:
            A list of new Trajectory objects, one for each possible action
            from the current end node. Returns empty list if trajectory is invalid.
        """
        if self.invalid:
            return []

        new_trajectories = []

        # Get the current end node
        current_node = self.get_last_node()

        if not current_node:
            return []

        # Create new trajectories for each possible action
        for action, child_node in current_node.children.items():
            # Create a new trajectory
            new_trajectory = self.copy()
            # Add the new action to the route
            new_trajectory.update(action)
            new_trajectories.append(new_trajectory)

        return new_trajectories

class TrajectoryTree:
    def __init__(self, init_coord: Point, init_direction: Direction = Direction.RIGHT, size: int = 16):
        """
        Initialize a TrajectoryTree with the agent's starting position and direction.

        Args:
            init_coord: Starting coordinate
            init_direction: Starting direction (default: RIGHT)
            size: Size of the grid environment (default: 16)
        """
        self.size = size
        self.node_lookup_table: DirectionalNodeTable = DirectionalNodeTable(size)

        # Initialize the node lookup table by populating children for each node
        for node in self.node_lookup_table.nodes.values():
            node.populate_possible_children(self.node_lookup_table)

        # Create the root node
        self.root_node = self.node_lookup_table.get_node(init_coord, init_direction)

        # Initialize with a single trajectory starting from the root
        self.trajectories: list[Trajectory] = [Trajectory(self.root_node)]

    # def expand_trajectories(self, max_depth: int = None):
    #     """
    #     Expand all trajectories to generate new possible paths.

    #     Args:
    #         max_depth: Maximum depth/length of trajectories to consider (None for unlimited)

    #     Returns:
    #         Number of new trajectories added
    #     """
    #     if not self.trajectories:
    #         return 0

    #     new_trajectories = []
    #     for traj in self.trajectories:
    #         # Skip trajectories that have reached max depth
    #         if max_depth is not None and len(traj.route) >= max_depth:
    #             continue

    #         # Generate new trajectories from this one
    #         new_trajs = traj.get_new_trajectories()
    #         new_trajectories.extend(new_trajs)

    #     # Store the new trajectories
    #     self.trajectories.extend(new_trajectories)
    #     return len(new_trajectories)

    # def get_terminal_nodes(self):
    #     """
    #     Get nodes that have no children (dead ends).

    #     Returns:
    #         List of DirectionalNodes that are terminal
    #     """
    #     terminal_nodes = []
    #     for node in self.node_lookup_table.nodes.values():
    #         if not node.children:
    #             terminal_nodes.append(node)
    #     return terminal_nodes

    # def find_trajectory_to(self, target_coord: Point, max_steps: int = 20):
    #     """
    #     Find a trajectory that reaches the target position.

    #     Args:
    #         target_coord: The target position to reach
    #         max_steps: Maximum number of steps to try before giving up

    #     Returns:
    #         Trajectory that reaches target, or None if not found
    #     """
    #     # Reset trajectories to just the starting point
    #     self.trajectories = [Trajectory(self.root_node)]

    #     for _ in range(max_steps):
    #         # Expand all trajectories
    #         new_count = self.expand_trajectories()
    #         if new_count == 0:
    #             # No more possible expansions
    #             break

    #         # Check if any trajectory reaches the target
    #         for traj in self.trajectories:
    #             if traj.invalid:
    #                 continue

    #             # Get the last node in this trajectory
    #             current_node = traj.get_last_node()
    #             if not current_node:
    #                 continue

    #             # Check if we've reached the target
    #             if current_node.coord == target_coord:
    #                 return traj

    #     return None

    # def get_shortest_trajectory(self, target_coords: list[Point]):
    #     """
    #     Find the shortest trajectory to any of the target coordinates.

    #     Args:
    #         target_coords: List of target coordinates to consider

    #     Returns:
    #         Shortest trajectory to any target, or None if none found
    #     """
    #     best_traj = None
    #     best_length = float('inf')

    #     for target in target_coords:
    #         traj = self.find_trajectory_to(target)
    #         if traj and not traj.invalid and len(traj.route) < best_length:
    #             best_traj = traj
    #             best_length = len(traj.route)

    #     return best_traj
