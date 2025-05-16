from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray

from ..utils import Direction, Action, Point, Tile
from ..utils.geometry import POINT_EQ_LOOKUP
from ..utils.pathfinding import manhattan_distance, find_path, get_directional_neighbors
from ..node import NodeRegistry, DirectionalNode


# @lru_cache(maxsize=2**16)
def get_trajectory_hash(root, *route):
    return hash((root, *route))

class Trajectory:
    def __init__(self, root_node: DirectionalNode, step: int):
        self.head: DirectionalNode = root_node
        self.route: list[Action] = []
        self.nodes: list[DirectionalNode] = [self.head]
        self.position_cache: set[Point] = {root_node.position}  # Cache positions for faster lookups

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self.pruned: bool = False
        self.discarded: bool = False # if it's valid and was not (yet) pruned, but discarded by  TrajectoryTree.step()

        self.created_at: int = step

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: dict[Action, 'Trajectory'] = {}

        self._hash: Optional[int] = None

    @classmethod
    def from_points(cls, points, registry, step=-1, start_direction=Direction.RIGHT):
        """
        Create a trajectory that passes through all given points using the fewest number of actions.
        Points can be visited in any order to minimize the total number of actions.

        Args:
            points: List of Point objects to visit in any order
            registry: NodeRegistry to use for node creation
            start_direction: Initial direction the agent is facing (default: Direction.RIGHT)

        Returns:
            Trajectory: A trajectory that visits all points with minimal actions
        """
        if not points:
            raise ValueError("Points list cannot be empty")

        # Remove duplicates while preserving order
        unique_points = []
        seen = set()
        for point in points:
            point_key = (point.x, point.y)
            if point_key not in seen:
                seen.add(point_key)
                unique_points.append(point)

        # If only one point is provided, create and return a simple trajectory
        if len(unique_points) == 1:
            root_node = registry.get_node(unique_points[0], start_direction)
            return cls(root_node, step)

        # Use nearest neighbor algorithm with a twist - try each point as a starting point
        best_trajectory = None
        min_actions = float('inf')

        for start_idx, start_point in enumerate(unique_points):
            # Create root node at this starting point
            root_node = registry.get_node(start_point, start_direction)
            trajectory = cls(root_node, step)

            current_pos = start_point
            current_dir = start_direction

            # Points to visit (excluding the starting point)
            to_visit = unique_points.copy()
            to_visit.pop(start_idx)

            # Use nearest neighbor algorithm to visit remaining points
            while to_visit:
                # Find the point that requires the fewest actions to reach
                best_next_idx = -1
                best_actions: list[Action] = []
                fewest_actions = float('inf')

                for i, next_point in enumerate(to_visit):
                    actions: list[Action] = cls._find_shortest_path(current_pos, current_dir, next_point)
                    if len(actions) < fewest_actions:
                        fewest_actions = len(actions)
                        best_next_idx = i
                        best_actions = actions

                # Move to the best next point
                for action in best_actions:
                    trajectory.update(action, step)
                    if trajectory.invalid:
                        break

                if trajectory.invalid:
                    break

                # Update current position and direction
                current_node = trajectory.tail
                current_pos = current_node.position
                current_dir = current_node.direction

                # Remove this point from the to-visit list
                to_visit.pop(best_next_idx)

            # If this is the best trajectory so far, save it
            if not trajectory.invalid and len(trajectory.route) < min_actions:
                min_actions = len(trajectory.route)
                best_trajectory = trajectory

        # If we couldn't find a valid trajectory, return the first attempt
        if best_trajectory is None:
            root_node = registry.get_node(unique_points[0], start_direction)
            return cls(root_node, step)

        return best_trajectory

    @staticmethod
    def _find_shortest_path(start_pos, start_dir, target_pos):
        """
        Find the shortest sequence of actions to move from start position and direction to target position.

        Args:
            start_pos: Starting Point
            start_dir: Starting Direction
            target_pos: Target Point to reach

        Returns:
            list[Action]: Sequence of actions that leads to the target with minimal steps
        """
        # If already at the target, return empty list
        if POINT_EQ_LOOKUP[start_pos.x, target_pos.x, start_pos.y, target_pos.y]:
            return []

        # Define a state as (position, direction)
        start_state = (start_pos, start_dir)

        # Define the goal check function
        def is_goal(state):
            pos, _ = state
            return pos.x == target_pos.x and pos.y == target_pos.y

        # Define the neighbor function
        def get_neighbors(state):
            return get_directional_neighbors(state)

        # Define the heuristic function
        def heuristic(state):
            pos, _ = state
            return manhattan_distance(pos, target_pos)

        # Define hash function for states
        def state_hash(state):
            pos, direction = state
            return hash((pos.x, pos.y, direction))

        # Find path using A* algorithm
        result = find_path(
            start_node=start_state,
            is_goal=is_goal,
            get_neighbors=get_neighbors,
            heuristic=heuristic,
            node_hash=state_hash
        )

        return result.actions if result.success else []

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = get_trajectory_hash(self.head, *self.route)
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False
        return self.__hash__() == hash(other)

    @lru_cache(maxsize=1024+256)
    def __contains__(self, item: DirectionalNode | Point) -> bool:
        """
        Check if a DirectionalNode or Point is in this trajectory.

        Args:
            item: Either a DirectionalNode or a Point object

        Returns:
            bool: True if the node or position is in this trajectory, False otherwise
        """
        if self.invalid:
            return False

        if isinstance(item, DirectionalNode):
            return item in self.nodes

        if isinstance(item, Point):
            return item in self.position_cache

        return False

    def copy(self):
        """Create a deep copy of this trajectory with the same root but new lists."""
        new_trajectory = Trajectory(self.head, self.created_at)
        new_trajectory.route = self.route[:]
        new_trajectory.nodes = self.nodes[:]
        new_trajectory.position_cache = self.position_cache.copy()
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        return new_trajectory

    def update(self, action: Action, step: int):
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)
        self.created_at = step

        # Check if this action is valid
        self.get_last_node()

    def prune(self):
        self.pruned = True

    def invalidate(self, propagate=False):
        self.invalid = True
        if propagate:
            self._mark_as_invalid()

    def _mark_as_invalid(self, invalid_action_idx: Optional[int] = None):
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
                    self._inherited_from._mark_as_invalid(invalid_action_idx)
            else:
                return
        else:
            self.invalid = True

        # Propagate to all trajectories that inherit from this one
        for traj in self._inherits_to.values():
            traj._mark_as_invalid()

    def get_last_node(self):
        """
        Get the last node in the trajectory.

        Returns:
            DirectionalNode: The last node in the trajectory, or None if invalid.
        """
        if self.invalid:
            return None

        populate_nodes = (len(self.nodes) - 1) != len(self.route)
        if populate_nodes:
            self.nodes = [self.head]
            # Reset position cache when repopulating nodes
            self.position_cache = {self.head.position}

        current_node = self.head
        for i, action in enumerate(self.route):
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                self._mark_as_invalid(i)  # Mark invalid at the specific action index
                return None

            if populate_nodes:
                self.nodes.append(current_node)
                # Add position to cache for faster lookups
                self.position_cache.add(current_node.position)

        return current_node

    @property
    def tail(self) -> DirectionalNode:
        return self.nodes[-1]

    def get_new_trajectories(self, step: int, max_backtrack: Optional[int] = None):
        """
        Get new trajectories by exploring all valid actions from the current endpoint.

        Args:
            max_backtrack: Maximum number of backtracking steps allowed.

        Returns:
            List[Trajectory]: List of new trajectories, each extending from this one
        """
        if self.invalid:
            return []

        # last_node = self.get_last_node()
        last_node = self.tail
        if not last_node:
            return []

        if max_backtrack is not None:
            if self.has_backtrack(max_backtrack):
                return []

        new_trajectories = []

        # Explore all valid actions from current node
        for action, next_node in last_node.children.items():
            if action not in self._inherits_to:
                # Create a new trajectory with this action
                new_trajectory = self.copy()
                new_trajectory.update(action, step)

                # Skip if this made the trajectory invalid
                if new_trajectory.invalid:
                    continue

                self._inherits_to[action] = new_trajectory

                new_trajectories.append(new_trajectory)

        return new_trajectories

    def get_endpoint_key(self, consider_direction: bool = True):
        """
        Get a key for this trajectory's endpoint, used for deduplication.

        When consider_direction=True, the key is (position, direction),
        otherwise it's just the position.

        Args:
            consider_direction: Whether to include direction in the key.

        Returns:
            Tuple containing the key for deduplication
        """
        if self.invalid:
            return None

        last_node = self.tail
        # last_node = self.get_last_node()
        if not last_node:
            return None

        if consider_direction:
            return (last_node.position, last_node.direction)
        else:
            return (last_node.position,)

    def __str__(self):
        """String representation of the trajectory."""
        if self.invalid:
            return f"Invalid Trajectory: {self.route}"

        last_node = self.tail
        if last_node:
            return f"Trajectory to {last_node.position} {last_node.direction} from {self.head.position} {self.head.direction} via {self.route}, created at: {self.created_at}"
        return f"Incomplete Trajectory: {self.route}"

    def __repr__(self):
        return self.__str__()

    def has_backtrack(
        self,
        max_backtrack: int = 3,
        consider_direction: bool = False
    ) -> bool:
        """
        Check if this trajectory backtracks more than the allowed number of steps.

        Args:
            max_backtrack: Maximum number of backtracking steps allowed.

        Returns:
            bool: True if trajectory has too much backtracking.
        """
        if not self.nodes or len(self.nodes) <= 1:
            return False  # Need at least 2 nodes to backtrack

        visited = set()
        backtrack_count = 0

        for node in self.nodes:
            current = node if consider_direction else node.position

            if current not in visited:
                visited.add(current)
            else:
                backtrack_count += 1
                if backtrack_count > max_backtrack:
                    return True

        return False

    @property
    def parent(self):
        return self._inherited_from

    @property
    def children(self):
        return self._inherits_to.values()

    @property
    def to_delete(self):
        return self.invalid or self.pruned
