from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray

from .utils import Direction, Action, Point, Tile
from .utils.geometry import POINT_EQ_LOOKUP
from .utils.pathfinding import manhattan_distance, find_path, get_directional_neighbors
from .node import NodeRegistry, DirectionalNode


# @lru_cache(maxsize=2**16)
def get_trajectory_hash(root, *route):
    return hash((root, *route))

class Trajectory:
    def __init__(self, root_node):
        self.head: DirectionalNode = root_node
        self.route: list[Action] = []
        self.nodes: list[DirectionalNode] = [self.head]
        self.position_cache: set[Point] = {root_node.position}  # Cache positions for faster lookups

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self.pruned: bool = False
        self.discarded: bool = False # if it's valid and was not (yet) pruned, but discarded by  TrajectoryTree.step()

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: dict[Action, 'Trajectory'] = {}

        self._hash: Optional[int] = None

    @classmethod
    def from_points(cls, points, registry, start_direction=Direction.RIGHT):
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
            return cls(root_node)

        # Use nearest neighbor algorithm with a twist - try each point as a starting point
        best_trajectory = None
        min_actions = float('inf')

        for start_idx, start_point in enumerate(unique_points):
            # Create root node at this starting point
            root_node = registry.get_node(start_point, start_direction)
            trajectory = cls(root_node)

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
                    trajectory.update(action)
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
            return cls(root_node)

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
        new_trajectory = Trajectory(self.head)
        new_trajectory.route = self.route[:]
        new_trajectory.nodes = self.nodes[:]
        new_trajectory.position_cache = self.position_cache.copy()
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        return new_trajectory

    def update(self, action):
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)

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

    def get_new_trajectories(self, max_backtrack: int = 3):
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

        if self.has_backtrack(max_backtrack):
            return []

        new_trajectories = []

        # Explore all valid actions from current node
        for action, next_node in last_node.children.items():
            if action not in self._inherits_to:
                # Create a new trajectory with this action
                new_trajectory = self.copy()
                new_trajectory.update(action)

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
            return f"Trajectory to {last_node.position} {last_node.direction} from {self.head.position} {self.head.direction} via {self.route}"
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

@dataclass
class Constraints:
    """
    Point constraints
    """
    contains: list[Point]
    excludes: list[Point]

    def __bool__(self):
        return len(self.contains) > 0 or len(self.excludes) > 0

    def copy(self):
        return Constraints(
            self.contains[:],
            self.excludes[:]
        )

@dataclass
class TrajectoryConstraints:
    route: Constraints
    tail: Constraints

    def __bool__(self):
        return bool(self.route) or bool(self.tail)

    def copy(self):
        return TrajectoryConstraints(
            self.route.copy(),
            self.tail.copy()
        )

class TrajectoryIndex:
    def __init__(self):
        self._index: dict[Any, set[Trajectory]] = {}

    def add(self, key: Any, trajectory: Trajectory) -> None:
        if key not in self._index:
            self._index[key] = set()
        self._index[key].add(trajectory)

    def remove(self, key: Any, trajectory: Trajectory) -> None:
        if key in self._index and trajectory in self._index[key]:
            trajectory.prune()
            self._index[key].remove(trajectory)
            # Clean up empty sets
            if not self._index[key]:
                del self._index[key]

    def clear(self):
        self._index.clear()

    def __contains__(self, key: Any):
        return key in self._index

    def __getitem__(self, key: Any):
        if key not in self._index:
            self._index[key] = set()
        return self._index[key]

class TrajectoryTree:
    def __init__(
        self,
        init_position: Point,
        init_direction: Optional[Direction] = None,
        size: int = 16,
        registry: Optional[NodeRegistry] = None,
        consider_direction: bool = True,
        regenerate_edge_threshold: int = 4,
        max_backtrack: int = 3,
    ):
        """
        Initialize a TrajectoryTree with the agent's starting position and direction.

        Args:
            init_position: Starting coordinate position
            init_direction: Starting direction (default: RIGHT)
            size: Size of the grid environment (default: 16)
            consider_direction: Whether to consider direction when reducing trajectories.
                               If True, trajectories with same position but different directions are considered distinct.
                               If False, only position matters, reducing the number of trajectories (default: True)
        """
        self.size = size
        self.consider_direction = consider_direction
        self.regenerate_edge_threshold = regenerate_edge_threshold
        self.max_backtrack = max_backtrack

        self.num_step = 0

        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        # Spatial index: maps positions to trajectories that pass through them
        self.position_index = TrajectoryIndex()
        self.tail_position_index = TrajectoryIndex()
        self.node_index = TrajectoryIndex()

        # Initialize the starting trajectories
        possible_init_directions: list[Direction] = []

        if init_direction is not None:
            possible_init_directions.append(init_direction)
        else:
            possible_init_directions.extend([d for d in Direction])

        self.trajectories: list[Trajectory] = []
        for direction in possible_init_directions:
            # This will either create a new node or use an existing one
            root_node = self.registry.get_or_create_node(init_position, direction)
            trajectory = Trajectory(root_node)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        self.edge_trajectories: list[Trajectory] = self.trajectories.copy()
        self.discard_edge_trajectories: list[tuple[int, Trajectory]] = [] # [(step it was discarded, edge_trajectory), ...]

        # Add a set to track ambiguous tiles (visited tiles we should ignore for pruning)
        self.ambiguous_tiles = set()

        # Flag to track if we've performed a reset (detected agent) yet
        self.has_reset = False

    def _register_trajectory_in_index(self, trajectory: Trajectory):
        """Register a trajectory in the position index."""
        if trajectory.to_delete:
            return

        for node in trajectory.nodes:
            self.position_index.add(node.position, trajectory)
            self.node_index.add(node, trajectory)

        self.tail_position_index.add(trajectory.tail.position, trajectory)

    def _unregister_trajectory_from_index(self, trajectory: Trajectory):
        """Remove a trajectory from the position index."""
        for node in trajectory.nodes:
            self.position_index.remove(node.position, trajectory)
            self.node_index.add(node, trajectory)

        # delete from end_index too
        self.tail_position_index.remove(trajectory.tail.position, trajectory)

    def _clear_all_trajectories(self):
        """Clear all trajectories and reset the position index."""
        self.trajectories.clear()
        self.edge_trajectories.clear()
        self.position_index.clear()
        self.tail_position_index.clear()

    def set_consider_direction(self, consider_direction: bool):
        """
        Update whether to consider direction when reducing trajectories.

        Args:
            consider_direction: If True, trajectories with same position but different
                               directions are considered distinct. If False, only position matters.
        """
        self.consider_direction = consider_direction

    def prune(self, information: list[tuple[Point, Tile]], seeking_scout: bool = True):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)
        4. when I encounter a tile and it has no agent, I remove all trajectories ending with that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
            seeking_scout (bool): If True, look for scout agent. If False, look for guard agent.
        """
        # Track newly discovered ambiguous tiles
        self._track_ambiguous_tiles(information)

        if seeking_scout:
            self._prune_by_tile_content(information)

        self._clean_up_trajectories()

        # Process agent sightings first (early exit)
        # agent_position = self._check_for_agent(information, seeking_scout)
        # if agent_position is not None:
        #     self._reset_trajectories_for_agent(agent_position)
        #     return

    def _check_for_agent(self, information, seeking_scout):
        """Check if an agent is present in the information."""
        for position, tile in information:
            if (seeking_scout and tile.has_scout) or (not seeking_scout and tile.has_guard):
                return position
        return None

    def _reset_trajectories_for_agent(self, agent_position):
        """
        Create trajectories that end with the agent's position.
        ._prune_by_tile_content() will have kept only trajectories ending in the agent's location, but that is not enough.
        """
        # If this is not the first reset, mark all currently known empty tiles as ambiguous
        if self.has_reset:
            # Add all currently tracked empty/visited tiles to ambiguous_tiles
            for trajectory in self.trajectories:
                for node in trajectory.nodes:
                    position = node.position
                    if position != agent_position:  # Don't mark agent's current position as ambiguous
                        self.ambiguous_tiles.add(position)
        else:
            # First reset - set the flag
            self.has_reset = True

        # Clear existing trajectories and their index
        # self._clear_all_trajectories()

        # Create new trajectories at the agent's position, considering all possible directions
        # for direction in Direction:
        #     root_node = self.registry.get_or_create_node(agent_position, direction)
        #     trajectory = Trajectory(root_node)
        #     self.trajectories.append(trajectory)
        #     self._register_trajectory_in_index(trajectory)

        # Update edge trajectories
        # self.edge_trajectories = self.trajectories.copy()

    def _track_ambiguous_tiles(self, information):
        """Track newly discovered empty tiles as ambiguous if we've already had a reset."""
        if not self.has_reset:
            return  # Only track ambiguous tiles after the first reset

        for position, tile in information:
            if tile.is_empty:  # Newly discovered visited/empty tile
                self.ambiguous_tiles.add(position)

    def debug_ambiguous_tiles(self):
        """Print debug information about ambiguous tiles."""
        print(f"Has reset: {self.has_reset}")
        print(f"Number of ambiguous tiles: {len(self.ambiguous_tiles)}")
        if self.ambiguous_tiles:
            print(f"Ambiguous tile positions: {sorted(self.ambiguous_tiles)}")
        print(f"Number of trajectories: {len(self.trajectories)}")

    def _prune_by_tile_content(self, information: list[tuple[Point, Tile]]):
        """Prune trajectories based on tile content."""
        # Group positions by condition to avoid redundant iterations
        route_constraints = Constraints([], [])
        tail_constraints = Constraints([], [])

        all_constraints = TrajectoryConstraints(
            route_constraints,
            tail_constraints
        )

        for position, tile in information:
            # Skip any tiles that are in our ambiguous set
            if position in self.ambiguous_tiles:
                continue

            # Case 1: agent detected - only keep trajectories containing this position
            if tile.has_scout:
                all_constraints.tail.contains.append(position)
            # Case 2: not visited - remove trajectories containing this position
            if tile.is_visible and (tile.is_recon or tile.is_mission):
                all_constraints.route.excludes.append(position)
            # Case 3: visited - only keep trajectories containing this position
            if tile.is_empty:
                all_constraints.route.contains.append(position)
            # Case 4: no agent - remove trajectories ending at this position
            if not tile.has_scout:
                all_constraints.tail.excludes.append(position)

        # Use the spatial index for efficient filtering
        if not all_constraints:
            return  # No filtering needed

        self._apply_filtering(all_constraints)

    def _apply_filtering(self, constraints: TrajectoryConstraints):
        """Apply filtering based on position constraints."""

        if Point(0, 0) in constraints.route.excludes:
            constraints.route.excludes.remove(Point(0, 0))

        if Point(0, 0) in constraints.tail.excludes:
            constraints.tail.excludes.remove(Point(0, 0))


        # Find trajectories to exclude (has any excluded position)
        trajectories_to_remove = set()
        for position in constraints.route.excludes:
            if position in self.position_index:
                trajectories_to_remove.update(self.position_index[position])

        for position in constraints.tail.excludes:
            if position in self.tail_position_index:
                trajectories_to_remove.update(self.tail_position_index[position])

        # Find trajectories to keep (has all required positions)
        trajectories_to_keep = set(self.trajectories)

        # Filter by required route positions
        for position in constraints.route.contains:
            if position in self.position_index:
                trajectories_with_position = self.position_index[position]
                trajectories_to_keep &= trajectories_with_position
            else:
                # If a required position doesn't exist in any trajectory, no trajectories can satisfy
                trajectories_to_keep = set()
                break

        # Filter by required tail positions
        for position in constraints.tail.contains:
            if position in self.tail_position_index:
                trajectories_with_tail_position = self.tail_position_index[position]
                trajectories_to_keep &= trajectories_with_tail_position
            else:
                # If a required tail position doesn't exist in any trajectory, no trajectories can satisfy
                trajectories_to_keep = set()
                break

        # Compute final valid trajectories
        valid_trajectories = [
            traj for traj in trajectories_to_keep if traj not in trajectories_to_remove
        ] if trajectories_to_keep else []

        # Remove invalid trajectories from the index
        removed_trajectories = set(self.trajectories) - set(valid_trajectories)
        for traj in removed_trajectories:
            traj.prune()
            # self._unregister_trajectory_from_index(traj)

        self._destroy_trajectory_families()
        self.trajectories = valid_trajectories

        # whether any edge trajectories have been deleted, may want to re-add them.
        trajectory_set = set(self.trajectories)
        has_deleted_edge = False
        for traj in self.edge_trajectories:
            if traj not in trajectory_set:
                has_deleted_edge = True
                break

        if has_deleted_edge:
            self.edge_trajectories = self._get_edge_trajectories()

    def step(self) -> int:
        """
        Expand all valid trajectories by one step, but only expanding the
        longest and shortest trajectory that reaches each endpoint.

        When consider_direction=True, trajectories are grouped by (position, direction).
        When consider_direction=False, trajectories are grouped only by position,
        significantly reducing the number of trajectories.

        For each endpoint, we select:
        1. The shortest trajectory (fewest steps to reach that point)
        2. The longest trajectory (if different from the shortest)

        Periodically validates a random sample of trajectories to ensure
        invalid trajectories are properly identified.

        Returns:
            Number of new trajectories added
        """
        if not self.trajectories:
            return 0

        before_len = len(self.trajectories)

        old_edge_trajectories = self.edge_trajectories
        print(f"Updating from {len(old_edge_trajectories)} trajectories")
        self.edge_trajectories = []  # Clear edge trajectories for this step

        # Mapping of endpoint keys to trajectories
        endpoint_to_trajectories = {}

        # Group trajectories by their endpoints
        for traj in old_edge_trajectories:
            # Skip invalid trajectories
            if traj.to_delete:
                continue

            traj.discarded = True

            # Get key for this trajectory's endpoint
            key = traj.get_endpoint_key(self.consider_direction)
            if not key:
                continue

            if key not in endpoint_to_trajectories:
                endpoint_to_trajectories[key] = []
            endpoint_to_trajectories[key].append(traj)

        # For each endpoint, select the shortest and longest trajectories
        selected_trajectories = []
        for key, trajectories in endpoint_to_trajectories.items():
            # Sort by trajectory length (shorter first)
            trajectories.sort(key=lambda t: len(t.route))

            # Always select the shortest trajectory
            shortest = trajectories[0]
            shortest.discarded = False
            selected_trajectories.append(shortest)

            # Also select the longest if different
            if len(trajectories) > 1 and len(trajectories[-1].route) > len(shortest.route):
                longest = trajectories[-1]
                longest.discarded = False
                selected_trajectories.append(longest)

        # Add discarded trajectories to discard_edge_trajectories
        for traj in old_edge_trajectories:
            if traj.discarded:
                self.discard_edge_trajectories.append((self.num_step, traj))

        print(f"Total discard: {len(self.discard_edge_trajectories)}")
        print(f"Valid discarded: {len([0 for traj in self.discard_edge_trajectories if not traj[-1].to_delete])}")

        # Expand selected trajectories
        for trajectory in selected_trajectories:
            new_trajectories = trajectory.get_new_trajectories(max_backtrack=self.max_backtrack)
            if new_trajectories:
                # Add valid new trajectories
                for new_traj in new_trajectories:
                    self.trajectories.append(new_traj)
                    self._register_trajectory_in_index(new_traj)
                    self.edge_trajectories.append(new_traj)

        self.num_step += 1

        # Return count of new trajectories
        return len(self.trajectories) - before_len

    @property
    def probability_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions.

        This method computes how likely each position is to contain an agent,
        based on the number of valid trajectories passing through it.

        Returns:
            numpy.ndarray: 2D array with probability for each position
        """
        # Initialize empty grid
        density = np.zeros((self.size, self.size), dtype=np.float32)

        if not self.trajectories:
            return density

        # Count trajectories passing through each cell
        for trajectory in self.trajectories:
            if trajectory.to_delete:
                continue

            # last_node = trajectory.get_last_node()
            last_node = trajectory.tail
            if not last_node:
                continue

            # Add 1 to the density at the endpoint position
            pos = last_node.position
            density[pos.y, pos.x] += 1

        # Normalize to get probability density
        total = density.sum()
        if total > 0:
            density = density / total

        return density

    @property
    def path_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions based on entire paths.

        Unlike probability_density which only considers endpoints, this method
        distributes the probability across all nodes in each trajectory's path.
        Each node in a trajectory gets 1/n of the trajectory's weight, where n is
        the number of nodes in that trajectory.

        Returns:
            numpy.ndarray: 2D array with path density for each position
        """
        # Initialize empty grid
        density = np.zeros((self.size, self.size), dtype=np.float32)

        if not self.trajectories:
            return density

        # Distribute density across all nodes in each trajectory
        for trajectory in self.trajectories:
            if trajectory.to_delete:
                continue

            # Skip empty trajectories
            if not trajectory.nodes:
                continue

            # Calculate weight per node (1 divided by number of nodes)
            weight_per_node = 1.0 / len(trajectory.nodes)

            # Add the weight to each position in the trajectory
            for node in trajectory.nodes:
                pos = node.position
                density[pos.y, pos.x] += weight_per_node

        # Normalize to get probability density
        total = density.sum()
        if total > 0:
            density = density / total

        return density

    def _destroy_trajectory_families(self):
        """
        Starting from any invalid trajectory, traverse up ._inherited_from,
        deleting invalids from ._inherits_to. traverse until no invalids in ._inherits_to are found
        """
        parent_traj: Optional[Trajectory] = None

        for traj in self.trajectories:
            if traj.to_delete:
                parent_traj = traj.parent

                if parent_traj:
                    if len(parent_traj.children) > 0:

                        parent_traj._inherits_to = {
                            action: child for action, child in parent_traj._inherits_to.items()
                            if not child.to_delete
                        }

    def _get_edge_trajectories(self):
        """
        Edge trajectories are basically any trajectory without children
        """
        edge_trajectories = []
        for traj in self.trajectories:
            if traj.to_delete:
                continue

            if len(traj.children) < self.regenerate_edge_threshold:
                edge_trajectories.append(traj)

        return edge_trajectories

    def _clean_up_trajectories(self):
        valid_trajectories = []
        for traj in self.trajectories:
            if traj.to_delete:
                self._unregister_trajectory_from_index(traj)
            else:
                valid_trajectories.append(traj)
        self.trajectories = valid_trajectories

    def check_wall_trajectories(self, node: DirectionalNode):
        trajectories = self.node_index[node]
        for traj in trajectories:
            traj.get_last_node()
