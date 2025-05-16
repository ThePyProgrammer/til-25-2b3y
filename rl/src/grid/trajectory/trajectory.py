from functools import lru_cache
from typing import Optional, Any

from ..utils import Direction, Action, Point
from ..utils.geometry import POINT_EQ_LOOKUP
from ..utils.pathfinding import manhattan_distance, find_path, get_directional_neighbors
from ..node import DirectionalNode, NodeRegistry


# @lru_cache(maxsize=2**16)
def get_trajectory_hash(root: DirectionalNode, *route: Action) -> int:
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

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = get_trajectory_hash(self.head, *self.route)
        return self._hash

    def __eq__(self, other: Any) -> bool:
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

    def copy(self) -> 'Trajectory':
        """Create a deep copy of this trajectory with the same root but new lists."""
        new_trajectory = Trajectory(self.head, self.created_at)
        new_trajectory.route = self.route[:]
        new_trajectory.nodes = self.nodes[:]
        new_trajectory.position_cache = self.position_cache.copy()
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        return new_trajectory

    def update(self, action: Action, step: int) -> None:
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)
        self.created_at = step

        # Check if this action is valid
        self.get_last_node()

    def prune(self) -> None:
        self.pruned = True

    def invalidate(self, propagate: bool = False) -> None:
        self.invalid = True
        if propagate:
            self._mark_as_invalid()

    def _mark_as_invalid(self, invalid_action_idx: Optional[int] = None) -> None:
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

    def get_last_node(self) -> Optional[DirectionalNode]:
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

    def __str__(self) -> str:
        """String representation of the trajectory."""
        if self.invalid:
            return f"Invalid Trajectory: {self.route}"

        last_node = self.tail
        if last_node:
            return f"Trajectory to {last_node.position} {last_node.direction} from {self.head.position} {self.head.direction} via {self.route}, created at: {self.created_at}"
        return f"Incomplete Trajectory: {self.route}"

    def __repr__(self) -> str:
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
    def parent(self) -> Optional['Trajectory']:
        return self._inherited_from

    @property
    def children(self):
        return self._inherits_to.values()

    @property
    def to_delete(self) -> bool:
        return self.invalid or self.pruned

    def get_new_trajectories(self, step: int, max_backtrack: Optional[int] = None) -> list['Trajectory']:
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

    @classmethod
    def from_points(
        cls,
        roots: list[DirectionalNode],
        points: list[Point],
        step: int,
        registry: NodeRegistry,
        budget: int = 100
    ) -> list['Trajectory']:
        """
        Generate all possible trajectories that visit all given points within a maximum budget of actions.

        Args:
            roots: List of DirectionalNode objects to use as starting points
            points: List of Point objects that must all be visited
            step: Step/time at which the trajectory is created
            registry: NodeRegistry to use for node creation
            budget: Maximum number of actions allowed (default: 100)

        Returns:
            list[Trajectory]: List of all valid trajectories that visit all points within the budget
        """
        # Handle empty points list
        if not points:
            return [cls(root, step) for root in roots]

        # Remove duplicates from points list
        unique_points = cls._deduplicate_points(points)
        # Find all valid trajectories
        valid_trajectories = []
        path_cache = {}  # Cache for path finding

        # Explore trajectories from each root node
        for root in roots:
            # Create initial trajectory with this root
            initial_trajectory = cls(root, step)

            # Check which points are already visited by this root trajectory
            initial_points_to_visit = cls._get_unvisited_points(initial_trajectory, unique_points)

            # If all points are already visited at the root, add this trajectory
            if not initial_points_to_visit:
                valid_trajectories.append(initial_trajectory.copy())
                continue

            # Start exploring from this root with the points that aren't already visited
            cls._explore_trajectories(
                initial_trajectory,
                initial_points_to_visit,
                budget,
                valid_trajectories,
                path_cache,
                step
            )

        return valid_trajectories

    @classmethod
    def _deduplicate_points(cls, points: list[Point]) -> list[Point]:
        """
        Remove duplicate points while preserving order.

        Args:
            points: List of Point objects

        Returns:
            list: Deduplicated list of points
        """
        unique_points = []
        seen = set()
        for point in points:
            point_key = (point.x, point.y)
            if point_key not in seen:
                seen.add(point_key)
                unique_points.append(point)
        return unique_points

    @classmethod
    def _is_point_visited(cls, trajectory: 'Trajectory', target_point: Point) -> bool:
        """
        Check if a point has been visited in the trajectory.

        Args:
            trajectory: Trajectory to check
            target_point: Point to look for

        Returns:
            bool: True if the point is visited, False otherwise
        """
        for node in trajectory.nodes:
            pos = node.position
            if POINT_EQ_LOOKUP[pos.x, target_point.x, pos.y, target_point.y]:
                return True
        return False

    @classmethod
    def _get_unvisited_points(cls, trajectory: 'Trajectory', points_list: list[Point]) -> list[Point]:
        """
        Get list of points not yet visited in the trajectory.

        Args:
            trajectory: Trajectory to check
            points_list: List of points to check

        Returns:
            list: Points that haven't been visited
        """
        unvisited = []
        for point in points_list:
            if not cls._is_point_visited(trajectory, point):
                unvisited.append(point)
        return unvisited

    @classmethod
    def _get_cached_path(
        cls,
        start_pos: Point,
        start_dir: Direction,
        target_point: Point,
        path_cache: dict[tuple[int, int, Direction, int, int], list[Action]]
    ) -> list[Action]:
        """
        Get cached path or compute new path between positions.

        Args:
            start_pos: Starting position
            start_dir: Starting direction
            target_point: Target position
            path_cache: Dictionary for caching paths

        Returns:
            list: Actions to take to reach target
        """
        cache_key = (start_pos.x, start_pos.y, start_dir, target_point.x, target_point.y)
        if cache_key not in path_cache:
            path_cache[cache_key] = cls._find_shortest_path(start_pos, start_dir, target_point)
        return path_cache[cache_key]

    @classmethod
    def _try_apply_actions(
        cls,
        trajectory: 'Trajectory',
        actions: list[Action],
        step: int
    ) -> tuple['Trajectory', bool]:
        """
        Try to apply a sequence of actions to a trajectory.

        Args:
            trajectory: Base trajectory
            actions: List of actions to apply
            step: Current step

        Returns:
            tuple: (new_trajectory, success_flag)
        """
        new_trajectory = trajectory.copy()
        for action in actions:
            new_trajectory.update(action, step)
            if new_trajectory.invalid:
                return new_trajectory, False
        return new_trajectory, True

    @classmethod
    def _explore_trajectories(
        cls,
        trajectory: 'Trajectory',
        points_to_visit: list[Point],
        remaining_budget: int,
        valid_trajectories: list['Trajectory'],
        path_cache: dict[tuple[int, int, Direction, int, int], list[Action]],
        step: int
    ) -> None:
        """
        Recursively explore all possible trajectories that visit all points within budget.

        Args:
            trajectory: Current trajectory
            points_to_visit: Points still to be visited
            remaining_budget: Remaining action budget
            valid_trajectories: List to store valid trajectories
            path_cache: Cache for path finding
            step: Current step
        """
        # Success case: all points have been visited
        if not points_to_visit:
            valid_trajectories.append(trajectory.copy())
            return

        # Base case: no more budget left
        if remaining_budget <= 0:
            return

        # Get current position and direction
        current_node = trajectory.tail
        current_pos = current_node.position
        current_dir = current_node.direction

        # Check for points satisfied at current position
        points_satisfied_here = cls._check_current_position(
            trajectory,
            points_to_visit,
            remaining_budget,
            valid_trajectories,
            path_cache,
            step
        )

        # If some points were satisfied at current position, stop further exploration
        if points_satisfied_here:
            return

        # Try to visit each remaining point
        cls._explore_paths_to_points(
            trajectory,
            points_to_visit,
            remaining_budget,
            valid_trajectories,
            path_cache,
            step
        )

    @classmethod
    def _check_current_position(
        cls,
        trajectory: 'Trajectory',
        points_to_visit: list[Point],
        remaining_budget: int,
        valid_trajectories: list['Trajectory'],
        path_cache: dict[tuple[int, int, Direction, int, int], list[Action]],
        step: int
    ) -> bool:
        """
        Check if any points are satisfied at the current position and handle accordingly.

        Returns:
            bool: True if some points were satisfied and further exploration should stop
        """
        current_pos = trajectory.tail.position

        # Find points already visited at current position
        current_points_to_visit = []
        for point in points_to_visit:
            if not POINT_EQ_LOOKUP[current_pos.x, point.x, current_pos.y, point.y]:
                current_points_to_visit.append(point)

        # If all points are now visited, add this trajectory
        if not current_points_to_visit:
            valid_trajectories.append(trajectory.copy())
            return True

        # If we've visited some points at the current position (but not all),
        # continue exploration with the reduced set
        if len(current_points_to_visit) < len(points_to_visit):
            cls._explore_trajectories(
                trajectory.copy(),
                current_points_to_visit,
                remaining_budget,
                valid_trajectories,
                path_cache,
                step
            )
            return True

        return False

    @classmethod
    def _explore_paths_to_points(
        cls, trajectory: 'Trajectory',
        points_to_visit: list[Point],
        remaining_budget: int,
        valid_trajectories: list['Trajectory'],
        path_cache: dict[tuple[int, int, Direction, int, int], list[Action]],
        step: int
    ) -> None:
        """
        Try paths to each target point and explore resulting trajectories.
        """
        current_pos = trajectory.tail.position
        current_dir = trajectory.tail.direction

        for target_point in points_to_visit:
            # Find shortest path to this point
            actions = cls._get_cached_path(current_pos, current_dir, target_point, path_cache)

            # Skip if path would exceed budget
            if len(actions) > remaining_budget:
                continue

            # Try to apply all actions in the path
            new_trajectory, path_valid = cls._try_apply_actions(trajectory, actions, step)
            if not path_valid:
                continue

            # Determine which points are still unvisited after taking this path
            unvisited_points = cls._get_unvisited_points(new_trajectory, points_to_visit)

            # Continue exploration with remaining points and reduced budget
            cls._explore_trajectories(
                new_trajectory,
                unvisited_points,
                remaining_budget - len(actions),
                valid_trajectories,
                path_cache,
                step
            )

    @staticmethod
    def _find_shortest_path(
        start_pos: Point,
        start_dir: Direction,
        target_pos: Point
    ) -> list[Action]:
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
        def is_goal(state: tuple[Point, Direction]) -> bool:
            pos, _ = state
            return pos.x == target_pos.x and pos.y == target_pos.y

        # Define the neighbor function
        def get_neighbors(state: tuple[Point, Direction]) -> dict[Action, tuple[Point, Direction]]:
            return get_directional_neighbors(state)

        # Define the heuristic function
        def heuristic(state: tuple[Point, Direction]) -> int:
            pos, _ = state
            return manhattan_distance(pos, target_pos)

        # Define hash function for states
        def state_hash(state: tuple[Point, Direction]) -> int:
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
