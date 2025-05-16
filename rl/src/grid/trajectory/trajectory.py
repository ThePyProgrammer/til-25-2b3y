from functools import lru_cache
from typing import Optional, Any

from ..utils import Direction, Action, Point
from ..utils.pathfinding import (
    manhattan_distance,
    get_node_neighbors,
    find_shortest_paths,
    find_path
)
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

        self._step: int = step

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
        self._step = step

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

    @property
    def created_at(self) -> int:
        if self._step:
            return self._step
        return len(self.route)

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
    def _create_point_nodes(cls, points: list[Point], registry: NodeRegistry) -> list[DirectionalNode]:
        """Create directional nodes for each point and direction."""
        point_nodes: list[DirectionalNode] = []
        for point in points:
            for direction in Direction:
                node: DirectionalNode = registry.get_or_create_node(point, direction)
                point_nodes.append(node)
        return point_nodes

    @classmethod
    def _compute_paths_from_root(
        cls,
        root: DirectionalNode,
        point_nodes: list[DirectionalNode],
        budget: int
    ) -> dict[DirectionalNode, Any]:
        """Compute shortest paths from root to all point nodes."""
        return find_shortest_paths(
            root,
            point_nodes,
            get_node_neighbors,
            node_hash=hash,
            max_iterations=budget * 20  # Increased max iterations for better exploration
        )

    @classmethod
    def _filter_reachable_paths(cls, paths_from_root: dict[DirectionalNode, Any], budget: int) -> dict[DirectionalNode, Any]:
        """Filter paths that are successful and within budget."""
        return {node: path for node, path in paths_from_root.items()
                if path.success and path.cost <= budget}

    @classmethod
    def _group_nodes_by_position(cls, nodes: dict[DirectionalNode, Any]) -> dict[Point, list[DirectionalNode]]:
        """Group nodes by their position for more efficient processing."""
        position_to_nodes: dict[Point, list[DirectionalNode]] = {}
        for node in nodes:
            if node.position not in position_to_nodes:
                position_to_nodes[node.position] = []
            position_to_nodes[node.position].append(node)
        return position_to_nodes

    @classmethod
    def _get_best_nodes_per_position(
        cls,
        position_to_nodes: dict[Point, list[DirectionalNode]],
        reachable_paths: dict[DirectionalNode, Any]
    ) -> dict[Point, DirectionalNode]:
        """Get the node with lowest path cost from root for each position."""
        best_nodes_per_position: dict[Point, DirectionalNode] = {}
        for pos, nodes in position_to_nodes.items():
            best_node: DirectionalNode = min(nodes, key=lambda n: reachable_paths[n].cost)
            best_nodes_per_position[pos] = best_node
        return best_nodes_per_position

    @classmethod
    def _compute_position_pairs(
        cls,
        positions_list: list[Point],
        best_nodes_per_position: dict[Point, DirectionalNode],
        budget: int
    ) -> tuple[dict[tuple[Point, Point], int], dict[tuple[DirectionalNode, DirectionalNode], Any]]:
        """Compute shortest paths between all position pairs."""
        position_pairs: dict[tuple[Point, Point], int] = {}
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any] = {}

        for i, pos1 in enumerate(positions_list):
            node1: DirectionalNode = best_nodes_per_position[pos1]
            for j, pos2 in enumerate(positions_list):
                if i == j:
                    continue
                node2: DirectionalNode = best_nodes_per_position[pos2]
                # Compute path from node1 to node2
                paths: dict[DirectionalNode, Any] = find_shortest_paths(
                    node1,
                    [node2],
                    get_node_neighbors,
                    node_hash=hash,
                    max_iterations=budget * 10
                )
                if node2 in paths and paths[node2].success:
                    position_pairs[(pos1, pos2)] = paths[node2].cost
                    node_pairs[(node1, node2)] = paths[node2]

        return position_pairs, node_pairs

    @classmethod
    def _create_greedy_trajectory(
        cls,
        root: DirectionalNode,
        start_pos: Point,
        start_node: DirectionalNode,
        reachable_paths: dict[DirectionalNode, Any],
        position_pairs: dict[tuple[Point, Point], int],
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any],
        best_nodes_per_position: dict[Point, DirectionalNode],
        positions_list: list[Point],
        points_set: set[Point],
        step: int,
        budget: int
    ) -> tuple[Optional['Trajectory'], int]:
        """Create a trajectory using greedy nearest neighbor approach."""
        path_result: Any = reachable_paths[start_node]

        # Skip if this point is unreachable
        if not path_result.success:
            return None, 0

        # Initialize trajectory with the first leg
        trajectory: 'Trajectory' = cls(root, step)

        # Add actions to get to the first point
        actions_left: int = budget
        for action in path_result.actions:
            trajectory.update(action, step)
            actions_left -= 1

        # Skip if trajectory became invalid
        if trajectory.invalid:
            return None, 0

        # Set of points we've already visited
        visited_positions: set[Point] = {start_pos}

        # Current position (for finding next point)
        current_pos: Point = start_pos
        current_node: DirectionalNode = start_node

        # Track the remaining positions to visit
        remaining_positions: set[Point] = set(positions_list) - visited_positions

        # Early termination check - can we visit all points?
        can_complete: bool = True
        for pos in remaining_positions:
            if not any((current_pos, pos) in position_pairs for pos in remaining_positions):
                can_complete = False
                break

        if not can_complete:
            return None, 0

        # While we haven't visited all points and have actions left
        while remaining_positions and actions_left > 0:
            closest_info = cls._find_closest_position(
                current_pos,
                current_node,
                remaining_positions,
                position_pairs,
                best_nodes_per_position,
                node_pairs,
                actions_left
            )

            if closest_info is None:
                break

            closest_pos, closest_node, closest_path = closest_info

            # Add actions to get to the next point
            for action in closest_path.actions:
                trajectory.update(action, step)
                actions_left -= 1
                if trajectory.invalid or actions_left <= 0:
                    break

            # If trajectory became invalid, break
            if trajectory.invalid:
                break

            # Update current position and visited positions
            current_pos = closest_pos
            current_node = closest_node
            visited_positions.add(current_pos)
            remaining_positions.remove(current_pos)

        # Check if we've visited all required points
        if len(visited_positions) == len(points_set) and not trajectory.invalid:
            return trajectory, budget - actions_left

        return None, 0

    @classmethod
    def _find_closest_position(
        cls,
        current_pos: Point,
        current_node: DirectionalNode,
        remaining_positions: set[Point],
        position_pairs: dict[tuple[Point, Point], int],
        best_nodes_per_position: dict[Point, DirectionalNode],
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any],
        actions_left: int
    ) -> Optional[tuple[Point, DirectionalNode, Any]]:
        """Find the closest unvisited position."""
        closest_pos: Optional[Point] = None
        closest_node: Optional[DirectionalNode] = None
        closest_cost: float = float('inf')
        closest_path: Optional[Any] = None
        
        for pos in remaining_positions:
            if (current_pos, pos) in position_pairs:
                cost: int = position_pairs[(current_pos, pos)]
                if cost < closest_cost and cost <= actions_left:
                    closest_pos = pos
                    closest_node = best_nodes_per_position[pos]
                    closest_cost = cost
                    closest_path = node_pairs[(current_node, closest_node)]
                    
        if closest_pos is None or closest_node is None or closest_path is None:
            return None
            
        return closest_pos, closest_node, closest_path

    @classmethod
    def _try_optimize_trajectory(
        cls,
        root: DirectionalNode,
        best_traj: 'Trajectory',
        best_cost: int,
        points_set: set[Point],
        best_nodes_per_position: dict[Point, DirectionalNode],
        reachable_paths: dict[DirectionalNode, Any],
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any],
        step: int,
        budget: int
    ) -> list[tuple['Trajectory', int]]:
        """Try to optimize trajectory using insertion heuristic."""
        improved_trajectories: list[tuple['Trajectory', int]] = []

        # Try to improve by permuting the order
        for i in range(3):  # Try a few permutations
            visited_in_order = cls._get_visited_positions(best_traj, points_set)

            # Try a different ordering by moving some positions
            if i > 0 and len(visited_in_order) > 3:
                # Move a position to a different spot
                pos_to_move: Point = visited_in_order[i]
                visited_in_order.remove(pos_to_move)
                visited_in_order.insert(min(i+2, len(visited_in_order)), pos_to_move)

            # Create the trajectory with this new ordering
            improved_result = cls._create_trajectory_with_ordering(
                root,
                visited_in_order,
                best_nodes_per_position,
                reachable_paths,
                node_pairs,
                points_set,
                step,
                budget
            )

            if improved_result:
                improved_traj, improved_cost = improved_result
                if improved_cost < best_cost:
                    improved_trajectories.append((improved_traj, improved_cost))

        return improved_trajectories

    @classmethod
    def _get_visited_positions(cls, trajectory: 'Trajectory', points_set: set[Point]) -> list[Point]:
        """Get the ordered list of visited positions from a trajectory."""
        visited_in_order: list[Point] = []
        for node in trajectory.nodes:
            if node.position in points_set and node.position not in visited_in_order:
                visited_in_order.append(node.position)
        return visited_in_order

    @classmethod
    def _create_trajectory_with_ordering(
        cls,
        root: DirectionalNode,
        visited_in_order: list[Point],
        best_nodes_per_position: dict[Point, DirectionalNode],
        reachable_paths: dict[DirectionalNode, Any],
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any],
        points_set: set[Point],
        step: int,
        budget: int
    ) -> Optional[tuple['Trajectory', int]]:
        """Create a trajectory with a specific ordering of positions."""
        improved_traj: 'Trajectory' = cls(root, step)
        remaining_budget: int = budget

        current_node: DirectionalNode = root

        for pos in visited_in_order:
            target_node: DirectionalNode = best_nodes_per_position[pos]
            actions = cls._get_actions_to_target(
                current_node,
                target_node,
                root,
                reachable_paths,
                node_pairs,
                remaining_budget
            )

            # Add actions to get to this point
            for action in actions:
                improved_traj.update(action, step)
                remaining_budget -= 1
                if improved_traj.invalid or remaining_budget <= 0:
                    break

            # If trajectory became invalid, break
            if improved_traj.invalid or remaining_budget <= 0:
                break

            # Update current node
            current_node = target_node

        # Check if this improved trajectory is valid and better
        if (not improved_traj.invalid and
            len(set(n.position for n in improved_traj.nodes if n.position in points_set)) == len(points_set)):
            return improved_traj, budget - remaining_budget

        return None

    @classmethod
    def _get_actions_to_target(
        cls,
        current_node: DirectionalNode,
        target_node: DirectionalNode,
        root: DirectionalNode,
        reachable_paths: dict[DirectionalNode, Any],
        node_pairs: dict[tuple[DirectionalNode, DirectionalNode], Any],
        remaining_budget: int
    ) -> list[Action]:
        """Get actions to move from current node to target node."""
        if current_node is root:
            # Use pre-computed path from root
            path_result: Any = reachable_paths[target_node]
            return path_result.actions
        else:
            # Use pre-computed path between nodes
            if (current_node, target_node) in node_pairs:
                path_result: Any = node_pairs[(current_node, target_node)]
                return path_result.actions
            else:
                # Compute path directly if not pre-computed
                path_result: Any = find_path(
                    current_node,
                    lambda n: bool(n.position == target_node.position),
                    get_node_neighbors,
                    lambda n: manhattan_distance(n.position, target_node.position),
                    hash,
                    max_iterations=remaining_budget * 5
                )
                return path_result.actions

    @classmethod
    def from_points(
        cls,
        roots: list[DirectionalNode],
        points: list[Point],
        registry: NodeRegistry,
        budget: int = 100
    ) -> list['Trajectory']:
        """
        Generate the top N shortest trajectories that visit all given points within a maximum budget of actions.

        Args:
            roots: List of DirectionalNode objects to use as starting points
            points: List of Point objects that must all be visited
            step: Step/time at which the trajectory is created
            registry: NodeRegistry to use for node creation
            budget: Maximum number of actions allowed (default: 100)

        Returns:
            list[Trajectory]: List of the top N shortest trajectories that visit all points within the budget
        """
        if not points or not roots:
            return []

        step: int = 0
        points_set: set[Point] = set(points)
        all_trajectories: list[tuple['Trajectory', int]] = []

        for root in roots:
            # Initialize a trajectory from this root
            initial_trajectory = cls(root, step)
            if initial_trajectory.invalid:
                continue

            # Create nodes for each point and direction
            point_nodes = cls._create_point_nodes(points, registry)

            # Calculate paths from root to each point
            paths_from_root = cls._compute_paths_from_root(root, point_nodes, budget)
            if not paths_from_root:
                continue

            # Filter paths that are reachable within budget
            reachable_paths = cls._filter_reachable_paths(paths_from_root, budget)
            if not reachable_paths:
                continue

            # Group nodes by position
            position_to_nodes = cls._group_nodes_by_position(reachable_paths)
            if len(position_to_nodes) < len(points_set):
                continue

            # Get best node for each position
            best_nodes_per_position = cls._get_best_nodes_per_position(position_to_nodes, reachable_paths)

            # Compute paths between all positions
            positions_list = list(best_nodes_per_position.keys())
            position_pairs, node_pairs = cls._compute_position_pairs(
                positions_list,
                best_nodes_per_position,
                budget
            )

            # Try different starting positions
            for start_pos, start_node in best_nodes_per_position.items():
                trajectory_result = cls._create_greedy_trajectory(
                    root,
                    start_pos,
                    start_node,
                    reachable_paths,
                    position_pairs,
                    node_pairs,
                    best_nodes_per_position,
                    positions_list,
                    points_set,
                    step,
                    budget
                )

                trajectory, cost = trajectory_result
                if trajectory is not None:
                    all_trajectories.append((trajectory, cost))

            # Try to optimize trajectories with insertion heuristic
            if all_trajectories:
                best_traj, best_cost = min(all_trajectories, key=lambda x: x[1])

                improved_trajectories = cls._try_optimize_trajectory(
                    root,
                    best_traj,
                    best_cost,
                    points_set,
                    best_nodes_per_position,
                    reachable_paths,
                    node_pairs,
                    step,
                    budget
                )

                all_trajectories.extend(improved_trajectories)

        # Sort trajectories by total cost (number of actions used)
        all_trajectories.sort(key=lambda x: x[1])

        # Return the top N trajectories
        return [t for t, _ in all_trajectories]
