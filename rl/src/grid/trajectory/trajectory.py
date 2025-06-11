from functools import lru_cache
from typing import Optional, Any

from ..utils import Direction, Action, Point
from ..utils.pathfinding import (
    manhattan_distance,
    get_node_neighbors,
    find_shortest_paths,
    find_path,
)
from ..node import DirectionalNode, NodeRegistry


def canonicalise_route(route: list[Action]) -> list[Action]:
    """
    Canonicalise a route by rebuilding patterns of (TURN, BACKWARD, TURN_R) into (TURN_R, FORWARD, TURN).
    This helps deduplicate routes that are equivalent in terms of movement.
    """
    start_turn_idx = 0
    turn_type = None
    canon_route = route[:]
    for i in range(len(canon_route)):
        action = canon_route[i]
        if action == Action.FORWARD:
            # reset
            turn_type = None
        elif action == Action.RIGHT or action == Action.LEFT:
            if turn_type == Action.LEFT and action == Action.RIGHT:
                # canonicalise
                if start_turn_idx < i - 1:
                    # rewrite
                    canon_route[start_turn_idx] = Action.RIGHT
                    for j in range(start_turn_idx + 1, i):
                        canon_route[j] = Action.FORWARD
                    canon_route[i] = Action.LEFT
            elif turn_type == Action.RIGHT and action == Action.LEFT:
                # canonicalise
                if start_turn_idx < i - 1:
                    # rewrite
                    canon_route[start_turn_idx] = Action.LEFT
                    for j in range(start_turn_idx + 1, i):
                        canon_route[j] = Action.FORWARD
                    canon_route[i] = Action.RIGHT
                else:
                    # turn no-op turns into LEFT, RIGHT
                    canon_route[i] = Action.RIGHT
                    canon_route[start_turn_idx] = Action.LEFT
            elif turn_type == Action.RIGHT and action == Action.RIGHT:
                # canonicalise
                if start_turn_idx < i - 1:
                    # rewrite
                    canon_route[start_turn_idx] = Action.LEFT
                    for j in range(start_turn_idx + 1, i):
                        canon_route[j] = Action.FORWARD
                    canon_route[i] = Action.LEFT
                else:
                    # turn 180 turns into LEFT, LEFT
                    canon_route[i] = Action.LEFT
                    canon_route[start_turn_idx] = Action.LEFT
            elif turn_type == Action.LEFT and action == Action.LEFT:
                # canonicalise
                if start_turn_idx < i - 1:
                    # rewrite
                    canon_route[start_turn_idx] = Action.RIGHT
                    for j in range(start_turn_idx + 1, i):
                        canon_route[j] = Action.FORWARD
                    canon_route[i] = Action.RIGHT
            start_turn_idx = i
            turn_type = canon_route[i]

    return canon_route


# @lru_cache(maxsize=2**16)
def get_trajectory_hash(root: DirectionalNode, route: list[Action]) -> int:
    # print(route)
    canon_route = canonicalise_route(route)
    return hash((root, *canon_route))


class Trajectory:
    def __init__(self, root_node: DirectionalNode, step: int):
        self.head: DirectionalNode = root_node
        self.route: list[Action] = []
        self.nodes: list[DirectionalNode] = [self.head]
        self.position_cache: set[Point] = {
            root_node.position
        }  # Cache positions for faster lookups

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self.pruned: bool = False
        self.discarded: bool = (
            False  # if it's valid and was not (yet) pruned, but discarded by  TrajectoryTree.step()
        )

        self._step: int = step

        self._inherited_from: Optional["Trajectory"] = None
        self._inherits_to: dict[Action, "Trajectory"] = {}

        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = get_trajectory_hash(self.head, self.route)
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Trajectory):
            return False
        return self.__hash__() == hash(other)

    @lru_cache(maxsize=1024 + 256)
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

    def copy(self) -> "Trajectory":
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
        self, max_backtrack: int = 3, consider_direction: bool = False
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
    def parent(self) -> Optional["Trajectory"]:
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

    def get_new_trajectories(
        self, step: int, max_backtrack: Optional[int] = None
    ) -> list["Trajectory"]:
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
            already_has_backtrack = self.has_backtrack(max_backtrack)

        new_trajectories = []

        # Explore all valid actions from current node
        for action, next_node in last_node.children.items():
            if action not in self._inherits_to:
                # Create a new trajectory with this action
                new_trajectory = self.copy()
                new_trajectory.update(action, step)

                if max_backtrack is not None:
                    new_has_backtrack = new_trajectory.has_backtrack(max_backtrack)

                    if not already_has_backtrack and new_has_backtrack:
                        continue

                # Skip if this made the trajectory invalid
                if new_trajectory.invalid:
                    continue

                self._inherits_to[action] = new_trajectory

                new_trajectories.append(new_trajectory)

        return new_trajectories
