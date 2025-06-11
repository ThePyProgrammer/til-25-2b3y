from dataclasses import dataclass
from typing import (
    TypeVar,
    Generic,
    Optional,
    Callable,
    Dict,
    Set,
    List,
    Tuple,
    Any,
    TYPE_CHECKING,
)
import heapq

import numpy as np
from numpy.typing import NDArray

from .geometry import Point, MOVEMENT_VECTORS
from .enums import Direction, Action

if TYPE_CHECKING:
    from ..node import DirectionalNode

# Type variables for generic pathfinding
T = TypeVar("T")  # Node type
C = TypeVar("C")  # Cost type


class PathNode(Generic[T]):
    """Wrapper for nodes in pathfinding algorithms with tracking info."""

    def __init__(
        self,
        node: T,
        parent: Optional["PathNode[T]"] = None,
        action: Optional[Action] = None,
        g_cost: float = 0,
        h_cost: float = 0,
    ):
        self.node = node
        self.parent = parent
        self.action = action  # Action taken to reach this node from parent
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal

    @property
    def f_cost(self) -> float:
        """Total estimated cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "PathNode[T]") -> bool:
        """Comparison for priority queue."""
        return self.f_cost < other.f_cost


@dataclass
class PathResult(Generic[T]):
    """Result of a pathfinding search."""

    path: List[T]
    actions: List[Action]
    cost: float
    first_action: Optional[Action]

    @property
    def success(self) -> bool:
        """Whether a path was found."""
        return len(self.path) > 0

    @property
    def nodes(self) -> List[T]:
        """Alias for path."""
        return self.path


def find_path(
    start_node: T,
    is_goal: Callable[[T], bool],
    get_neighbors: Callable[[T], Dict[Action, T]],
    heuristic: Callable[[T], float] = lambda _: 0,
    node_hash: Callable[[T], Any] = hash,
    max_iterations: int = 10000,
) -> PathResult[T]:
    """
    Generic A* pathfinding algorithm.

    Args:
        start_node: The starting node
        is_goal: Function that returns True if the node is a goal
        get_neighbors: Function that returns dict of {action: neighbor_node}
        heuristic: Optional function to estimate cost to goal
        node_hash: Function to convert nodes to hashable objects
        max_iterations: Maximum number of nodes to explore

    Returns:
        PathResult containing the path, actions, cost, and first action
    """
    # Initialize open and closed sets
    open_set: List[PathNode[T]] = []
    closed_set: Set[Any] = set()

    # Node lookup for reconstructing path
    node_lookup: Dict[Any, PathNode[T]] = {}

    # Add start node to open set
    start_path_node = PathNode(start_node, None, None, 0, heuristic(start_node))
    heapq.heappush(open_set, start_path_node)
    node_lookup[node_hash(start_node)] = start_path_node

    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1

        # Get node with lowest f_cost
        current = heapq.heappop(open_set)

        # Mark as processed
        closed_set.add(node_hash(current.node))

        # Check if goal reached
        if is_goal(current.node):
            return reconstruct_path(current)

        # Process neighbors
        for action, neighbor in get_neighbors(current.node).items():
            neighbor_hash = node_hash(neighbor)

            # Skip if already processed
            if neighbor_hash in closed_set:
                continue

            # Calculate costs
            g_cost = current.g_cost + 1  # Assuming unit cost for each step
            h_cost = heuristic(neighbor)

            # Check if better path found
            if neighbor_hash not in [node_hash(n.node) for n in open_set]:
                neighbor_node = PathNode(neighbor, current, action, g_cost, h_cost)
                heapq.heappush(open_set, neighbor_node)
                node_lookup[neighbor_hash] = neighbor_node
            else:
                # Get existing node from open set
                for i, node in enumerate(open_set):
                    if node_hash(node.node) == neighbor_hash:
                        # Update if better path found
                        if g_cost < node.g_cost:
                            # Replace in open set
                            open_set[i] = PathNode(
                                neighbor, current, action, g_cost, h_cost
                            )
                            heapq.heapify(open_set)
                            node_lookup[neighbor_hash] = open_set[i]
                        break

    # No path found
    return PathResult([], [], float("inf"), None)


def reconstruct_path(end_node: PathNode[T]) -> PathResult[T]:
    """
    Reconstructs the path from start to end.

    Args:
        end_node: The final node in the path

    Returns:
        PathResult containing the path, actions, cost, and first action
    """
    path = []
    actions = []
    current = end_node
    first_action = None

    while current:
        path.append(current.node)
        if current.action:
            actions.append(current.action)
            # Save the first action (will be the last one we process)
            if not current.parent or not current.parent.parent:
                first_action = current.action
        current = current.parent

    # Reverse to get path from start to end
    path.reverse()
    actions.reverse()

    return PathResult(path, actions, end_node.g_cost, first_action)


def find_shortest_paths(
    start_node: T,
    goal_nodes: List[T],
    get_neighbors: Callable[[T], Dict[Action, T]],
    node_hash: Callable[[T], Any] = hash,
    max_iterations: int = 10000,
) -> Dict[T, PathResult[T]]:
    """
    Finds shortest paths from start to multiple goals using Dijkstra's algorithm.

    Args:
        start_node: Starting node
        goal_nodes: List of goal nodes to find paths to
        get_neighbors: Function that returns dict of {action: neighbor_node}
        node_hash: Function to convert nodes to hashable objects
        max_iterations: Maximum nodes to explore

    Returns:
        Dict mapping goal nodes to their PathResults
    """
    # Set of goal node hashes
    goal_hashes = {node_hash(goal) for goal in goal_nodes}

    # Maps node hash -> (distance, first_action, parent_hash)
    distances: Dict[Any, Tuple[float, Optional[Action], Optional[Any]]] = {
        node_hash(start_node): (0, None, None)
    }

    # Set of visited node hashes
    visited: Set[Any] = set()

    # Priority queue stores (distance, node_hash)
    pq = [(0, node_hash(start_node))]

    # Map of node hashes to actual nodes
    hash_to_node: Dict[Any, T] = {node_hash(start_node): start_node}

    # Maps goal node -> PathResult
    results: Dict[T, PathResult[T]] = {}

    iterations = 0

    while pq and iterations < max_iterations and len(results) < len(goal_nodes):
        iterations += 1

        # Get node with lowest distance
        current_dist, current_hash = heapq.heappop(pq)

        # Skip if already processed
        if current_hash in visited:
            continue

        # Mark as visited
        visited.add(current_hash)
        current_node = hash_to_node[current_hash]

        # Check if this is a goal node
        if current_hash in goal_hashes:
            # Reconstruct path
            path = reconstruct_dijkstra_path(current_hash, distances, hash_to_node)

            # Find the goal this node corresponds to
            for goal in goal_nodes:
                if node_hash(goal) == current_hash:
                    # Get first action
                    _, first_action, _ = distances[current_hash]

                    # Create PathResult
                    results[goal] = PathResult(
                        path=path,
                        actions=get_actions_from_path(path, get_neighbors, node_hash),
                        cost=current_dist,
                        first_action=first_action,
                    )
                    break

            # If we've found all goals, we can stop
            if len(results) == len(goal_nodes):
                break

        # Process neighbors
        for action, neighbor in get_neighbors(current_node).items():
            neighbor_hash = node_hash(neighbor)

            # Skip if already processed
            if neighbor_hash in visited:
                continue

            # Calculate new distance
            new_dist = current_dist + 1  # Assuming unit cost

            # Store the node for later
            hash_to_node[neighbor_hash] = neighbor

            # If this is a shorter path or a new node
            if neighbor_hash not in distances or new_dist < distances[neighbor_hash][0]:
                # Determine first action in the path
                first_action = (
                    action
                    if current_hash == node_hash(start_node)
                    else distances[current_hash][1]
                )

                # Update distance and path info
                distances[neighbor_hash] = (new_dist, first_action, current_hash)

                # Add to priority queue
                heapq.heappush(pq, (new_dist, neighbor_hash))

    return results


def reconstruct_dijkstra_path(
    end_hash: Any,
    distances: Dict[Any, Tuple[float, Optional[Action], Optional[Any]]],
    hash_to_node: Dict[Any, T],
) -> List[T]:
    """
    Reconstructs path from Dijkstra's algorithm results.

    Args:
        end_hash: Hash of the end node
        distances: Map of node hash to (distance, first_action, parent_hash)
        hash_to_node: Map of node hash to actual node

    Returns:
        List of nodes from start to end
    """
    path = []
    current_hash = end_hash

    while current_hash is not None:
        path.append(hash_to_node[current_hash])
        _, _, parent_hash = distances[current_hash]
        current_hash = parent_hash

    # Reverse to get path from start to end
    path.reverse()
    return path


def get_actions_from_path(
    path: List[T],
    get_neighbors: Callable[[T], Dict[Action, T]],
    node_hash: Callable[[T], Any] = hash,
) -> List[Action]:
    """
    Determines the actions taken along a path.

    Args:
        path: List of nodes from start to end
        get_neighbors: Function that returns neighbors of a node
        node_hash: Function to convert nodes to hashable objects

    Returns:
        List of actions taken along the path
    """
    actions = []

    if len(path) < 2:
        return actions

    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        next_hash = node_hash(next_node)

        # Find the action that leads from current to next
        for action, neighbor in get_neighbors(current).items():
            if node_hash(neighbor) == next_hash:
                actions.append(action)
                break

    return actions


def manhattan_distance(p1: Point, p2: Point) -> int:
    """
    Calculate Manhattan distance between two points.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Manhattan distance (|x1-x2| + |y1-y2|)
    """
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def get_directional_neighbors(
    state: Tuple[Point, Direction],
) -> Dict[Action, Tuple[Point, Direction]]:
    """
    Gets neighboring states for a positional state with direction.

    Args:
        state: A tuple of (position, direction)

    Returns:
        Dictionary mapping actions to resulting (position, direction) states
    """
    pos, direction = state
    neighbors = {}

    # STAY action (no change)
    neighbors[Action.STAY] = (Point(pos.x, pos.y), direction)

    # FORWARD action
    dx, dy = MOVEMENT_VECTORS[direction]
    neighbors[Action.FORWARD] = (Point(pos.x + dx, pos.y + dy), direction)

    # BACKWARD action
    neighbors[Action.BACKWARD] = (Point(pos.x - dx, pos.y - dy), direction)

    # LEFT action (turn left)
    neighbors[Action.LEFT] = (Point(pos.x, pos.y), direction.turn_left())

    # RIGHT action (turn right)
    neighbors[Action.RIGHT] = (Point(pos.x, pos.y), direction.turn_right())

    return neighbors


def get_node_neighbors(node: "DirectionalNode") -> Dict[Action, "DirectionalNode"]:
    """
    Gets neighboring nodes for a DirectionalNode.

    Args:
        node: A DirectionalNode object

    Returns:
        Dictionary mapping actions to resulting DirectionalNode objects
    """
    return node.children


def find_reward_positions(
    density: NDArray[np.float32], threshold: float = 0.0
) -> List[Tuple[Point, float]]:
    """
    Finds positions with rewards above a threshold in a density map.

    Args:
        density: 2D array of probability density values
        threshold: Minimum value to consider as a reward

    Returns:
        List of (Point, value) tuples for positions with rewards
    """
    height, width = density.shape
    reward_positions = []

    for y in range(height):
        for x in range(width):
            if density[y, x] > threshold:
                reward_positions.append((Point(x, y), density[y, x]))

    return reward_positions
