from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .map import Map
from .trajectory import TrajectoryTree
from .utils import Point, Direction, Action, find_shortest_paths, find_reward_positions, get_node_neighbors
from .node import DirectionalNode, NodeRegistry


@dataclass
class PathfinderConfig:
    """Configuration for the Pathfinder class.

    Attributes:
        use_viewcone: Whether to use viewcone for action selection. If False, only path efficiency is considered.
        use_path_density: Whether to use path density instead of the default (random walk) probability density
    """
    use_viewcone: bool = False
    use_path_density: bool = False

class Pathfinder:
    def __init__(
        self,
        map: Map,
        config: PathfinderConfig
    ) -> None:
        self.map = map
        self.config = config

    @property
    def registry(self) -> NodeRegistry:
        return self.map.registry

    @property
    def trees(self) -> list[TrajectoryTree]:
        return self.map.trees

    def get_optimal_action(
        self,
        position: Point,
        direction: Direction,
        tree_index: int = 0,
        destination: Optional[Point] = None,
    ) -> Action:
        """
        Get the optimal action to take from the current position and direction.
        Prioritizes seeing tiles with high probability densities, then reaching reward positions.

        Behavior is modified by PathfinderConfig options:
        - use_viewcone: When True, considers field of view in action selection

        Args:
            position: Current position of the agent
            direction: Current direction the agent is facing
            tree_index: Index of the trajectory tree to use
            destination: Path find to some point instead of using the tree

        Returns:
            Action: The optimal action to take
        """
        if destination is None:
            # Validate and get trajectory tree
            tree = self._validate_and_get_tree(tree_index)
            self.density: NDArray[np.float32] = tree.probability_density
        elif isinstance(destination, Point):
            self.density: NDArray[np.float32] = np.zeros((self.map.size, self.map.size), dtype=np.float32)
            self.density[destination.y, destination.x] = 1
        else:
            raise TypeError(f"destination should be of type Optional[Point] not {type(destination)}")

        # Get the current node from the registry
        start_node = self.registry.get_or_create_node(position, direction)

        # Find positions with rewards
        reward_positions = self._find_reward_positions()

        # If no rewards found, return an action that maximizes visibility of high probability areas
        if not reward_positions:
            # Try all possible actions and choose the one with best view
            candidate_actions = []
            for action in Action:
                if action in start_node.children:
                    child_node = start_node.children[action]
                    candidate_actions.append((action, child_node))

            if candidate_actions:
                if self.config.use_viewcone:
                    raise NotImplementedError("use_viewcone is not implemented")
                else:
                    # If not using viewcone, just pick the first available action
                    return candidate_actions[0][0]
            return self._get_default_action(start_node)

        # Find shortest paths to all rewards
        best_paths_to_rewards = self._find_paths_to_rewards(start_node, reward_positions)

        # Choose the best action prioritizing visibility of high probability areas
        # then considering reward/distance ratio
        best_action = self._select_best_action(reward_positions, best_paths_to_rewards)

        # If no path found to any reward, choose action with best view
        if best_action is None:
            # Try all possible actions and choose the one with best view
            candidate_actions = []
            for action in Action:
                if action in start_node.children:
                    child_node = start_node.children[action]
                    candidate_actions.append((action, child_node))

            if candidate_actions:
                if self.config.use_viewcone:
                    raise NotImplementedError("use_viewcone is not implemented")
                else:
                    # If not using viewcone, just pick the first available action
                    return candidate_actions[0][0]
            return self._get_default_action(start_node)

        return best_action

    def _validate_and_get_tree(self, tree_index: int) -> 'TrajectoryTree':
        """Validates the tree index and returns the corresponding tree."""
        if tree_index >= len(self.trees):
            raise ValueError(f"Invalid tree_index: {tree_index}. Only {len(self.trees)} trees available.")
        return self.trees[tree_index]

    def _find_reward_positions(self) -> list[tuple[Point, float]]:
        """Finds all positions with positive rewards in the reward density."""
        return find_reward_positions(self.density, threshold=0.0)

    def _get_default_action(self, node: DirectionalNode) -> Action:
        """Returns a default valid action from the given node."""
        # Return a valid action if possible
        for action in Action:
            if action in node.children:
                return action
        return Action.STAY  # Default if no valid actions

    def _find_paths_to_rewards(
        self,
        start_node: DirectionalNode,
        reward_positions: list[tuple[Point, float]]
    ) -> dict[Point, tuple[DirectionalNode, int, Action]]:
        """
        Uses Dijkstra's algorithm to find shortest paths to all reward positions.

        Returns:
            A dictionary mapping reward positions to (node, distance, first_action)
        """
        # Extract reward positions as DirectionalNode objects
        goal_points = [pos for pos, _ in reward_positions]

        # Create goal nodes (one for each direction at each reward position)
        goal_nodes = []
        for point in goal_points:
            for direction in Direction:
                if (node := self.registry.get_or_create_node(point, direction)) is not None:
                    goal_nodes.append(node)

        # Find shortest paths to all goals
        path_results = find_shortest_paths(
            start_node=start_node,
            goal_nodes=goal_nodes,
            get_neighbors=get_node_neighbors,
            node_hash=hash
        )

        # Convert to expected format: Point -> (node, distance, first_action)
        best_paths_to_rewards = {}

        for node, result in path_results.items():
            # Skip if no path found or no first action
            if not result.success or result.first_action is None:
                continue

            # Get the point for this node
            point = node.position

            # Check if we already have a better path to this point
            if (
                point in best_paths_to_rewards
                and best_paths_to_rewards[point][1] <= result.cost
            ):
                continue

            # Add or update the path
            best_paths_to_rewards[point] = (node, int(result.cost), result.first_action)

        return best_paths_to_rewards

    # These methods have been replaced by using find_shortest_paths utility function

    def _select_best_action(
        self,
        reward_positions: list[tuple[Point, float]],
        best_paths_to_rewards: dict[Point, tuple[DirectionalNode, int, Action]]
    ) -> Optional[Action]:
        """
        Selects the best action prioritizing visibility of tiles with high probability density,
        then considering the reward/distance ratio.

        If use_viewcone is False, only the path efficiency (reward/distance) is considered.
        """
        # First, evaluate all candidate actions
        scored_actions = []

        for reward_pos, reward_value in reward_positions:
            if reward_pos in best_paths_to_rewards:
                node, distance, first_action = best_paths_to_rewards[reward_pos]

                if first_action is None:
                    continue

                # Calculate path efficiency score
                if distance == 0:
                    path_ratio = float('inf')
                else:
                    path_ratio = reward_value / distance

                # Calculate view score if using viewcone
                view_score = 0
                if self.config.use_viewcone:
                    raise NotImplementedError("use_viewcone is not implemented")
                    # view_tiles = self._get_viewcone_tiles(node.position, node.direction)

                # Store action with its scores
                scored_actions.append((first_action, node, view_score, path_ratio))

        # If no candidates found, return None
        if not scored_actions:
            return None

        # Sort actions by appropriate criteria based on use_viewcone flag
        if self.config.use_viewcone:
            # Sort by view score (primary) and path ratio (secondary)
            scored_actions.sort(key=lambda x: (x[2], x[3]), reverse=True)
        else:
            # Sort only by path ratio when not using viewcone
            scored_actions.sort(key=lambda x: x[3], reverse=True)

        # Return the best action
        return scored_actions[0][0]

    def _get_viewcone_tiles(self, position: Point, direction: Direction) -> list[Point]:
        """
        Returns a list of tile positions in the viewcone from a given position and direction.

        The viewcone is 7x5, stretching out:
        - 4 tiles forward
        - 2 tiles to the left, right, and backward
        for a total of 35 tiles.

        Args:
            position: Position of the agent
            direction: Direction the agent is facing

        Returns:
            List of Point objects representing tile positions in the viewcone
        """
        viewcone_tiles = []

        if direction == Direction.RIGHT:
            # Forward = +x, width along y-axis
            for dx in range(-2, 5):  # -2 to 4 (2 back, position, 4 forward)
                for dy in range(-2, 3):  # -2 to 2 (2 up, position, 2 down)
                    viewcone_tiles.append(Point(position.x + dx, position.y + dy))

        elif direction == Direction.LEFT:
            # Forward = -x, width along y-axis
            for dx in range(-4, 3):  # -4 to 2 (4 forward, position, 2 back)
                for dy in range(-2, 3):  # -2 to 2 (2 up, position, 2 down)
                    viewcone_tiles.append(Point(position.x + dx, position.y + dy))

        elif direction == Direction.DOWN:
            # Forward = +y, width along x-axis
            for dy in range(-2, 5):  # -2 to 4 (2 back, position, 4 forward)
                for dx in range(-2, 3):  # -2 to 2 (2 left, position, 2 right)
                    viewcone_tiles.append(Point(position.x + dx, position.y + dy))

        elif direction == Direction.UP:
            # Forward = -y, width along x-axis
            for dy in range(-4, 3):  # -4 to 2 (4 forward, position, 2 back)
                for dx in range(-2, 3):  # -2 to 2 (2 left, position, 2 right)
                    viewcone_tiles.append(Point(position.x + dx, position.y - dy))

        return viewcone_tiles
