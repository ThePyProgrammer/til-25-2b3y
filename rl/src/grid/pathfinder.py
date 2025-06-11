from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .map import Map
from .trajectory import TrajectoryTree
from .utils import (
    Point,
    Direction,
    Action,
    find_shortest_paths,
    find_reward_positions,
    get_node_neighbors,
)
from .node import DirectionalNode, NodeRegistry

# VIEWCONE_INDICES[direction] = [x neg offset, x pos offset, y neg offset, y pos offset]
VIEWCONE_INDICES = [
    [-2, 4, -2, 2],  # RIGHT
    [-2, 2, -2, 4],  # DOWN
    [-4, 2, -2, 2],  # LEFT
    [-2, 2, -4, 2],  # UP
]
LAST_POSITION_PENALTY = 0.5  # Penalty for revisits


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
    def __init__(self, map: Map, config: PathfinderConfig) -> None:
        self.map = map
        self.config = config

        self.last_positions_counter = Counter()

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
        is_guard: bool = True,
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
            is_guard: Whether the agent is a guard (will avoid other guards if True)

        Returns:
            Action: The optimal action to take
        """
        # Get the current node from the registry
        start_node = self.registry.get_or_create_node(position, direction)

        def _inner():
            if destination is None:
                # Validate and get trajectory tree
                tree = self._validate_and_get_tree(tree_index)
                self.density: NDArray[np.float32] = tree.probability_density
            elif isinstance(destination, Point):
                self.density: NDArray[np.float32] = np.zeros(
                    (self.map.size, self.map.size), dtype=np.float32
                )
                self.density[destination.y, destination.x] = 1
            else:
                raise TypeError(
                    f"destination should be of type Optional[Point] not {type(destination)}"
                )

            # Get guard positions if this is a guard agent
            self.guard_positions = []
            if is_guard:
                guards_map = self.map.get_guards()
                for y in range(self.map.size):
                    for x in range(self.map.size):
                        # Add guard position to avoid list, excluding current position
                        if guards_map[y, x] and not (
                            x == position.x and y == position.y
                        ):
                            self.guard_positions.append(Point(x, y))

            # Find positions with rewards
            reward_positions = self._find_reward_positions()

            # If no rewards found, return an action that maximizes visibility of high probability areas
            if not reward_positions:
                # Try all possible actions and choose the one with best view
                candidate_actions = []
                for action in Action:
                    if action in start_node.children:
                        child_node = start_node.children[action]
                        # Skip actions that would lead to guard positions
                        if self.guard_positions:
                            if self._is_blocked_by_guard(child_node.position):
                                continue
                        candidate_actions.append((action, child_node))

                if candidate_actions:
                    if self.config.use_viewcone:
                        # no rewards, just pick the first available action
                        # return candidate_actions[0][0]
                        raise NotImplementedError("use_viewcone is not implemented")
                    else:
                        # If not using viewcone, just pick the first available action
                        return candidate_actions[0][0]
                return self._get_default_action(start_node)

            # Find shortest paths to all rewards
            best_paths_to_rewards = self._find_paths_to_rewards(
                start_node, reward_positions
            )

            # Choose the best action prioritizing visibility of high probability areas
            # then considering reward/distance ratio
            best_action = self._select_best_action(
                start_node, reward_positions, best_paths_to_rewards
            )

            # If no path found to any reward, choose action with best view
            if best_action is None:
                # Try all possible actions and choose the one with best view
                candidate_actions = []
                for action in Action:
                    if action in start_node.children:
                        child_node = start_node.children[action]
                        # Skip actions that would lead to guard positions
                        if self.guard_positions:
                            if self._is_blocked_by_guard(child_node.position):
                                continue
                        candidate_actions.append((action, child_node))

                if candidate_actions:
                    if self.config.use_viewcone:
                        # If using viewcone, sort by view score
                        candidate_actions.sort(
                            key=lambda x: self._get_view_score(x[1]), reverse=True
                        )
                        # Return the action with the best view score
                        return candidate_actions[0][0]
                    else:
                        # If not using viewcone, just pick the first available action
                        return candidate_actions[0][0]
                return self._get_default_action(start_node)

            return best_action

        action = _inner()
        self.last_positions_counter[start_node.children[action]] += 1
        return action

    def _validate_and_get_tree(self, tree_index: int) -> "TrajectoryTree":
        """Validates the tree index and returns the corresponding tree."""
        if tree_index >= len(self.trees):
            raise ValueError(
                f"Invalid tree_index: {tree_index}. Only {len(self.trees)} trees available."
            )
        return self.trees[tree_index]

    def _find_reward_positions(self, threshold=0.0) -> list[tuple[Point, float]]:
        """Finds all positions with positive rewards in the reward density."""
        return find_reward_positions(self.density, threshold=threshold)

    def _get_default_action(self, node: DirectionalNode) -> Action:
        """Returns a default valid action from the given node."""
        # Return a valid action if possible
        for action in Action:
            if action in node.children:
                # Skip actions that would lead to guard positions
                if self.guard_positions:
                    child_node = node.children[action]
                    if self._is_blocked_by_guard(child_node.position):
                        continue
                return action
        return Action.STAY  # Default if no valid actions

    def _is_blocked_by_guard(self, position: Point) -> bool:
        """
        Check if a position is blocked by a guard.

        Args:
            position: The position to check

        Returns:
            bool: True if the position is blocked, False otherwise
        """
        for guard_pos in self.guard_positions:
            if position.x == guard_pos.x and position.y == guard_pos.y:
                return True
        return False

    def _get_view_score(self, node: DirectionalNode) -> float:
        indices = VIEWCONE_INDICES[node.direction]
        view_score = self.density[
            max(0, node.position.x + indices[0]) : min(
                node.position.x + indices[1] + 1, 16
            ),
            max(0, node.position.y + indices[2]) : min(
                node.position.y + indices[3] + 1, 16
            ),
        ].mean()
        return view_score

    def _find_paths_to_rewards(
        self, start_node: DirectionalNode, reward_positions: list[tuple[Point, float]]
    ) -> dict[Point, tuple[DirectionalNode, int, Action]]:
        """
        Uses Dijkstra's algorithm to find shortest paths to all reward positions.
        Avoids paths that would lead to guard positions if the agent is a guard.

        Returns:
            A dictionary mapping reward positions to (node, distance, first_action)
        """
        # Extract reward positions as DirectionalNode objects
        goal_points = [pos for pos, _ in reward_positions]

        # Create goal nodes (one for each direction at each reward position)
        goal_nodes = []
        for point in goal_points:
            for direction in Direction:
                if (
                    node := self.registry.get_or_create_node(point, direction)
                ) is not None:
                    goal_nodes.append(node)

        # Custom neighbor function that avoids guard positions
        def get_neighbors_avoiding_guards(node):
            neighbors = get_node_neighbors(node)
            # Filter out neighbors that would move to a guard position
            if self.guard_positions:
                neighbors = {
                    action: neighbor
                    for action, neighbor in neighbors.items()
                    if not self._is_blocked_by_guard(neighbor.position)
                }
            return neighbors

        # Find shortest paths to all goals
        path_results = find_shortest_paths(
            start_node=start_node,
            goal_nodes=goal_nodes,
            get_neighbors=get_neighbors_avoiding_guards,
            node_hash=hash,
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
        start_node: DirectionalNode,
        reward_positions: list[tuple[Point, float]],
        best_paths_to_rewards: dict[Point, tuple[DirectionalNode, int, Action]],
    ) -> Optional[Action]:
        """
        Selects the best action prioritizing visibility of tiles with high probability density,
        then considering the reward/distance ratio.

        If use_viewcone is False, only the path efficiency (reward/distance) is considered.
        When the agent is a guard, also avoids actions that would lead to other guard positions.
        """
        # First, evaluate all candidate actions and group by first_action
        action_data_map: dict[
            Action, list[tuple[Optional[DirectionalNode], float, float]]
        ] = {}

        for reward_pos, reward_value in reward_positions:
            if reward_pos in best_paths_to_rewards:
                node, distance, first_action = best_paths_to_rewards[reward_pos]

                if first_action is None:
                    continue

                # Skip actions that would lead to guard positions
                if self.guard_positions and first_action in node.children:
                    next_node = node.children[first_action]
                    if self._is_blocked_by_guard(next_node.position):
                        continue

                # Calculate path efficiency score
                if distance == 0:  # Agent is already at the reward or a path of 0 cost
                    path_ratio = (
                        float("inf") if reward_value > 0 else 0
                    )  # Infinite if positive reward, 0 otherwise
                else:
                    path_ratio = reward_value / distance

                # Calculate view score if using viewcone
                view_score = 0.0  # Default view_score
                # if self.config.use_viewcone:
                #     view_score = self._get_view_score(node)

                if first_action not in action_data_map:
                    action_data_map[first_action] = []
                # Store action with its scores: (originating_node_for_this_path_segment, view_score, path_ratio)
                action_data_map[first_action].append((node, view_score, path_ratio))

        # If no candidates found, return None
        if not action_data_map:
            return None

        # Calculate average scores for each action
        # Action -> (avg_view_score, avg_path_ratio)
        averaged_scores: list[tuple[Action, float, float]] = []
        for action, scores_list in action_data_map.items():
            if not scores_list:
                continue
            # if self.config.use_viewcone:
            #     if action in start_node.children:
            #         # Get the next node in the path
            #         next_node = start_node.children[action]
            #         print(next_node)
            #         avg_view_score = self._get_view_score(next_node)
            # else:
            avg_view_score = sum(s[1] for s in scores_list) / len(scores_list)
            avg_path_ratio = sum(s[2] for s in scores_list) / len(scores_list)
            print(action, avg_view_score, avg_path_ratio)
            averaged_scores.append((action, avg_view_score, avg_path_ratio))

        if not averaged_scores:
            return None

        # Sort actions by appropriate criteria based on use_viewcone flag
        if self.config.use_viewcone:
            # Sort by average view score (primary, descending) and average path ratio (secondary, descending)
            # Current view_score is always 0 due to NotImplementedError, so this effectively sorts by path_ratio only for now.
            averaged_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        else:
            # apply penalties for revisiting last positions
            for i in range(len(averaged_scores)):
                action, view_score, path_ratio = averaged_scores[i]
                if start_node.children[action] in self.last_positions_counter:
                    penalty = (
                        LAST_POSITION_PENALTY
                        ** self.last_positions_counter[start_node.children[action]]
                    )
                    averaged_scores[i] = (action, view_score, path_ratio * penalty)
            # Sort only by average path ratio (descending) when not using viewcone
            averaged_scores.sort(key=lambda x: x[2], reverse=True)

        # Return the best action
        return averaged_scores[0][0]

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
