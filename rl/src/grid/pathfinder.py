from dataclasses import dataclass

import heapq

from .map import Map
from .utils import Point, Direction, Action
from .node import DirectionalNode


@dataclass
class PathfinderConfig:
    """Configuration for the Pathfinder class.

    Attributes:
        use_viewcone: Whether to use viewcone for action selection. If False, only path efficiency is considered.
    """
    use_viewcone: bool = False

class Pathfinder:
    def __init__(
        self,
        map: Map,
        config: PathfinderConfig
    ):
        self.map = map
        self.config = config

    @property
    def registry(self):
        return self.map.registry

    @property
    def trees(self):
        return self.map.trees

    def get_optimal_action(self, position: Point, direction: Direction, tree_index: int = 0) -> Action:
        """
        Get the optimal action to take from the current position and direction.
        Prioritizes seeing tiles with high probability densities, then reaching reward positions.

        Behavior is modified by PathfinderConfig options:
        - use_viewcone: When True, considers field of view in action selection

        Args:
            position: Current position of the agent
            direction: Current direction the agent is facing
            tree_index: Index of the trajectory tree to use

        Returns:
            Action: The optimal action to take
        """
        # Validate and get trajectory tree
        tree = self._validate_and_get_tree(tree_index)

        # Get the current node from the registry
        start_node = self.registry.get_or_create_node(position, direction)

        # Find positions with rewards
        reward_positions = self._find_reward_positions(tree)

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
        best_action = self._select_best_action(reward_positions, best_paths_to_rewards, tree.probability_density)

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

    def _validate_and_get_tree(self, tree_index: int):
        """Validates the tree index and returns the corresponding tree."""
        if tree_index >= len(self.trees):
            raise ValueError(f"Invalid tree_index: {tree_index}. Only {len(self.trees)} trees available.")
        return self.trees[tree_index]

    def _find_reward_positions(self, tree):
        """Finds all positions with positive rewards in the reward density."""
        reward_positions = []
        reward_density = tree.probability_density

        for y in range(self.map.size):
            for x in range(self.map.size):
                if reward_density[y, x] > 0:
                    reward_positions.append((Point(x, y), reward_density[y, x]))

        return reward_positions

    def _get_default_action(self, node: DirectionalNode):
        """Returns a default valid action from the given node."""
        # Return a valid action if possible
        for action in Action:
            if action in node.children:
                return action
        return Action.STAY  # Default if no valid actions

    def _find_paths_to_rewards(self, start_node, reward_positions):
        """
        Uses Dijkstra's algorithm to find shortest paths to all reward positions.

        Returns:
            A dictionary mapping reward positions to (node, distance, first_action)
        """
        # Maps node hash -> (distance, first_action, previous_node_hash)
        distances = {hash(start_node): (0, None, None)}
        visited = set()  # Set of visited node hashes

        # Priority queue stores (distance, node_hash)
        pq = [(0, hash(start_node))]

        # Maps reward position -> (closest node, distance, first_action)
        best_paths_to_rewards = {}

        while pq:
            current_dist, current_hash = heapq.heappop(pq)

            # Skip if already processed this node with a shorter path
            if current_hash in visited:
                continue

            # Mark as visited
            visited.add(current_hash)
            current_node = self.registry.nodes[current_hash]

            # Check if this node is at a reward position
            self._update_paths_if_at_reward(
                current_node, current_hash, current_dist,
                reward_positions, distances, best_paths_to_rewards
            )

            # If we've found paths to all rewards, we can stop
            if len(best_paths_to_rewards) == len(reward_positions):
                break

            # Process neighbors
            self._process_neighbors(
                current_node, current_hash, current_dist,
                start_node, distances, visited, pq
            )

        return best_paths_to_rewards

    def _update_paths_if_at_reward(self, current_node, current_hash, current_dist,
                                  reward_positions, distances, best_paths_to_rewards):
        """Updates best_paths_to_rewards if the current node is at a reward position."""
        for reward_pos, reward_value in reward_positions:
            if current_node.position == reward_pos:
                # Get the first action that led to this path
                _, first_action, _ = distances[current_hash]

                # Update best path to this reward if better
                if reward_pos not in best_paths_to_rewards or current_dist < best_paths_to_rewards[reward_pos][1]:
                    best_paths_to_rewards[reward_pos] = (current_node, current_dist, first_action)

    def _process_neighbors(self, current_node, current_hash, current_dist,
                          start_node, distances, visited, pq):
        """Processes all neighbors of the current node in Dijkstra's algorithm."""
        for action, next_node in current_node.children.items():
            next_hash = hash(next_node)

            if next_hash in visited:
                continue

            # Calculate new distance
            new_dist = current_dist + 1

            # If this is a shorter path or a new node
            if next_hash not in distances or new_dist < distances[next_hash][0]:
                # Determine first action in the path
                first_action = action if current_hash == hash(start_node) else distances[current_hash][1]

                # Update distance and path information
                distances[next_hash] = (new_dist, first_action, current_hash)

                # Add to priority queue
                heapq.heappush(pq, (new_dist, next_hash))

    def _select_best_action(self, reward_positions, best_paths_to_rewards, probability_density):
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
