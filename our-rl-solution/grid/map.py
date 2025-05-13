from typing import Optional

import numpy as np
import heapq

from .utils import (
    Direction,
    Action,
    Point,
    rotate_wall_bits,
    view_to_world,
    Tile,
    TileContent
)
from .node import NodeRegistry, DirectionalNode
from .trajectory import TrajectoryTree


class Map:
    EMPTY = TileContent.EMPTY
    RECON = TileContent.RECON
    MISSION = TileContent.MISSION

    def __init__(self, use_viewcone=False):
        """
        Initialize an empty map of the environment.

        Args:
            use_viewcone: Whether to use viewcone for action selection (default: True)
        """
        self.size = 16  # Assume a 16x16 environment
        self.map = np.zeros((self.size, self.size), dtype=np.uint8)
        self.viewed = np.zeros((self.size, self.size), dtype=bool)
        self.last_updated = np.zeros((self.size, self.size), dtype=np.int32)  # Timestamp for last update
        self.recently_updated = np.zeros((self.size, self.size), dtype=np.uint8)
        self.step_counter = 0  # Count of steps/updates to use as timestamp
        self.use_viewcone = use_viewcone  # Feature flag for viewcone-based action selection

        self.registry = NodeRegistry(self.size)
        self._populate_nodes()

        self.trees: list[TrajectoryTree] = []

    def _populate_nodes(self):
        """Populate all possible nodes in the grid (assuming no walls)."""
        # Create nodes for all positions and directions
        for y in range(self.size):
            for x in range(self.size):
                for direction in Direction:
                    self.registry.get_or_create_node(Point(x, y), direction)

    def get_node(self, position: Point, direction: Direction) -> DirectionalNode:
        """
        Get a node from the registry.

        Args:
            x, y: Coordinates of the position
            direction: Direction the agent is facing

        Returns:
            DirectionalNode: The node at the specified position and direction
        """
        return self.registry.get_or_create_node(position, direction)

    def __call__(self, observation):
        """Update the map with a new observation."""
        return self.update(observation)

    def update(self, observation):
        """Update the map with a new observation.

        Args:
            observation: Dict containing 'viewcone', 'direction', 'location', etc.

        Returns:
            Updated map
        """
        viewcone = observation['viewcone']
        direction = observation['direction']
        agent_loc = observation['location']

        # Increment step counter
        self.step_counter += 1
        self.recently_updated = np.zeros((self.size, self.size), dtype=np.uint8)

        updated_cells = []
        observed_cells = []

        # Viewcone is 7x5 with agent at (2, 2)
        for i in range(viewcone.shape[0]):
            for j in range(viewcone.shape[1]):
                tile_value = viewcone[i, j]

                # Create a Tile instance for easy property access
                tile = Tile(tile_value)

                # Skip tiles with no vision, possible to hear agents without vision
                if not tile.is_visible and not tile.has_scout and not tile.has_guard:
                    continue

                # Convert viewcone coordinates to world coordinates
                # Agent is at position (2, 2) in the viewcone
                view_coord = np.array([i - 2, j - 2])  # Offset from agent position in viewcone
                world_coord = view_to_world(agent_loc, direction, view_coord)

                # Convert coordinates to integers to use as array indices
                x, y = int(world_coord[0]), int(world_coord[1])

                # Check if coordinates are within bounds
                if (0 <= x < self.size and 0 <= y < self.size and tile_value != 0):
                    # Create a Tile instance and rotate its walls to maintain global orientation
                    rotated_tile_value = rotate_wall_bits(tile_value, direction)

                    # Clear agent bits when processing the agent's own position in viewcone
                    if i == 2 and j == 2:
                        # Clear bits 2-3 (agent bits) but keep all other bits
                        rotated_tile_value = rotated_tile_value & ~0b1100

                    info = (Point(x, y), Tile(rotated_tile_value))
                    observed_cells.append(info)

                    # Check if wall information has changed
                    old_value = self.map[x, y]
                    if old_value != rotated_tile_value:
                        # Wall configuration might have changed
                        updated_cells.append(info)

                    # Update map with tile information
                    self.map[x, y] = rotated_tile_value
                    self.viewed[x, y] = True
                    self.last_updated[x, y] = self.step_counter  # Record when this cell was updated

        # Update node connections for cells with updated wall information
        for position, tile in updated_cells:
            self._update_node_connections(position, tile)

        for tree in self.trees:
            tree.step()
            tree.prune(observed_cells)

        return self.map

    def get_walls(self):
        """
        Extract wall information from the map.

        Returns:
            np.ndarray: Array of shape (size, size, 4) containing wall information.
                        The last dimension corresponds to [right, bottom, left, top] walls.
        """
        walls = np.zeros((self.size, self.size, 4), dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if self.viewed[y, x]:
                    # Use Tile utility to extract wall information
                    tile = Tile(self.map[y, x])
                    walls[y, x, 0] = tile.has_right_wall
                    walls[y, x, 1] = tile.has_bottom_wall
                    walls[y, x, 2] = tile.has_left_wall
                    walls[y, x, 3] = tile.has_top_wall

        return walls

    def get_tile_type(self):
        """
        Extract tile type information (empty, recon, mission) from the map.

        Returns:
            np.ndarray: Array of shape (size, size) containing tile type information.
        """
        tile_types: list[list[Optional[TileContent]]] = [[None for _ in range(16)] for _ in range(16)]

        for x in range(self.size):
            for y in range(self.size):
                if self.viewed[x, y]:
                    # Use Tile utility to extract tile type
                    tile = Tile(self.map[x, y])
                    tile_types[x][y] = tile.tile_content

        return tile_types

    def get_agents(self):
        """
        Extract agent information (scouts and guards) from the map.

        Returns:
            tuple: Two arrays of shape (size, size) for scouts and guards.
        """
        scouts = np.zeros((self.size, self.size), dtype=bool)
        guards = np.zeros((self.size, self.size), dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if self.viewed[y, x]:
                    # Use Tile utility to extract agent information
                    tile = Tile(self.map[y, x])
                    scouts[y, x] = tile.has_scout
                    guards[y, x] = tile.has_guard

        return scouts, guards

    def get_visited(self):
        """
        Return the visited locations.

        Returns:
            np.ndarray: Boolean array of shape (size, size) for visited locations.
        """
        return self.viewed

    @property
    def time_since_update(self):
        """
        Calculate how many steps have passed since each cell was last updated.

        Returns:
            np.ndarray: Array of shape (size, size) with the number of steps since last update.
                       Cells that have never been updated will have a large value.
        """
        # For cells that have never been updated (last_updated is 0),
        # we'll use a large value to indicate "not updated"
        result = np.ones((self.size, self.size), dtype=np.int32) * (self.step_counter + 1)

        # For visited cells, calculate the actual time difference
        mask = self.last_updated > 0
        result[mask] = self.step_counter - self.last_updated[mask]

        return result

    def _update_node_connections(self, position: Point, tile: Tile):
        """
        Update node connections based on wall information.

        Args:
            position: Coordinates of the updated cell
            tile_value: Updated tile value containing wall information
        """
        # Extract wall information from the Tile object
        walls = {
            'right': tile.has_right_wall,
            'bottom': tile.has_bottom_wall,
            'left': tile.has_left_wall,
            'top': tile.has_top_wall
        }

        # Define a mapping from direction + action to the wall that would block it
        # Format: {direction: {action: wall_location}}
        blocking_walls = {
            Direction.RIGHT: {
                Action.FORWARD: 'right',
                Action.BACKWARD: 'left',
                Action.LEFT: 'top',
                Action.RIGHT: 'bottom'
            },
            Direction.DOWN: {
                Action.FORWARD: 'bottom',
                Action.BACKWARD: 'top',
                Action.LEFT: 'right',
                Action.RIGHT: 'left'
            },
            Direction.LEFT: {
                Action.FORWARD: 'left',
                Action.BACKWARD: 'right',
                Action.LEFT: 'bottom',
                Action.RIGHT: 'top'
            },
            Direction.UP: {
                Action.FORWARD: 'top',
                Action.BACKWARD: 'bottom',
                Action.LEFT: 'left',
                Action.RIGHT: 'right'
            }
        }

        # Update each directional node at this position
        for direction in Direction:
            node = self.registry.get_or_create_node(position, direction)

            # Determine which actions are blocked by walls
            invalid_actions = []

            # Check each action
            for action in [Action.FORWARD, Action.BACKWARD, Action.LEFT, Action.RIGHT]:
                # Get the wall location that would block this action from this direction
                wall_location = blocking_walls[direction][action]

                # If there's a wall in that location and the action exists in node children
                if walls[wall_location] and action in node.children:
                    invalid_actions.append(action)

            # Remove invalid actions from node children
            for action in invalid_actions:
                if action in node.children:
                    del node.children[action]

        # Define wall relationships (opposites and position offsets)
        wall_relationships = {
            'right': ('left', Point(1, 0)),   # If right wall here, there's a left wall at x+1,y
            'left': ('right', Point(-1, 0)),  # If left wall here, there's a right wall at x-1,y
            'bottom': ('top', Point(0, 1)),   # If bottom wall here, there's a top wall at x,y+1
            'top': ('bottom', Point(0, -1))   # If top wall here, there's a bottom wall at x,y-1
        }

        # Update adjacent cells for walls that exist in the current cell
        for wall_dir, has_wall in walls.items():
            if has_wall and wall_dir in wall_relationships:
                # Get opposite wall direction and position offset
                opposite_wall, offset = wall_relationships[wall_dir]
                adjacent_pos = Point(position.x + offset.x, position.y + offset.y)

                # Only process if adjacent position is within grid bounds
                if 0 <= adjacent_pos.x < self.size and 0 <= adjacent_pos.y < self.size:
                    # For each direction at the adjacent cell
                    for adj_direction in Direction:
                        adj_node = self.registry.get_or_create_node(adjacent_pos, adj_direction)

                        # Find and remove actions that would be blocked by this wall
                        for adj_action in list(adj_node.children.keys()):
                            # Check if this action would be blocked by the opposite wall
                            if blocking_walls[adj_direction].get(adj_action) == opposite_wall:
                                # Remove the blocked action
                                if adj_action in adj_node.children:
                                    del adj_node.children[adj_action]

    def create_trajectory_tree(self, position, direction=None):
        """
        Create a trajectory tree starting from a specific position and direction.

        Args:
            position: Tuple (x, y) or Point representing the starting position
            direction: Optional starting direction (if None, will create trees for all directions)

        Returns:
            TrajectoryTree: A trajectory tree populated with valid paths
        """
        # Convert tuple to Point if necessary
        if isinstance(position, tuple):
            position = Point(position[0], position[1])

        # Create a trajectory tree with the current map's registry
        tree = TrajectoryTree(position, direction, self.size, registry=self.registry)

        self.trees.append(tree)

        return tree

    def get_optimal_action(self, position: Point, direction: Direction, tree_index: int = 0) -> Action:
        """
        Get the optimal action to take from the current position and direction.
        Prioritizes seeing tiles with high probability densities, then reaching reward positions.

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
                if self.use_viewcone:
                    return self._select_action_for_best_view(candidate_actions, tree.probability_density)
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
                if self.use_viewcone:
                    return self._select_action_for_best_view(candidate_actions, tree.probability_density)
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

        for y in range(self.size):
            for x in range(self.size):
                if reward_density[y, x] > 0:
                    reward_positions.append((Point(x, y), reward_density[y, x]))

        return reward_positions

    def _get_default_action(self, node):
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
                if self.use_viewcone:
                    view_tiles = self._get_viewcone_tiles(node.position, node.direction)
                    view_score = self._calculate_view_score(view_tiles, probability_density)

                # Store action with its scores
                scored_actions.append((first_action, node, view_score, path_ratio))

        # If no candidates found, return None
        if not scored_actions:
            return None

        # Sort actions by appropriate criteria based on use_viewcone flag
        if self.use_viewcone:
            # Sort by view score (primary) and path ratio (secondary)
            scored_actions.sort(key=lambda x: (x[2], x[3]), reverse=True)
        else:
            # Sort only by path ratio when not using viewcone
            scored_actions.sort(key=lambda x: x[3], reverse=True)

        # Return the best action
        return scored_actions[0][0]

    def _calculate_view_score(self, view_tiles, probability_density):
        """
        Calculates the view score based on probability densities and unexplored tiles in the viewcone.

        Args:
            view_tiles: List of Point objects representing positions in the viewcone
            probability_density: 2D array of probability densities from the trajectory tree

        Returns:
            float: Score representing the exploration value of the viewcone
        """
        view_score = 0

        # Count unexplored tiles in viewcone and weight by probability density
        for tile_pos in view_tiles:
            # Skip tiles outside grid boundaries
            if (tile_pos.x < 0 or tile_pos.x >= self.size or
                tile_pos.y < 0 or tile_pos.y >= self.size):
                continue

            # Get the probability density for this tile
            # Note: probability_density array is indexed as [y, x]
            prob_value = probability_density[tile_pos.y, tile_pos.x]

            view_score += prob_value * 10

        return view_score

    def _select_action_for_best_view(self, candidate_actions, probability_density):
        """
        Selects the action that maximizes the guard's field of view from candidate actions,
        prioritizing tiles with high probability density.

        The guard's viewcone is shaped like:
        - 4 tiles forward
        - 2 tiles to the left, right, and backward
        - Making the viewcone 7x5 in total

        Args:
            candidate_actions: List of tuples (action, resulting_node) with same reward/distance ratio
            probability_density: 2D array of probability densities from the trajectory tree

        Returns:
            Action: The action that maximizes view of high probability areas
        """
        best_action = candidate_actions[0][0]  # Default to first action
        max_view_score = -1

        for action, node in candidate_actions:
            # Get all tiles in the potential view cone
            view_tiles = self._get_viewcone_tiles(node.position, node.direction)

            # Calculate view score considering probability density
            view_score = self._calculate_view_score(view_tiles, probability_density)

            # Update best action if better view score found
            if view_score > max_view_score:
                max_view_score = view_score
                best_action = action

        return best_action

    def _get_viewcone_tiles(self, position, direction):
        """
        Returns a list of tile positions in the viewcone from a given position and direction.

        The viewcone is 7x5:
        - 4 tiles forward
        - 2 tiles to the left, right, and backward

        Args:
            position: Position of the agent
            direction: Direction the agent is facing

        Returns:
            List of Point objects representing tile positions in the viewcone
        """
        viewcone_tiles = []

        # Define direction vectors for each direction
        direction_vectors = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1)
        }

        # Get the direction vector for the current direction
        dx, dy = direction_vectors[direction]

        # Forward section: 4 tiles in the facing direction
        for i in range(1, 5):
            viewcone_tiles.append(Point(position.x + i * dx, position.y + i * dy))

        # Left and right sections (perpendicular to the direction)
        # For horizontal directions (RIGHT, LEFT), left/right means up/down
        # For vertical directions (DOWN, UP), left/right means left/right
        perpendicular_dx, perpendicular_dy = -dy, dx  # Rotate 90 degrees

        # Side tiles (left and right of forward direction)
        for side in range(-2, 3):
            if side == 0:
                continue  # Skip center line (handled in forward section)

            # Add tiles in a line perpendicular to the direction
            for fwd in range(0, 3):
                # Calculate the position: start + forward_component + side_component
                x = position.x + fwd * dx + side * perpendicular_dx
                y = position.y + fwd * dy + side * perpendicular_dy
                viewcone_tiles.append(Point(x, y))

        # Backward section: 2 tiles in the opposite direction
        for i in range(-2, 0):
            viewcone_tiles.append(Point(position.x + i * dx, position.y + i * dy))

        return viewcone_tiles
