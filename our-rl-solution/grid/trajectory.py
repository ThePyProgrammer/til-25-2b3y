from typing import Optional

import numpy as np

from .utils import Direction, Action, Point
from .node import NodeRegistry, DirectionalNode


class Trajectory:
    def __init__(self, root_node):
        self.root: DirectionalNode = root_node
        self.last: Optional[DirectionalNode] = None
        self.route: list[Action] = []
        self.nodes: list[DirectionalNode] = [self.root]

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: list['Trajectory'] = []

    def __hash__(self):
        return hash(tuple(self.route))

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False
        return (self.root == other.root and
                self.route == other.route)

    def copy(self):
        """Create a deep copy of this trajectory with the same root but new lists."""
        new_trajectory = Trajectory(self.root)
        new_trajectory.route = self.route[:]
        new_trajectory.nodes = self.nodes[:]
        new_trajectory.last = self.last
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        self._inherits_to.append(new_trajectory)
        return new_trajectory

    def update(self, action):
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)

        # Check if this action is valid
        self.get_last_node()

    def mark_as_invalid(self, invalid_action_idx: Optional[int] = None):
        """
        Mark this trajectory as invalid and propagate to inheriting trajectories.

        Args:
            invalid_action_idx: The index of the action in the route that is invalid.
                                If None, the entire trajectory is marked invalid.
        """
        print(invalid_action_idx)
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
                    self._inherited_from.mark_as_invalid(invalid_action_idx)
            else:
                return
        else:
            self.invalid = True

        # Propagate to all trajectories that inherit from this one
        for trajectory in self._inherits_to:
            trajectory.mark_as_invalid(invalid_action_idx)

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
            self.nodes = [self.root]

        current_node = self.root
        for i, action in enumerate(self.route):
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                self.mark_as_invalid(i)  # Mark invalid at the specific action index
                return None

            if populate_nodes:
                self.nodes.append(current_node)

        self.last = current_node
        return current_node

    def get_new_trajectories(self, expansion_cache=None) -> list['Trajectory']:
        """
        Generate all possible new trajectories from the current state.
        Uses memoization if an expansion cache is provided.

        Args:
            expansion_cache: Optional dictionary to cache expansion results

        Returns:
            A list of new Trajectory objects, one for each possible action
            from the current end node. Returns empty list if trajectory is invalid.
        """
        if self.invalid:
            return []

        new_trajectories = []

        # Get the current end node
        current_node = self.get_last_node()

        if not current_node:
            return []

        # Check if we've already calculated expansions for this node
        # We only use the node for caching, not the full route to save memory
        if expansion_cache is not None and current_node in expansion_cache:
            # Use cached results
            cached_children = expansion_cache[current_node]
            for action, child_node in cached_children.items():
                # Create a new trajectory
                new_trajectory = self.copy()
                # Add the new action to the route
                new_trajectory.update(action)
                new_trajectories.append(new_trajectory)
            return new_trajectories

        # Create new trajectories for each possible action
        node_expansions = {}
        for action, child_node in current_node.children.items():
            # Create a new trajectory
            new_trajectory = self.copy()
            # Add the new action to the route
            new_trajectory.update(action)
            new_trajectories.append(new_trajectory)

            # Store for caching
            node_expansions[action] = child_node

        # Cache the results if a cache is provided
        if expansion_cache is not None:
            expansion_cache[current_node] = node_expansions

        return new_trajectories

    def get_endpoint_key(self, consider_direction=True):
        """
        Get a key that represents the endpoint state (position and direction).
        Used for grouping trajectories by their endpoints.

        Args:
            consider_direction: Whether to include direction in the key.
                               Set to False to only consider position, reducing
                               the number of unique endpoints.

        Returns:
            A tuple of (position, direction) or (position,) or None if trajectory is invalid
        """
        last_node = self.get_last_node()
        if last_node:
            if consider_direction:
                return (last_node.position, last_node.direction)
            else:
                return (last_node.position,)
        return None

    def __str__(self):
        status = "INVALID" if self.invalid else "VALID"
        invalid_info = f", Invalid at action {self.invalid_action_idx}" if self.invalid and self.invalid_action_idx is not None else ""

        route_str = " ".join([f"{action} {node}" for node, action in zip(self.nodes[1:], self.route)]) if self.nodes else ""

        return f"Trajectory({status}{invalid_info}): {self.root} {route_str}"

    def __repr__(self):
        return self.__str__()

class TrajectoryTree:
    def __init__(self, init_position: Point, init_direction: Optional[Direction] = None, size: int = 16, consider_direction: bool = True):
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
        # Create a node registry for this trajectory tree
        self.registry = NodeRegistry(size)

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
            self.trajectories.append(Trajectory(root_node))
        self.edge_trajectories: list[Trajectory] = self.trajectories

        # Add expansion cache for memoization
        self.expansion_cache: dict[DirectionalNode, dict[Action, DirectionalNode]] = {}

    def set_consider_direction(self, consider_direction: bool):
        """
        Update whether to consider direction when reducing trajectories.

        Args:
            consider_direction: If True, trajectories with same position but different
                               directions are considered distinct. If False, only position matters.
        """
        self.consider_direction = consider_direction

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

        Returns:
            Number of new trajectories added
        """
        if not self.trajectories:
            return 0

        before_len = len(self.trajectories)

        old_edge_trajectories = self.edge_trajectories.copy()

        # Group trajectories by endpoint
        endpoint_trajectories = {}
        for traj in old_edge_trajectories:
            if traj.invalid:
                continue

            key = traj.get_endpoint_key(self.consider_direction)
            if key:
                if key not in endpoint_trajectories:
                    endpoint_trajectories[key] = []
                endpoint_trajectories[key].append(traj)

        # For each endpoint, select only the longest and shortest trajectories
        selected_trajectories = []
        for trajectories in endpoint_trajectories.values():
            if not trajectories:
                continue

            # Find shortest and longest trajectories without full sorting
            shortest = min(trajectories, key=lambda t: len(t.route))
            longest = max(trajectories, key=lambda t: len(t.route))

            # Add shortest trajectory
            selected_trajectories.append(shortest)

            # Add longest trajectory if it's different from the shortest
            # We compare the actual trajectory objects, not just their lengths
            if shortest != longest:
                selected_trajectories.append(longest)

        # Generate new trajectories from selected ones
        new_trajectories = []
        for traj in selected_trajectories:
            expanded_trajectories = traj.get_new_trajectories(self.expansion_cache)
            if expanded_trajectories:
                new_trajectories.extend(expanded_trajectories)

        # filter invalid trajectories out.
        self.trajectories = [traj for traj in self.trajectories if not traj.invalid]

        self.edge_trajectories = new_trajectories
        self.trajectories.extend(new_trajectories)

        # Perform deduplication (existing method)
        self.deduplicate()

        after_len = len(self.trajectories)
        return after_len - before_len

    def deduplicate(self):
        """Remove duplicate trajectories."""
        self.trajectories = list(set(self.trajectories))
        # Also update edge_trajectories to maintain consistency
        self.edge_trajectories = list(set(self.edge_trajectories))

    @property
    def probability_density(self):
        """
        Calculate the probability density map based on trajectory endpoints.

        Returns:
            numpy.ndarray: A 2D array of shape (size, size) where each cell
                          represents the probability of the agent being at that location.
        """
        probas = np.zeros((self.size, self.size))

        # Count valid trajectories at each position
        valid_trajectories = [t for t in self.trajectories if not t.invalid]

        # Skip if no valid trajectories
        if not valid_trajectories:
            return probas

        # Count trajectories at each position
        for trajectory in valid_trajectories:
            endpoint_key = trajectory.get_endpoint_key(consider_direction=False)
            if endpoint_key:  # Make sure the trajectory has a valid endpoint
                position = endpoint_key[0]  # Extract the position from the tuple
                probas[position.y, position.x] += 1

        # Normalize to get probability distribution
        total = np.sum(probas)
        if total > 0:  # Avoid division by zero
            probas = probas / total

        return probas
