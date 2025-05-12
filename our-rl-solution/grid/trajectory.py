from typing import Optional

from .utils import Direction, Action, Point
from .node import NodeRegistry, DirectionalNode

class Trajectory:
    def __init__(self, root_node):
        self.root: DirectionalNode = root_node
        self.route: list[Action] = []

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: list['Trajectory'] = []

    def copy(self):
        """Create a copy of this trajectory with the same root but a new route list."""
        new_trajectory = Trajectory(self.root)
        new_trajectory.route = self.route.copy()
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
        current_node = self.get_last_node()
        if current_node and action not in current_node.children:
            self.mark_as_invalid()

    def mark_as_invalid(self, invalid_action_idx: Optional[int] = None):
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

        current_node = self.root
        for i, action in enumerate(self.route):
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                self.mark_as_invalid(i)  # Mark invalid at the specific action index
                return None
        return current_node

    def get_new_trajectories(self) -> list['Trajectory']:
        """
        Generate all possible new trajectories from the current state.

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

        # Create new trajectories for each possible action
        for action, child_node in current_node.children.items():
            # Create a new trajectory
            new_trajectory = self.copy()
            # Add the new action to the route
            new_trajectory.update(action)
            new_trajectories.append(new_trajectory)

        return new_trajectories

class TrajectoryTree:
    def __init__(self, init_coord: Point, init_direction: Optional[Direction] = None, size: int = 16):
        """
        Initialize a TrajectoryTree with the agent's starting position and direction.

        Args:
            init_coord: Starting coordinate
            init_direction: Starting direction (default: RIGHT)
            size: Size of the grid environment (default: 16)
        """
        self.size = size
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
            root_node = self.registry.get_or_create_node(init_coord, direction)
            self.trajectories.append(Trajectory(root_node))

    def step(self):
        """
        Expand all valid trajectories by one step.

        Returns:
            Number of new trajectories added
        """
        if not self.trajectories:
            return 0

        new_trajectories = []
        current_trajectories = self.trajectories.copy()
        self.trajectories = []

        for traj in current_trajectories:
            # Skip invalid trajectories
            if traj.invalid:
                continue

            # Generate new trajectories from this one
            expanded_trajectories = traj.get_new_trajectories()

            if expanded_trajectories:
                new_trajectories.extend(expanded_trajectories)
            else:
                # Keep trajectories that can't be expanded (terminal nodes)
                self.trajectories.append(traj)

        # Store the new trajectories
        self.trajectories.extend(new_trajectories)
        return len(new_trajectories)
