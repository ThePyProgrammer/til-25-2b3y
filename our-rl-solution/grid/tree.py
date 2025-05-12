from typing import Optional

from .utils import Direction, Action, Point, get_hash

class NodeRegistry:
    """Registry that manages DirectionalNode instances to ensure uniqueness."""

    def __init__(self, grid_size: int = 16):
        """Initialize a new registry with specified grid size."""
        self.grid_size = grid_size
        self.nodes: dict[int, 'DirectionalNode'] = {}

    def get_or_create_node(self, coord: Point, direction: Direction) -> 'DirectionalNode':
        """Get an existing node from the registry or create a new one if it doesn't exist."""
        node_hash = get_hash(coord, direction)

        # Check if we already have a node with this hash
        if node_hash in self.nodes:
            return self.nodes[node_hash]

        # Create a new instance if it doesn't exist
        node = DirectionalNode(coord, direction, self)
        self.nodes[node_hash] = node
        return node

class DirectionalNode:
    """A node in the path tree that tracks both position and direction."""

    def __init__(self, coord: Point, direction: Direction, registry: NodeRegistry):
        self.coord = coord
        self.direction = direction
        self.registry = registry
        self.children: dict[Action, 'DirectionalNode'] = {}
        # Populate children on first initialization
        self._populate_children()

    def _get_next_state(self, action: Action):
        """
        Calculate the next position and direction based on an action.

        Args:
            action: The action to take (FORWARD, BACKWARD, LEFT, RIGHT, STAY)

        Returns:
            tuple: (next_coord, next_direction)
        """
        # Movement vectors for each direction: (dx, dy)
        movement_vectors = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1)
        }

        # Direction changes for turns
        direction_changes = {
            Action.LEFT: {  # Left turns
                Direction.RIGHT: Direction.UP,
                Direction.DOWN: Direction.RIGHT,
                Direction.LEFT: Direction.DOWN,
                Direction.UP: Direction.LEFT
            },
            Action.RIGHT: {  # Right turns
                Direction.RIGHT: Direction.DOWN,
                Direction.DOWN: Direction.LEFT,
                Direction.LEFT: Direction.UP,
                Direction.UP: Direction.RIGHT
            }
        }

        # Start with current position and direction
        next_coord = Point(self.coord.x, self.coord.y)
        next_direction = self.direction

        if action == Action.STAY:
            # No change in position or direction
            pass

        elif action == Action.FORWARD:
            # Move forward in current direction
            dx, dy = movement_vectors[self.direction]
            next_coord.x += dx
            next_coord.y += dy

        elif action == Action.BACKWARD:
            # Move backward (opposite of current direction)
            dx, dy = movement_vectors[self.direction]
            next_coord.x -= dx
            next_coord.y -= dy

        elif action in (Action.LEFT, Action.RIGHT):
            # Turn and move in the new direction
            next_direction = direction_changes[action][self.direction]
            dx, dy = movement_vectors[next_direction]
            next_coord.x += dx
            next_coord.y += dy

        return next_coord, next_direction

    def _populate_children(self):
        """
        Populate the children dictionary with nodes that can be reached from this node.
        Does NOT take into account walls so those actions will have to be pruned.
        """
        self.children = {}

        for action in Action:
            # Calculate next position and direction
            next_coord, next_direction = self._get_next_state(action)

            # Apply boundary constraints
            next_coord.x = max(0, min(next_coord.x, self.registry.grid_size - 1))
            next_coord.y = max(0, min(next_coord.y, self.registry.grid_size - 1))

            # if the move is effectively STAY, then don't add it.
            if (
                next_coord.x == self.coord.x
                and next_coord.y == self.coord.y
                and next_direction == self.direction
            ):
                continue

            # Create the next node (will use existing one if it exists)
            next_node = self.registry.get_or_create_node(next_coord, next_direction)
            self.children[action] = next_node

    def __str__(self):
        dir_names = {
            Direction.RIGHT: "→",
            Direction.DOWN: "↓",
            Direction.LEFT: "←",
            Direction.UP: "↑"
        }
        return f"({self.coord} {dir_names[self.direction]})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return get_hash(self.coord, self.direction)

    def is_same_state(self, coord, direction):
        """Check if this node has the same state (position and direction)"""
        return self.coord == coord and self.direction == direction

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
