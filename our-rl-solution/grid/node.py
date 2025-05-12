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

        node.populate_children()
        return node

class DirectionalNode:
    """A node in the path tree that tracks both position and direction."""

    def __init__(self, coord: Point, direction: Direction, registry: NodeRegistry):
        self.coord = coord
        self.direction = direction
        self.registry = registry
        self.children: dict[Action, 'DirectionalNode'] = {}

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

    def populate_children(self):
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
        return f"({self.coord} {self.direction})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return get_hash(self.coord, self.direction)

    def is_same_state(self, coord, direction):
        """Check if this node has the same state (position and direction)"""
        return self.coord == coord and self.direction == direction
