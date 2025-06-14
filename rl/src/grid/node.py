from functools import lru_cache
import sys
from typing import Optional

from .utils import Direction, Action, Point, get_hash
from .utils.geometry import MOVEMENT_VECTORS


class NodeRegistry:
    """Registry that manages DirectionalNode instances to ensure uniqueness."""

    def __init__(self, grid_size: int = 16):
        """Initialize a new registry with specified grid size."""
        self.grid_size = grid_size
        self.nodes: dict[int, 'DirectionalNode'] = {}
        sys.setrecursionlimit(1500)

    @lru_cache(maxsize=1024)
    def get_or_create_node(self, position: Point, direction: Direction) -> 'DirectionalNode':
        """Get an existing node from the registry or create a new one if it doesn't exist."""
        node_hash = get_hash(position, direction)

        # Check if we already have a node with this hash
        if node_hash in self.nodes:
            return self.nodes[node_hash]

        # Create a new instance if it doesn't exist
        node = DirectionalNode(position, direction, self)
        self.nodes[node_hash] = node

        node.populate_children()
        return node

class DirectionalNode:
    """A node in the path tree that tracks both position and direction."""

    def __init__(self, position: Point, direction: Direction, registry: NodeRegistry):
        self.position = position
        self.direction = direction
        self.registry = registry
        self.children: dict[Action, 'DirectionalNode'] = {}

        self._hash: Optional[int] = None

    def _get_next_state(self, action: Action):
        """
        Calculate the next position and direction based on an action.

        Args:
            action: The action to take (FORWARD, BACKWARD, LEFT, RIGHT, STAY)

        Returns:
            tuple: (next_position, next_direction)
        """
        # Movement vectors for each direction: (dx, dy)

        # Start with current position and direction
        next_position = Point(self.position.x, self.position.y)
        next_direction = self.direction

        if action == Action.STAY:
            # No change in position or direction
            pass

        elif action == Action.FORWARD:
            # Move forward in current direction
            dx, dy = MOVEMENT_VECTORS[self.direction]
            next_position.x += dx
            next_position.y += dy

        elif action == Action.BACKWARD:
            # Move backward (opposite of current direction)
            dx, dy = MOVEMENT_VECTORS[self.direction]
            next_position.x -= dx
            next_position.y -= dy

        elif action == Action.LEFT:
            next_direction = self.direction.turn_left()

        elif action == Action.RIGHT:
            next_direction = self.direction.turn_right()

        return next_position, next_direction

    def populate_children(self):
        """
        Populate the children dictionary with nodes that can be reached from this node.
        Does NOT take into account walls so those actions will have to be pruned.
        """
        self.children = {}

        for action in Action:
            # Calculate next position and direction
            next_position, next_direction = self._get_next_state(action)

            # Apply boundary constraints
            next_position.x = max(0, min(next_position.x, self.registry.grid_size - 1))
            next_position.y = max(0, min(next_position.y, self.registry.grid_size - 1))

            # if the move is effectively STAY, then don't add it.
            if (
                next_position.x == self.position.x
                and next_position.y == self.position.y
                and next_direction == self.direction
            ):
                continue

            # Create the next node (will use existing one if it exists)
            next_node = self.registry.get_or_create_node(next_position, next_direction)
            self.children[action] = next_node

    def __str__(self):
        return f"({self.position} {self.direction})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = get_hash(self.position, self.direction)
        return self._hash

    def is_same_state(self, position, direction):
        """Check if this node has the same state (position and direction)"""
        return self.position == position and self.direction == direction
