import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class Direction(IntEnum):
    """Direction enum with values matching the environment."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def __str__(self):
        DIRECTION_ICONS = {
            Direction.RIGHT: "→",
            Direction.DOWN: "↓",
            Direction.LEFT: "←",
            Direction.UP: "↑"
        }

        return DIRECTION_ICONS[self]

    def __repr__(self):
        return self.__str__()

class Action(IntEnum):
    """Action enum with values matching the environment."""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    def __str__(self):
        ACTION_ICONS = {
            Action.FORWARD: "↑",
            Action.BACKWARD: "↓",
            Action.LEFT: "↺",
            Action.RIGHT: "↻",
            Action.STAY: "⊙"
        }
        return ACTION_ICONS[self]

    def __repr__(self):
        return self.__str__()


class Wall:
    """Wall bit positions."""
    RIGHT = 4  # Bit 4, value 16
    BOTTOM = 5  # Bit 5, value 32
    LEFT = 6   # Bit 6, value 64
    TOP = 7    # Bit 7, value 128


@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()


def get_hash(init_position: Point, direction: Direction) -> int:
    """Generate a unique hash for a coordinate and direction combination."""
    return hash((init_position, direction))


def rotate_wall_bits(tile_value, direction):
    """
    Rotates the wall bits based on the agent's direction to maintain global orientation.

    Args:
        tile_value (int): The observed tile value
        direction (int): Direction the agent is facing (0-3)

    Returns:
        int: Tile value with rotated wall bits
    """
    # Extract different parts of the tile value
    item_bits = tile_value & 0b11  # Last 2 bits (tile type)
    agent_bits = tile_value & 0b1100  # Bits 2-3 (agent type)
    wall_bits = tile_value & 0b11110000  # Bits 4-7 (walls)

    if direction == Direction.RIGHT:
        # No rotation needed for RIGHT direction (0)
        return tile_value

    # Extract individual wall bits
    right_wall = 1 if wall_bits & (1 << Wall.RIGHT) else 0
    bottom_wall = 1 if wall_bits & (1 << Wall.BOTTOM) else 0
    left_wall = 1 if wall_bits & (1 << Wall.LEFT) else 0
    top_wall = 1 if wall_bits & (1 << Wall.TOP) else 0

    # Initialize rotated walls
    rotated_right = right_wall
    rotated_bottom = bottom_wall
    rotated_left = left_wall
    rotated_top = top_wall

    if direction == Direction.DOWN:
        # Rotate clockwise once
        rotated_right = top_wall
        rotated_bottom = right_wall
        rotated_left = bottom_wall
        rotated_top = left_wall
    elif direction == Direction.LEFT:
        # Rotate clockwise twice
        rotated_right = left_wall
        rotated_bottom = top_wall
        rotated_left = right_wall
        rotated_top = bottom_wall
    elif direction == Direction.UP:
        # Rotate clockwise three times
        rotated_right = bottom_wall
        rotated_bottom = left_wall
        rotated_left = top_wall
        rotated_top = right_wall

    # Combine the rotated wall bits with the original agent and item bits
    rotated_wall_bits = (
        (rotated_right << Wall.RIGHT) |
        (rotated_bottom << Wall.BOTTOM) |
        (rotated_left << Wall.LEFT) |
        (rotated_top << Wall.TOP)
    )

    return item_bits | agent_bits | rotated_wall_bits


def view_to_world(agent_loc, agent_dir, view_coord):
    """
    Maps viewcone coordinate to world coordinate

    Args:
        agent_loc (np.ndarray): Agent's location in world coordinates
        agent_dir (int): Direction the agent is facing (0-3)
        view_coord (np.ndarray): Viewcone coordinate relative to agent

    Returns:
        np.ndarray: World coordinate
    """
    agent_dir = Direction(agent_dir)

    if agent_dir == Direction.RIGHT:
        return agent_loc + view_coord
    elif agent_dir == Direction.DOWN:
        return agent_loc - np.array((view_coord[1], -view_coord[0]))
    elif agent_dir == Direction.LEFT:
        return agent_loc - view_coord
    else:  # Direction.UP
        return agent_loc + np.array((view_coord[1], -view_coord[0]))
