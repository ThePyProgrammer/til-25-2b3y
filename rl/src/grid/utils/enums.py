from enum import IntEnum
from dataclasses import dataclass


class TileContent(IntEnum):
    """Enum for tile types based on the 2 least significant bits."""
    NO_VISION = 0
    EMPTY = 1
    RECON = 2  # 1 point
    MISSION = 3  # 5 points

    def __str__(self):
        TILE_TYPE_ICONS = {
            TileContent.NO_VISION: "?",
            TileContent.EMPTY: "·",
            TileContent.RECON: "R",
            TileContent.MISSION: "M"
        }
        return TILE_TYPE_ICONS[self]

    def __repr__(self):
        return self.__str__()

class Agent(IntEnum):
    """Enum for agent types."""
    NONE = 0
    SCOUT = 1
    GUARD = 2

    def __str__(self):
        AGENT_TYPE_ICONS = {
            Agent.NONE: " ",
            Agent.SCOUT: "S",
            Agent.GUARD: "G",
        }
        return AGENT_TYPE_ICONS[self]

    def __repr__(self):
        return self.__str__()

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

    def turn_right(self):
        """Returns the direction after turning right (clockwise)."""
        return Direction((self.value + 1) % 4)

    def turn_left(self):
        """Returns the direction after turning left (counterclockwise)."""
        return Direction((self.value - 1) % 4)

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

@dataclass
class Wall:
    """Wall bit positions."""
    RIGHT = 4  # Bit 4, value 16
    BOTTOM = 5  # Bit 5, value 32
    LEFT = 6   # Bit 6, value 64
    TOP = 7    # Bit 7, value 128
