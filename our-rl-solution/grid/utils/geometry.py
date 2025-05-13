import numpy as np
from dataclasses import dataclass
from .enums import Direction


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