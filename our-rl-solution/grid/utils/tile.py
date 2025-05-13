from dataclasses import dataclass
from .enums import TileContent, Agent, Direction, Wall


@dataclass
class Tile:
    """
    Representation of a tile from an 8-bit unsigned integer.

    Bit layout:
    - Bits 0-1: Tile type (0=No vision, 1=Empty, 2=Recon, 3=Mission)
    - Bit 2: Scout presence
    - Bit 3: Guard presence
    - Bit 4: Right wall
    - Bit 5: Bottom wall
    - Bit 6: Left wall
    - Bit 7: Top wall
    """
    value: int  # uint8 value

    def __post_init__(self):
        # Ensure the value is treated as an 8-bit unsigned integer
        self.value = int(self.value) & 0xFF

    @property
    def tile_content(self) -> TileContent:
        """Get the type of the tile from the 2 least significant bits."""
        return TileContent(self.value & 0b11)

    @property
    def has_scout(self) -> bool:
        """Check if a scout is present on this tile (bit 2)."""
        return bool(self.value & (1 << 2))

    @property
    def has_guard(self) -> bool:
        """Check if a guard is present on this tile (bit 3)."""
        return bool(self.value & (1 << 3))

    @property
    def agent_type(self) -> Agent:
        """Get the agent type present on this tile."""
        scout_bit = 1 if self.has_scout else 0
        guard_bit = 2 if self.has_guard else 0
        return Agent(scout_bit | guard_bit)

    @property
    def has_right_wall(self) -> bool:
        """Check if there's a wall on the right (bit 4)."""
        return bool(self.value & (1 << Wall.RIGHT))

    @property
    def has_bottom_wall(self) -> bool:
        """Check if there's a wall on the bottom (bit 5)."""
        return bool(self.value & (1 << Wall.BOTTOM))

    @property
    def has_left_wall(self) -> bool:
        """Check if there's a wall on the left (bit 6)."""
        return bool(self.value & (1 << Wall.LEFT))

    @property
    def has_top_wall(self) -> bool:
        """Check if there's a wall on the top (bit 7)."""
        return bool(self.value & (1 << Wall.TOP))

    @property
    def wall_bits(self) -> int:
        """Get just the wall bits (bits 4-7)."""
        return self.value & 0b11110000

    @property
    def is_visible(self) -> bool:
        """Check if the tile is visible (not NO_VISION)."""
        return self.tile_content != TileContent.NO_VISION

    @property
    def is_recon(self) -> bool:
        """Check if the tile is a recon tile."""
        return self.tile_content == TileContent.RECON

    @property
    def is_mission(self) -> bool:
        """Check if the tile is a mission tile."""
        return self.tile_content == TileContent.MISSION

    @property
    def is_reward_tile(self) -> bool:
        """Check if the tile offers a reward (recon or mission)."""
        return self.is_recon or self.is_mission

    @property
    def reward_value(self) -> int:
        """Get the reward value of the tile."""
        if self.is_recon:
            return 1
        elif self.is_mission:
            return 5
        return 0

    def has_wall_in_direction(self, direction: Direction) -> bool:
        """Check if there's a wall in the specified direction."""
        if direction == Direction.RIGHT:
            return self.has_right_wall
        elif direction == Direction.DOWN:
            return self.has_bottom_wall
        elif direction == Direction.LEFT:
            return self.has_left_wall
        elif direction == Direction.UP:
            return self.has_top_wall
        return False

    def rotate_walls(self, direction: Direction) -> 'Tile':
        """Return a new Tile with wall bits rotated based on the agent's direction."""
        rotated_value = rotate_wall_bits(self.value, direction)
        return Tile(rotated_value)

    def __str__(self) -> str:
        """String representation of the tile."""
        # Show tile type and agent if present
        base = f"{self.tile_content}"
        if self.agent_type != Agent.NONE:
            base += f"({self.agent_type})"

        # Add wall indicators
        walls = []
        if self.has_right_wall:
            walls.append("R")
        if self.has_bottom_wall:
            walls.append("B")
        if self.has_left_wall:
            walls.append("L")
        if self.has_top_wall:
            walls.append("T")

        if walls:
            base += f"[{''.join(walls)}]"

        return base

    def __repr__(self) -> str:
        return f"Tile({self.value:#x})"  # Show hex value in representation


def rotate_wall_bits(tile_value, direction):
    """
    Rotates the wall bits based on the agent's direction to maintain global orientation.

    Args:
        tile_value (int): The observed tile value
        direction (int): Direction the agent is facing (0-3)

    Returns:
        int: Tile value with rotated wall bits
    """
    # If it's a Tile object, extract its value
    if hasattr(tile_value, 'value'):
        tile_value = tile_value.value

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


def int_to_tile(value: int) -> Tile:
    """Convert an integer value to a Tile object."""
    return Tile(value)