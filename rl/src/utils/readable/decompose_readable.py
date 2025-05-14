import numpy as np
from typing import Dict, Any

from structs import Tile, Player, Wall

# Recreating the relevant enums for reference

def decompose_tile(tile_value: int) -> Dict[str, Any]:
    """
    Decomposes a tile value from the TIL environment's observation space into its components.

    Args:
        tile_value: An integer representing the tile (uint8, 0-255)

    Returns:
        A dictionary with the following keys:
        - point_type: Base tile type (0-3)
        - point_name: String representation of the point type
        - has_scout: Boolean indicating if a scout is present
        - has_guard: Boolean indicating if a guard is present
        - walls: Dictionary of wall presence by direction
        - wall_list: List of wall directions that are present
    """
    # Extract point type (bits 0-1)
    point_type = tile_value & 0x3  # Same as tile_value % 4

    # Check for players (bits 2-3)
    has_scout = bool((tile_value >> Player.SCOUT) & 1)
    has_guard = bool((tile_value >> Player.GUARD) & 1)

    # Map point type to name
    point_names = {
        Tile.NO_VISION: "No Vision",
        Tile.EMPTY: "Empty",
        Tile.RECON: "Recon Point",
        Tile.MISSION: "Mission Point"
    }
    point_name = point_names.get(point_type, "Unknown")

    # Extract wall information (bits 4-7)
    walls = {
        "right": bool((tile_value >> Wall.RIGHT) & 1),
        "bottom": bool((tile_value >> Wall.BOTTOM) & 1),
        "left": bool((tile_value >> Wall.LEFT) & 1),
        "top": bool((tile_value >> Wall.TOP) & 1)
    }

    # Create a list of present walls for convenience
    wall_list = [direction for direction, present in walls.items() if present]

    return {
        "point_type": point_type,
        "point_name": point_name,
        "has_scout": has_scout,
        "has_guard": has_guard,
        "walls": walls,
        "wall_list": wall_list
    }

def decompose_viewcone(viewcone: np.ndarray) -> np.ndarray:
    """
    Applies the decompose_tile function to an entire viewcone array.

    Args:
        viewcone: A 2D numpy array of tile values

    Returns:
        A numpy object array of the same shape, containing dictionaries with decomposed tile information
    """
    result = np.empty(viewcone.shape, dtype=object)
    for i in range(viewcone.shape[0]):
        for j in range(viewcone.shape[1]):
            result[i, j] = decompose_tile(viewcone[i, j])
    return result
