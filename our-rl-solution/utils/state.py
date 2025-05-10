import numpy as np

def encode_viewcone(viewcone):
    """
    Encodes the viewcone into multiple channels suitable for CNN processing.

    Returns:
        A multi-channel tensor with shape [channels, height, width]
    """
    height, width = viewcone.shape

    # Create 10 binary channels (one-hot for point types + players + 4 wall directions)
    encoded = np.zeros((10, height, width), dtype=np.float32)

    # Process each tile
    for y in range(height):
        for x in range(width):
            tile = viewcone[y, x]
            encoded[:, y, x] = encode_tile(tile)

    return encoded


def encode_tile(tile: int):
    channel = np.zeros((10), dtype=np.float32)
    # Channel 0: Visibility mask (0 = not visible, 1 = visible)
    channel[0] = 0.0 if (tile & 0x3) == 0 else 1.0

    # Channels 1-3: Point type (one-hot encoding)
    point_type = tile & 0x3  # Extract bits 0-1
    if point_type > 0:  # Only encode if the tile is visible
        channel[point_type] = 1.0

    # Channel 4: Scout presence
    channel[4] = 1.0 if (tile >> 2) & 1 else 0.0

    # Channel 5: Guard presence
    channel[5] = 1.0 if (tile >> 3) & 1 else 0.0

    # Channels 6-9: Wall directions (binary encoding)
    channel[6] = 1.0 if (tile >> 4) & 1 else 0.0  # Right wall
    channel[7] = 1.0 if (tile >> 5) & 1 else 0.0  # Bottom wall
    channel[8] = 1.0 if (tile >> 6) & 1 else 0.0  # Left wall
    channel[9] = 1.0 if (tile >> 7) & 1 else 0.0  # Top wall

    return channel
