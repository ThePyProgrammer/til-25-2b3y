import numpy as np


class Map:
    # Constants for tile information
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

    # Constants for tile occupancy and wall information
    SCOUT = 4      # Bit 2
    GUARD = 8      # Bit 3
    RIGHT_WALL = 16  # Bit 4
    BOTTOM_WALL = 32  # Bit 5
    LEFT_WALL = 64   # Bit 6
    TOP_WALL = 128   # Bit 7

    def __init__(self):
        # Initialize maps for different aspects
        self.size = 16
        self.map = np.zeros((self.size, self.size), dtype=np.uint8)
        self.tile_type = np.zeros((self.size, self.size), dtype=np.uint8)  # 0: unknown, 1: empty, 2: recon, 3: mission
        self.walls = np.zeros((self.size, self.size, 4), dtype=np.uint8)  # [right, bottom, left, top]
        self.scouts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.guards = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited = np.zeros((self.size, self.size), dtype=np.uint8)

    def __call__(self, observation):
        return self.update(observation)

    def update(self, observation):
        """Update the map based on current observation"""
        if 'viewcone' not in observation or 'direction' not in observation or 'location' not in observation:
            return self.map

        viewcone = observation['viewcone']
        direction = observation['direction']
        x, y = observation['location']

        # Mark current position as visited
        self.visited[y, x] = 1

        # Define viewcone dimensions
        vc_height, vc_width = viewcone.shape

        assert 0 <= direction <= 3

        # Calculate the offset to align the viewcone with the agent's position and direction
        start_x = start_y = None

        if direction == 0:  # Right
            start_x = x - 2
            start_y = y - 2
            dx, dy = 1, 0
            forward_x, forward_y = 1, 0
        elif direction == 1:  # Down
            start_x = x - 2
            start_y = y - 2
            dx, dy = 0, 1
            forward_x, forward_y = 0, 1
        elif direction == 2:  # Left
            start_x = x + 2
            start_y = y - 2
            dx, dy = -1, 0
            forward_x, forward_y = -1, 0
        elif direction == 3:  # Up
            start_x = x - 2
            start_y = y + 2
            dx, dy = 0, -1
            forward_x, forward_y = 0, -1

        assert start_x is not None and start_y is not None

        # Process the viewcone
        for vc_y in range(vc_height):
            for vc_x in range(vc_width):
                # Transform coordinates based on direction
                if direction == 0:  # Right
                    map_x = start_x + vc_x
                    map_y = start_y + vc_y
                elif direction == 1:  # Down
                    map_x = start_x + vc_y
                    map_y = start_y + vc_x
                elif direction == 2:  # Left
                    map_x = start_x - vc_x
                    map_y = start_y + vc_y
                else:  # Up
                    map_x = start_x + vc_y
                    map_y = start_y - vc_x

                # Check if the transformed coordinates are within map bounds
                if 0 <= map_x < self.size and 0 <= map_y < self.size:
                    tile_value = viewcone[vc_y, vc_x]

                    # Extract tile type (last 2 bits)
                    tile_type = tile_value & 0b11
                    if tile_type != self.NO_VISION:
                        # Update the main map (combining all information)
                        self.map[map_y, map_x] = tile_value

                        # misc
                        self.tile_type[map_y, map_x] = tile_type
                        self.walls[map_y, map_x, 0] = 1 if tile_value & self.RIGHT_WALL else 0
                        self.walls[map_y, map_x, 1] = 1 if tile_value & self.BOTTOM_WALL else 0
                        self.walls[map_y, map_x, 2] = 1 if tile_value & self.LEFT_WALL else 0
                        self.walls[map_y, map_x, 3] = 1 if tile_value & self.TOP_WALL else 0
                        self.scouts[map_y, map_x] = 1 if tile_value & self.SCOUT else 0
                        self.guards[map_y, map_x] = 1 if tile_value & self.GUARD else 0


        return self.map

    def get_map(self):
        """Return the current map state"""
        return self.map

    def get_tile_type(self):
        """Return map of tile types"""
        return self.tile_type

    def get_walls(self):
        """Return map of walls"""
        return self.walls

    def get_agents(self):
        """Return maps of scouts and guards"""
        return self.scouts, self.guards

    def get_visited(self):
        """Return visited locations"""
        return self.visited
