import numpy as np
import pygame
import warnings
from enum import IntEnum
from ..map import Map
from ..utils import TileContent, Wall as WallEnum, int_to_tile


class Tile(IntEnum):
    NO_VISION = TileContent.NO_VISION
    EMPTY = TileContent.EMPTY
    RECON = TileContent.RECON
    MISSION = TileContent.MISSION

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
        vision=False,
    ):
        match self:
            case Tile.NO_VISION if vision:
                pygame.draw.rect(
                    canvas,
                    (80, 80, 80),
                    pygame.Rect(
                        x_corner + x * square_size,
                        y_corner + y * square_size,
                        square_size,
                        square_size,
                    ),
                )
            case Tile.RECON:
                pygame.draw.circle(
                    canvas,
                    (255, 165, 0),
                    (
                        x_corner + (x + 0.5) * square_size,
                        y_corner + (y + 0.5) * square_size,
                    ),
                    square_size / 10,
                )
            case Tile.MISSION:
                pygame.draw.circle(
                    canvas,
                    (147, 112, 219),
                    (
                        x_corner + (x + 0.5) * square_size,
                        y_corner + (y + 0.5) * square_size,
                    ),
                    square_size / 6,
                )


class Player(IntEnum):
    SCOUT = 2  # 2**2
    GUARD = 3  # 2**3

    @property
    def color(self) -> tuple[int, int, int]:
        match self:
            case Player.GUARD:
                return (255, 0, 0)
            case Player.SCOUT:
                return (0, 0, 255)

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
    ):
        pygame.draw.circle(
            canvas,
            self.color,
            (x_corner + (x + 0.5) * square_size, y_corner + (y + 0.5) * square_size),
            square_size / 3,
        )


class Wall(IntEnum):
    RIGHT = WallEnum.RIGHT  # 2**4 = 16
    BOTTOM = WallEnum.BOTTOM  # 2**5 = 32
    LEFT = WallEnum.LEFT  # 2**6 = 64
    TOP = WallEnum.TOP  # 2**7 = 128

    @property
    def orientation(self):
        # returns start x, start y, end x, end y
        # of the line for the wall to be drawn
        match self:
            case Wall.RIGHT:
                return (1, 0, 1, 1)
            case Wall.BOTTOM:
                return (0, 1, 1, 1)
            case Wall.LEFT:
                return (0, 0, 0, 1)
            case Wall.TOP:
                return (0, 0, 1, 0)

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
        width: int = 7,
    ):
        x1, y1, x2, y2 = self.orientation
        pygame.draw.line(
            canvas,
            0,
            (x_corner + square_size * (x + x1), y_corner + square_size * (y + y1)),
            (x_corner + square_size * (x + x2), y_corner + square_size * (y + y2)),
            width=width,
        )


def get_bit(value, bit_position):
    """Helper function to check if a bit is set at a specific position."""
    return (value & (1 << bit_position)) > 0
    
def get_tile_obj(value):
    """Convert an integer value to a Tile instance for easier property access."""
    return int_to_tile(value)


class MapVisualizer:
    def __init__(self, map_obj: Map, window_size=512, caption="Map Visualization"):
        """
        Initialize a visualizer for the map.

        Args:
            map_obj: An instance of the Map class to visualize
            window_size: Size of the window (square)
            caption: Window caption
        """
        self.map = map_obj
        self.window_size = window_size
        self.caption = caption
        
        # Initialize pygame components
        self.window = None
        self.clock = None
        self.font = None
        
        # FPS for rendering
        self.fps = 30

    def _draw_text(self, text, text_col="black", **kwargs):
        """Utility to draw text on the pygame surface."""
        if self.font is not None:
            img = self.font.render(text, True, text_col)
            rect = img.get_rect(**kwargs)
            self.window.blit(img, rect)

    def _draw_gridlines(
        self,
        max_x: int,
        max_y: int,
        square_size: float,
        x_corner: int = 0,
        y_corner: int = 0,
        width: int = 3,
    ):
        """Draw grid lines on the pygame surface."""
        for x in range(max_x + 1):
            pygame.draw.line(
                self.window,
                (211, 211, 211),
                (x_corner + square_size * x, y_corner),
                (x_corner + square_size * x, y_corner + square_size * max_y),
                width=width,
            )
        for y in range(max_y + 1):
            pygame.draw.line(
                self.window,
                (211, 211, 211),
                (x_corner, y_corner + square_size * y),
                (x_corner + square_size * max_x, y_corner + square_size * y),
                width=width,
            )

    def render(self, human_mode=True):
        """
        Render the map visualization.

        Args:
            human_mode: If True, updates the display and manages timing. 
                      If False, returns an RGB array.
        """
        # Initialize pygame components if not already done
        if self.window is None:
            pygame.init()
            if human_mode:
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption(self.caption)
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))

        if self.clock is None and human_mode:
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            try:
                self.font = pygame.font.Font("freesansbold.ttf", 12)
            except:
                warnings.warn("unable to import font")

        # Fill with white background
        self.window.fill((255, 255, 255))
        
        # Calculate pixel size for a grid square
        pix_square_size = self.window_size / self.map.size

        # Draw gridlines
        self._draw_gridlines(self.map.size, self.map.size, pix_square_size)

        # Get map data
        tile_types = self.map.get_tile_type()
        walls = self.map.get_walls()
        scouts, guards = self.map.get_agents()
        visited = self.map.get_visited()

        # Draw all tiles and walls
        for x, y in np.ndindex((self.map.size, self.map.size)):
            if not visited[x, y]:
                continue
            
            # Get tile object from raw value for easier property access
            tile_obj = get_tile_obj(self.map.map[x, y])
                
            # Draw the tile type
            tile_type = tile_types[x, y]
            if tile_type > 0:  # Skip NO_VISION
                Tile(tile_type).draw(self.window, x, y, int(pix_square_size))
            
            # Draw walls - can use tile_obj properties but continue to use wall array for consistency
            if walls[x, y, 0]:  # Right wall
                Wall.RIGHT.draw(self.window, x, y, int(pix_square_size))
            if walls[x, y, 1]:  # Bottom wall
                Wall.BOTTOM.draw(self.window, x, y, int(pix_square_size))
            if walls[x, y, 2]:  # Left wall
                Wall.LEFT.draw(self.window, x, y, int(pix_square_size))
            if walls[x, y, 3]:  # Top wall
                Wall.TOP.draw(self.window, x, y, int(pix_square_size))
            
            # Draw agents - use scouts/guards arrays for consistency
            if scouts[x, y]:
                Player.SCOUT.draw(self.window, x, y, int(pix_square_size))
            if guards[x, y]:
                Player.GUARD.draw(self.window, x, y, int(pix_square_size))

        # Display time since last update for each cell
        if self.font is not None:
            time_since_update = self.map.time_since_update
            for x, y in np.ndindex((self.map.size, self.map.size)):
                if visited[x, y] and time_since_update[x, y] < self.map.step_counter:
                    # Only display time for visited cells that have been updated
                    center = (np.array([x, y]) + 0.5) * pix_square_size
                    self._draw_text(
                        f"{time_since_update[x, y]}",
                        "gray",
                        center=(int(center[0]), int(center[1])),
                    )

        if human_mode:
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """Close the pygame window and clean up."""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None