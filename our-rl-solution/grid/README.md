# Grid Module

This module implements grid-based reinforcement learning environment utilities, including map representation, node-based pathfinding, trajectory planning, and visualization tools.

## Core Components

### `map.py`

The `Map` class is the central data structure representing the environment grid:

- Maintains the state of each grid cell (walls, agents, reward tiles)
- Handles updates from observations
- Tracks visibility and time since last update for each cell
- Provides pathfinding functionality
- Supports trajectory tree creation for optimal path planning
- Calculates optimal actions based on path analysis

### `node.py`

Implements directional node-based navigation:

- `DirectionalNode`: Represents positions in the grid with directional information
- `NodeRegistry`: Manages node instances to ensure uniqueness
- Supports movement planning with consideration for direction and valid actions
- Provides position transitions based on specific actions (FORWARD, BACKWARD, LEFT, RIGHT, STAY)

### `trajectory.py`

Handles path planning through the environment:

- `Trajectory`: Represents a sequence of actions through the grid
  - Tracks validity of paths
  - Enables copying and updating trajectories
  - Detects backtracking in paths

- `TrajectoryTree`: Manages and evaluates multiple trajectory possibilities
  - Indexes trajectories by endpoints for efficient lookup
  - Supports pruning invalid trajectories
  - Provides probability density estimation
  - Optimizes agent movement with filtering and validation

## Visualization

### `viz/base.py`

Provides pygame-based visualization of the grid environment:

- `MapVisualizer`: Main class for rendering the grid state
- Visual representations for tiles, players, and walls
- Support for both human viewing and RGB array output
- Displays time since last update for each cell

## Utility Modules

### `utils/enums.py`

Contains enumerations used throughout the module:

- `TileContent`: Types of grid cells (NO_VISION, EMPTY, RECON, MISSION)
- `Agent`: Types of agents (NONE, SCOUT, GUARD)
- `Direction`: Movement directions (RIGHT, DOWN, LEFT, UP)
- `Action`: Possible actions (FORWARD, BACKWARD, LEFT, RIGHT, STAY)
- `Wall`: Wall positions (RIGHT, BOTTOM, LEFT, TOP)

### `utils/geometry.py`

Geometric utilities for the grid:

- `Point`: 2D coordinate representation with equality and hashing
- Functions to convert between viewcone and world coordinates
- Hash generation for nodes based on position and direction

### `utils/tile.py`

Bit-based representation of grid tiles:

- `Tile`: Class representing a single grid cell
  - Encodes information in an 8-bit integer
  - Tracks tile type, agent presence, and wall configuration
  - Provides properties for all tile attributes
  - Handles wall rotation based on agent direction
