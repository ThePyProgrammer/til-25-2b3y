from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..utils import Direction, Point, Tile
from ..node import NodeRegistry, DirectionalNode

from .trajectory import Trajectory
from .index import TrajectoryIndex
from .constraints import Constraints, TrajectoryConstraints, TemporalTrajectoryConstraints
from .utils import fast_forward_trajectories, expand_trajectories
from .factory import create_trajectories_from_constraints

from utils.profiler import start_profiling, stop_profiling

class TrajectoryTree:
    def __init__(
        self,
        init_position: Point,
        init_direction: Optional[Direction] = None,
        size: int = 16,
        registry: Optional[NodeRegistry] = None,
        consider_direction: bool = True,
        regenerate_edge_threshold: int = 4,
        max_backtrack: int = 3,
        num_samples: int = 2,  # Number of trajectory samples per endpoint
    ):
        """
        Initialize a TrajectoryTree with the agent's starting position and direction.

        Args:
            init_position: Starting coordinate position
            init_direction: Starting direction (default: RIGHT)
            size: Size of the grid environment (default: 16)
            consider_direction: Whether to consider direction when reducing trajectories.
                               If True, trajectories with same position but different directions are considered distinct.
                               If False, only position matters, reducing the number of trajectories (default: True)
        """
        self.size = size
        self.consider_direction = consider_direction
        self.regenerate_edge_threshold = regenerate_edge_threshold
        self.max_backtrack = max_backtrack
        self.num_samples = num_samples

        self.num_step = 0

        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        # Spatial index: maps positions to trajectories that pass through them
        self.position_index = TrajectoryIndex()
        self.tail_position_index = TrajectoryIndex()
        self.node_index = TrajectoryIndex()

        # Initialize the starting trajectories
        self.roots: list[DirectionalNode] = []
        possible_init_directions: list[Direction] = []
        if init_direction is not None:
            possible_init_directions.append(init_direction)
        else:
            possible_init_directions.extend([d for d in Direction])

        self.trajectories: list[Trajectory] = []
        for direction in possible_init_directions:
            # This will either create a new node or use an existing one
            root = self.registry.get_or_create_node(init_position, direction)
            self.roots.append(root)
            trajectory = Trajectory(root, self.num_step)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        self.edge_trajectories: list[Trajectory] = self.trajectories.copy()
        # self.discard_edge_trajectories: list[Trajectory] = [] # [(step it was discarded, edge_trajectory), ...]

        self.temporal_constraints: TemporalTrajectoryConstraints = TemporalTrajectoryConstraints()

        # Add a set to track ambiguous tiles (visited tiles we should ignore for pruning)
        self.ambiguous_tiles = set()

        # Flag to track if we've performed a reset (detected agent) yet
        self.has_reset = False
        self.next_density: Optional[tuple[int, NDArray]] = None


    def step(self) -> int:
        """
        Expand all valid trajectories by one step, selecting n evenly spaced
        trajectory samples for each endpoint.

        When consider_direction=True, trajectories are grouped by (position, direction).
        When consider_direction=False, trajectories are grouped only by position,
        significantly reducing the number of trajectories.

        For each endpoint, we select:
        1. The shortest trajectory (fewest steps to reach that point)
        2. Up to (num_samples-2) evenly spaced trajectories
        3. The longest trajectory (if different from the shortest)

        Periodically validates a random sample of trajectories to ensure
        invalid trajectories are properly identified.

        Returns:
            Number of new trajectories added
        """
        self.num_step += 1

        if not self.trajectories:
            # self._restart_from_discarded()
            return len(self.trajectories)

        if len(self.edge_trajectories) < 16:
            self.edge_trajectories = self.trajectories.copy()
        old_edge_trajectories = self.edge_trajectories
        # print(f"Updating from {len(old_edge_trajectories)} trajectories")
        self.edge_trajectories = []  # Clear edge trajectories for this step

        # Mark all edge trajectories as discarded initially
        for traj in old_edge_trajectories:
            traj.discarded = True

        # Use the enhanced expand_trajectories function to select and expand trajectories
        expanded_trajectories = expand_trajectories(
            old_edge_trajectories,
            self.num_step,
            max_backtrack=self.max_backtrack,
            num_samples_per_trajectory=self.num_samples,
            consider_direction=self.consider_direction
        )


        # Process the newly expanded trajectories
        # new_trajectory_count = 0
        # for traj in expanded_trajectories:
            # Skip the original trajectories that were used for expansion
            # if traj in old_edge_trajectories:
                # traj.discarded = False
                # continue

            # Add this new trajectory
            # new_trajectory_count += 1

        self.trajectories.extend(expanded_trajectories)
        self.edge_trajectories.extend(expanded_trajectories)
        self._register_trajectory_in_index(expanded_trajectories)

        # # Add the remaining trajectories to discard_edge_trajectories
        # for traj in old_edge_trajectories:
        #     if traj.discarded:
        #         self.discard_edge_trajectories.append(traj)

        # print(f"Total discard: {len(self.discard_edge_trajectories)}")

        # Return count of new trajectories
        return len(expanded_trajectories)

    def prune(self, information: list[tuple[Point, Tile]], seeking_scout: bool = True, before_step: bool = False):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)
        4. when I encounter a tile and it has no agent, I remove all trajectories ending with that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
            seeking_scout (bool): If True, look for scout agent. If False, look for guard agent.
        """
        # Track newly discovered ambiguous tiles
        self._track_ambiguous_tiles(information)

        if seeking_scout:
            self._prune_by_tile_content(information, before_step=before_step)

        self._clean_up_trajectories()

        if len(self.trajectories) == 0 and not before_step:
            self.trajectories = create_trajectories_from_constraints(
                self.roots,
                self.temporal_constraints,
                self.num_step,
                self.registry,
            )

            # print(f"Fit {len(self.trajectories)} trajectories to visited points.")

            self.trajectories = fast_forward_trajectories(
                self.trajectories,
                self.num_step,
                self.temporal_constraints,
                self.registry
            )

            self.edge_trajectories = self.trajectories.copy()

        if len(self.trajectories) == 0 and not before_step:
            # Process agent sightings first (early exit)
            agent_position = self._check_for_agent(information, seeking_scout)
            if agent_position is not None:
                self._reset_trajectories_for_agent(agent_position)


    @property
    def probability_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions.

        This method computes how likely each position is to contain an agent,
        based on the number of valid trajectories passing through it.

        Returns:
            numpy.ndarray: 2D array with probability for each position
        """
        if self.next_density is not None:
            if self.next_density[0] == self.num_step:
                return self.next_density[1]

        # Initialize empty grid
        density = np.zeros((self.size, self.size), dtype=np.float32)

        if not self.trajectories:
            return density

        # Count trajectories passing through each cell
        for trajectory in self.trajectories:
            if trajectory.to_delete:
                continue

            # last_node = trajectory.get_last_node()
            last_node = trajectory.tail
            if not last_node:
                continue

            # Add 1 to the density at the endpoint position
            pos = last_node.position
            density[pos.y, pos.x] += 1

        # Normalize to get probability density
        total = density.sum()
        if total > 0:
            density = density / total

        return density

    @property
    def path_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions based on entire paths.

        Unlike probability_density which only considers endpoints, this method
        distributes the probability across all nodes in each trajectory's path.
        Each node in a trajectory gets 1/n of the trajectory's weight, where n is
        the number of nodes in that trajectory.

        Returns:
            numpy.ndarray: 2D array with path density for each position
        """
        # Initialize empty grid
        density = np.zeros((self.size, self.size), dtype=np.float32)

        if not self.trajectories:
            return density

        # Distribute density across all nodes in each trajectory
        for trajectory in self.trajectories:
            if trajectory.to_delete:
                continue

            # Skip empty trajectories
            if not trajectory.nodes:
                continue

            # Calculate weight per node (1 divided by number of nodes)
            weight_per_node = 1.0 / len(trajectory.nodes)

            # Add the weight to each position in the trajectory
            for node in trajectory.nodes:
                pos = node.position
                density[pos.y, pos.x] += weight_per_node

        # Normalize to get probability density
        total = density.sum()
        if total > 0:
            density = density / total

        return density

    def check_wall_trajectories(self, key: Point | DirectionalNode):
        """
        Instead of checking EVERY trajectory for validity, only check trajectories which
        contain point/node with updated wall information.

        Args:
            key (Point | DirectionalNode)
        """
        if isinstance(key, DirectionalNode):
            trajectories = self.node_index[key]
        elif isinstance(key, Point):
            trajectories = self.position_index[key]

        for traj in trajectories:
            traj.get_last_node()

    def _register_trajectory_in_index(self, t: Trajectory | list[Trajectory]):
        """Register a trajectory or list of trajectories in the position index."""
        if isinstance(t, list):
            # Handle list of trajectories
            # Filter out trajectories marked for deletion
            valid_trajectories = [traj for traj in t if not traj.to_delete]

            # We need to group trajectories by position to use the update method effectively
            position_to_trajectories = {}
            node_to_trajectories = {}
            tail_position_to_trajectories = {}

            # First, collect trajectories by their positions
            for trajectory in valid_trajectories:
                for node in trajectory.nodes:
                    position = node.position
                    if position not in position_to_trajectories:
                        position_to_trajectories[position] = []
                    position_to_trajectories[position].append(trajectory)

                    if node not in node_to_trajectories:
                        node_to_trajectories[node] = []
                    node_to_trajectories[node].append(trajectory)

                tail_position = trajectory.tail.position
                if tail_position not in tail_position_to_trajectories:
                    tail_position_to_trajectories[tail_position] = []
                tail_position_to_trajectories[tail_position].append(trajectory)

            # Now use the update method to register trajectories in bulk
            for position, trajectories in position_to_trajectories.items():
                self.position_index.update(position, trajectories)

            for node, trajectories in node_to_trajectories.items():
                self.node_index.update(node, trajectories)

            for position, trajectories in tail_position_to_trajectories.items():
                self.tail_position_index.update(position, trajectories)
        else:
            # Handle single trajectory
            trajectory = t
            if not trajectory.to_delete:
                # For a single trajectory, using add directly is simpler
                for node in trajectory.nodes:
                    self.position_index.add(node.position, trajectory)
                    self.node_index.add(node, trajectory)
                self.tail_position_index.add(trajectory.tail.position, trajectory)

    def _unregister_trajectory_from_index(self, t: Trajectory | list[Trajectory]):
        """Remove a trajectory or list of trajectories from the position index."""
        if isinstance(t, list):
            # Handle list of trajectories
            for trajectory in t:
                for node in trajectory.nodes:
                    self.position_index.remove(node.position, trajectory)
                    self.node_index.remove(node, trajectory)
                # Delete from tail_position_index too
                self.tail_position_index.remove(trajectory.tail.position, trajectory)
        else:
            # Handle single trajectory
            trajectory = t
            for node in trajectory.nodes:
                self.position_index.remove(node.position, trajectory)
                self.node_index.remove(node, trajectory)
            # Delete from tail_position_index too
            self.tail_position_index.remove(trajectory.tail.position, trajectory)

    def _clear_all_trajectories(self):
        """Clear all trajectories and reset the position index."""
        self.trajectories.clear()
        self.edge_trajectories.clear()
        self.position_index.clear()
        self.tail_position_index.clear()

    def _check_for_agent(self, information: list[tuple[Point, Tile]], seeking_scout: bool) -> Optional[Point]:
        """Check if an agent is present in the information."""
        for position, tile in information:
            if ((seeking_scout and tile.has_scout) or (not seeking_scout and tile.has_guard)):
                return position
        return None

    def _reset_trajectories_for_agent(self, agent_position):
        """
        Create trajectories that end with the agent's position.
        ._prune_by_tile_content() will have kept only trajectories ending in the agent's location, but that is not enough.
        """
        # If this is not the first reset, mark all currently known empty tiles as ambiguous
        if self.has_reset:
            # Add all currently tracked empty/visited tiles to ambiguous_tiles
            for trajectory in self.trajectories:
                for node in trajectory.nodes:
                    position = node.position
                    if position != agent_position:  # Don't mark agent's current position as ambiguous
                        self.ambiguous_tiles.add(position)
        else:
            # First reset - set the flag
            self.has_reset = True

        # Clear existing trajectories and their index
        self._clear_all_trajectories()

        # Create new trajectories at the agent's position, considering all possible directions
        for direction in Direction:
            root_node = self.registry.get_or_create_node(agent_position, direction)
            trajectory = Trajectory(root_node, self.num_step)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        # Update edge trajectories
        self.edge_trajectories = self.trajectories.copy()

    def _track_ambiguous_tiles(self, information):
        """Track newly discovered empty tiles as ambiguous if we've already had a reset."""
        if not self.has_reset:
            return  # Only track ambiguous tiles after the first reset

        for position, tile in information:
            if tile.is_empty:  # Newly discovered visited/empty tile
                self.ambiguous_tiles.add(position)

    # def debug_ambiguous_tiles(self):
    #     """Print debug information about ambiguous tiles."""
    #     print(f"Has reset: {self.has_reset}")
    #     print(f"Number of ambiguous tiles: {len(self.ambiguous_tiles)}")
    #     if self.ambiguous_tiles:
    #         print(f"Ambiguous tile positions: {sorted(self.ambiguous_tiles)}")
    #     print(f"Number of trajectories: {len(self.trajectories)}")

    def _prune_by_tile_content(self, information: list[tuple[Point, Tile]], before_step: bool = False):
        """Prune trajectories based on tile content."""
        # Group positions by condition to avoid redundant iterations
        route_constraints = Constraints([], [])
        tail_constraints = Constraints([], [])

        constraints = TrajectoryConstraints(
            route_constraints,
            tail_constraints
        )

        for position, tile in information:
            # Skip any tiles that are in our ambiguous set
            # if position in self.ambiguous_tiles:
            #     continue

            # Case 1: agent detected - only keep trajectories containing this position
            if tile.has_scout:
                constraints.tail.contains.append(position)
            # Case 2: not visited - remove trajectories containing this position
            if tile.is_visible and (tile.is_recon or tile.is_mission):
                constraints.route.excludes.append(position)
            # Case 3: visited - only keep trajectories containing this position
            if tile.is_empty and not self.has_reset:
                constraints.route.contains.append(position)
            # Case 4: no agent - remove trajectories ending at this position
            if not tile.has_scout:
                constraints.tail.excludes.append(position)

        # Use the spatial index for efficient filtering
        if not constraints:
            return  # No filtering needed

        # bugged? scout doesn't collect point at spawn location
        if Point(0, 0) in constraints.route.excludes:
            constraints.route.excludes.remove(Point(0, 0))

        if Point(0, 0) in constraints.tail.excludes:
            constraints.tail.excludes.remove(Point(0, 0))

        self._apply_filtering(constraints, before_step=before_step)
        self.temporal_constraints.update(constraints)

    def _apply_filtering(self, constraints: TrajectoryConstraints, before_step):
        """Apply filtering based on position constraints."""

        # Find trajectories to exclude (has any excluded position)
        trajectories_to_remove = set()
        for position in constraints.route.excludes:
            if position in self.position_index:
                trajectories_to_remove.update(self.position_index[position])

        if not before_step:
            for position in constraints.tail.excludes:
                if position in self.tail_position_index:
                    trajectories_to_remove.update(self.tail_position_index[position])

        # Find trajectories to keep (has all required positions)
        trajectories_to_keep = set(self.trajectories)

        # Filter by required route positions
        for position in constraints.route.contains:
            if position in self.position_index:
                trajectories_with_position = self.position_index[position]
                trajectories_to_keep &= trajectories_with_position
            else:
                # If a required position doesn't exist in any trajectory, no trajectories can satisfy
                trajectories_to_keep = set()
                break

        # Filter by required tail positions
        if not before_step:
            for position in constraints.tail.contains:
                if position in self.tail_position_index:
                    trajectories_with_tail_position = self.tail_position_index[position]
                    trajectories_to_keep &= trajectories_with_tail_position
                else:
                    # If a required tail position doesn't exist in any trajectory, no trajectories can satisfy
                    trajectories_to_keep = set()
                    break

        # Compute final valid trajectories
        valid_trajectories = [
            traj for traj in trajectories_to_keep if traj not in trajectories_to_remove
        ] if trajectories_to_keep else []

        # Remove invalid trajectories from the index
        removed_trajectories = set(self.trajectories) - set(valid_trajectories)
        for traj in removed_trajectories:
            traj.prune()
            # self._unregister_trajectory_from_index(traj)

        self._destroy_trajectory_families()
        self.trajectories = valid_trajectories

        # whether any edge trajectories have been deleted, may want to re-add them.
        trajectory_set = set(self.trajectories)
        has_deleted_edge = False
        for traj in self.edge_trajectories:
            if traj not in trajectory_set:
                has_deleted_edge = True
                break

        if has_deleted_edge:
            self.edge_trajectories = self._get_edge_trajectories()

    def _destroy_trajectory_families(self):
        """
        Starting from any invalid trajectory, traverse up ._inherited_from,
        deleting invalids from ._inherits_to. traverse until no invalids in ._inherits_to are found
        """
        parent_traj: Optional[Trajectory] = None

        for traj in self.trajectories:
            if traj.to_delete:
                parent_traj = traj.parent

                if parent_traj:
                    if len(parent_traj.children) > 0:

                        parent_traj._inherits_to = {
                            action: child for action, child in parent_traj._inherits_to.items()
                            if not child.to_delete
                        }

    def _get_edge_trajectories(self):
        """
        Edge trajectories are basically any trajectory without children
        """
        edge_trajectories = []
        for traj in self.trajectories:
            if traj.to_delete:
                continue

            if len(traj.children) < self.regenerate_edge_threshold:
                edge_trajectories.append(traj)

        return edge_trajectories

    def _clean_up_trajectories(self):
        valid_trajectories = []
        for traj in self.trajectories:
            if traj.to_delete:
                self._unregister_trajectory_from_index(traj)
            else:
                valid_trajectories.append(traj)
        self.trajectories = valid_trajectories
        # self.discard_edge_trajectories = [traj for traj in self.discard_edge_trajectories if not traj.to_delete]
