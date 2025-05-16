from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..utils import Direction, Point, Tile
from ..node import NodeRegistry, DirectionalNode

from .trajectory import Trajectory
from .utils import Constraints, TrajectoryConstraints, TemporalTrajectoryConstraints, TrajectoryIndex

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

        self.num_step = 0

        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        # Spatial index: maps positions to trajectories that pass through them
        self.position_index = TrajectoryIndex()
        self.tail_position_index = TrajectoryIndex()
        self.node_index = TrajectoryIndex()

        # Initialize the starting trajectories
        possible_init_directions: list[Direction] = []

        if init_direction is not None:
            possible_init_directions.append(init_direction)
        else:
            possible_init_directions.extend([d for d in Direction])

        self.trajectories: list[Trajectory] = []
        for direction in possible_init_directions:
            # This will either create a new node or use an existing one
            root_node = self.registry.get_or_create_node(init_position, direction)
            trajectory = Trajectory(root_node, self.num_step)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        self.edge_trajectories: list[Trajectory] = self.trajectories.copy()
        self.discard_edge_trajectories: list[Trajectory] = [] # [(step it was discarded, edge_trajectory), ...]

        self.temporal_constraints: TemporalTrajectoryConstraints = TemporalTrajectoryConstraints()

        # Add a set to track ambiguous tiles (visited tiles we should ignore for pruning)
        self.ambiguous_tiles = set()

        # Flag to track if we've performed a reset (detected agent) yet
        self.has_reset = False

    def _register_trajectory_in_index(self, trajectory: Trajectory):
        """Register a trajectory in the position index."""
        if trajectory.to_delete:
            return

        for node in trajectory.nodes:
            self.position_index.add(node.position, trajectory)
            self.node_index.add(node, trajectory)

        self.tail_position_index.add(trajectory.tail.position, trajectory)

    def _unregister_trajectory_from_index(self, trajectory: Trajectory):
        """Remove a trajectory from the position index."""
        for node in trajectory.nodes:
            self.position_index.remove(node.position, trajectory)
            self.node_index.add(node, trajectory)

        # delete from end_index too
        self.tail_position_index.remove(trajectory.tail.position, trajectory)

    def _clear_all_trajectories(self):
        """Clear all trajectories and reset the position index."""
        self.trajectories.clear()
        self.edge_trajectories.clear()
        self.position_index.clear()
        self.tail_position_index.clear()

    def prune(self, information: list[tuple[Point, Tile]], seeking_scout: bool = True):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)
        4. when I encounter a tile and it has no agent, I remove all trajectories ending with that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
            seeking_scout (bool): If True, look for scout agent. If False, look for guard agent.
        """
        # # Process agent sightings first (early exit)
        # agent_position = self._check_for_agent(information, seeking_scout)
        # if agent_position is not None:
        #     self._reset_trajectories_for_agent(agent_position)
        #     return

        # Track newly discovered ambiguous tiles
        self._track_ambiguous_tiles(information)

        if seeking_scout:
            self._prune_by_tile_content(information)

        self._clean_up_trajectories()

    def _check_for_agent(self, information, seeking_scout):
        """Check if an agent is present in the information."""
        for position, tile in information:
            if (seeking_scout and tile.has_scout) or (not seeking_scout and tile.has_guard):
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

    def debug_ambiguous_tiles(self):
        """Print debug information about ambiguous tiles."""
        print(f"Has reset: {self.has_reset}")
        print(f"Number of ambiguous tiles: {len(self.ambiguous_tiles)}")
        if self.ambiguous_tiles:
            print(f"Ambiguous tile positions: {sorted(self.ambiguous_tiles)}")
        print(f"Number of trajectories: {len(self.trajectories)}")

    def _prune_by_tile_content(self, information: list[tuple[Point, Tile]]):
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
            if position in self.ambiguous_tiles:
                continue

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

        self._apply_filtering(constraints)
        self.temporal_constraints.update(constraints)

    def _apply_filtering(self, constraints: TrajectoryConstraints):
        """Apply filtering based on position constraints."""

        # Find trajectories to exclude (has any excluded position)
        trajectories_to_remove = set()
        for position in constraints.route.excludes:
            if position in self.position_index:
                trajectories_to_remove.update(self.position_index[position])

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

    def step(self) -> int:
        """
        Expand all valid trajectories by one step, but only expanding the
        longest and shortest trajectory that reaches each endpoint.

        When consider_direction=True, trajectories are grouped by (position, direction).
        When consider_direction=False, trajectories are grouped only by position,
        significantly reducing the number of trajectories.

        For each endpoint, we select:
        1. The shortest trajectory (fewest steps to reach that point)
        2. The longest trajectory (if different from the shortest)

        Periodically validates a random sample of trajectories to ensure
        invalid trajectories are properly identified.

        Returns:
            Number of new trajectories added
        """
        self.num_step += 1

        if not self.trajectories:
            # self._restart_from_discarded()
            return len(self.trajectories)

        before_len = len(self.trajectories)

        old_edge_trajectories = self.edge_trajectories
        print(f"Updating from {len(old_edge_trajectories)} trajectories")
        self.edge_trajectories = []  # Clear edge trajectories for this step

        # Mapping of endpoint keys to trajectories
        endpoint_to_trajectories = {}

        # Group trajectories by their endpoints
        for traj in old_edge_trajectories:
            # Skip invalid trajectories
            if traj.to_delete:
                continue

            traj.discarded = True

            # Get key for this trajectory's endpoint
            key = traj.get_endpoint_key(self.consider_direction)
            if not key:
                continue

            if key not in endpoint_to_trajectories:
                endpoint_to_trajectories[key] = []
            endpoint_to_trajectories[key].append(traj)

        # For each endpoint, select the shortest and longest trajectories
        selected_trajectories: list[Trajectory] = []
        for key, trajectories in endpoint_to_trajectories.items():
            # Sort by trajectory length (shorter first)
            trajectories.sort(key=lambda t: len(t.route))

            # Always select the shortest trajectory
            shortest = trajectories[0]
            shortest.discarded = False
            selected_trajectories.append(shortest)

            # Also select the longest if different
            if len(trajectories) > 1 and len(trajectories[-1].route) > len(shortest.route):
                longest = trajectories[-1]
                longest.discarded = False
                selected_trajectories.append(longest)

        # Add discarded trajectories to discard_edge_trajectories
        for traj in old_edge_trajectories:
            if traj.discarded:
                self.discard_edge_trajectories.append(traj)

        print(f"Total discard: {len(self.discard_edge_trajectories)}")
        # print(f"Valid discarded: {len([0 for traj in self.discard_edge_trajectories if not traj.to_delete])}")

        # Expand selected trajectories
        for trajectory in selected_trajectories:
            new_trajectories = trajectory.get_new_trajectories(self.num_step, max_backtrack=self.max_backtrack)
            if new_trajectories:
                # Add valid new trajectories
                for new_traj in new_trajectories:
                    self.trajectories.append(new_traj)
                    self._register_trajectory_in_index(new_traj)
                    self.edge_trajectories.append(new_traj)

        # Return count of new trajectories
        return len(self.trajectories) - before_len

    @property
    def probability_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions.

        This method computes how likely each position is to contain an agent,
        based on the number of valid trajectories passing through it.

        Returns:
            numpy.ndarray: 2D array with probability for each position
        """
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
        self.discard_edge_trajectories = [traj for traj in self.discard_edge_trajectories if not traj.to_delete]

    def check_wall_trajectories(self, key: Point | DirectionalNode):
        if isinstance(key, DirectionalNode):
            trajectories = self.node_index[key]
        elif isinstance(key, Point):
            trajectories = self.position_index[key]

        for traj in trajectories:
            traj.get_last_node()

    def _restart_from_discarded(self):
        """
        When all trajectories have been eliminated, this method tries to resurrect
        discarded edge trajectories from previous steps and propagate them forward
        to the current step, applying appropriate temporal constraints.

        Temporal constraints are applied as follows:
        - excludes: applied to all steps
        - contains: applied only to the current step

        Returns:
            bool: True if any trajectories were successfully resurrected, False otherwise
        """
        if len(self.trajectories) > 0:
            return True  # No need to restart if we still have valid trajectories

        # Clear any existing trajectories marked for deletion
        self._clean_up_trajectories()

        print(self.temporal_constraints[-1])

        # Go backwards through time steps looking for discarded trajectories
        for backward_step in range(self.num_step, -1, -1):
            candidates: list[Trajectory] = [traj for traj in self.discard_edge_trajectories if traj.created_at == backward_step]

            if backward_step == 0:
                for direction in Direction:
                    # This will either create a new node or use an existing one
                    root_node = self.registry.get_or_create_node(Point(0, 0), direction)
                    trajectory = Trajectory(root_node, self.num_step)
                    candidates.append(trajectory)

            if not candidates:
                continue

            print(f"Restarting from {len(candidates)} candidates at {backward_step}")

            for forward_step in range(backward_step + 1, self.num_step + 1):
                new_candidates: list[Trajectory] = []
                new_candidates.extend(candidates)

                for candidate in candidates:
                    new_trajectories = candidate.get_new_trajectories(forward_step, max_backtrack=3)
                    new_candidates.extend(new_trajectories)

                candidates = new_candidates

                print(f"Expanded to {len(candidates)} candidates")

                for traj in candidates:
                    # Get constraints for this trajectory at this forward step
                    constraints = self.temporal_constraints[forward_step] if forward_step < len(self.temporal_constraints) else None

                    if constraints:
                        for node in traj.nodes:
                            if node in constraints.route.excludes:
                                traj.prune()
                                break

                        for node in constraints.route.contains:
                            if node not in traj.nodes:
                                traj.prune()
                                break

                        if traj.tail.position in constraints.tail.excludes:
                            traj.prune()
                            break

                        if traj.tail.position not in constraints.tail.contains:
                            traj.prune()
                            break

                candidates = [traj for traj in candidates if not traj.to_delete]
                print(f"Filtered to {len(candidates)} candidates")

                if not candidates:
                    continue

            # for traj in candidates:
            #     constraints = self.temporal_constraints[-1]

            if len(candidates) > 0:
                print(candidates[-1])
                print(candidates[-1].nodes)

            candidates = [traj for traj in candidates if not traj.to_delete]
            print(f"Filtered to {len(candidates)} candidates")

            # print(f"Found {len(candidates)} candidates")

            if len(candidates) > 0:
                break

        self.trajectories = candidates
        self.edge_trajectories = self.trajectories.copy()
