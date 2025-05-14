from typing import Optional
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager

import numpy as np

from .utils import Direction, Action, Point, Tile
from .node import NodeRegistry, DirectionalNode

# Profiling globals
_profiler = None

def start_profiling():
    """Start profiling session"""
    global _profiler
    _profiler = cProfile.Profile()
    _profiler.enable()
    return _profiler

def stop_profiling(print_stats=True, sort_by='cumulative', lines=20):
    """Stop profiling and optionally print stats"""
    global _profiler
    if _profiler is not None:
        _profiler.disable()
        if print_stats:
            s = io.StringIO()
            ps = pstats.Stats(_profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(lines)
            print(s.getvalue())
        return _profiler
    return None

@contextmanager
def profile_section(section_name):
    """Context manager for profiling specific code sections"""
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    print(f"\n--- Profiling results for {section_name} ---")
    ps.print_stats(20)
    print(s.getvalue())

def profile(func):
    """Decorator to profile a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        print(f"\n--- Profiling results for {func.__name__} ---")
        ps.print_stats(20)
        print(s.getvalue())
        return result
    return wrapper


class Trajectory:
    def __init__(self, root_node):
        self.root: DirectionalNode = root_node
        self.last: Optional[DirectionalNode] = None
        self.route: list[Action] = []
        self.nodes: list[DirectionalNode] = [self.root]
        self.position_cache: set[Point] = {root_node.position}  # Cache positions for faster lookups

        self.invalid: bool = False
        self.invalid_action_idx: Optional[int] = None

        self._inherited_from: Optional['Trajectory'] = None
        self._inherits_to: dict[Action, 'Trajectory'] = {}

    def __hash__(self):
        return hash(tuple(self.route))

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False
        return (self.root == other.root and
                self.route == other.route)

    def __contains__(self, item: DirectionalNode | Point) -> bool:
        """
        Check if a DirectionalNode or Point is in this trajectory.

        Args:
            item: Either a DirectionalNode or a Point object

        Returns:
            bool: True if the node or position is in this trajectory, False otherwise
        """
        if self.invalid:
            return False

        if isinstance(item, DirectionalNode):
            return item in self.nodes

        if isinstance(item, Point):
            return item in self.position_cache

        return False

    def copy(self):
        """Create a deep copy of this trajectory with the same root but new lists."""
        new_trajectory = Trajectory(self.root)
        new_trajectory.route = self.route[:]
        new_trajectory.nodes = self.nodes[:]
        new_trajectory.last = self.last
        new_trajectory.position_cache = self.position_cache.copy()
        # Set up inheritance relationship
        new_trajectory._inherited_from = self
        return new_trajectory

    def update(self, action):
        """
        Add an action to the route and validate it.
        If the action is invalid, mark this trajectory as invalid.
        """
        self.route.append(action)

        # Check if this action is valid
        self.get_last_node()

    def mark_as_invalid(self, invalid_action_idx: Optional[int] = None):
        """
        Mark this trajectory as invalid and propagate to inheriting trajectories.

        Args:
            invalid_action_idx: The index of the action in the route that is invalid.
                                If None, the entire trajectory is marked invalid.
        """
        # avoid propagating if already done.
        if self.invalid:
            return

        # Store the index of the invalid action
        if invalid_action_idx is not None:
            if invalid_action_idx < len(self.route):
                self.invalid = True
                self.invalid_action_idx = invalid_action_idx

                # If this has a parent trajectory, check if the parent should be invalidated
                if self._inherited_from is not None:
                    self._inherited_from.mark_as_invalid(invalid_action_idx)
            else:
                return
        else:
            self.invalid = True

        # Propagate to all trajectories that inherit from this one
        for traj in self._inherits_to.values():
            traj.mark_as_invalid(invalid_action_idx)

    def get_last_node(self):
        """
        Get the last node in the trajectory.

        Returns:
            DirectionalNode: The last node in the trajectory, or None if invalid.
        """
        if self.invalid:
            return None

        populate_nodes = (len(self.nodes) - 1) != len(self.route)
        if populate_nodes:
            self.nodes = [self.root]
            # Reset position cache when repopulating nodes
            self.position_cache = {self.root.position}

        current_node = self.root
        for i, action in enumerate(self.route):
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                self.mark_as_invalid(i)  # Mark invalid at the specific action index
                return None

            if populate_nodes:
                self.nodes.append(current_node)
                # Add position to cache for faster lookups
                self.position_cache.add(current_node.position)

        self.last = current_node
        return current_node

    def get_new_trajectories(self, max_backtrack: int = 3):
        """
        Get new trajectories by exploring all valid actions from the current endpoint.

        Args:
            max_backtrack: Maximum number of backtracking steps allowed.

        Returns:
            List[Trajectory]: List of new trajectories, each extending from this one
        """
        if self.invalid:
            return []

        last_node = self.get_last_node()
        if not last_node:
            return []

        if self.has_backtrack(max_backtrack):
            return []

        new_trajectories = []

        # Explore all valid actions from current node
        for action, next_node in last_node.children.items():
            if action not in self._inherits_to:
                # Create a new trajectory with this action
                new_trajectory = self.copy()
                new_trajectory.update(action)

                # Skip if this made the trajectory invalid
                if new_trajectory.invalid:
                    continue

                self._inherits_to[action] = new_trajectory

                new_trajectories.append(new_trajectory)

        return new_trajectories

    def get_endpoint_key(self, consider_direction: bool = True):
        """
        Get a key for this trajectory's endpoint, used for deduplication.

        When consider_direction=True, the key is (position, direction),
        otherwise it's just the position.

        Args:
            consider_direction: Whether to include direction in the key.

        Returns:
            Tuple containing the key for deduplication
        """
        if self.invalid:
            return None

        last_node = self.get_last_node()
        if not last_node:
            return None

        if consider_direction:
            return (last_node.position, last_node.direction)
        else:
            return (last_node.position,)

    def __str__(self):
        """String representation of the trajectory."""
        if self.invalid:
            return f"Invalid Trajectory: {self.route}"

        last_node = self.get_last_node()
        if last_node:
            return f"Trajectory to {last_node.position} {last_node.direction} from {self.root.position} {self.root.direction} via {self.route}"
        return f"Incomplete Trajectory: {self.route}"

    def __repr__(self):
        return self.__str__()

    def has_backtrack(self, max_backtrack: int = 3) -> bool:
        """
        Check if this trajectory backtracks more than the allowed number of steps.

        Args:
            max_backtrack: Maximum number of backtracking steps allowed.

        Returns:
            bool: True if trajectory has too much backtracking.
        """
        if not self.nodes or len(self.nodes) <= 1:
            return False  # Need at least 2 nodes to backtrack

        visited_positions = set()
        backtrack_count = 0

        for node in self.nodes:
            pos = node.position

            if pos not in visited_positions:
                visited_positions.add(pos)
            else:
                backtrack_count += 1
                if backtrack_count > max_backtrack:
                    return True

        return False

    @property
    def parent(self):
        return self._inherited_from

    @property
    def children(self):
        return self._inherits_to.values()


class TrajectoryTree:
    def __init__(
        self,
        init_position: Point,
        init_direction: Optional[Direction] = None,
        size: int = 16,
        consider_direction: bool = True,
        registry: Optional[NodeRegistry] = None
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
        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        # Spatial index: maps positions to trajectories that pass through them
        self.position_index: dict[Point, set[Trajectory]] = {}

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
            trajectory = Trajectory(root_node)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        self.edge_trajectories: list[Trajectory] = self.trajectories.copy()

        # Add expansion cache for memoization
        self.expansion_cache: dict[DirectionalNode, dict[Action, DirectionalNode]] = {}

    def _register_trajectory_in_index(self, trajectory):
        """Register a trajectory in the position index."""
        if trajectory.invalid:
            return

        for position in trajectory.position_cache:
            if position not in self.position_index:
                self.position_index[position] = set()
            self.position_index[position].add(trajectory)

    def _unregister_trajectory_from_index(self, trajectory):
        """Remove a trajectory from the position index."""
        for position in trajectory.position_cache:
            if position in self.position_index and trajectory in self.position_index[position]:
                self.position_index[position].remove(trajectory)
                # Clean up empty sets
                if not self.position_index[position]:
                    del self.position_index[position]

    def _clear_all_trajectories(self):
        """Clear all trajectories and reset the position index."""
        self.trajectories.clear()
        self.edge_trajectories.clear()
        self.position_index.clear()

    def set_consider_direction(self, consider_direction: bool):
        """
        Update whether to consider direction when reducing trajectories.

        Args:
            consider_direction: If True, trajectories with same position but different
                               directions are considered distinct. If False, only position matters.
        """
        self.consider_direction = consider_direction

    def prune(self, information: list[tuple[Point, Tile]], seeking_scout: bool = True):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
            seeking_scout (bool): If True, look for scout agent. If False, look for guard agent.
        """
        # Process agent sightings first (early exit)
        agent_position = self._check_for_agent(information, seeking_scout)
        if agent_position is not None:
            self._reset_trajectories_for_agent(agent_position)
            self.trajectories = self._get_valid_trajectories()
            return

        if seeking_scout:
            self._prune_by_tile_content(information)
            self.trajectories = self._get_valid_trajectories()

    def _check_for_agent(self, information, seeking_scout):
        """Check if an agent is present in the information."""
        for position, tile in information:
            if (seeking_scout and tile.has_scout) or (not seeking_scout and tile.has_guard):
                return position
        return None

    def _reset_trajectories_for_agent(self, agent_position):
        """Reset all trajectories to start from the agent position."""
        # Clear existing trajectories and their index
        self._clear_all_trajectories()

        # Create new trajectories at the agent's position, considering all possible directions
        for direction in Direction:
            root_node = self.registry.get_or_create_node(agent_position, direction)
            trajectory = Trajectory(root_node)
            self.trajectories.append(trajectory)
            self._register_trajectory_in_index(trajectory)

        # Update edge trajectories
        self.edge_trajectories = self.trajectories.copy()

    def _prune_by_tile_content(self, information):
        """Prune trajectories based on tile content."""
        # Group positions by condition to avoid redundant iterations
        positions_to_exclude = []  # Positions trajectories should NOT contain
        positions_must_contain = []  # Positions trajectories MUST contain

        for position, tile in information:
            # Case 2: not visited - remove trajectories containing this position
            if tile.is_visible and (tile.is_recon or tile.is_mission):
                positions_to_exclude.append(position)
            # Case 3: visited - only keep trajectories containing this position
            if tile.is_empty:
                positions_must_contain.append(position)

        # Use the spatial index for efficient filtering
        if not positions_to_exclude and not positions_must_contain:
            return  # No filtering needed

        self._apply_filtering(positions_to_exclude, positions_must_contain)

    def _apply_filtering(self, positions_to_exclude, positions_must_contain):
        """Apply filtering based on position constraints."""
        # Find trajectories to exclude (has any excluded position)
        trajectories_to_remove = set()
        for position in positions_to_exclude:
            if position in self.position_index:
                trajectories_to_remove.update(self.position_index[position])

        # Find trajectories to keep (has all required positions)
        trajectories_to_keep = None
        for position in positions_must_contain:
            if position in self.position_index:
                position_trajs = self.position_index[position]
                if trajectories_to_keep is None:
                    trajectories_to_keep = position_trajs.copy()
                else:
                    trajectories_to_keep.intersection_update(position_trajs)
            else:
                # If any required position has no trajectories, none can be kept
                trajectories_to_keep = set()
                break

        # Default to all trajectories if no must-contain positions
        if not positions_must_contain:
            trajectories_to_keep = set(self.trajectories)

        # Compute final valid trajectories
        valid_trajectories = [
            traj for traj in trajectories_to_keep if traj not in trajectories_to_remove
        ] if trajectories_to_keep else []

        # Remove invalid trajectories from the index
        removed_trajectories = set(self.trajectories) - set(valid_trajectories)
        for traj in removed_trajectories:
            traj.mark_as_invalid()
            self._unregister_trajectory_from_index(traj)

        self._destroy_trajectory_families()
        self.trajectories = valid_trajectories

        # Update edge trajectories - use a set operation for efficiency
        trajectory_set = set(self.trajectories)

        has_deleted_edge = False
        for traj in self.edge_trajectories:
            if traj not in trajectory_set:
                has_deleted_edge = True
                break

        if has_deleted_edge:
            self.edge_trajectories = self._get_edge_trajectories()

    @profile
    def _get_valid_trajectories(self):
        """
        Returns a list of all valid trajectories (not marked as invalid).

        Returns:
            list[Trajectory]: Valid trajectories
        """
        return [traj for traj in self.trajectories if not traj.invalid]

    @profile
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
        if not self.trajectories:
            return 0

        before_len = len(self.trajectories)

        old_edge_trajectories = self.edge_trajectories
        print(f"Updating from {len(old_edge_trajectories)} trajectories")
        self.edge_trajectories = []  # Clear edge trajectories for this step

        # Mapping of endpoint keys to trajectories
        endpoint_to_trajectories = {}

        # Group trajectories by their endpoints
        for traj in old_edge_trajectories:
            # Skip invalid trajectories
            if traj.invalid:
                continue

            # Get key for this trajectory's endpoint
            key = traj.get_endpoint_key(self.consider_direction)
            if not key:
                continue

            if key not in endpoint_to_trajectories:
                endpoint_to_trajectories[key] = []
            endpoint_to_trajectories[key].append(traj)

        # For each endpoint, select the shortest and longest trajectories
        selected_trajectories = []
        for key, trajectories in endpoint_to_trajectories.items():
            # Sort by trajectory length (shorter first)
            trajectories.sort(key=lambda t: len(t.route))

            # Always select the shortest trajectory
            shortest = trajectories[0]
            selected_trajectories.append(shortest)

            # Also select the longest if different
            if len(trajectories) > 1 and len(trajectories[-1].route) > len(shortest.route):
                longest = trajectories[-1]
                selected_trajectories.append(longest)

        # Expand selected trajectories
        for trajectory in selected_trajectories:
            new_trajectories = trajectory.get_new_trajectories()
            if new_trajectories:
                # Add valid new trajectories
                for new_traj in new_trajectories:
                    self.trajectories.append(new_traj)
                    self._register_trajectory_in_index(new_traj)
                    self.edge_trajectories.append(new_traj)

        # # Periodically validate a random sample of trajectories
        # # to ensure nothing that should be invalidated wasn't caught
        # if random.random() < 0.1:  # 10% chance to run validation
        #     sample_size = min(500, len(self.trajectories))  # Cap at 500 trajectories
        #     if sample_size > 0:
        #         sampled_trajectories = random.sample(self.trajectories, sample_size)
        #         for traj in sampled_trajectories:
        #             traj.get_last_node()

        #         # Filter out invalid trajectories
        #         self.trajectories = self._get_valid_trajectories()

        # Return count of new trajectories
        return len(self.trajectories) - before_len

    @property
    def probability_density(self):
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
            if trajectory.invalid:
                continue

            last_node = trajectory.get_last_node()
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

    def _destroy_trajectory_families(self):
        """
        Starting from any invalid trajectory, traverse up ._inherited_from,
        deleting invalids from ._inherits_to. traverse until no invalids in ._inherits_to are found
        """
        parent_traj: Optional[Trajectory] = None

        for traj in self.trajectories:
            if traj.invalid:
                parent_traj = traj.parent

                if parent_traj:
                    if len(parent_traj.children) > 0:

                        parent_traj._inherits_to = {
                            action: child for action, child in parent_traj._inherits_to.items()
                            if not child.invalid
                        }

    def _get_edge_trajectories(self):
        """
        Edge trajectories are basically any trajectory without children
        """
        edge_trajectories = []
        for traj in self.trajectories:
            if traj.invalid:
                continue

            if len(traj.children) < 4:
                edge_trajectories.append(traj)

        return edge_trajectories
