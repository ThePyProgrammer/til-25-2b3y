import heapq
import sys
from typing import Optional

from ..node import DirectionalNode, NodeRegistry
from ..utils import Point, Action
from ..utils.pathfinding import manhattan_distance

from .trajectory import Trajectory
from .constraints import TemporalTrajectoryConstraints


def create_trajectories_from_constraints(
    roots: list[DirectionalNode],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    registry: NodeRegistry,
    beam_width: int = 100,  # Maximum queue size before pruning
    enable_modifications: bool = True,  # Whether to enable trajectory modifications
) -> list[Trajectory]:
    """
    Creates a list of trajectories that fulfill the conditions:
        1. starts at one of the roots
        2. visits all the points in in temporal_constraints[-1].route.contains
        3. does not contain any nodes in temporal_constraints[i].route.excludes at step i (a step is considered to be taken for each action)
            temporal_constraints[i].route.excludes can only have more members at i+1, not less
        4. the tail node at step i must match the conditions in temporal_constraints[i].tail
    Return the trajectories that fulfill 1-4, it does not need to use up all of budget which is the maximum number of steps

    This is somewhat like the travelling salesman problem but with more conditions.
    """
    # Handle edge cases
    if not roots or len(temporal_constraints) == 0:
        return []

    valid_trajectories: list[Trajectory] = []

    # Initialize priority queue with root trajectories
    trajectories_to_explore, counter = _initialize_trajectory_queue(roots, temporal_constraints, budget)

    # Best-first search approach using priority queue
    explored_count = 0
    while trajectories_to_explore:
        # Pop trajectory with lowest priority value (best trajectory to explore)
        priority, _, current_trajectory = heapq.heappop(trajectories_to_explore)
        explored_count += 1

        # Log progress periodically
        _log_exploration_progress(
            explored_count, priority, current_trajectory,
            trajectories_to_explore, temporal_constraints, budget
        )

        # Process current trajectory and check if it should be expanded
        should_expand, valid_trajectories = _process_trajectory(
            current_trajectory, temporal_constraints, budget, valid_trajectories
        )

        if should_expand:
            # Expand and add new trajectories
            current_step = len(current_trajectory.route)
            trajectories_to_explore, counter = _expand_and_add_trajectories(
                current_trajectory, current_step, temporal_constraints,
                budget, trajectories_to_explore, counter
            )

            # Apply beam search if needed
            trajectories_to_explore = _apply_beam_search(trajectories_to_explore, beam_width)

    print(f"Search complete. Found {len(valid_trajectories)} valid trajectories after exploring {explored_count} paths.")

    # If modifications are enabled and we found some valid trajectories, try to improve them
    if enable_modifications:
        valid_trajectories = _apply_trajectory_modifications(
            valid_trajectories, temporal_constraints, budget, registry
        )

    return valid_trajectories


def _apply_trajectory_modifications(
    valid_trajectories: list[Trajectory],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    registry: NodeRegistry
) -> list[Trajectory]:
    """Apply modifications to trajectories to improve them."""
    if not valid_trajectories:
        return valid_trajectories

    print("Attempting to create modified trajectories with detours...")

    # Find trajectories that visit some but not all required points
    partial_trajectories: list[Trajectory] = []
    for traj in valid_trajectories:
        if not _has_visited_all_required_points(traj, temporal_constraints):
            partial_trajectories.append(traj)

    if partial_trajectories:
        print(f"Found {len(partial_trajectories)} partial trajectories to modify")
        modified_trajectories = _create_trajectories_with_modifications(
            partial_trajectories,
            temporal_constraints,
            budget,
            registry
        )

        # Add valid modified trajectories
        modified_count = 0
        for traj in modified_trajectories:
            if _has_visited_all_required_points(traj, temporal_constraints):
                valid_trajectories.append(traj)
                modified_count += 1

        print(f"Added {modified_count} modified trajectories")

    return valid_trajectories


def _initialize_trajectory_queue(
    roots: list[DirectionalNode],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int
) -> tuple[list[tuple[float, int, Trajectory]], int]:
    """Initialize the priority queue with trajectories starting from root nodes."""
    trajectories_to_explore: list[tuple[float, int, Trajectory]] = []
    counter = 0

    for root in roots:
        trajectory = Trajectory(root, 0)
        priority = _calculate_trajectory_priority(trajectory, temporal_constraints, budget)
        heapq.heappush(trajectories_to_explore, (priority, counter, trajectory))
        counter += 1

    print(f"Starting search with {len(roots)} root nodes")
    return trajectories_to_explore, counter


def _log_exploration_progress(
    explored_count: int,
    priority: float,
    current_trajectory: Trajectory,
    trajectories_to_explore: list[tuple[float, int, Trajectory]],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int
) -> None:
    """Log progress information during trajectory exploration."""
    if explored_count % 100 != 0:
        return

    print(f"Explored {explored_count} trajectories, queue size: {len(trajectories_to_explore)}")
    print(f"Current best priority: {priority}, trajectory length: {len(current_trajectory.route)}")

    if len(temporal_constraints) > 0:
        required_points = temporal_constraints[-1].route.contains
        visited = sum(1 for p in required_points if p in current_trajectory.position_cache)
        print(f"Visited {visited}/{len(required_points)} required points")

        # Print minimum distance to any unvisited point
        if visited < len(required_points):
            current_pos = current_trajectory.tail.position
            unvisited = [p for p in required_points if p not in current_trajectory.position_cache]
            min_dist = min(abs(current_pos.x - p.x) + abs(current_pos.y - p.y) for p in unvisited)
            print(f"Minimum distance to unvisited point: {min_dist}, remaining budget: {budget - len(current_trajectory.route)}")

    sys.stdout.flush()  # Ensure output is displayed immediately


def _process_trajectory(
    current_trajectory: Trajectory,
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    valid_trajectories: list[Trajectory]
) -> tuple[bool, list[Trajectory]]:
    """Process a trajectory, checking validity and constraints."""
    # Skip invalid trajectories
    if current_trajectory.invalid or current_trajectory.pruned:
        return False, valid_trajectories

    # Get current step
    current_step = len(current_trajectory.route)

    # Check if we've exceeded budget
    if current_step >= budget:
        return False, valid_trajectories

    # Check constraints for current step
    if not _check_trajectory_constraints(current_trajectory, current_step, temporal_constraints):
        return False, valid_trajectories

    # Check if trajectory has visited all required points
    if _has_visited_all_required_points(current_trajectory, temporal_constraints):
        valid_trajectories.append(current_trajectory)
        return False, valid_trajectories  # No need to expand further

    return True, valid_trajectories


def _expand_and_add_trajectories(
    current_trajectory: Trajectory,
    current_step: int,
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    trajectories_to_explore: list[tuple[float, int, Trajectory]],
    counter: int
) -> tuple[list[tuple[float, int, Trajectory]], int]:
    """Expand a trajectory and add new trajectories to the queue."""
    # Expand trajectory with all possible actions
    new_trajectories = current_trajectory.get_new_trajectories(current_step + 1)

    # Add new trajectories to priority queue with calculated priorities
    for new_traj in new_trajectories:
        if not new_traj.invalid and not new_traj.pruned:
            priority = _calculate_trajectory_priority(new_traj, temporal_constraints, budget)
            heapq.heappush(trajectories_to_explore, (priority, counter, new_traj))
            counter += 1

    return trajectories_to_explore, counter


def _apply_beam_search(
    trajectories_to_explore: list[tuple[float, int, Trajectory]],
    beam_width: int
) -> list[tuple[float, int, Trajectory]]:
    """Apply beam search to limit queue size."""
    if len(trajectories_to_explore) > beam_width:
        print(f"Applying beam search: pruning queue from {len(trajectories_to_explore)} to {beam_width}")
        trajectories_to_explore = heapq.nsmallest(beam_width, trajectories_to_explore)
        heapq.heapify(trajectories_to_explore)
    return trajectories_to_explore


def _check_trajectory_constraints(
    trajectory: Trajectory,
    step: int,
    temporal_constraints: TemporalTrajectoryConstraints
) -> bool:
    """Helper function to check if a trajectory satisfies constraints at a given step."""
    if step >= len(temporal_constraints):
        return True

    current_constraints = temporal_constraints[step]

    # Check if current state violates route excludes
    if any(node.position in current_constraints.route.excludes for node in trajectory.nodes):
        trajectory.prune()
        return False

    # Check tail constraints
    if current_constraints.tail.contains and trajectory.tail.position not in current_constraints.tail.contains:
        trajectory.prune()
        return False

    if trajectory.tail.position in current_constraints.tail.excludes:
        trajectory.prune()
        return False

    return True


def _has_visited_all_required_points(
    trajectory: Trajectory,
    temporal_constraints: TemporalTrajectoryConstraints
) -> bool:
    """Helper function to check if a trajectory has visited all required points."""
    if len(temporal_constraints) == 0:
        return True

    required_points: list[Point] = temporal_constraints[-1].route.contains
    return all(point in trajectory.position_cache for point in required_points)


def _calculate_trajectory_priority(
    trajectory: Trajectory,
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int
) -> float:
    """
    Calculate the priority of a trajectory for exploration.
    Lower priority value = higher priority for exploration.

    The priority is based on:
    1. Number of unvisited required points (fewer is better)
    2. Minimum Manhattan distance to next required point (closer is better)
    3. Efficiency ratio (distance vs remaining budget - lower is better)

    Args:
        trajectory: The trajectory to evaluate
        temporal_constraints: The temporal constraints containing required points
        budget: Maximum number of steps allowed

    Returns:
        float: Priority value (lower is better)
    """
    # Early return if no constraints or invalid trajectory
    if len(temporal_constraints) == 0 or trajectory.invalid or trajectory.pruned:
        return float('inf')  # Lowest priority

    # Get required points and check which ones are still unvisited
    required_points = temporal_constraints[-1].route.contains

    # If no required points, return based on trajectory length (shorter is better)
    if not required_points:
        return len(trajectory.route)

    # Find points that haven't been visited yet
    unvisited_points: list[Point] = []
    for point in required_points:
        if point not in trajectory.position_cache:
            unvisited_points.append(point)

    # If all required points visited, this is a valid trajectory - give it high priority
    if not unvisited_points:
        return 0  # Highest priority

    # Calculate minimum Manhattan distance to any unvisited required point
    current_pos = trajectory.tail.position
    min_distance = float('inf')
    for point in unvisited_points:
        distance = abs(current_pos.x - point.x) + abs(current_pos.y - point.y)
        if distance < min_distance:
            min_distance = distance

    # Calculate remaining budget
    remaining_steps = budget - len(trajectory.route)

    # If not enough steps left to reach even the closest point, give low priority
    if min_distance > remaining_steps:
        return float('inf')

    # Additional penalty for trajectories that make little progress toward goals
    backtrack_penalty = 0
    if trajectory.has_backtrack(max_backtrack=4):
        backtrack_penalty = 500  # Significant penalty for excessive backtracking

    # Priority based on multiple factors (lower is better)
    priority = (
        len(unvisited_points) * 1000 +  # Primary factor: number of unvisited points
        min_distance * 10 +             # Secondary factor: distance to closest point
        (min_distance / max(1, remaining_steps) * 5) +  # Efficiency factor
        backtrack_penalty  # Penalty for excessive backtracking
    )

    return priority


def _create_trajectories_with_modifications(
    base_trajectories: list[Trajectory],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    registry: NodeRegistry,
    max_modifications: int = 50
) -> list[Trajectory]:
    """
    Take existing trajectories and modify them by inserting detours to reach additional points.

    Args:
        base_trajectories: List of existing trajectories to modify
        temporal_constraints: The temporal constraints to satisfy
        budget: Maximum number of steps allowed
        registry: The node registry
        max_modifications: Maximum number of modified trajectories to create

    Returns:
        List of modified trajectories
    """
    if not base_trajectories or len(temporal_constraints) == 0:
        return []

    # Find which trajectories already visit some but not all required points
    required_points = temporal_constraints[-1].route.contains

    # Score and filter trajectories
    scored_trajectories: list[tuple[float, Trajectory, list[Point]]] = []
    for traj in base_trajectories:
        if traj.invalid or traj.pruned:
            continue

        score, visited, unvisited = _score_trajectory_for_modification(traj, required_points)
        if score > 0:  # Only consider trajectories that visit some but not all points
            scored_trajectories.append((score, traj, unvisited))

    # Sort by score (highest first) and limit candidates
    scored_trajectories.sort(reverse=True, key=lambda x: x[0])
    base_candidates = scored_trajectories[:min(20, len(scored_trajectories))]

    # Process each candidate trajectory
    modified_trajectories: list[Trajectory] = []
    for _, trajectory, unvisited in base_candidates:
        modified_trajectories = _process_trajectory_modifications(
            trajectory, unvisited, temporal_constraints, budget,
            modified_trajectories, max_modifications
        )
        if len(modified_trajectories) >= max_modifications:
            break

    return modified_trajectories


def _score_trajectory_for_modification(
    trajectory: Trajectory,
    required_points: list[Point]
) -> tuple[float, list[Point], list[Point]]:
    """
    Score a trajectory based on how suitable it is for modification.

    Args:
        trajectory: The trajectory to score
        required_points: List of required points

    Returns:
        Tuple of (score, visited_points, unvisited_points)
    """
    # Count visited and unvisited points
    visited: list[Point] = [p for p in required_points if p in trajectory.position_cache]
    unvisited: list[Point] = [p for p in required_points if p not in trajectory.position_cache]

    # Calculate score - prefer trajectories that visit more points with fewer steps
    score = len(visited) * 1000 - len(trajectory.route) if visited and unvisited else -1

    return score, visited, unvisited


def _process_trajectory_modifications(
    trajectory: Trajectory,
    unvisited: list[Point],
    temporal_constraints: TemporalTrajectoryConstraints,
    budget: int,
    modified_trajectories: list[Trajectory],
    max_modifications: int
) -> list[Trajectory]:
    """
    Process a single trajectory and create detours to reach unvisited points.

    Args:
        trajectory: Base trajectory to modify
        unvisited: List of unvisited points
        temporal_constraints: Constraints to satisfy
        budget: Maximum budget
        modified_trajectories: Current list of modified trajectories
        max_modifications: Maximum number of modifications to create

    Returns:
        Updated list of modified trajectories
    """
    # Find potential cut points
    cut_points = _find_cut_points(trajectory, unvisited)

    # For each cut point, try to create detours to each unvisited point
    for cut_index, cut_node in cut_points:
        for target_point in unvisited:
            # Calculate remaining budget for detour
            remaining_budget = budget - len(trajectory.route)
            if remaining_budget <= 0:
                continue

            # Find a detour to the target point and back
            detour = _find_detour(
                cut_node,
                target_point,
                remaining_budget,
                trajectory,
                temporal_constraints,
                cut_index
            )

            if detour:
                # Create modified trajectory with the detour
                modified = _create_modified_trajectory(trajectory, cut_index, detour, cut_index)

                # Check if modified trajectory is valid and satisfies constraints
                if (not modified.invalid and not modified.pruned and
                    len(modified.route) <= budget and
                    _check_trajectory_constraints(modified, len(modified.route), temporal_constraints)):

                    modified_trajectories.append(modified)

                    if len(modified_trajectories) >= max_modifications:
                        return modified_trajectories

    return modified_trajectories


def _find_cut_points(
    trajectory: Trajectory,
    unvisited_points: list[Point],
    max_cuts: int = 5,
    spacing: int = 3
) -> list[tuple[int, DirectionalNode]]:
    """
    Find promising points in the trajectory where we could cut and insert a detour.

    Args:
        trajectory: The trajectory to analyze
        unvisited_points: Points that still need to be visited
        max_cuts: Maximum number of cut points to return
        spacing: Minimum number of steps between cut points

    Returns:
        List of (index, node) tuples representing potential cut points
    """
    if len(trajectory.nodes) <= 1:
        return []

    # Calculate distances from each node to unvisited points
    distances = []
    for i, node in enumerate(trajectory.nodes):
        min_distance = float('inf')
        for point in unvisited_points:
            distance = manhattan_distance(node.position, point)
            min_distance = min(min_distance, distance)
        distances.append((i, node, min_distance))

    # Sort potential cut points by distance (closest first)
    distances.sort(key=lambda x: x[2])

    # Take the best cut points with spacing between them
    cut_points = []
    used_indices = set()

    for i, node, distance in distances:
        # Skip if too close to other cut points
        if any(abs(i - prev_i) < spacing for prev_i, _ in cut_points):
            continue

        cut_points.append((i, node))
        used_indices.add(i)

        if len(cut_points) >= max_cuts:
            break

    return cut_points


def _find_detour(
    start_node: DirectionalNode,
    target_point: Point,
    budget: int,
    existing_trajectory: Trajectory,
    temporal_constraints: TemporalTrajectoryConstraints,
    step_offset: int
) -> Optional[list[Action]]:
    """
    Find a detour from a cut point to a target point and back.

    Args:
        start_node: The node where the detour starts
        target_point: The point we want to visit
        budget: Maximum steps for the detour
        existing_trajectory: The original trajectory we're modifying
        temporal_constraints: Constraints to follow
        step_offset: The current step in the original trajectory

    Returns:
        A list of actions forming the detour, or None if no valid detour found
    """
    # Create a mini trajectory starting at the cut point
    detour = Trajectory(start_node, step_offset)

    # Use priority queue to find paths
    queue = []
    counter = 0
    heapq.heappush(queue, (0, counter, detour, False))  # (priority, counter, trajectory, reached_target)

    visited_states = {}  # (position, direction, reached_target) -> steps used
    max_explored = 1000  # Limit search space for detours

    explored = 0
    while queue and explored < max_explored:
        _, _, current, reached_target = heapq.heappop(queue)
        explored += 1

        # Get current state
        current_pos = current.tail.position
        current_dir = current.tail.direction
        state_key = (current_pos, current_dir, reached_target)

        # Check if we've seen this state with fewer steps
        current_steps = len(current.route)
        if state_key in visited_states and visited_states[state_key] <= current_steps:
            continue

        # Record this state
        visited_states[state_key] = current_steps

        # Check if we've reached the target
        if current_pos == target_point:
            reached_target = True

        # If we've reached the target and returned to start node (or close to it)
        if reached_target and current_pos == start_node.position:
            return current.route

        # Check budget
        if current_steps >= budget:
            continue

        # Check constraints at this step
        step = step_offset + current_steps
        if not _check_trajectory_constraints(current, step, temporal_constraints):
            continue

        # Expand trajectory
        new_trajectories = current.get_new_trajectories(step + 1)

        for new_traj in new_trajectories:
            if new_traj.invalid or new_traj.pruned:
                continue

            # Calculate priority - distance to target if not reached,
            # distance back to start if target reached
            if not reached_target:
                target_dist = manhattan_distance(new_traj.tail.position, target_point)
                priority = target_dist
            else:
                start_dist = manhattan_distance(new_traj.tail.position, start_node.position)
                priority = start_dist

            counter += 1
            heapq.heappush(queue, (priority, counter, new_traj, reached_target))

    return None


def _create_modified_trajectory(
    original: Trajectory,
    cut_index: int,
    detour_actions: list[Action],
    step_offset: int
) -> Trajectory:
    """
    Create a new trajectory that follows the original up to the cut point,
    then takes the detour, and continues from the cut point.

    Args:
        original: The original trajectory
        cut_index: Index where to cut the trajectory
        detour_actions: List of actions for the detour
        step_offset: Current step in the original trajectory

    Returns:
        A new trajectory with the detour inserted
    """
    # Start with the root node
    new_trajectory = Trajectory(original.head, 0)

    # Follow original trajectory up to the cut point
    for i, action in enumerate(original.route[:cut_index]):
        new_trajectory.update(action, i + 1)

    # Take the detour
    for i, action in enumerate(detour_actions):
        new_trajectory.update(action, cut_index + i + 1)

    # Continue with the rest of the original trajectory
    for i, action in enumerate(original.route[cut_index:]):
        new_trajectory.update(action, cut_index + len(detour_actions) + i + 1)

    return new_trajectory
