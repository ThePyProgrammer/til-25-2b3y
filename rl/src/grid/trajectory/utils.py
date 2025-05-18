from typing import Optional, Any
from .trajectory import Trajectory
from ..utils.geometry import Point
from ..utils.enums import Direction
from ..node import NodeRegistry
from .constraints import filter_trajectories_by_constraints, TrajectoryConstraints, TemporalTrajectoryConstraints


def fast_forward_trajectories(
    trajectories: list[Trajectory],
    budget: int,
    temporal_constraints: TemporalTrajectoryConstraints,
    registry: NodeRegistry,
    max_backtrack: int = 1,
    num_samples_per_trajectory: int = 2,
) -> list[Trajectory]:
    """
    When all trajectories have been eliminated, this method tries to resurrect
    discarded edge trajectories from previous steps and propagate them forward
    to the current step, applying appropriate temporal constraints.

    Args:
        discard_edge_trajectories: List of previously discarded trajectories.
        budget: The maximum number of steps to consider.
        temporal_constraints: List of constraints for each time step.
        registry: Node registry for creating new nodes.

    Returns:
        list[Trajectory]: List of resurrected trajectories
    """
    # Go backwards through time steps looking for discarded trajectories
    for backward_step in range(budget - 1, -1, -1):
        # Get candidates from this time step
        candidates: list[Trajectory] = get_initial_candidates(
            trajectories,
            backward_step,
            budget,
            registry
        )

        if not candidates:
            continue

        print(f"Restarting from {len(candidates)} candidates at {backward_step}")

        # Forward propagate candidates
        candidates = propagate_candidates_forward(
            candidates,
            backward_step,
            budget,
            temporal_constraints,
            max_backtrack=max_backtrack,
            num_samples_per_trajectory=num_samples_per_trajectory
        )

        if candidates:
            print(f"Found {len(candidates)} valid candidates")
            return candidates

    return []

def get_initial_candidates(
    trajectories: list[Trajectory],
    step: int,
    budget: int,
    registry: Optional[Any] = None
) -> list[Trajectory]:
    """Get initial trajectory candidates for the given step."""
    candidates = [traj for traj in trajectories if traj.created_at == step]

    # If at step 0 and no candidates, create new root trajectories
    if step == 0 and registry:
        for direction in Direction:
            root_node = registry.get_or_create_node(Point(0, 0), direction)
            trajectory = Trajectory(root_node, budget)
            candidates.append(trajectory)

    return candidates

def propagate_candidates_forward(
    candidates: list[Trajectory],
    start_step: int,
    budget: int,
    temporal_constraints: TemporalTrajectoryConstraints,
    max_backtrack: int = 1,
    num_samples_per_trajectory: int = 1,
) -> list[Trajectory]:
    """Propagate candidates forward through time, applying constraints at each step."""
    for forward_step in range(start_step + 1, budget + 1):
        # Expand trajectories
        new_candidates: list[Trajectory] = expand_trajectories(
            candidates,
            forward_step,
            max_backtrack=max_backtrack,
            num_samples_per_trajectory=num_samples_per_trajectory
        )

        # Apply constraints for this step
        step_constraints: Optional[TrajectoryConstraints] = (
            temporal_constraints[forward_step] if forward_step < len(temporal_constraints)
            else None
        )
        # step_constraints = None
        candidates = (
            filter_trajectories_by_constraints(
                new_candidates,
                step_constraints,
                use_route_excludes=False
            ) if step_constraints
            else new_candidates
        )

        print(f"Step {forward_step}: {len(candidates)} candidates remain after filtering")

        if not candidates:
            break

    return candidates

def expand_trajectories(
    trajectories: list[Trajectory],
    step: int,
    max_backtrack: int = 3,
    num_samples_per_trajectory: int = 2,
    consider_direction: bool = True
) -> list[Trajectory]:
    """
    Expand trajectories by generating new ones.

    Args:
        trajectories: List of trajectories to expand
        step: Current step number
        max_backtrack: Maximum number of backtracking steps allowed
        num_samples: Number of evenly spaced samples to select for each endpoint
                     (including shortest and longest)
        consider_direction: Whether to consider direction when grouping endpoints

    Returns:
        List of expanded trajectories
    """
    if num_samples_per_trajectory <= 0:
        return []

    # If num_samples is 1, just keep the shortest trajectory for each endpoint
    # If num_samples is 2, keep the shortest and longest (original behavior)
    # If num_samples > 2, keep evenly spaced samples including shortest and longest

    # Group trajectories by endpoint
    endpoint_to_trajectories = {}
    for traj in trajectories:
        if traj.to_delete:
            continue

        key = traj.get_endpoint_key(consider_direction)
        if not key:
            continue

        if key not in endpoint_to_trajectories:
            endpoint_to_trajectories[key] = []
        endpoint_to_trajectories[key].append(traj)

    # Select trajectories to expand
    selected_trajectories: list[Trajectory] = []
    for key, group in endpoint_to_trajectories.items():
        if not group:
            continue

        # Sort by trajectory length (shorter first)
        group.sort(key=lambda t: len(t.route))

        if num_samples_per_trajectory == 1:
            # Just select the shortest
            selected_trajectories.append(group[0])
        elif num_samples_per_trajectory == 2 or len(group) <= 2:
            # Select shortest and longest (original behavior)
            selected_trajectories.append(group[0])
            if len(group) > 1 and len(group[-1].route) > len(group[0].route):
                selected_trajectories.append(group[-1])
        else:
            # Select n evenly spaced samples
            if len(group) <= num_samples_per_trajectory:
                # If we have fewer trajectories than requested samples, use them all
                selected_trajectories.extend(group)
            else:
                # Get num_samples evenly spaced samples
                indices = [
                    int(i * (len(group) - 1) / (num_samples_per_trajectory - 1))
                    for i in range(num_samples_per_trajectory)
                ]
                for idx in indices:
                    selected_trajectories.append(group[idx])

    # Now expand the selected trajectories
    expanded: list[Trajectory] = []
    for trajectory in selected_trajectories:
        # Add the original trajectory
        expanded.append(trajectory)

        # Generate new trajectories
        new_trajectories = trajectory.get_new_trajectories(
            step,
            max_backtrack=max_backtrack
        )
        expanded.extend(new_trajectories)

    # print(f"Expanded to {len(expanded)} candidates from {len(selected_trajectories)} selected trajectories")
    return expanded
