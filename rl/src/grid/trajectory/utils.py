from typing import Optional, Any
from .trajectory import Trajectory
from ..utils.geometry import Point
from ..utils.enums import Direction
from ..node import NodeRegistry
from .constraints import (
    filter_trajectories_by_constraints,
    TrajectoryConstraints,
    TemporalTrajectoryConstraints,
)

MIN_EXPAND_TRAJECTORIES = 2


def fast_forward_trajectories(
    trajectories: list[Trajectory],
    budget: int,
    temporal_constraints: TemporalTrajectoryConstraints,
    registry: NodeRegistry,
    max_backtrack: int = 1,
    trajectory_budget: int = 2000,
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
            trajectories, backward_step, budget, registry
        )

        if not candidates:
            continue

        # print(f"Restarting from {len(candidates)} candidates at {backward_step}")

        # Forward propagate candidates
        candidates = propagate_candidates_forward(
            candidates,
            backward_step,
            budget,
            temporal_constraints,
            max_backtrack=max_backtrack,
            trajectory_budget=trajectory_budget,
        )

        if candidates:
            # print(f"Found {len(candidates)} valid candidates")
            return candidates

    return []


def get_initial_candidates(
    trajectories: list[Trajectory],
    step: int,
    budget: int,
    registry: Optional[NodeRegistry] = None,
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
    trajectory_budget: int = 2000,
) -> list[Trajectory]:
    """Propagate candidates forward through time, applying constraints at each step."""
    for forward_step in range(start_step + 1, budget + 1):
        # Expand trajectories
        new_candidates: list[Trajectory] = expand_trajectories(
            candidates,
            forward_step,
            max_backtrack=max_backtrack,
            trajectory_budget=trajectory_budget,
        )

        # Apply constraints for this step
        step_constraints: Optional[TrajectoryConstraints] = (
            temporal_constraints[forward_step]
            if forward_step < len(temporal_constraints)
            else None
        )
        # step_constraints = None
        candidates = (
            filter_trajectories_by_constraints(
                new_candidates, step_constraints, use_route_contains=False
            )
            if step_constraints
            else new_candidates
        )

        # print(f"Step {forward_step}: {len(candidates)} candidates remain after filtering")

        if not candidates:
            break

    return candidates


def expand_trajectories(
    trajectories: list[Trajectory],
    step: int,
    max_backtrack: int = 3,
    trajectory_budget: Optional[int] = 2000,
    consider_direction: bool = True,
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
    expanded: list[Trajectory] = []

    # Filter out deleted trajectories
    valid_trajectories = {traj for traj in trajectories if not traj.to_delete}

    selected_trajectories: set[Trajectory] = set()

    if trajectory_budget is None:
        # If no trajectory budget, select all valid trajectories
        selected_trajectories = valid_trajectories
    else:
        # If num_samples is 1, just keep the shortest trajectory for each endpoint
        # If num_samples is 2, keep the shortest and longest (original behavior)
        # If num_samples > 2, keep evenly spaced samples including shortest and longest

        # Group valid trajectories by endpoint
        endpoint_to_trajectories = {}
        for traj in valid_trajectories:
            key = traj.get_endpoint_key(consider_direction)
            if not key:
                continue

            if key not in endpoint_to_trajectories:
                endpoint_to_trajectories[key] = []
            endpoint_to_trajectories[key].append(traj)

        # Select trajectories from grouped valid trajectories
        num_trajectories = sum(map(lambda g: len(g), endpoint_to_trajectories.values()))
        # allocate the minimum to all groups first
        extra_trajectory_budget = trajectory_budget - MIN_EXPAND_TRAJECTORIES * len(
            endpoint_to_trajectories
        )
        print(num_trajectories, extra_trajectory_budget)

        for key, group in endpoint_to_trajectories.items():
            if not group:
                continue

            num_samples = (
                round(len(group) / num_trajectories * extra_trajectory_budget)
                + MIN_EXPAND_TRAJECTORIES
            )

            if len(group) <= num_samples:
                # If we have fewer trajectories than requested samples, use them all
                selected_trajectories.update(group)
                continue

            # Sort by trajectory length (shorter first)
            group.sort(key=lambda t: len(t.route))

            selected_group_trajectories: list[Trajectory] = (
                []
            )  # Use a temporary list for selection in this group

            if num_samples == 1:
                # Just select the shortest
                selected_group_trajectories.append(group[0])
            elif num_samples == 2 or len(group) <= 2:
                # Select shortest and longest (original behavior)
                selected_group_trajectories.append(group[0])
                if len(group) > 1 and len(group[-1].route) > len(group[0].route):
                    selected_group_trajectories.append(group[-1])
            else:
                # Get num_samples evenly spaced samples
                indices = [
                    int(i * (len(group) - 1) / (num_samples - 1))
                    for i in range(num_samples)
                ]
                for idx in indices:
                    selected_group_trajectories.append(group[idx])

            # Add selected trajectories from this group to the main list
            selected_trajectories.update(selected_group_trajectories)

    print(len(selected_trajectories))
    # Now expand the selected trajectories
    for trajectory in selected_trajectories:
        # Add the original trajectory
        expanded.append(trajectory)

        # Generate new trajectories
        new_trajectories = trajectory.get_new_trajectories(
            step, max_backtrack=max_backtrack
        )
        expanded.extend(new_trajectories)

    # print(f"Expanded to {len(expanded)} candidates from {len(selected_trajectories)} selected trajectories")
    return expanded
