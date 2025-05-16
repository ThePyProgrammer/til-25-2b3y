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
    registry: NodeRegistry
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
            temporal_constraints
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
    temporal_constraints: TemporalTrajectoryConstraints
) -> list[Trajectory]:
    """Propagate candidates forward through time, applying constraints at each step."""
    for forward_step in range(start_step + 1, budget + 1):
        # Expand trajectories
        new_candidates: list[Trajectory] = expand_trajectories(candidates, forward_step)

        # Apply constraints for this step
        step_constraints: Optional[TrajectoryConstraints] = (
            temporal_constraints[forward_step] if forward_step < len(temporal_constraints)
            else None
        )
        candidates = (
            filter_trajectories_by_constraints(new_candidates, step_constraints) if step_constraints
            else new_candidates
        )

        print(f"Step {forward_step}: {len(candidates)} candidates remain after filtering")

        if not candidates:
            break

    return candidates

def expand_trajectories(
    trajectories: list[Trajectory],
    step: int,
    max_backtrack: int = 3
) -> list[Trajectory]:
    """Expand trajectories by generating new ones."""
    expanded: list[Trajectory] = trajectories.copy()

    for trajectory in trajectories:
        new_trajectories: list[Trajectory] = trajectory.get_new_trajectories(
            step,
            max_backtrack=max_backtrack
        )
        expanded.extend(new_trajectories)

    print(f"Expanded to {len(expanded)} candidates")
    return expanded
