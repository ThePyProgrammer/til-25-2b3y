from dataclasses import dataclass
from typing import Optional

from ..utils import Point
from .trajectory import Trajectory  # Importing Trajectory for type annotations
from .index import TrajectoryIndex


@dataclass
class Constraints:
    """
    Point constraints
    """

    contains: list[Point]
    excludes: list[Point]

    def __bool__(self) -> bool:
        return len(self.contains) > 0 or len(self.excludes) > 0

    def copy(self) -> "Constraints":
        return Constraints(self.contains[:], self.excludes[:])


@dataclass
class TrajectoryConstraints:
    route: Constraints
    tail: Constraints

    def __bool__(self) -> bool:
        return bool(self.route) or bool(self.tail)

    def copy(self) -> "TrajectoryConstraints":
        return TrajectoryConstraints(self.route.copy(), self.tail.copy())


class TemporalTrajectoryConstraints:
    def __init__(self) -> None:
        self._observed_constraints: list[TrajectoryConstraints] = []
        self._temporal_constraints: list[TrajectoryConstraints] = []

    def update(self, constraints: TrajectoryConstraints) -> None:
        self._observed_constraints.append(constraints)
        self._temporal_constraints.append(constraints)

        for historical_constraints in self._temporal_constraints:
            historical_constraints.route.excludes.extend(constraints.route.excludes)
            historical_constraints.route.contains = list(
                set(historical_constraints.route.contains)
            )
            # historical_constraints.tail.excludes.extend(constraints.tail.excludes)

        current_step = len(self._observed_constraints) - 1
        previous_step = current_step - 1

        if previous_step >= 0:
            self._temporal_constraints[current_step].route.contains.extend(
                self._temporal_constraints[previous_step].route.contains
            )
            self._temporal_constraints[current_step].route.contains = list(
                set(self._temporal_constraints[current_step].route.contains)
            )

    def __len__(self) -> int:
        return len(self._temporal_constraints)

    def __getitem__(self, step: int) -> TrajectoryConstraints:
        return self._temporal_constraints[step]


def apply_constraints(
    trajectory: Trajectory,
    constraints: TrajectoryConstraints,
    use_route_excludes: bool = True,
    use_route_contains: bool = True,
    use_tail_excludes: bool = True,
    use_tail_contains: bool = True,
) -> bool:
    """
    Apply constraints to a trajectory and mark it for deletion if it violates any constraint.

    Args:
        trajectory: The trajectory to check
        constraints: The constraints to apply

    Returns:
        bool: True if the trajectory satisfies all constraints, False otherwise
    """
    if not constraints:
        return True

    # Check route exclude constraints
    if use_route_excludes:
        for point in constraints.route.excludes:
            if point in trajectory.position_cache:
                trajectory.prune()
                return False

    # Check route contain constraints
    if use_route_contains:
        for point in constraints.route.contains:
            if point not in trajectory.position_cache:
                trajectory.prune()
                return False

    # Check tail exclude constraints
    if use_tail_excludes:
        if trajectory.tail.position in constraints.tail.excludes:
            trajectory.prune()
            return False

    # Check tail contain constraints
    if use_tail_contains:
        if (
            constraints.tail.contains
            and trajectory.tail.position not in constraints.tail.contains
        ):
            trajectory.prune()
            return False

    return True


def build_position_indices_from_trajectories(trajectories: list[Trajectory]):
    """
    Build indices for trajectory positions and tail positions for efficient constraint filtering.

    Args:
        trajectories: List of trajectories to index

    Returns:
        tuple: (position_index, tail_position_index)
            - position_index: Maps positions to trajectories that contain that position
            - tail_position_index: Maps positions to trajectories with tails at that position
    """
    position_index = TrajectoryIndex()
    tail_position_index = TrajectoryIndex()

    for traj in trajectories:
        for node in traj.nodes:
            position_index.add(node.position, traj)
        tail_position_index.add(traj.tail.position, traj)

    return position_index, tail_position_index


def _apply_route_exclude_constraints(
    candidates: set[Trajectory],
    position_index: TrajectoryIndex,
    exclude_points: list[Point],
) -> None:
    """Apply route exclude constraints to the candidates."""
    for excluded_point in exclude_points:
        if excluded_point in position_index:
            # Remove all trajectories that contain this excluded point
            for traj in position_index[excluded_point]:
                candidates.discard(traj)
                traj.prune()  # Mark for deletion


def _apply_route_contain_constraints(
    candidates: set[Trajectory],
    position_index: TrajectoryIndex,
    contain_points: list[Point],
) -> bool:
    """
    Apply route contain constraints to the candidates.
    Returns False if all trajectories fail, True otherwise.
    """
    for required_point in contain_points:
        if required_point not in position_index:
            # If a required point isn't in any trajectory, all trajectories fail
            for traj in candidates.copy():
                traj.prune()
            return False

        # Keep only trajectories that contain this required point
        have_point = set(position_index[required_point])
        for traj in candidates.copy():
            if traj not in have_point:
                candidates.discard(traj)
                traj.prune()

    return True


def _apply_tail_exclude_constraints(
    candidates: set[Trajectory],
    tail_position_index: TrajectoryIndex,
    exclude_points: list[Point],
) -> None:
    """Apply tail exclude constraints to the candidates."""
    for excluded_tail_point in exclude_points:
        if excluded_tail_point in tail_position_index:
            # Remove all trajectories with tail at this excluded point
            for traj in tail_position_index[excluded_tail_point]:
                candidates.discard(traj)
                traj.prune()


def _apply_tail_contain_constraints(
    candidates: set[Trajectory],
    tail_position_index: TrajectoryIndex,
    contain_points: list[Point],
) -> bool:
    """
    Apply tail contain constraints to the candidates.
    Returns False if all trajectories fail, True otherwise.
    """
    if not contain_points:
        return True

    valid_tails = set()
    for required_tail_point in contain_points:
        if required_tail_point in tail_position_index:
            valid_tails.update(tail_position_index[required_tail_point])

    if not valid_tails:
        # If there are contain constraints but no valid tails, all trajectories fail
        for traj in candidates.copy():
            traj.prune()
        return False

    # Keep only trajectories with valid tail positions
    for traj in candidates.copy():
        if traj not in valid_tails:
            candidates.discard(traj)
            traj.prune()

    return True


def filter_trajectories_by_constraints(
    trajectories: list[Trajectory],
    constraints: Optional[TrajectoryConstraints],
    use_route_excludes: bool = True,
    use_route_contains: bool = True,
    use_tail_excludes: bool = True,
    use_tail_contains: bool = True,
) -> list[Trajectory]:
    """
    Filter a list of trajectories based on the given constraints.

    Args:
        trajectories: List of trajectories to filter
        constraints: Constraints to apply
        use_route_excludes: Whether to apply route exclude constraints
        use_route_contains: Whether to apply route contain constraints
        use_tail_excludes: Whether to apply tail exclude constraints
        use_tail_contains: Whether to apply tail contain constraints

    Returns:
        list: Filtered list of trajectories that satisfy all constraints
    """
    if not constraints:
        return trajectories

    # Build indices for efficient lookup
    position_index, tail_position_index = build_position_indices_from_trajectories(
        trajectories
    )

    # Start with all trajectories as potential candidates
    candidates = set(trajectories)

    # Apply constraints in order
    if use_route_excludes and constraints.route.excludes:
        _apply_route_exclude_constraints(
            candidates, position_index, constraints.route.excludes
        )

    if use_route_contains and constraints.route.contains:
        if not _apply_route_contain_constraints(
            candidates, position_index, constraints.route.contains
        ):
            return []

    if use_tail_excludes and constraints.tail.excludes:
        _apply_tail_exclude_constraints(
            candidates, tail_position_index, constraints.tail.excludes
        )

    if use_tail_contains and constraints.tail.contains:
        if not _apply_tail_contain_constraints(
            candidates, tail_position_index, constraints.tail.contains
        ):
            return []

    return list(candidates)
