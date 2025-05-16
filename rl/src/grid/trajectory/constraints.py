from dataclasses import dataclass
from typing import Optional

from ..utils import Point
from .trajectory import Trajectory  # Importing Trajectory for type annotations


@dataclass
class Constraints:
    """
    Point constraints
    """
    contains: list[Point]
    excludes: list[Point]

    def __bool__(self) -> bool:
        return len(self.contains) > 0 or len(self.excludes) > 0

    def copy(self) -> 'Constraints':
        return Constraints(
            self.contains[:],
            self.excludes[:]
        )

@dataclass
class TrajectoryConstraints:
    route: Constraints
    tail: Constraints

    def __bool__(self) -> bool:
        return bool(self.route) or bool(self.tail)

    def copy(self) -> 'TrajectoryConstraints':
        return TrajectoryConstraints(
            self.route.copy(),
            self.tail.copy()
        )

class TemporalTrajectoryConstraints:
    def __init__(self) -> None:
        self._observed_constraints: list[TrajectoryConstraints] = []
        self._temporal_constraints: list[TrajectoryConstraints] = []

    def update(self, constraints: TrajectoryConstraints) -> None:
        self._observed_constraints.append(constraints)
        self._temporal_constraints.append(constraints)

        for historical_constraints in self._temporal_constraints:
            historical_constraints.route.excludes.extend(constraints.route.excludes)
            historical_constraints.route.contains = list(set(
                historical_constraints.route.contains
            ))
            # historical_constraints.tail.excludes.extend(constraints.tail.excludes)

        current_step = len(self._observed_constraints) - 1
        previous_step = current_step - 1

        if previous_step >= 0:
            self._temporal_constraints[current_step].route.contains.extend(
                self._temporal_constraints[previous_step].route.contains
            )
            self._temporal_constraints[current_step].route.contains = list(set(
                self._temporal_constraints[current_step].route.contains
            ))

    def __len__(self) -> int:
        return len(self._temporal_constraints)

    def __getitem__(self, step: int) -> TrajectoryConstraints:
        return self._temporal_constraints[step]


def apply_constraints(trajectory: Trajectory, constraints: TrajectoryConstraints) -> bool:
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
    for node in trajectory.nodes:
        if node in constraints.route.excludes:
            trajectory.prune()
            return False

    # Check route contain constraints
    for node in constraints.route.contains:
        if node not in trajectory.nodes:
            trajectory.prune()
            return False

    # Check tail exclude constraints
    if trajectory.tail.position in constraints.tail.excludes:
        trajectory.prune()
        return False

    # Check tail contain constraints
    if constraints.tail.contains and trajectory.tail.position not in constraints.tail.contains:
        trajectory.prune()
        return False

    return True


def filter_trajectories_by_constraints(
    trajectories: list[Trajectory],
    constraints: Optional[TrajectoryConstraints]
) -> list[Trajectory]:
    """
    Filter a list of trajectories based on the given constraints.

    Args:
        trajectories: List of trajectories to filter
        constraints: Constraints to apply

    Returns:
        list: Filtered list of trajectories that satisfy all constraints
    """
    if constraints:
        valid_trajectories: List[Trajectory] = []

        for traj in trajectories:
            if apply_constraints(traj, constraints):
                valid_trajectories.append(traj)

        return valid_trajectories
    else:
        return trajectories
