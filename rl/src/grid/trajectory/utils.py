from dataclasses import dataclass
from typing import Any

from ..utils import Point
from .trajectory import Trajectory

@dataclass
class Constraints:
    """
    Point constraints
    """
    contains: list[Point]
    excludes: list[Point]

    def __bool__(self):
        return len(self.contains) > 0 or len(self.excludes) > 0

    def copy(self):
        return Constraints(
            self.contains[:],
            self.excludes[:]
        )

@dataclass
class TrajectoryConstraints:
    route: Constraints
    tail: Constraints

    def __bool__(self):
        return bool(self.route) or bool(self.tail)

    def copy(self):
        return TrajectoryConstraints(
            self.route.copy(),
            self.tail.copy()
        )

class TemporalTrajectoryConstraints:
    def __init__(self):
        self._observed_constraints: list[TrajectoryConstraints] = []
        self._temporal_constraints: list[TrajectoryConstraints] = []

    def update(self, constraints: TrajectoryConstraints):
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

    def __len__(self):
        return len(self._temporal_constraints)

    def __getitem__(self, step: int) -> TrajectoryConstraints:
        return self._temporal_constraints[step]

class TrajectoryIndex:
    def __init__(self):
        self._index: dict[Any, set[Trajectory]] = {}

    def add(self, key: Any, trajectory: Trajectory) -> None:
        if key not in self._index:
            self._index[key] = set()
        self._index[key].add(trajectory)

    def remove(self, key: Any, trajectory: Trajectory) -> None:
        if key in self._index and trajectory in self._index[key]:
            trajectory.prune()
            self._index[key].remove(trajectory)
            # Clean up empty sets
            if not self._index[key]:
                del self._index[key]

    def clear(self):
        self._index.clear()

    def __contains__(self, key: Any):
        return key in self._index

    def __getitem__(self, key: Any):
        if key not in self._index:
            self._index[key] = set()
        return self._index[key]
