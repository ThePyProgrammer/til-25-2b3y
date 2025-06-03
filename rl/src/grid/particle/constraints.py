from dataclasses import dataclass, field

from ..utils import Point

@dataclass
class Constraints:
    """
    Point constraints
    """
    contains: set[Point] = field(default_factory=set)
    excludes: set[Point] = field(default_factory=set)

    def __bool__(self) -> bool:
        return len(self.contains) > 0 or len(self.excludes) > 0

    def copy(self) -> 'Constraints':
        return Constraints(
            set(self.contains),
            set(self.excludes)
        )

@dataclass
class ParticleConstraints:
    route: Constraints = field(default_factory=Constraints)
    tail: Constraints = field(default_factory=Constraints)

    def __bool__(self) -> bool:
        return bool(self.route) or bool(self.tail)

    def copy(self) -> 'ParticleConstraints':
        return ParticleConstraints(
            self.route.copy(),
            self.tail.copy()
        )

class TemporalConstraints:
    def __init__(self) -> None:
        self._observed_constraints: list[ParticleConstraints] = []
        self._temporal_constraints: list[ParticleConstraints] = []

    def update(self, constraints: ParticleConstraints) -> None:
        self._observed_constraints.append(constraints)
        self._temporal_constraints.append(constraints)

        for historical_constraints in self._temporal_constraints:
            historical_constraints.route.excludes.update(constraints.route.excludes)
            # historical_constraints.tail.excludes.update(constraints.tail.excludes)

        current_step = len(self._observed_constraints) - 1
        previous_step = current_step - 1

        if previous_step >= 0:
            self._temporal_constraints[current_step].route.contains.update(
                self._temporal_constraints[previous_step].route.contains
            )

    def __len__(self) -> int:
        return len(self._temporal_constraints)

    def __getitem__(self, step: int) -> ParticleConstraints:
        return self._temporal_constraints[step]
