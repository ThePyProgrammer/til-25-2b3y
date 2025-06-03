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
            self.contains.copy(),
            self.excludes.copy()
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
        self.hard_constraints: list[ParticleConstraints] = []
        self.soft_constraints: list[ParticleConstraints] = []

    def update(self, constraints: ParticleConstraints) -> None:
        self._observed_constraints.append(constraints)
        self.hard_constraints.append(constraints.copy())
        self.soft_constraints.append(constraints.copy())

        for historical_hard_constraints in self.hard_constraints:
            historical_hard_constraints.route.excludes.update(constraints.route.excludes)
            # historical_constraints.tail.excludes.update(constraints.tail.excludes)

        for historical_soft_constraints in self.soft_constraints:
            historical_soft_constraints.route.excludes.update(constraints.route.excludes)
            historical_soft_constraints.route.contains.update(constraints.route.contains)
            # historical_constraints.tail.excludes.update(constraints.tail.excludes)

        current_step = len(self._observed_constraints) - 1
        previous_step = current_step - 1

        if previous_step >= 0:
            self.hard_constraints[current_step].route.contains.update(
                self.hard_constraints[previous_step].route.contains
            )

        if previous_step >= 0:
            self.soft_constraints[current_step].route.contains.update(
                self.soft_constraints[previous_step].route.contains
            )

    def __len__(self) -> int:
        return len(self.hard_constraints)

    def __getitem__(self, step: int) -> ParticleConstraints:
        return self.hard_constraints[step]
