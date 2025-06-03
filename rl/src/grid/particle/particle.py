from dataclasses import dataclass, field

from ..node import DirectionalNode
from ..utils import Action, Point


@dataclass
class NodeParticles:
    count: int
    individual_probability: float

    node: DirectionalNode

    incoming_action_counts: dict[Action, int] = field(
        default_factory=lambda: {
            Action.FORWARD: 0,
            Action.BACKWARD: 0,
            Action.LEFT: 0,
            Action.RIGHT: 0,
            Action.STAY: 0,
        }
    )

    previous_positions: set[Point] = field(default_factory=set)

    @property
    def probability(self):
        return self.count * self.individual_probability
