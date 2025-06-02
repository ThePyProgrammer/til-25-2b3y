from dataclasses import dataclass

from ..node import DirectionalNode
from ..utils import Action


@dataclass
class Particles:
    count: int
    individual_probability: float

    node: DirectionalNode

    incoming_action_counts: dict[Action, int]

    @property
    def probability(self):
        return self.count * self.individual_probability
