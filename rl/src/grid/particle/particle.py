from collections import Counter
from dataclasses import dataclass, field

import numpy as np
# from scipy.special import softmax

from ..node import DirectionalNode
from ..utils import Action

from .counter import VisitCounter

EPS = 1e-8

class ActionPreference:
    CONTINUATION_WEIGHT = 2.0    # Weight for continuing same action (forward/backward)
    MOVEMENT_WEIGHT = 1.5        # Weight for movement actions after left/right or stay
    BASE_WEIGHT = 1.0            # Base weight for other actions

def get_action_preference_bonus(
    prev_action: Action,
    current_action: Action,
) -> float:
    """Calculate preference bonus for action transition."""
    if prev_action in [Action.FORWARD, Action.BACKWARD] and current_action == prev_action:
        # Previous was forward/backward: prefer continuing same action
        return ActionPreference.CONTINUATION_WEIGHT

    elif prev_action in [Action.LEFT, Action.RIGHT] and current_action in [Action.FORWARD, Action.BACKWARD]:
        # Previous was left/right: prefer forward/backward
        return ActionPreference.MOVEMENT_WEIGHT

    elif prev_action == Action.STAY and current_action != Action.STAY:
        # Previous was stay: prefer movement actions
        return ActionPreference.MOVEMENT_WEIGHT

    return ActionPreference.BASE_WEIGHT

ACTION_PREFERENCE_MAT = np.zeros((len(Action), len(Action)), dtype=np.float32)

for prev in Action:
    for curr in Action:
        ACTION_PREFERENCE_MAT[prev, curr] = get_action_preference_bonus(prev, curr)

@dataclass
class NodeParticles:
    count: int
    individual_probability: float

    node: DirectionalNode

    incoming_action_counts: Counter[Action] = field(
        default_factory=lambda: Counter()
    )

    previous_visit_counts: VisitCounter = field(
        default_factory=lambda: VisitCounter(16)
    )

    def __post_init__(self):
        self.previous_visit_counts[self.node.position] += 1

    def copy(self) -> 'NodeParticles':
        return NodeParticles(
            count = self.count,
            individual_probability = self.individual_probability,
            node = self.node,
            incoming_action_counts = self.incoming_action_counts.copy(),
            previous_visit_counts = self.previous_visit_counts.copy()
        )

    @property
    def probability(self):
        return self.count * self.individual_probability

    @property
    def outgoing_actions(self) -> list[Action]:
        return list(self.node.children.keys()) + [Action.STAY]

    @property
    def outgoing_action_probas(self):
        # Create incoming action proportions vector
        incoming_proportions = np.zeros(len(Action), dtype=np.float32)
        for action, count in self.incoming_action_counts.items():
            incoming_proportions[action] = count

        # Calculate action weights using matrix multiplication
        # action_weights[i, j] gives preference for transitioning from action i to action j
        action_weights = incoming_proportions @ ACTION_PREFERENCE_MAT
        action_weights = action_weights[self.outgoing_actions]
        total_weights = action_weights.sum()
        if total_weights == 0:
            action_probas = [1 / len(action_weights) for _ in action_weights]
        else:
            action_probas = action_weights / action_weights.sum()

        action_probas = {action: proba for proba, action in zip(action_probas, self.outgoing_actions)}

        return action_probas

    def merge_into(self, other_particles: 'NodeParticles', action: Action, count: int):
        """Merge this particle with another particle, updating counts and visit history."""
        # Merge other particle's visit history
        self.previous_visit_counts.update(other_particles.previous_visit_counts)
        self.incoming_action_counts[action] = count

        self.count = self.count + count

    def create_child_particle(self, child_node: DirectionalNode, count: int) -> 'NodeParticles':
        """Create a new child particle from this parent particle."""

        child_particle = NodeParticles(
            count=count,
            individual_probability=self.individual_probability,
            node=child_node,
            previous_visit_counts=self.previous_visit_counts.copy()
        )

        return child_particle

    def add_incoming_action(self, action: Action, count: int):
        """Add incoming action count."""
        self.incoming_action_counts[action] = count

    def scale_count(self, new_count: int, total_count: int):
        """Scale the count and update individual probability based on total."""
        self.count = new_count
        self.individual_probability = 1 / total_count
