from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# from scipy.special import softmax

from ..node import DirectionalNode
from ..utils import Action, Point, Direction

from .counter import VisitCounter

EPS = 1e-8
POSINF_WEIGHT = 10.0  # Maximum weight for actions with no previous visits
MAX_WEIGHT = 9.9  # Maximum weight for actions that are not posinf
OOB_WEIGHT = 0.1  # Weight for actions that will lead to out-of-bounds


class ActionPreference:
    CONTINUATION_BONUS = 2.0  # Weight for continuing same action (forward/backward)
    MOVEMENT_BONUS = 1.5  # Weight for movement actions after left/right or stay
    STATIONARY_PENALTY = 0.1
    BASE = 1.0  # Base weight for other actions


def get_action_preference_bonus(
    prev_action: Action,
    current_action: Action,
) -> float:
    """Calculate preference bonus for action transition."""
    pref = ActionPreference.BASE

    if (
        prev_action in [Action.FORWARD, Action.BACKWARD]
        and current_action == prev_action
    ):
        # Previous was forward/backward: prefer continuing same action
        pref *= ActionPreference.CONTINUATION_BONUS

    elif prev_action in [Action.LEFT, Action.RIGHT] and current_action in [
        Action.FORWARD,
        Action.BACKWARD,
    ]:
        # Previous was left/right: prefer forward/backward
        pref *= ActionPreference.MOVEMENT_BONUS

    elif (prev_action, current_action) == (Action.LEFT, Action.RIGHT) or (
        (prev_action, current_action) == (Action.RIGHT, Action.LEFT)
    ):
        # Penalise switching between left and right, this is effectively staying stationary
        pref *= ActionPreference.STATIONARY_PENALTY

    elif prev_action == Action.STAY and current_action != Action.STAY:
        # Previous was stay: prefer movement actions
        pref *= ActionPreference.MOVEMENT_BONUS

    elif current_action == Action.STAY:
        pref *= ActionPreference.STATIONARY_PENALTY

    return pref


ACTION_PREFERENCE_MAT = np.zeros((len(Action), len(Action)), dtype=np.float32)

for prev in Action:
    for curr in Action:
        ACTION_PREFERENCE_MAT[prev, curr] = get_action_preference_bonus(prev, curr)

# slightly different because we care more about the likely next move from a turn
MOVEMENT_VECTORS = {
    Action.FORWARD: {
        Direction.RIGHT: (1, 0),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
        Direction.UP: (0, -1),
    },
    Action.BACKWARD: {
        Direction.RIGHT: (-1, 0),
        Direction.DOWN: (0, -1),
        Direction.LEFT: (1, 0),
        Direction.UP: (0, 1),
    },
    Action.LEFT: {
        Direction.RIGHT: (0, -1),
        Direction.DOWN: (1, 0),
        Direction.LEFT: (0, 1),
        Direction.UP: (-1, 0),
    },
    Action.RIGHT: {
        Direction.RIGHT: (0, 1),
        Direction.DOWN: (-1, 0),
        Direction.LEFT: (0, -1),
        Direction.UP: (1, 0),
    },
}


@dataclass
class NodeParticles:
    count: int
    individual_probability: float

    node: DirectionalNode

    _outgoing_actions: list[Action] = field(default_factory=list)
    _weight_indices: tuple[NDArray[np.int32]] = tuple()
    _oob: NDArray[np.bool_] = field(default_factory=lambda: np.array([]))

    incoming_action_counts: Counter[Action] = field(default_factory=lambda: Counter())

    previous_visit_counts: VisitCounter = field(
        default_factory=lambda: VisitCounter(16)
    )

    def __post_init__(self):
        self._outgoing_actions = list(self.node.children.keys()) + [Action.STAY]
        curr_x, curr_y = self.node.position.x, self.node.position.y
        self._weight_indices = tuple(
            np.array(
                [
                    (
                        MOVEMENT_VECTORS[action][self.node.direction][0] + curr_x,
                        MOVEMENT_VECTORS[action][self.node.direction][1] + curr_y,
                    )
                    for action in self.node.children.keys()
                ]
                + [(curr_x, curr_y)]  # STAY
            ).T
        )
        self._oob = (
            (self._weight_indices[0] < 0)
            | (self._weight_indices[0] > 15)
            | (self._weight_indices[1] < 0)  # type: ignore
            | (self._weight_indices[1] > 15)  # type: ignore
        )  # Out of bounds check
        # make indices safe
        self._weight_indices[0][self._oob] = curr_x
        self._weight_indices[1][self._oob] = curr_y  # type: ignore

    def copy(self) -> "NodeParticles":
        return NodeParticles(
            count=self.count,
            individual_probability=self.individual_probability,
            node=self.node,
            incoming_action_counts=self.incoming_action_counts.copy(),
            previous_visit_counts=self.previous_visit_counts.copy(),
        )

    @property
    def probability(self):
        return self.count * self.individual_probability

    @property
    def outgoing_actions(self) -> list[Action]:
        # for backward compatibility
        return self._outgoing_actions

    # @property
    # def outgoing_weights_explore(self) -> NDArray[np.float32]:
    #     curr_x, curr_y = self.node.position.x, self.node.position.y
    #     wts = 1 / self.previous_visit_probas
    #     wts[curr_x, curr_y] = 1
    #     # make indices safe
    #     action_weights = wts[self._weight_indices]
    #     is_posinf = np.isinf(action_weights)
    #     action_weights = np.clip(
    #         action_weights, 0, MAX_WEIGHT
    #     )  # make posinf the maximum weight, sigmoid scaling
    #     action_weights[self._oob] = OOB_WEIGHT  # Set out-of-bounds weights
    #     action_weights[is_posinf] = POSINF_WEIGHT  # Set posinf weights\
    #     return action_weights

    @property
    def outgoing_action_probas(self):
        # Create incoming action proportions vector
        incoming_proportions = np.zeros(len(Action), dtype=np.float32)
        for action, count in self.incoming_action_counts.items():
            incoming_proportions[action] = count

        # Calculate action weights using matrix multiplication
        # action_weights[i, j] gives preference for transitioning from action i to action j
        action_weights = incoming_proportions @ ACTION_PREFERENCE_MAT
        action_weights = action_weights[self._outgoing_actions]
        # print(self.outgoing_weights_explore)
        # action_weights *= self.outgoing_weights_explore
        total_weights = action_weights.sum()
        if total_weights == 0:
            action_probas = [1 / len(action_weights) for _ in action_weights]
        else:
            action_probas = action_weights / action_weights.sum()

        action_probas = {
            action: proba
            for proba, action in zip(action_probas, self._outgoing_actions)
        }

        return action_probas

    def merge_into(self, other_particles: "NodeParticles", action: Action, count: int):
        """Merge this particle with another particle, updating counts and visit history."""
        # Merge other particle's visit history
        other_previous_visit_counts = other_particles.previous_visit_counts.copy()
        other_previous_visit_counts.scale(count / other_particles.count)

        self.previous_visit_counts.update(other_previous_visit_counts)
        self.incoming_action_counts[action] += count

        self.count = self.count + count

    def create_child_particle(
        self, child_node: DirectionalNode, count: int
    ) -> "NodeParticles":
        """Create a new child particle from this parent particle."""

        previous_visit_counts = self.previous_visit_counts.copy()
        previous_visit_counts.scale(count / self.count)
        previous_visit_counts[self.node.position] += self.count

        child_particle = NodeParticles(
            count=count,
            individual_probability=self.individual_probability,
            node=child_node,
            previous_visit_counts=previous_visit_counts,
        )

        return child_particle

    def add_incoming_action(self, action: Action, count: int):
        """Add incoming action count."""
        self.incoming_action_counts[action] = count

    def scale_count(self, new_count: int, total_count: int):
        """Scale the count and update individual probability based on total."""
        old_count = self.count
        self.count = new_count
        self.individual_probability = 1 / total_count

        self.previous_visit_counts.scale(new_count / old_count)

    @property
    def previous_visit_probas(self) -> NDArray[np.float32]:
        expected_count = self.previous_visit_counts.counts / self.count
        visit_probas = expected_count / (1 + expected_count)
        return visit_probas.astype(np.float32)
