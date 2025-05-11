from collections import deque
from dataclasses import dataclass
import random
from typing import Optional, Any

import torch

Observation = tuple[torch.Tensor, torch.Tensor]

@dataclass
class Step:
    observation: Observation
    action: int
    reward: float
    next_observation: Optional[Observation]
    done: bool

class State:
    """
    Class representing a state as a sequence of historical observations.

    This is used for environments where temporal information (history)
    is needed for the agent to make decisions.
    """
    def __init__(self, max_history: int = 10):
        """
        Initialize a state with an empty history.

        Args:
            max_history: Maximum number of historical observations to store
        """
        self.max_history = max_history
        self.observation_history = deque(maxlen=max_history)

    def __getitem__(self, idx):
        return self.observation_history[idx]

    def update(self, observation: Observation) -> None:
        """
        Add a new observation to the state history.

        Args:
            observation: New observation dictionary from the environment
        """
        self.observation_history.append(observation)

    def get_encoded_sequence(self) -> list[Observation]:
        """
        Get the full sequence of encoded observations.

        Returns:
            List of (spatial_tensor, static_tensor) tuples as expected by the encoder
        """
        return list(self.observation_history)

    def get_last_encoded(self) -> Optional[Observation]:
        """
        Get the most recent encoded observation.

        Returns:
            Tuple of (spatial_tensor, static_tensor) or None if history is empty
        """
        if not self.observation_history:
            return None
        return self.observation_history[-1]

    def most_recent(self, n: int = 1) -> list[dict[str, Any]]:
        """
        Get the n most recent raw observations.

        Args:
            n: Number of recent observations to return

        Returns:
            List of the n most recent observations
        """
        return list(self.observation_history)[-n:]

    def is_empty(self) -> bool:
        """Check if the state has no observations."""
        return len(self.observation_history) == 0

    def clone(self) -> 'State':
        """Create a copy of this state."""
        new_state = State(max_history=self.max_history)
        new_state.observation_history = deque(self.observation_history, maxlen=self.max_history)
        return new_state

    def __len__(self) -> int:
        """Return the number of observations in this state."""
        return len(self.observation_history)

    @classmethod
    def from_observations(cls, observations: list[Observation], max_history: int = 10):
        state = cls(max_history=max_history)
        state.observation_history = observations
        return state

@dataclass
class Transition:
    state: State
    action: int
    reward: float
    next_state: Optional[State]
    done: bool

class Episode:
    def __init__(self):
        self.steps: list[Step] = []

    def update(self, step: Step):
        self.steps.append(step)

    def sample_one(self, idx: Optional[int] = None, max_history: int = 10) -> Transition:
        """
        Sample a single transition from the episode at a specific index.

        Args:
            idx: Index to sample from. If None, a random valid index will be chosen.
            max_history: Maximum history length for the state

        Returns:
            A Transition object containing state, action, reward, next_state, and done
        """
        if idx is None:
            idx = random.randint(1, len(self.steps))
        else:
            idx = max(1, min(idx, len(self.steps)-1))

        steps = self.steps[:idx]
        current_step = steps[-1]
        observations = [step.observation for step in steps]
        state = State.from_observations(observations, max_history=max_history)

        if idx < (len(self.steps) - 1):
            next_observation = self.steps[-1].observation
            next_state = State.from_observations(observations + [next_observation], max_history=max_history)
        else:
            next_state = None

        return Transition (
            state,
            current_step.action,
            current_step.reward,
            next_state,
            current_step.done,
        )

    def sample_many(self, n: int, idxs: Optional[list[int]] = None, max_history: int = 10) -> list[Transition]:
        """
        Sample multiple transitions from the episode.

        Args:
            n: Number of transitions to sample
            idxs: Optional list of specific indices to sample. If provided, n is ignored.
            max_history: Maximum history length for each state

        Returns:
            List of sampled transitions
        """
        if idxs is None:
            idxs = [random.randint(1, len(self.steps) - 1) for _ in range(n)]

        transitions = [self.sample_one(idx, max_history) for idx in idxs]
        return transitions
