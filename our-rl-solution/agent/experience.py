import random
from typing import Optional

import torch

from .episode import (
    Observation,
    Step,
    State,
    StateConfig,
    DEFAULT_STATE_CONFIG,
    Transition,
    Episode,
)

class Experience:
    """
    Experience replay buffer for reinforcement learning agents with sequence-based states.

    Stores transitions (state, action, reward, next_observation, done) and provides
    methods for adding experiences and sampling batches for training.
    Supports both individual transitions and episode-based sampling.
    """
    def __init__(
        self,
        capacity: int = 10000,
        episode_length: int = 100,
        state_config: StateConfig = DEFAULT_STATE_CONFIG,
    ):
        """
        Initialize the experience replay buffer.

        Args:
            capacity: Maximum number of transitions to store in the buffer.
                     When buffer is full, oldest transitions are overwritten.
            sequence_length: Maximum length of sequences to sample
        """
        self.capacity: int = capacity
        self.episode_length = episode_length
        self.episode_capacity: int = self.capacity // self.episode_length
        self.episode_buffer: list[Episode] = []
        self.position: int = 0

        self.current_episode: Episode = Episode()

        self.state_config = state_config

    def push(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Optional[Observation],
        done: bool
    ):
        """
        Store a step in the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next state (None if terminal)
            done: Whether this transition ended the episode
        """

        step = Step(observation, action, reward, next_observation, done)
        self.position += 1

        # Also track in the current episode
        self.current_episode.update(step)

        # If episode is done, store it and start a new one
        if done:
            self.end_episode()

    def end_episode(self):
        """Store the current episode and start a new one."""
        if not self.current_episode:
            return

        self.episode_buffer.append(self.current_episode)

        # Limit episode buffer size (by removing oldest episodes)
        while len(self.episode_buffer) > self.episode_capacity:
            self.episode_buffer.pop(0)

        self.current_episode = Episode()

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Sample a random batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions
        """
        episodes = random.sample(self.episode_buffer, batch_size)
        transitions = [ep.sample_one(state_config=self.state_config) for ep in episodes]
        return transitions

    def __len__(self) -> int:
        """Return the current number of episodes in buffer"""
        return len(self.episode_buffer)

    def get_batch(
        self,
        batch_size: int
    ) -> tuple[
        list[list[Observation]],
        torch.Tensor,
        torch.Tensor,
        list[list[Observation]],
        torch.Tensor
    ]:
        """
        Sample a batch from the experience buffer in a format compatible with the encoder.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple containing:
            - states: List[List[Observation]] - Batch of observation sequences for states
            - actions: torch.Tensor - Tensor of actions taken
            - rewards: torch.Tensor - Tensor of rewards received
            - next_states: List[List[Observation]] - Batch of observation sequences for next states
            - dones: torch.Tensor - Tensor indicating if transitions were terminal
        """
        # Sample batch of transitions
        transitions = self.sample(batch_size)

        # Extract states, actions, rewards, next_states, and dones
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        for transition in transitions:
            # Get the sequence of observations for current state
            state_observations = transition.state.get_encoded_sequence()
            states.append(state_observations)

            # Get action, reward, and done flag
            actions.append(transition.action)
            rewards.append(transition.reward)
            dones.append(float(transition.done))

            # Get the sequence of observations for next state if available
            if transition.next_state is not None:
                next_state_observations = transition.next_state.get_encoded_sequence()
                next_states.append(next_state_observations)
            else:
                # If there is no next state (end of episode), use an empty sequence
                next_states.append([])

        # Convert actions, rewards, and dones to tensors
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)
        dones_tensor = torch.tensor(dones, dtype=torch.float)

        return states, actions_tensor, rewards_tensor, next_states, dones_tensor

    def __call__(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Optional[Observation],
        done: bool
    ):
        """Convenience method to add a transition to the buffer."""
        self.push(observation, action, reward, next_observation, done)

    def force_end_episode(self):
        """
        Manually end the current episode.
        Useful when environment resets without a done signal.
        """
        if self.current_episode:
            self.end_episode()
