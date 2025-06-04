from typing import Optional

import torch
from tensordict.tensordict import TensorDict


def compute_gae(
    episode_experiences: list[TensorDict],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    next_value = torch.tensor(0.0)

    for step in reversed(range(len(episode_experiences))):
        delta = (
            episode_experiences[step]['reward']
            + gamma * next_value * (1 - dones[step])
            - episode_experiences[step]['value']
        )
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = episode_experiences[step]['value']

    return advantages

class PPOExperienceBuffer:
    """
    A buffer for collecting training experiences (state, action, reward, etc.)
    for Proximal Policy Optimization (PPO).
    """
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            gamma (float): for rewards discounting under temporal differencing.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.experiences = []
        self._current_episode = []

    def add(
        self,
        experience: TensorDict,
        done: bool
    ):
        self._current_episode.append(experience)

        if done:
            dones = [False] * (len(self._current_episode) - 1) + [True]  # Only last step is done

            # Compute advantages using GAE
            advantages = compute_gae(
                episode_experiences=self._current_episode,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

            # Add advantages to each experience and move to experiences buffer
            for i, exp in enumerate(self._current_episode):
                exp['advantage'] = advantages[i]
                exp['return'] = advantages[i] + exp['value']  # GAE + value = return
                self.experiences.append(exp)

            # Clear current episode
            self._current_episode = []

    def get_batch(self) -> Optional[TensorDict]:
        """
        Collects all stored experiences into batch tensors.

        Returns:
            A dictionary containing batched tensors for map_inputs,
            actions, log_probs, values, rewards, and dones.
            Returns None if the buffer is empty.
        """
        if not self.experiences:
            return None # Buffer is empty

        return torch.stack(self.experiences)

    def clear(self):
        """
        Clears all stored experiences from the buffer.
        """
        self.experiences = []

    def __len__(self):
        """
        Returns the number of steps currently stored in the buffer.
        """
        return len(self.experiences)

    def is_empty(self):
        """
        Checks if the buffer is empty.
        """
        return len(self) == 0
