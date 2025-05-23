from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class DiscretePolicyConfig:
    input_dim: int
    action_dim: int
    hidden_dims: list[int]

class DiscretePolicy(nn.Module):
    def __init__(self, config: DiscretePolicyConfig):
        super().__init__()

        self.config = config

        layers = []
        in_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, self.config.action_dim)

    def forward(self, x):
        """Forward pass to get action logits."""
        features = self.network(x)
        action_logits = self.action_head(features)
        return action_logits

    def get_distribution(self, x):
        """Get action distribution from embeddings."""
        action_logits = self.forward(x)
        return Categorical(logits=action_logits)

    def evaluate_actions(self, x, actions):
        """
        Evaluate log probability and entropy of given actions.

        Args:
            x: Encoded state embeddings
            actions: Actions to evaluate

        Returns:
            log_probs: Log probabilities of the actions
            entropy: Entropy of the action distribution
        """
        dist = self.get_distribution(x)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy

    def sample_action(self, x, deterministic=False):
        """
        Sample an action from the policy.

        Args:
            x: Encoded state embeddings
            greedy: If True, return the most likely action

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
        """
        dist = self.get_distribution(x)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

@dataclass
class ValueNetworkConfig:
    input_dim: int
    hidden_dims: list[int]

class ValueNetwork(nn.Module):
    def __init__(self, config: ValueNetworkConfig):
        super().__init__()

        self.config = config

        layers = []
        in_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass to get state value."""
        return self.network(x)

class PPOActorCritic(nn.Module):
    """Combined actor-critic network for PPO algorithm."""

    def __init__(
        self,
        actor_encoder: nn.Module,
        critic_encoder: nn.Module,
        actor: DiscretePolicy,
        critic: ValueNetwork,
    ):
        super().__init__()

        # Store encoders
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder

        self.actor = actor
        self.critic = critic

    def get_value(self, map_input):
        """Get state value estimate from critic."""
        critic_embedding = self.critic_encoder(map_input)
        return self.critic(critic_embedding)

    def get_action_and_value(
        self,
        actor_input,
        critic_input=None,
        action=None,
        deterministic=False
    ):
        """
        Forward pass to get action, log probability, entropy, and value.

        Args:
            actor_input: Actor's observation tensor
            critic_input: Critic's observation tensor (if None, uses actor_input)
            action: If provided, evaluate this action instead of sampling
            deterministic: If True and action is None, sample deterministically

        Returns:
            action: Sampled or evaluated action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        # Use actor_input for critic if critic_input not provided
        critic_input = critic_input if critic_input is not None else actor_input

        # Get embeddings
        actor_embedding = self.actor_encoder(actor_input)
        critic_embedding = self.critic_encoder(critic_input)

        # Get state value
        value = self.critic(critic_embedding)

        # Sample or evaluate action
        if action is None:
            action, log_prob, entropy = self.actor.sample_action(
                actor_embedding,
                deterministic=deterministic
            )
        else:
            dist = self.actor.get_distribution(actor_embedding)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(self, actor_input, critic_input, actions):
        """
        Evaluate actions for PPO update.

        Args:
            actor_input: Actor's observation tensor
            critic_input: Critic's observation tensor
            actions: Actions to evaluate

        Returns:
            log_probs: Log probabilities of the actions
            entropy: Entropy of the action distribution
            values: State value estimates
        """
        # Get embeddings
        actor_embedding = self.actor_encoder(actor_input)
        critic_embedding = self.critic_encoder(critic_input)

        # Get action log probs and entropy
        log_probs, entropy = self.actor.evaluate_actions(actor_embedding, actions)

        # Get state values
        values = self.critic(critic_embedding)

        return log_probs, entropy, values.squeeze(-1)
