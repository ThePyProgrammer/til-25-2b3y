from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoder import create_encoder


def _orthogonal_init(scale=1.0):
    def _init(tensor):
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")

        nn.init.orthogonal_(tensor)
        tensor.mul_(scale)
    return _init

def orthogonal_init(module):
    """
    Applies orthogonal initialization to nn.Linear layers' weights
    and initializes biases to zero.
    """
    orthogonal_init_fn = _orthogonal_init(scale=1.4)

    if isinstance(module, nn.Linear):
        orthogonal_init_fn(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class PPODiscretePolicy(nn.Module):
    """Policy network for discrete action spaces in PPO algorithm."""

    def __init__(self, embedding_dim: int, action_dim: int, hidden_dims: list[int] = [256]):
        super().__init__()

        # Build policy network
        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        # Final layer for action logits
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, action_dim)

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


class PPOValueNetwork(nn.Module):
    """Value network (critic) for PPO algorithm."""

    def __init__(self, embedding_dim: int, hidden_dims: list[int] = [256]):
        super().__init__()

        # Build value network
        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        # Final layer for value output
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass to get state value."""
        return self.network(x)


class PPOActorCritic(nn.Module):
    """Combined actor-critic network for PPO algorithm."""

    def __init__(
        self,
        action_dim: int,
        map_size: int = 16,
        channels: int = 14,
        encoder_type: str = "large",
        embedding_dim: int = 256,
        actor_hidden_dims: list[int] = [256],
        critic_hidden_dims: list[int] = [256],
        shared_encoder: bool = True,
        encoder_kwargs: Optional[dict] = None
    ):
        super().__init__()

        # Initialize default encoder kwargs if not provided
        if encoder_kwargs is None:
            encoder_kwargs = {}

        # Set up encoder(s)
        self.shared_encoder = shared_encoder

        # Actor's encoder
        self.actor_encoder = create_encoder(
            encoder_type,
            map_size=map_size,
            channels=channels,
            embedding_dim=embedding_dim,
            **encoder_kwargs
        )

        # Critic's encoder (shared or separate)
        if self.shared_encoder:
            self.critic_encoder = self.actor_encoder
        else:
            self.critic_encoder = create_encoder(
                encoder_type,
                map_size=map_size,
                channels=channels,
                embedding_dim=embedding_dim,
                **encoder_kwargs
            )

        # Actor (policy) network
        self.actor = PPODiscretePolicy(
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims
        )

        # Critic (value) network
        self.critic = PPOValueNetwork(
            embedding_dim=embedding_dim,
            hidden_dims=critic_hidden_dims
        )

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
            map_input: Map observation tensor
            action: If provided, evaluate this action instead of sampling
            deterministic: If True and action is None, sample deterministically

        Returns:
            action: Sampled or evaluated action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        # Get actor embedding
        actor_embedding = self.actor_encoder(actor_input)

        # Get critic embedding (same as actor if shared)
        if self.shared_encoder:
            critic_embedding = actor_embedding
        else:
            critic_input = critic_input if critic_input is not None else actor_input
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
            actor_input: Actor's observation tensor:
            critic_input: Critic's observation tensor
            actions: Actions to evaluate

        Returns:
            log_probs: Log probabilities of the actions
            entropy: Entropy of the action distribution
            values: State value estimates
        """
        # Get actor embedding
        actor_embedding = self.actor_encoder(actor_input)

        # Get critic embedding (same as actor if shared)
        if self.shared_encoder:
            critic_embedding = actor_embedding
        else:
            critic_embedding = self.critic_encoder(critic_input)

        # Get action log probs and entropy
        log_probs, entropy = self.actor.evaluate_actions(actor_embedding, actions)

        # Get state values
        values = self.critic(critic_embedding)

        return log_probs, entropy, values.squeeze(-1)


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    map_size = 16
    channels = 12
    action_dim = 4

    # Sample inputs
    map_input = torch.rand(batch_size, channels, map_size, map_size)

    # Create model
    model = PPOActorCritic(
        action_dim=action_dim,
        map_size=map_size,
        channels=channels,
        encoder_type="small"
    )

    # Test forward pass
    action, log_prob, entropy, value = model.get_action_and_value(map_input)

    print(f"Action shape: {action.shape}, Value shape: {value.shape}")
    print(f"Log prob shape: {log_prob.shape}, Entropy shape: {entropy.shape}")

    # Test evaluation
    sampled_actions = torch.randint(0, action_dim, (batch_size,))
    log_probs, entropy, values = model.evaluate_actions(map_input, sampled_actions)

    print(f"Evaluated actions shape: {sampled_actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Values shape: {values.shape}")
