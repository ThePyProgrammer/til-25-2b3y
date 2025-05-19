from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.replay_buffer import ReplayBuffer
from .recurrent_encoder import StateEncoder


class DQN(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        act_cls: type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.LeakyReLU] = nn.ReLU,
        dropout: float = 0.1,
        encoder_kwargs: Optional[dict] = None,
        dueling: bool = True,  # Whether to use dueling architecture
        lr: float = 3e-4,      # Learning rate
        gamma: float = 0.99,   # Discount factor
        buffer_size: int = 100000  # Replay buffer size
    ):
        super().__init__()

        # Initialize the state encoder
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        self.encoder = StateEncoder(act_cls=act_cls, dropout=dropout, **encoder_kwargs)

        # Dueling network architecture (if enabled)
        self.dueling = dueling
        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # Standard Q-network head
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_dim)
            )

        # Training parameters
        self.action_dim = action_dim
        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def forward(
        self,
        states: list[list[tuple[torch.Tensor, torch.Tensor]]],
        h_n: Optional[torch.Tensor] = None,
        c_n: Optional[torch.Tensor] = None,
        use_cached_states: bool = False
    ) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Forward pass through the DQN network.

        Args:
            states: List of state sequences, each containing tuples of (spatial, static) tensors
            h_n: Optional hidden state for LSTM
            c_n: Optional cell state for LSTM
            use_cached_states: Whether to use cached LSTM states

        Returns:
            Tuple of (q_values, (h_n, c_n)) where:
                - q_values is the Q-values for each action
                - h_n and c_n are the final hidden and cell states of the LSTM
        """
        # Process through the encoder
        outputs, (h_n, c_n) = self.encoder(
            states, h_n, c_n, use_cached_states=use_cached_states
        )

        # If we have no outputs, return zeros for Q-values
        if len(outputs) == 0:
            action_dim = self.advantage_stream[-1].out_features if self.dueling else self.q_head[-1].out_features
            batch_size = len(states)
            q_values = torch.zeros(batch_size, action_dim)
            return q_values, (h_n, c_n)

        # Get the final state representation from each sequence
        batch_final_states = []
        for seq in outputs:
            # Use the last state in each sequence
            batch_final_states.append(seq[-1])

        # Stack the final states
        if len(batch_final_states) > 0:
            x = torch.stack(batch_final_states)

            # Compute Q-values
            if self.dueling:
                # Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
                values = self.value_stream(x)
                advantages = self.advantage_stream(x)
                q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
            else:
                # Standard Q-network
                q_values = self.q_head(x)
        else:
            # Handle empty batch case
            action_dim = self.advantage_stream[-1].out_features if self.dueling else self.q_head[-1].out_features
            q_values = torch.zeros(0, action_dim)

        return q_values, (h_n, c_n)

    def act(
        self,
        state: list[tuple[torch.Tensor, torch.Tensor]],
        epsilon: float = 0.0,
        h_n: Optional[torch.Tensor] = None,
        c_n: Optional[torch.Tensor] = None,
        use_cached_states: bool = True
    ) -> Tuple[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state: Current state as a list of (spatial, static) tuples
            epsilon: Exploration probability
            h_n: Optional hidden state for LSTM
            c_n: Optional cell state for LSTM
            use_cached_states: Whether to use cached LSTM states

        Returns:
            Tuple of (action, (h_n, c_n))
        """
        # Convert single state to batch format expected by forward
        state_batch = [state]

        with torch.no_grad():
            q_values, (h_n, c_n) = self.forward(
                state_batch, h_n, c_n, use_cached_states=use_cached_states
            )

            # Epsilon-greedy action selection
            if torch.rand(1).item() < epsilon:
                action_dim = q_values.shape[1]
                action = torch.randint(0, action_dim, (1,)).item()
            else:
                action = int(q_values.argmax(dim=1).item())

            return int(action), (h_n, c_n)

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=None):
        """
        Compute the loss for a batch of transitions.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor (if None, use self.gamma)

        Returns:
            Loss value
        """
        gamma = gamma if gamma is not None else self.gamma

        # Convert actions to tensor if not already
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        else:
            actions = actions.unsqueeze(1)

        # Convert rewards and dones to tensors if not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float)

        # Get current Q-values
        current_q_values, _ = self(states)
        current_q_values = current_q_values.gather(1, actions).squeeze(1)

        # Get next Q-values
        with torch.no_grad():
            next_q_values, _ = self(next_states)
            next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * gamma * next_q_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        return loss

    def update(self, batch_size: int) -> float:
        """
        Update the network using a batch of transitions from the replay buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Loss value
        """
        if not self.replay_buffer.can_sample(batch_size):
            return 0.0

        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(batch_size)

        # Unpack the batch
        batch = list(zip(*transitions))
        state_batch = list(batch[0])
        action_batch = torch.tensor(batch[1], dtype=torch.long)
        reward_batch = torch.tensor(batch[2], dtype=torch.float)
        next_state_batch = list(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float)

        # Use update_with_batch for consistency
        return self.update_with_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'model': self.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])

    def update_with_batch(self, states, actions, rewards, next_states, dones, gamma=None):
        """
        Update the network using an explicit batch of transitions.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor (if None, use self.gamma)

        Returns:
            Loss value
        """
        # Convert actions to tensor if not already
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)

        # Convert rewards and dones to tensors if not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float)

        # Calculate current Q values
        current_q_values, _ = self(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate next Q values
        with torch.no_grad():
            next_q_values, _ = self(next_states)
            next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * (self.gamma if gamma is None else gamma) * next_q_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

class DoubleDQN(nn.Module):
    """
    Implementation of Double DQN to reduce overestimation of Q-values
    """
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        act_cls: type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.LeakyReLU] = nn.ReLU,
        dropout: float = 0.1,
        encoder_kwargs: Optional[dict] = None,
        dueling: bool = True,
        tau: float = 0.005,  # Soft update parameter
        lr: float = 3e-4,    # Learning rate
        gamma: float = 0.99, # Discount factor
        buffer_size: int = 100000  # Replay buffer size
    ):
        super().__init__()

        # Online network
        self.online_network = DQN(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            act_cls=act_cls,
            dropout=dropout,
            encoder_kwargs={} if encoder_kwargs is None else encoder_kwargs,
            dueling=dueling,
            lr=lr,
            gamma=gamma,
            buffer_size=buffer_size
        )

        # Target network (initialized with same weights as online network)
        self.target_network = DQN(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            act_cls=act_cls,
            dropout=dropout,
            encoder_kwargs=encoder_kwargs,
            dueling=dueling,
            lr=lr,
            gamma=gamma,
            buffer_size=buffer_size
        )
        self.target_network.load_state_dict(self.online_network.state_dict())

        # Disable gradient updates for target network
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.tau = tau
        self.gamma = gamma
        self.action_dim = action_dim

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the online network
        """
        return self.online_network(*args, **kwargs)

    # act method is implemented through the __getattr__ mechanism below

    def target_forward(self, *args, **kwargs):
        """
        Forward pass through the target network
        """
        return self.target_network(*args, **kwargs)

    def soft_update(self):
        """
        Soft update of target network parameters:
        θ_target = τ*θ_online + (1-τ)*θ_target
        """
        for target_param, online_param in zip(
            self.target_network.parameters(), self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self, batch_size: int) -> float:
        """
        Update the network using a batch of transitions from the replay buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Loss value
        """
        if not self.replay_buffer.can_sample(batch_size):
            return 0.0

        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(batch_size)

        # Unpack the batch
        batch = list(zip(*transitions))
        state_batch = list(batch[0])
        action_batch = torch.tensor(batch[1], dtype=torch.long)
        reward_batch = torch.tensor(batch[2], dtype=torch.float)
        next_state_batch = list(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float)

        # Use update_with_batch for consistency
        return self.update_with_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def update_with_batch(self, states, actions, rewards, next_states, dones, gamma=None):
        """
        Update the network using an explicit batch of transitions.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor (if None, use self.gamma)

        Returns:
            Loss value
        """
        # Convert actions to tensor if not already
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)

        # Convert rewards and dones to tensors if not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float)

        # Calculate current Q values
        current_q_values, _ = self.online_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate next Q values using Double Q-learning approach
        with torch.no_grad():
            next_q_values, _ = self.online_network(next_states)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)

            next_q_values_target, _ = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

            expected_q_values = rewards + (1 - dones) * (self.gamma if gamma is None else gamma) * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize
        self.online_network.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.online_network.optimizer.step()

        # Update target network
        self.soft_update()

        return loss.item()

    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])

    # Make the interface consistent with DQN by forwarding methods
    def __getattr__(self, name):
        """
        Forward method calls to the online network if they're not defined in DoubleDQN.
        This avoids duplicating method implementations and ensures consistency.
        """
        if hasattr(self.online_network, name):
            return getattr(self.online_network, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
