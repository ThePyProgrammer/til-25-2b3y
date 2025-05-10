from typing import Optional, Any, Tuple, List, Deque, Union, Callable
from collections import deque
import random
from tqdm.auto import tqdm
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .encoder import StateEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DQN")

class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size

class DQN(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        act_cls: type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.LeakyReLU] = nn.ReLU,
        dropout: float = 0.1,
        encoder_kwargs: dict = None,
        dueling: bool = True,  # Whether to use dueling architecture
        lr: float = 3e-4,      # Learning rate
        gamma: float = 0.99    # Discount factor
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

        return action, (h_n, c_n)

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

    def update(self, states, actions, rewards, next_states, dones, gamma=None):
        """
        Update the network using a batch of transitions.

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
        # Compute loss
        loss = self.compute_loss(states, actions, rewards, next_states, dones, gamma)

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
        encoder_kwargs: dict = None,
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
            gamma=gamma
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
            gamma=gamma
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

    def act(self, *args, **kwargs):
        """
        Select action using the online network
        """
        return self.online_network.act(*args, **kwargs)

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
        action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float)
        next_state_batch = list(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float)

        # Calculate current Q values
        current_q_values, _ = self.online_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch).squeeze(1)

        # Calculate next Q values using Double Q-learning approach
        # 1. Get actions from online network
        with torch.no_grad():
            next_q_values, _ = self.online_network(next_state_batch)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)

            # 2. Evaluate those actions using target network
            next_q_values_target, _ = self.target_network(next_state_batch)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

            # 3. Compute expected Q values
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize
        self.online_network.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
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


class DQNTrainer:
    """
    Training helper for DQN agents
    """
    def __init__(
        self,
        agent: Union[DQN, DoubleDQN],
        env: Any,
        state_processor: Callable = None,
        action_processor: Callable = None,
        reward_processor: Callable = None,
        checkpoint_dir: str = "checkpoints",
        max_iter_per_episode: int = 1000
    ):
        """
        Initialize the DQN trainer.
        
        Args:
            agent: DQN or DoubleDQN agent
            env: Environment to train on
            state_processor: Function to process environment states into network format
            action_processor: Function to convert network actions to environment actions
            reward_processor: Function to process environment rewards
            checkpoint_dir: Directory to save checkpoints
            max_iter_per_episode: Maximum iterations per episode
        """
        self.agent = agent
        self.env = env
        self.max_iter_per_episode = max_iter_per_episode
        
        # Default processors (identity functions)
        def default_processor(x): return x
        self.state_processor: Callable = default_processor if state_processor is None else state_processor
        self.action_processor: Callable = default_processor if action_processor is None else action_processor
        self.reward_processor: Callable = default_processor if reward_processor is None else reward_processor
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.train_losses = []

    def train(
        self,
        num_episodes: int,
        batch_size: int = 64,
        update_freq: int = 4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        eval_freq: int = 20,
        checkpoint_freq: int = 100,
        eval_episodes: int = 5,
        use_tqdm: bool = True
    ):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train for
            batch_size: Batch size for training
            update_freq: How often to update the network (in steps)
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Minimum epsilon value
            epsilon_decay: Epsilon decay factor
            eval_freq: How often to evaluate the agent (in episodes)
            checkpoint_freq: How often to save checkpoints (in episodes)
            eval_episodes: Number of episodes to evaluate on
            use_tqdm: Whether to use tqdm progress bars
        
        Returns:
            Training metrics (episode rewards, lengths, losses)
        """
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.train_losses = []
        
        # Current epsilon value
        epsilon = epsilon_start
        
        # Total number of training steps
        total_steps = 0
        
        # Training loop
        episode_iterator = tqdm(range(num_episodes)) if use_tqdm else range(num_episodes)
        for episode in episode_iterator:
            # Reset environment and agent states
            self.env.reset(seed=episode)  # Use episode number as seed for reproducibility
            observation, reward, termination, truncation, info = self.env.last()
            state = self.state_processor(observation)
            h_n, c_n = None, None
            
            episode_reward = 0
            episode_loss = 0
            step = 0
            
            # Episode loop
            for agent_id in self.env.agent_iter(max_iter=self.max_iter_per_episode):
                step += 1
                
                # Get current state
                observation, reward, termination, truncation, info = self.env.last()
                state = self.state_processor(observation)
                processed_reward = self.reward_processor(reward)
                
                # Check if episode is done
                done = termination or truncation
                
                if not done:
                    # Select action
                    action, (h_n, c_n) = self.agent.act(
                        state, epsilon=epsilon, h_n=h_n, c_n=c_n, use_cached_states=True
                    )
                    env_action = self.action_processor(action)
                    
                    # Store current state before taking action
                    current_state = state
                    
                    # Take action in environment
                    self.env.step(env_action)
                    
                    # Get next state
                    next_observation, next_reward, next_termination, next_truncation, next_info = self.env.last()
                    next_state = self.state_processor(next_observation)
                    next_done = next_termination or next_truncation
                    
                    # Store transition in replay buffer
                    if hasattr(self.agent, 'replay_buffer'):
                        self.agent.replay_buffer.push(current_state, action, processed_reward, next_state, next_done)
                    
                    # Update metrics
                    episode_reward += processed_reward
                    total_steps += 1
                    
                    # Update network if it's time
                    if total_steps % update_freq == 0:
                        if hasattr(self.agent, 'update'):
                            loss = self.agent.update(batch_size)
                        else:
                            loss = 0.0
                        
                        if loss > 0:
                            episode_loss += loss
                            self.train_losses.append(loss)
                else:
                    # Episode is done
                    break

            # Update episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)

            # Update exploration rate
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Log progress
            if use_tqdm and isinstance(episode_iterator, tqdm):
                episode_iterator.set_description(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Length: {step+1} | "
                    f"Epsilon: {epsilon:.4f}"
                )
            else:
                if (episode + 1) % 10 == 0:
                    logger.info(
                        f"Episode {episode+1}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Length: {step+1} | "
                        f"Epsilon: {epsilon:.4f}"
                    )

            # Evaluate agent
            if (episode + 1) % eval_freq == 0:
                eval_rewards = self.evaluate(eval_episodes)
                logger.info(
                    f"Evaluation at episode {episode+1}: "
                    f"Average reward: {eval_rewards:.4f}"
                )

            # Save checkpoint
            if (episode + 1) % checkpoint_freq == 0:
                self._save_checkpoint(episode + 1)

        # Final evaluation
        final_eval = self.evaluate(eval_episodes)
        logger.info(f"Final evaluation: Average reward: {final_eval:.4f}")

        # Final checkpoint
        self._save_checkpoint("final")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'train_losses': self.train_losses,
            'final_eval': final_eval
        }

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            
        Returns:
            Average episode reward
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            self.env.reset(seed=10000 + episode)  # Use different seeds than training
            observation, reward, termination, truncation, info = self.env.last()
            state = self.state_processor(observation)
            h_n, c_n = None, None
            
            episode_reward = 0
            
            # Episode loop
            for agent_id in self.env.agent_iter(max_iter=self.max_iter_per_episode):
                # Get current state
                observation, reward, termination, truncation, info = self.env.last()
                state = self.state_processor(observation)
                processed_reward = self.reward_processor(reward)
                
                # Update metrics
                episode_reward += processed_reward
                
                # Check if episode is done
                if termination or truncation:
                    break
                
                # Select action (no exploration)
                action, (h_n, c_n) = self.agent.act(
                    state, epsilon=0.0, h_n=h_n, c_n=c_n, use_cached_states=True
                )
                env_action = self.action_processor(action)
                
                # Take action in environment
                self.env.step(env_action)
            
            eval_rewards.append(episode_reward)

        return sum(eval_rewards) / len(eval_rewards)

    def _save_checkpoint(self, identifier):
        """Save checkpoint with identifier"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"dqn_checkpoint_{identifier}.pt")
        if hasattr(self.agent, 'save'):
            self.agent.save(checkpoint_path)
        else:
            torch.save(self.agent.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
