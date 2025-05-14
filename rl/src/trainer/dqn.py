import logging
import os
from typing import Union, Callable, Any, Optional

from tqdm.auto import tqdm

from networks.dqn import DQN, DoubleDQN


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DQN")


class DQNTrainer:
    """
    Training helper for DQN agents
    """
    def __init__(
        self,
        agent: Union[DQN, DoubleDQN],
        env: Any,
        state_processor: Optional[Callable] = None,
        action_processor: Optional[Callable] = None,
        reward_processor: Optional[Callable] = None,
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
