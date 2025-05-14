import os
import sys
import pathlib
import numpy as np
import torch
import random
from typing import Tuple, List, Any

# Add parent directory to path to import local modules
sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

try:
    from til_environment import gridworld
except ImportError:
    print("Warning: Could not import gridworld environment. This example assumes you have the TIL environment installed.")
    print("You may need to adjust this example to your specific environment.")

from utils.state import encode_observation
from networks.encoder import StateEncoder
from networks.dqn import DoubleDQN, DQNTrainer

# Set seeds for reproducibility
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# State processing function for the environment
def process_state(observation: dict) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Process environment observation into the format expected by the DQN."""
    if observation is None:
        return []
    
    # Encode observation using our utility function
    spatial, static = encode_observation(observation)
    
    # DQN expects a list of sequences, but we have a single state
    return [(spatial, static)]

# Action processing function for the environment
def process_action(action: int) -> int:
    """Process DQN action into the format expected by the environment."""
    # In this simple example, we assume actions are the same format
    return action

# Reward processing function for the environment
def process_reward(reward: float) -> float:
    """Process environment reward into the format expected by the DQN."""
    # In this simple example, we use the reward as is
    return reward

if __name__ == "__main__":
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create environment
        env = gridworld.env(
            env_wrappers=[],  # clear out default env wrappers
            debug=False,  # Disable debug mode for training
            novice=True,  # Use same map layout every time (for Novice teams only)
        )
        
        # Get action space size
        action_dim = 5  # Matches the TIL environment (0: stay, 1-4: move)
        print(f"Action space size: {action_dim}")
        
        # Create DQN agent
        agent = DoubleDQN(
            action_dim=action_dim,
            hidden_dim=128,
            act_cls=torch.nn.ReLU,
            dropout=0.1,
            dueling=True,
            tau=0.005,
            lr=3e-4,
            gamma=0.99,
            buffer_size=10000
        ).to(device)
        
        print("DQN agent created successfully!")
        
        # Create trainer
        trainer = DQNTrainer(
            agent=agent,
            env=env,
            state_processor=process_state,
            action_processor=process_action,
            reward_processor=process_reward,
            checkpoint_dir="checkpoints",
            max_iter_per_episode=1000
        )
        
        print("Starting training...")
        
        # Train agent
        metrics = trainer.train(
            num_episodes=500,
            batch_size=64,
            update_freq=4,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            eval_freq=20,
            checkpoint_freq=100,
            eval_episodes=5
        )
        
        print("Training complete!")
        print(f"Final evaluation average reward: {metrics['final_eval']:.4f}")
        
        # Test trained agent
        print("\nRunning test episodes...")
        test_rewards = []
        
        for i in range(10):
            # Reset environment
            env.reset(seed=42 + i)
            observation, reward, termination, truncation, info = env.last()
            state = process_state(observation)
            h_n, c_n = None, None
            
            episode_reward = 0
            step = 0
            
            # Episode loop
            for agent_id in env.agent_iter(max_iter=1000):
                # Get current state
                observation, reward, termination, truncation, info = env.last()
                state = process_state(observation)
                
                # Check if episode is done
                if termination or truncation:
                    break
                
                # Select action
                action, (h_n, c_n) = agent.act(state, epsilon=0.0, h_n=h_n, c_n=c_n)
                
                # Take action in environment
                env.step(action)
                
                # Update metrics
                episode_reward += reward
                step += 1
            
            test_rewards.append(episode_reward)
            print(f"Test episode {i+1} - Reward: {episode_reward:.2f}, Steps: {step}")
        
        print(f"Average test reward: {sum(test_rewards) / len(test_rewards):.4f}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("This example assumes you have the TIL environment installed.")
    except Exception as e:
        print(f"Unexpected error: {e}")