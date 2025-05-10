import os
import sys
import pathlib
import numpy as np
import torch
import random

# Add parent directory to path to import local modules
sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

try:
    from til_environment import gridworld
except ImportError:
    print("Error: Could not import gridworld environment. Please make sure the TIL environment is installed.")
    sys.exit(1)

from utils.state import encode_observation
from networks.encoder import StateEncoder
from networks.dqn import DoubleDQN

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

# State processing function
def process_state(observation):
    """Process environment observation into the format expected by the DQN."""
    if observation is None:
        return []
    
    # Encode observation using our utility function
    spatial, static = encode_observation(observation)
    
    # DQN expects a list of sequences, but we have a single state
    return [(spatial, static)]

def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with human rendering
    env = gridworld.env(
        env_wrappers=[],  # clear out default env wrappers
        render_mode="human",  # Render the map in a window
        debug=True,  # Enable debug mode to see more information
        novice=True,  # Use same map layout every time (for Novice teams only)
    )
    
    # Action space size
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
    
    # Try to load a pre-trained model
    try:
        agent.load("checkpoints/dqn_checkpoint_final.pt")
        print("Loaded pre-trained model")
    except FileNotFoundError:
        print("No pre-trained model found, using random policy")
    
    # Reset environment
    env.reset(seed=42)
    
    # Get initial state
    observation, reward, termination, truncation, info = env.last()
    state = process_state(observation)
    h_n, c_n = None, None
    
    print("Starting demonstration...")
    print("Press Ctrl+C to stop")
    
    total_reward = 0
    steps = 0
    
    try:
        # Episode loop
        for agent_id in env.agent_iter(max_iter=1000):
            # Get current state
            observation, reward, termination, truncation, info = env.last()
            state = process_state(observation)
            total_reward += reward
            steps += 1
            
            # Print info
            print(f"Step: {steps}, Reward: {reward}, Total Reward: {total_reward}")
            
            # Check if episode is done
            if termination or truncation:
                print("Episode finished!")
                break
            
            # Select action (no exploration)
            action, (h_n, c_n) = agent.act(state, epsilon=0.0, h_n=h_n, c_n=c_n)
            
            # Print action
            action_names = ["Stay", "Up", "Right", "Down", "Left"]
            print(f"Action: {action} ({action_names[action]})")
            
            # Execute action
            env.step(action)
    
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")
    finally:
        env.close()
        print(f"Demonstration finished. Total reward: {total_reward}")

if __name__ == "__main__":
    main()