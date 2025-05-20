import sys
import os
import pathlib
import random
import numpy as np

# Add parent directories to path
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld


def setup_environment(args, rewards_dict):
    """
    Initialize and setup the gridworld environment

    Args:
        args: Command line arguments
        rewards_dict: Optional custom rewards dictionary

    Returns:
        env: Initialized gridworld environment
    """
    # Create environment
    env = gridworld.env(
        env_wrappers=[],  # clear out default env wrappers
        render_mode="human" if args.render else None,  # Render the map if requested
        debug=False,  # Enable debug mode
        novice=False,  # Use same map layout every time (for Novice teams only)
        rewards_dict=rewards_dict
    )

    # Reset the environment with seed
    env.reset(seed=args.seed)

    return env

def set_seeds(seed):
    """
    Set random seeds for reproducibility

    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    # Uncomment if deterministic behavior is critical
    # torch.backends.cudnn.deterministic = True

def get_scout_initial_observation(env):
    """
    Get the initial observation for the scout agent

    Args:
        env: Gridworld environment

    Returns:
        scout_initial_observation: Initial observation for the scout
    """
    scout_initial_observation = None

    # Iterate agents until we get scout's first observation after reset
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if agent == env.scout:
            scout_initial_observation = obs
            break  # Found scout's first observation

        # Step other agents to get to scout's turn
        env.step(env.action_space(agent).sample())  # Step other agents randomly

    if scout_initial_observation is None:
        raise RuntimeError("Could not get initial observation for the scout after reset.")

    return scout_initial_observation

def extract_observation_shape(obs_tensor):
    """
    Extract shape information from the observation tensor

    Args:
        obs_tensor: Observation tensor from scout's map

    Returns:
        tuple: (channels, map_size, action_dim)
    """
    # The tensor shape is (C, H, W)
    if len(obs_tensor.shape) != 3:
        raise RuntimeError(f"Expected map tensor shape (C, H, W), but got {obs_tensor.shape}")

    channels, map_size, _ = obs_tensor.shape
    if map_size != obs_tensor.shape[2]:
        print(f"Warning: Map tensor is not square. Assuming MAP_SIZE = {map_size}")

    # Assuming Action_DIM is 5 for movement (Forward, Backward, Turn left, Turn right, Stay)
    action_dim = 5

    return channels, map_size, action_dim
