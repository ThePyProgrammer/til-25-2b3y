import sys
import os
import pathlib
import random
import numpy as np

# Add parent directories to path
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld

from utils.wrapper import CustomDictWrapper


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
        env_wrappers=[CustomDictWrapper],  # clear out default env wrappers
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
