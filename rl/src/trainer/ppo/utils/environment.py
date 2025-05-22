import random
import numpy as np

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
