import os
import torch
import torch.optim as optim
from networks.ppo import PPOActorCritic


def initialize_model(action_dim, map_size, channels, encoder_type="large", shared_encoder=False, device=None):
    """
    Initialize the PPO model

    Args:
        action_dim: Dimension of the action space
        map_size: Size of the map
        channels: Number of channels in the map tensor
        encoder_type: Type of encoder to use ("small", "medium", "large")
        shared_encoder: Whether to use a shared encoder for actor and critic
        device: Device to place the model on (cpu/cuda)

    Returns:
        model: Initialized PPO model
    """
    model = PPOActorCritic(
        action_dim=action_dim,
        map_size=map_size,
        channels=channels,
        encoder_type=encoder_type,
        shared_encoder=shared_encoder,
    )

    if device is not None:
        model.to(device)

    return model

def initialize_optimizer(model, optimizer_name, learning_rate):
    """
    Initialize the optimizer

    Args:
        model: PPO model
        optimizer_name: Name of the optimizer (adam, adamw, sgd)
        learning_rate: Learning rate

    Returns:
        optimizer: Initialized optimizer
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}'. Using Adam.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer

def save_checkpoint(model, optimizer, scheduler, timesteps_elapsed, save_dir, filename='latest.pt'):
    """
    Save a checkpoint of the model, optimizer, and scheduler state

    Args:
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        timesteps_elapsed: Current timestep count
        save_dir: Directory to save the checkpoint in
        filename: Name of the checkpoint file

    Returns:
        str: Path to the saved checkpoint
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'timesteps_elapsed': timesteps_elapsed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, checkpoint_path)

    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """
    Load a checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)

    Returns:
        int: Timesteps elapsed from the checkpoint
        bool: Whether the checkpoint was loaded successfully
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return 0, False

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if it exists and a scheduler was provided
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state.")

    timesteps_elapsed = checkpoint['timesteps_elapsed']
    print(f"Loaded checkpoint from {checkpoint_path} at timestep {timesteps_elapsed}")

    return timesteps_elapsed, True

def apply_model_precision(model, use_bfloat16=False):
    """
    Apply precision setting to model

    Args:
        model: PPO model
        use_bfloat16: Whether to use bfloat16 precision

    Returns:
        model: Model with updated precision
    """
    if use_bfloat16:
        torch.set_default_dtype(torch.bfloat16)
        model.to(torch.bfloat16)
        print("Using bfloat16 precision.")

    return model
