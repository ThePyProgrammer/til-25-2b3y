import os
import sys
import pathlib

import torch

sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment.types import RewardNames

# Import utility modules
from trainer.ppo.utils.args import parse_args
from trainer.ppo.utils.environment import set_seeds
from trainer.ppo.utils.model_utils import initialize_model, initialize_optimizer, load_checkpoint, apply_model_precision
from trainer.ppo.utils.training_loop import train_scout
from trainer.ppo.utils.buffer import ExperienceBuffer
from trainer.ppo.utils.scheduler import create_scheduler

from networks.ppo import orthogonal_init

from utils import count_parameters
from utils.wrapper import ScoutWrapper, CustomRewardsWrapper, CustomStateWrapper, TimeoutResetWrapper

from til_environment import gridworld


REWARDS_DICT = {
    RewardNames.GUARD_CAPTURES: 10,
    RewardNames.SCOUT_CAPTURED: -10,
    RewardNames.SCOUT_RECON: 0.2,
    RewardNames.SCOUT_MISSION: 1,
    # RewardNames.WALL_COLLISION: -0.4,
    # RewardNames.SCOUT_TRUNCATION: 2.5,
    # RewardNames.STATIONARY_PENALTY: -0.2,
    # RewardNames.SCOUT_STEP: 0.2
}

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    set_seeds(args.seed)

    env = gridworld.env(
        env_wrappers=[TimeoutResetWrapper, CustomStateWrapper, GuardWrapper],
        render_mode="human" if args.render else None,  # Render the map if requested
        debug=False,  # Enable debug mode
        novice=False,  # Use same map layout every time (for Novice teams only)
        rewards_dict=REWARDS_DICT
    )

    # Reset the environment with seed
    env.set_num_active_guards(args.num_guards)
    env.reset(seed=args.seed)

    # Extract observation shape information
    CHANNELS, MAP_SIZE, ACTION_DIM = 12, 31, 5
    print(f"Detected Map size: {MAP_SIZE}, Channels: {CHANNELS}, Action Dim: {ACTION_DIM}")

    # Initialize model
    model = initialize_model(
        action_dim=ACTION_DIM,
        map_size=MAP_SIZE,
        channels=CHANNELS,
        hidden_dims=[32, 32],
        encoder_type="small",
        shared_encoder=False,
        device=device,
        use_center_only=True,
    )

    if args.orthogonal_init:
        model.apply(orthogonal_init)

    print(f"Model has {count_parameters(model):,} parameters")

    print(model)

    # Apply precision setting
    model = apply_model_precision(model, args.bfloat16)

    # Initialize optimizer
    optimizer = initialize_optimizer(model, args.optim, args.lr)

    # Create learning rate scheduler
    scheduler = create_scheduler(optimizer, args, total_steps=args.timesteps)

    # Resume training if requested
    start_timesteps = 0
    if args.resume_from:
        checkpoint_path = os.path.join(args.save_dir, args.resume_from)
        start_timesteps, success = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # Initialize experience buffer
    buffer = ExperienceBuffer()

    # Run training loop
    train_scout(env, model, optimizer, scheduler, buffer, args)

    # Close environment
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
