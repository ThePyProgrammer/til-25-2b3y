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
from trainer.ppo.utils.model_utils import initialize_optimizer, load_checkpoint, apply_model_precision
from trainer.ppo.utils.training_loop import train_scout
from trainer.ppo.utils.buffer import ExperienceBuffer
from trainer.ppo.utils.scheduler import create_scheduler

from networks.v2.init import orthogonal_init
from networks.v2.encoder import MapEncoderConfig, TemporalMapEncoderConfig
from networks.v2.ppo import (
    DiscretePolicyConfig,
    ValueNetworkConfig
)
from networks.v2.utils import initialize_model

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
        env_wrappers=[TimeoutResetWrapper, CustomStateWrapper, ScoutWrapper],
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

    if args.temporal_state:
        assert args.temporal_frames

        encoder_config = TemporalMapEncoderConfig(
            map_size = 16,
            channels = 12,
            output_dim = 32,
            frames = 3,

            conv3d_channels = [16, 24, 32],
            conv3d_kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3)],
            conv3d_strides = [(1, 1, 1), (1, 1, 1), (1, 1, 1)],
            conv3d_paddings = [(0, 0, 0), (0, 0, 0), (0, 0, 0)],

            conv_layers = [32, 32],
            kernel_sizes = [3, 3],
            strides = [1, 1],
            paddings = [0, 0],

            use_batch_norm = True,
            dropout_rate = 0.1,
            use_layer_norm = True,
            use_center_only = True,
        )
    else:
        encoder_config = MapEncoderConfig(
            kernel_sizes=[7, 3, 3, 3],
            output_dim=32
        )

    actor_config = DiscretePolicyConfig(
        input_dim=32,
        action_dim=ACTION_DIM,
        hidden_dims=[32, 32]
    )

    critic_config = ValueNetworkConfig(
        input_dim=32,
        hidden_dims=[32, 32]
    )

    model = initialize_model(
        encoder_config,
        actor_config,
        critic_config,
        "cuda"
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
