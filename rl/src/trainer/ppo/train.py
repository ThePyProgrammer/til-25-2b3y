import os
import sys
import pathlib

import torch

sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

# Import utility modules
from trainer.ppo.utils.args import parse_args
from trainer.ppo.utils.environment import setup_environment, set_seeds, get_scout_initial_observation, extract_observation_shape
from trainer.ppo.utils.agent_utils import init_agents
from trainer.ppo.utils.model_utils import initialize_model, initialize_optimizer, load_checkpoint, apply_model_precision
from trainer.ppo.utils.training_loop import train
from trainer.ppo.utils.buffer import ExperienceBuffer
from trainer.ppo.utils.scheduler import create_scheduler

# Main function definition below


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    set_seeds(args.seed)

    # Set up environment
    env = setup_environment(args)

    # Initialize agents
    agents = init_agents(env, args.num_guards)

    # Get initial observation to determine input shape
    scout_initial_observation = get_scout_initial_observation(env)

    # Update scout's map with the initial observation to determine tensor shape
    agents['maps']['scout'](scout_initial_observation)
    dummy_map_tensor = agents['maps']['scout'].get_tensor()

    # Extract observation shape information
    CHANNELS, MAP_SIZE, ACTION_DIM = extract_observation_shape(dummy_map_tensor)
    print(f"Detected Map size: {MAP_SIZE}, Channels: {CHANNELS}, Action Dim: {ACTION_DIM}")

    # Initialize model
    model = initialize_model(
        action_dim=ACTION_DIM,
        map_size=MAP_SIZE,
        channels=CHANNELS,
        encoder_type="large",
        shared_encoder=False,
        device=device
    )

    # Apply precision setting
    model = apply_model_precision(model, args.bfloat16)

    # Initialize optimizer
    optimizer = initialize_optimizer(model, args.optim, args.lr)

    # Create learning rate scheduler
    scheduler = create_scheduler(optimizer, args, total_steps=args.timesteps)

    # Resume training if requested
    start_timesteps = 0
    if args.resume:
        checkpoint_path = os.path.join(args.save_dir, 'latest.pt')
        start_timesteps, success = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # Initialize experience buffer
    buffer = ExperienceBuffer()

    # Run training loop
    train(env, agents, model, optimizer, scheduler, buffer, args)

    # Close environment
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
