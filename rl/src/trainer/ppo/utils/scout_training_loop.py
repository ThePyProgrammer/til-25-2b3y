import random

import torch
import numpy as np
from tqdm import tqdm

from .model_utils import save_checkpoint
from .ppo_update import ppo_update, RewardScaling

from grid.utils import Point, Direction, Action
from grid.map import tiles_to_tensor, map_to_tiles
from grid.node import DirectionalNode

from til_environment.types import RewardNames


def train_scout_episode(
    env,
    model,
    optimizer,
    scheduler,
    buffer,
    args,
    device,
    seed = None
) -> tuple[int, float]:
    """
    Run a single training episode for scout

    Args:
        env: Gridworld environment
        agents: Dictionary containing agent information
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        buffer: Experience buffer
        args: Command line arguments
        device: Compute device (CPU/CUDA)
        seed: seed

    """
    if seed is None:
        seed = random.randint(0, 999999)

    num_guards = 0
    for i in range(args.num_guards):
        if random.random() < args.guards_spawnrate:
            num_guards += 1
    env.set_num_active_guards(num_guards)
    env.reset(seed=seed)

    last_scout_step_info = None  # Clear previous step
    model.to(device)

    scout_reward = 0

    steps = 0

    done = False

    if args.temporal_state:
        actor_input = torch.zeros((1, 12, args.temporal_frames, 31, 31))
        critic_input = torch.zeros((1, 12, args.temporal_frames, 31, 31))

    # Inner loop for one episode
    while not done:
        # Get observation, reward, etc. for the current agent's turn
        observation, reward, termination, truncation, info = env.last()

        done = termination or truncation
        if termination:
            reward = env.rewards[env.scout]

        scout_reward += reward

        # Check if the current agent is done
        if done:
            # If the scout had an unfinished step when the episode ended,
            # finalize it with the final reward/done
            if last_scout_step_info is not None:
                buffer.add(
                    actor_input=last_scout_step_info['actor_input'],
                    critic_input=last_scout_step_info['critic_input'],
                    action=last_scout_step_info['action'],
                    log_prob=last_scout_step_info['log_prob'],
                    value=last_scout_step_info['value'],
                    reward=reward,
                    done=True  # Episode ended
                )
                last_scout_step_info = None

            break  # End episode
        else:
            previous_action = None
            if last_scout_step_info is not None:
                previous_action = last_scout_step_info['action']
                # S_{t-1}, A_{t-1}, log_prob_{t-1}, V_{t-1} are in last_scout_step_info
                # R_{t-1}, done_{t-1} are available now from env.last()
                buffer.add(
                    actor_input=last_scout_step_info['actor_input'],
                    critic_input=last_scout_step_info['critic_input'],
                    action=last_scout_step_info['action'],
                    log_prob=last_scout_step_info['log_prob'],
                    value=last_scout_step_info['value'],
                    reward=reward,  # Reward for the step that ended just before this turn
                    done=termination or truncation  # Done for the step that ended just before this turn
                )
                last_scout_step_info = None  # Clear as step is finalized

        location = observation["location"]
        position = Point(int(location[0]), int(location[1]))
        direction = Direction(observation["direction"])
        env.maps[env.scout](observation)

        map_state = env.maps[env.scout].get_tensor().unsqueeze(0)  # Get tensor and add batch dim

        if args.temporal_state:
            for i in range(args.temporal_frames - 1):
                actor_input[:, :, i] = actor_input[:, :, i+1]
            actor_input[:, :, args.temporal_frames-1] = map_state
        else:
            actor_input = map_state

        global_state = tiles_to_tensor(
            map_to_tiles(env.state().transpose()),
            location,
            direction,
            16,
            np.zeros((16, 16)),
            env.maps[env.scout].step_counter
        ).unsqueeze(0)

        node: DirectionalNode = env.maps[env.scout].get_node(position, direction)
        valid_actions = set(node.children.keys())

        if args.global_critic:
            if args.temporal_state:
                for i in range(args.temporal_frames - 1):
                    critic_input[:, :, i] = critic_input[:, :, i+1]
                critic_input[:, :, args.temporal_frames-1] = global_state
            else:
                critic_input = global_state
        else:
            critic_input = actor_input

        # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
        if args.bfloat16:
            actor_input = actor_input.to(torch.bfloat16)
            critic_input = critic_input.to(torch.bfloat16)

        actor_input = actor_input.to(device)
        critic_input = critic_input.to(device)

        # Get action (A_t), log_prob, value (V(S_t)) from the model
        action = None

        max_retries = 3
        tries = 0

        while action is None or (
            action == previous_action
            and previous_action in [Action.LEFT, Action.RIGHT]
            and args.prevent_180_turns
        ) or (
            action not in valid_actions
            and args.prevent_invalid_actions
        ):
            with torch.no_grad():

                action, log_prob, entropy, value = model.get_action_and_value(actor_input, critic_input, deterministic=False)

            # Store info for this step to be finalized in the next scout turn
            last_scout_step_info = {
                'actor_input': actor_input,
                'critic_input': critic_input,
                'action': action.item(),
                'log_prob': log_prob.item(),
                'value': value.item(),
            }

            action = action.item()

            tries += 1

            if tries >= max_retries:
                break

        env.step(action)
        steps += 1

    # Print episode summary
    # print(f"Collected {len(buffer)} scout steps in this episode with a reward of {scout_reward:.1f}.")

    return steps, scout_reward

def update_model(buffer, model, optimizer, scheduler, reward_scaler, args, device):
    """
    Update the model using PPO

    Args:
        buffer: Experience buffer
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Command line arguments
        device: Compute device (CPU/CUDA)

    Returns:
        dict: Dictionary containing loss values
    """
    model.to(device)

    # Get batch from buffer
    training_data = buffer.get_batch()

    assert training_data, "Buffer empty? This shouldn't happen."

    # Move training data to device
    for k, v in training_data.items():
        if isinstance(v, torch.Tensor):
            training_data[k] = v.to(device)

    # Perform PPO update
    update_losses = ppo_update(model, optimizer, training_data, reward_scaler, args)

    # Step the learning rate scheduler if it exists
    if scheduler:
        scheduler.step(len(buffer))

    return update_losses

def evaluate_scout(env, model, args, device, seed):
    """
    Evaluate the model over multiple episodes without training

    Args:
        env: Gridworld environment
        model: PPO model
        args: Command line arguments
        device: Compute device (CPU/CUDA)
        seed: Random seed for evaluation episodes

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    total_rewards = 0
    total_steps = 0
    num_episodes = 32

    # Collect reward metrics by type
    rewards_by_type = {reward_type: 0.0 for reward_type in RewardNames}

    # Run multiple evaluation episodes
    for eval_ep in range(num_episodes):
        # Set number of guards based on spawnrate
        num_guards = 0
        for i in range(args.num_guards):
            if random.random() < args.guards_spawnrate:
                num_guards += 1
        env.set_num_active_guards(num_guards)

        # Reset environment with a varying seed
        env.reset(seed=seed + eval_ep)

        episode_reward = 0
        episode_steps = 0
        previous_action = None
        done = False

        if args.temporal_state:
            actor_input = torch.zeros((1, 12, args.temporal_frames, 31, 31))
            critic_input = torch.zeros((1, 12, args.temporal_frames, 31, 31))

        # Run one episode
        while not done:
            # Get observation, reward, etc. for the current agent's turn
            observation, reward, termination, truncation, info = env.last()

            done = termination or truncation
            if termination:
                reward = env.rewards[env.scout]

                # Track reward types if available in info
                if 'reward_breakdown' in info:
                    for reward_type, value in info['reward_breakdown'].items():
                        if reward_type in rewards_by_type:
                            rewards_by_type[reward_type] += value

            # Check if the current agent is done
            if done:
                episode_reward += reward
            else:
                episode_reward += reward
                episode_steps += 1

            location = observation["location"]
            position = Point(int(location[0]), int(location[1]))
            direction = Direction(observation["direction"])

            env.maps[env.scout](observation)

            map_state = env.maps[env.scout].get_tensor().unsqueeze(0)  # Get tensor and add batch dim

            if args.temporal_state:
                for i in range(args.temporal_frames - 1):
                    actor_input[:, :, i] = actor_input[:, :, i+1]
                actor_input[:, :, args.temporal_frames-1] = map_state
            else:
                actor_input = map_state

            global_state = tiles_to_tensor(
                map_to_tiles(env.state().transpose()),
                location,
                direction,
                16,
                np.zeros((16, 16)),
                env.maps[env.scout].step_counter
            ).unsqueeze(0)

            node: DirectionalNode = env.maps[env.scout].get_node(position, direction)
            valid_actions = set(node.children.keys())

            if args.global_critic:
                if args.temporal_state:
                    for i in range(args.temporal_frames - 1):
                        critic_input[:, :, i] = critic_input[:, :, i+1]
                    critic_input[:, :, args.temporal_frames-1] = global_state
                else:
                    critic_input = global_state
            else:
                critic_input = actor_input

            # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
            if args.bfloat16:
                actor_input = actor_input.to(torch.bfloat16)
                critic_input = critic_input.to(torch.bfloat16)

            actor_input = actor_input.to(device)
            critic_input = critic_input.to(device)

            # Get action from the model
            action = None
            max_retries = 3
            tries = 0

            while action is None or (
                action == previous_action
                and previous_action in [Action.LEFT, Action.RIGHT]
                and args.prevent_180_turns
            ) or (
                action not in valid_actions
                and args.prevent_invalid_actions
            ):
                with torch.no_grad():
                    action, log_prob, entropy, value = model.get_action_and_value(actor_input, critic_input, deterministic=True)

                action = action.item()
                tries += 1

                if tries >= max_retries:
                    break

            previous_action = action
            env.step(action)

        # Track episode statistics
        total_rewards += episode_reward
        total_steps += episode_steps
        print(f"  Episode {eval_ep+1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

    # Set model back to training mode
    model.train()

    # Calculate average metrics
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes

    # Calculate average rewards by type
    avg_rewards_by_type = {k: v / num_episodes for k, v in rewards_by_type.items()}

    print("Evaluation results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average episode steps: {avg_steps:.2f}")

    # Print reward breakdown if available
    if any(avg_rewards_by_type.values()):
        print("  Reward breakdown:")
        for reward_type, value in avg_rewards_by_type.items():
            if value != 0:
                print(f"    {reward_type}: {value:.2f}")

    return {
        "avg_reward": avg_reward,
        "avg_episode_length": avg_steps,
        "total_rewards": total_rewards,
        "total_steps": total_steps,
        "rewards_by_type": avg_rewards_by_type,
        "success_rate": success_rate
    }

def train_scout(env, model, optimizer, scheduler, buffer, args):
    """
    Main training loop

    Args:
        env: Gridworld environment
        agents: Dictionary containing agent information
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        buffer: Experience buffer
        args: Command line arguments

    Returns:
        tuple: (model, optimizer, scheduler, timesteps_elapsed)
    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tracking variables
    rewards_since_update = 0
    steps_since_save = 0
    steps_since_eval = 0
    steps_since_update = 0
    episodes_since_update = 0
    episodes_in_buffer = 0
    timesteps_elapsed = 0  # Will be overridden if resuming

    reward_scaler = RewardScaling()
    reward_scaler.to(device)

    # Main training loop
    while timesteps_elapsed < args.timesteps:

        for e in tqdm(range(args.episodes_per_update), desc="Collecting episodes"):
            steps_elapsed, rewards = train_scout_episode(
                env,
                model,
                optimizer,
                scheduler,
                buffer,
                args,
                device,
            )

            rewards_since_update += rewards
            timesteps_elapsed += steps_elapsed
            steps_since_eval += steps_elapsed
            steps_since_save += steps_elapsed
            steps_since_update += steps_elapsed
            episodes_since_update += 1
            episodes_in_buffer += 1

        if steps_since_eval >= args.eval_interval:
            steps_since_eval = 0
            evaluate_scout(env, model, args, device, args.seed + timesteps_elapsed + 10000)

        # Perform PPO update if enough episodes have been collected
        if not buffer.is_empty() and episodes_since_update >= args.episodes_per_update:

            print(f"Performing PPO update with \n\t{episodes_since_update} episodes\n\t{steps_since_update} steps\n\t{(rewards_since_update / episodes_since_update):.2f} average episode reward")
            update_losses = update_model(buffer, model, optimizer, scheduler, reward_scaler, args, device)

            rewards_since_update = 0
            episodes_since_update = 0
            steps_since_update = 0

            # Log losses
            print(f"Timestep {timesteps_elapsed}: "
                 f"Policy Loss = {update_losses['policy_loss']:.4f}, "
                 f"Value Loss = {update_losses['value_loss']:.4f}, "
                 f"Entropy = {update_losses['entropy_bonus']:.4f}, "
                 f"Total Loss = {update_losses['total_loss']:.4f}")

            if episodes_in_buffer >= args.episodes_in_buffer:
                episodes_in_buffer = 0
                # Clear buffer after training
                buffer.clear()

            # Save checkpoint if needed
            if steps_since_save > args.save_interval or timesteps_elapsed >= args.timesteps:
                steps_since_save = 0
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    timesteps_elapsed,
                    args.save_dir,
                    args.experiment_name
                )
                print(f"Checkpoint saved at {checkpoint_path}")

        # Break if we've reached the timestep limit
        if timesteps_elapsed > args.timesteps:
            break

    print("Training finished.")

    return model, optimizer, scheduler, timesteps_elapsed
