import random

import torch
from tensordict.tensordict import TensorDict
import numpy as np
from tqdm import tqdm

from .model_utils import save_checkpoint
from .ppo_update import ppo_update, RewardScaling
from .buffer import PPOExperienceBuffer

from grid.utils import Point, Direction, Action
from grid.node import DirectionalNode

from utils.state import StateManager

from til_environment.types import RewardNames


def run_episode(
    env,
    model,
    optimizer,
    scheduler,
    buffer: PPOExperienceBuffer,
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

    previous_experience = None  # Clear previous step
    model.to(device)

    scout_reward = 0

    steps = 0

    done = False

    local_state_manager = StateManager(args.temporal_frames if args.temporal_state else None)
    global_state_manager = StateManager(args.temporal_frames if args.temporal_state else None)

    while not done:
        observation, reward, termination, truncation, info = env.last()

        done = termination or truncation

        if termination:
            reward = env.rewards[env.scout]

        scout_reward += reward

        if previous_experience is not None:
            previous_experience['reward'] = reward

        # Check if the current agent is done
        if done:
            # If the scout had an unfinished step when the episode ended,
            # finalize it with the final reward/done
            if previous_experience is not None:
                buffer.add(
                    previous_experience,
                    done=True
                )
                previous_experience = None

            break  # End episode
        else:
            previous_action = None
            if previous_experience is not None:
                previous_action = previous_experience['action'].item()

                buffer.add(
                    previous_experience,
                    done=False
                )
                previous_experience = None

        env.maps[env.scout](observation)

        local_state_manager.update(observation, env.maps[env.scout].map)
        global_state_manager.update(observation, env.state().transpose()) # convert [x, y] to [y, x]

        actor_input = local_state_manager[-1]
        critic_input = global_state_manager[-1] if args.global_critic else actor_input

        # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
        if args.bfloat16:
            actor_input = actor_input.to(torch.bfloat16)
            critic_input = critic_input.to(torch.bfloat16)

        actor_input = actor_input.to(device)
        critic_input = critic_input.to(device)

        max_retries = 3
        tries = 0

        location = observation["location"]
        position = Point(int(location[0]), int(location[1]))
        direction = Direction(observation["direction"])
        node: DirectionalNode = env.maps[env.scout].get_node(position, direction)
        valid_actions = set(node.children.keys())
        action = None

        while action is None or (
            action == previous_action
            and previous_action in [Action.LEFT, Action.RIGHT]
            and args.prevent_180_turns
        ) or (
            action not in valid_actions
            and args.prevent_invalid_actions
        ):
            with torch.no_grad():

                action, log_prob, entropy, value = model.get_action_and_value(
                    actor_input.unsqueeze(0),
                    critic_input.unsqueeze(0),
                    deterministic=False
                )

            # Store info for this step to be finalized in the next scout turn
            previous_experience = TensorDict({
                'actor_input': actor_input,
                'critic_input': critic_input,
                'action': action.squeeze(),
                'log_prob': log_prob.squeeze(),
                'value': value.squeeze(),
            })

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

    assert len(training_data), "Buffer empty? This shouldn't happen."

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

def evaluate(env, model, args, device, seed):
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

    # Track individual episode statistics for detailed analysis
    episode_rewards = []
    episode_lengths = []

    # Collect reward metrics by type
    rewards_by_type = {reward_type: 0.0 for reward_type in RewardNames}

    # Run multiple evaluation episodes
    for eval_ep in range(num_episodes):
        # Set number of guards based on spawnrate (align with training)
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

        local_state_manager = StateManager(args.temporal_frames if args.temporal_state else None)
        global_state_manager = StateManager(args.temporal_frames if args.temporal_state else None)

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

            # Update episode reward
            episode_reward += reward

            # Check if the current agent is done
            if done:
                break  # End episode
            else:
                episode_steps += 1

            location = observation["location"]
            position = Point(int(location[0]), int(location[1]))
            direction = Direction(observation["direction"])

            env.maps[env.scout](observation)

            local_state_manager.update(observation, env.maps[env.scout].map)
            global_state_manager.update(observation, env.state().transpose()) # convert [x, y] to [y, x]

            node: DirectionalNode = env.maps[env.scout].get_node(position, direction)
            valid_actions = set(node.children.keys())

            actor_input = local_state_manager[-1]
            critic_input = global_state_manager[-1] if args.global_critic else actor_input
            actor_input = actor_input.unsqueeze(0)
            critic_input = critic_input.unsqueeze(0)

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

        # Track individual episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        total_rewards += episode_reward
        total_steps += episode_steps
        print(f"  Episode {eval_ep+1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

    # Set model back to training mode
    model.train()

    # Calculate comprehensive statistics
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes

    # Convert to numpy arrays for statistical calculations
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)

    # Calculate detailed statistics
    reward_stats = {
        "mean": float(np.mean(rewards_array)),
        "median": float(np.median(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "q25": float(np.percentile(rewards_array, 25)),
        "q75": float(np.percentile(rewards_array, 75))
    }

    length_stats = {
        "mean": float(np.mean(lengths_array)),
        "median": float(np.median(lengths_array)),
        "std": float(np.std(lengths_array)),
        "min": float(np.min(lengths_array)),
        "max": float(np.max(lengths_array)),
        "q25": float(np.percentile(lengths_array, 25)),
        "q75": float(np.percentile(lengths_array, 75))
    }

    # Calculate average rewards by type
    avg_rewards_by_type = {k: v / num_episodes for k, v in rewards_by_type.items()}

    print("Evaluation results:")
    print(f"  Episode Rewards - Mean: {reward_stats['mean']:.2f}, Median: {reward_stats['median']:.2f}, Std: {reward_stats['std']:.2f}")
    print(f"                    Min: {reward_stats['min']:.2f}, Max: {reward_stats['max']:.2f}")
    print(f"                    Q25: {reward_stats['q25']:.2f}, Q75: {reward_stats['q75']:.2f}")
    print(f"  Episode Length  - Mean: {length_stats['mean']:.1f}, Median: {length_stats['median']:.1f}, Std: {length_stats['std']:.1f}")
    print(f"                    Min: {length_stats['min']:.0f}, Max: {length_stats['max']:.0f}")
    print(f"                    Q25: {length_stats['q25']:.1f}, Q75: {length_stats['q75']:.1f}")

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
        "reward_stats": reward_stats,
        "length_stats": length_stats,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "rewards_by_type": avg_rewards_by_type
    }

def train(env, model, optimizer, scheduler, buffer, args):
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
            steps_elapsed, rewards = run_episode(
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
            evaluate(env, model, args, device, args.seed + timesteps_elapsed + 10000)

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
