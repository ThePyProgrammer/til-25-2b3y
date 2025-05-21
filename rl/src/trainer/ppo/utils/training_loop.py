import threading

import torch

from .agent_utils import process_scout_step, process_guard_step, process_other_agents, init_agents
from .model_utils import save_checkpoint
from .ppo_update import ppo_update


class TimeoutError(Exception):
    pass

def timeout(timeout_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result
            result = None
            exception = None

            # Define the thread function
            def worker():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e

            # Create and start the thread
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()

            # Wait for completion or timeout
            thread.join(timeout_seconds)
            if thread.is_alive():
                # Thread is still running after timeout
                raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

            # If there was an exception in the thread, raise it
            if exception:
                raise exception

            return result
        return wrapper
    return decorator

@timeout(10)  # Set timeout to 10 seconds
def reset_environment(env, seed):
    return env.reset(seed=seed)

def train_episode(
    env,
    model,
    optimizer,
    scheduler,
    buffer,
    args,
    device,
    last_scout_step_info,
    timesteps_elapsed,
    steps_since_save
):
    """
    Run a single training episode

    Args:
        env: Gridworld environment
        agents: Dictionary containing agent information
        model: PPO model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        buffer: Experience buffer
        args: Command line arguments
        device: Compute device (CPU/CUDA)
        last_scout_step_info: Info about the last scout step
        timesteps_elapsed: Current timestep count
        steps_since_save: Steps since the last checkpoint save

    Returns:
        tuple: (updated last_scout_step_info, updated timesteps_elapsed,
                updated steps_since_save, updated episodes_since_update,
                scout_reward, whether training should continue)
    """
    seed = args.seed + timesteps_elapsed

    # Reset environment with a varying seed
    while True:
        try:
            reset_environment(env, seed)
            break
        except TimeoutError as _:
            seed += 1

    last_scout_step_info = None  # Clear previous step
    agents = init_agents(env, args.num_guards)
    model.to(device)

    # Clear the buffer at the start of each episode
    buffer.clear()
    scout_reward = 0

    steps = 0

    # Inner loop for one episode
    for agent in env.agent_iter():
        # Get observation, reward, etc. for the current agent's turn
        observation, reward, termination, truncation, info = env.last()

        # Check if the current agent is done
        if termination or truncation:
            reward = env.rewards[env.scout]
            scout_reward += reward

            # If the scout had an unfinished step when the episode ended,
            # finalize it with the final reward/done
            if last_scout_step_info is not None:
                buffer.add(
                    map_input=last_scout_step_info['map_input'],
                    action=last_scout_step_info['action'],
                    log_prob=last_scout_step_info['log_prob'],
                    value=last_scout_step_info['value'],
                    reward=reward,
                    done=True  # Episode ended
                )
                last_scout_step_info = None
            break  # End episode

        # Process based on agent type
        if agent == env.scout:
            scout_reward += reward

            # Process scout step
            last_scout_step_info, action = process_scout_step(
                agent, observation, reward, termination, truncation,
                agents, model, device, buffer, last_scout_step_info, args
            )

            # Perform the action
            env.step(action)
            steps += 1

            # End if we've reached the maximum timesteps
            if timesteps_elapsed >= args.timesteps:
                should_continue = False
                break

        elif agent in agents['names']['guards']:
            # Process guard step
            action = process_guard_step(agent, observation, agents, env, args)
            env.step(action)

        else:
            # Process other agents
            action = process_other_agents(agent, env)
            env.step(action)

    # Print episode summary
    print(f"Collected {len(buffer)} scout steps in this episode with a reward of {scout_reward}.")

    return (
        last_scout_step_info,
        scout_reward,
        steps
    )

def update_model(buffer, model, optimizer, scheduler, args, device):
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
    update_losses = ppo_update(model, optimizer, training_data, args)

    # Step the learning rate scheduler if it exists
    if scheduler:
        scheduler.step(len(buffer))

    return update_losses

def evaluate(env, model, args, device, current_timestep):
    """
    Evaluate the model over multiple episodes without training

    Args:
        env: Gridworld environment
        model: PPO model
        args: Command line arguments
        device: Compute device (CPU/CUDA)
        current_timestep: Current timestep count for seed generation

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()

    total_rewards = 0
    total_episode_length = 0
    num_episodes = 8

    print(f"\nEvaluating model at timestep {current_timestep}...")

    # Run multiple evaluation episodes
    for eval_ep in range(num_episodes):
        # Reset environment with a varying seed
        env.reset(seed=args.seed + current_timestep + eval_ep + 10000)  # Offset to avoid training seeds
        agents = init_agents(env, args.num_guards)
        episode_reward = 0
        episode_length = 0

        # Run one episode
        for agent in env.agent_iter():
            # Get observation, reward, etc. for the current agent's turn
            observation, reward, termination, truncation, info = env.last()

            # Check if the current agent is done
            if termination or truncation:
                reward = env.rewards[env.scout]
                episode_reward += reward
                break

            # Process based on agent type
            if agent == env.scout:
                episode_reward += reward
                episode_length += 1

                agents['maps']['scout'](observation)
                map_input = agents['maps']['scout'].get_tensor().unsqueeze(0)

                if args.bfloat16:
                    map_input = map_input.to(torch.bfloat16)

                map_input = map_input.to(device)

                # Process scout action without storing in buffer
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(map_input, deterministic=True)

                action = action.item()

                env.step(action)

            elif agent in agents['names']['guards']:
                # Process guard step
                action = process_guard_step(agent, observation, agents, env, args)
                env.step(action)

            else:
                # Process other agents
                action = process_other_agents(agent, env)
                env.step(action)

        # Track episode statistics
        total_rewards += episode_reward
        total_episode_length += episode_length

    # Set model back to training mode
    model.train()

    # Calculate average metrics
    avg_reward = total_rewards / num_episodes
    avg_episode_length = total_episode_length / num_episodes

    print("Evaluation results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average episode length: {avg_episode_length:.2f}")

    return {
        "avg_reward": avg_reward,
        "avg_episode_length": avg_episode_length
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
    last_scout_step_info = None
    steps_since_save = 0
    steps_since_eval = 0
    episodes_since_update = 0
    episodes_in_buffer = 0
    timesteps_elapsed = 0  # Will be overridden if resuming

    # Main training loop
    while timesteps_elapsed < args.timesteps:
        # Run one episode
        (
            last_scout_step_info,
            scout_reward,
            steps_elapsed
        ) = train_episode(
            env,
            model,
            optimizer,
            scheduler,
            buffer,
            args,
            device,
            last_scout_step_info,
            timesteps_elapsed,
            steps_since_save
        )

        # Update step counters
        # Calculate the steps taken in this episode
        timesteps_elapsed += steps_elapsed
        steps_since_eval += steps_elapsed
        steps_since_save += steps_elapsed
        episodes_since_update += 1
        episodes_in_buffer += 1

        # Run evaluation if needed
        if steps_since_eval >= args.eval_interval:
            steps_since_eval = 0
            evaluate(env, model, args, device, timesteps_elapsed)
            # Here you could log these metrics to a file or tracking system if needed

        # Perform PPO update if enough episodes have been collected
        if not buffer.is_empty() and episodes_since_update >= args.episodes_per_update:
            episodes_since_update = 0

            print("Performing PPO update.")
            update_losses = update_model(buffer, model, optimizer, scheduler, args, device)

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
