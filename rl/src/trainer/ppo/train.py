import os
import sys
import pathlib
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from networks.ppo import PPOActorCritic
from grid.map import Map
from grid.utils import Point
from til_environment import gridworld
from trainer.ppo.utils.args import parse_args # Import parse_args

from grid.map import Map
from grid.pathfinder import Pathfinder, PathfinderConfig
from utils.profiler import start_profiling, stop_profiling
from grid.utils import Point
from grid.map import Direction
from grid.viz import MapVisualizer # May not be needed for training, but keep for now
from trainer.ppo.utils.buffer import ExperienceBuffer
from trainer.ppo.utils.ppo_update import ppo_update # Import the ppo_update function
from trainer.ppo.utils.scheduler import create_scheduler # Import create_scheduler

def main(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True # Might slow down training

    # Set default tensor type if bfloat16 is requested
    if args.bfloat16:
        torch.set_default_dtype(torch.bfloat16)
        print("Using bfloat16 for training.")


    env = gridworld.env(
        env_wrappers=[],  # clear out default env wrappers
        render_mode="human" if args.render else None,  # Render the map if requested
        debug=True,  # Enable debug mode
        novice=True,  # Use same map layout every time (for Novice teams only)
    )
    # Reset the environment with seed
    env.reset(seed=args.seed)

    # Initialize scout's map and guard maps/pathfinders for the new episode
    scout_map = Map()
    guard_maps = {}
    guard_pathfinders = {}

    # Identify guard agents
    guards = [a for a in env.agents if a != env.scout]

    # Initialize maps and pathfinders for guards
    for guard_agent in guards:
        guard_maps[guard_agent] = Map()
        # Guards might need trajectory trees or specific pathfinder configs
        # Based on demo.py, using basic pathfinding without viewcone or density for simplicity
        guard_pathfinders[guard_agent] = Pathfinder(
            guard_maps[guard_agent],
            PathfinderConfig(
                use_viewcone=False,
                use_path_density=False # Can be made configurable later
            )
        )

    # Get initial observation to determine tensor shapes
    # We need to iterate through agents until we get scout's first observation after reset
    # The first observation is available via env.last() for the agent whose turn it is first.
    # We need the scout's observation to initialize its map dimensions correctly.
    # Let's reset and then iterate until we see the scout.

    # env.reset(seed=args.seed + timesteps_elapsed) # Already reset above
    scout_initial_observation = None

    # Need to iterate through agents once to get the scout's initial observation
    # This is a bit awkward with the pettingzoo API, might need a better way
    # For simplicity now, let's assume the first agent is the scout or we can find it.
    # A more robust way would be to loop until scout's turn after reset.
    # For this implementation, let's force a step until scout's turn to get its initial observation
    # Or, simpler, just get the observation for the first agent and assume it's sufficient to build the map once.
    # Let's try getting the observation after reset and update the map with that.
    # The observation structure can vary per agent. Let's assume we can get the scout's observation
    # from the first agent's turn and update the map. This might not be correct.

    # A better approach: Find the scout's turn after reset and get its observation
    # The first observation is available via env.last() for the agent whose turn it is first.
    # We need the scout's observation to initialize its map dimensions correctly.
    # Let's reset and then iterate until we see the scout.

    env.reset(seed=args.seed)
    scout_initial_observation = None
    # Iterate agents until we get scout's first observation after reset
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if agent == env.scout:
            scout_initial_observation = obs
            break # Found scout's first observation

        # Step other agents to get to scout's turn
        # This might interfere with training, but is needed to get the scout's observation
        # In a real training setup, you might handle the initial observation differently
        # or collect full trajectories before training.
        # For this example, we just need the shape once.
        env.step(env.action_space(agent).sample()) # Step other agents randomly

    if scout_initial_observation is None:
        raise RuntimeError("Could not get initial observation for the scout after reset.")

    # Update scout's map with the initial observation to determine tensor shape
    scout_map(scout_initial_observation)
    dummy_map_tensor = scout_map.get_tensor()
    print(f"Scout map tensor shape: {dummy_map_tensor.shape}")

    # Determine MAP_SIZE and CHANNELS from the map tensor shape
    # The tensor shape is (C, H, W) or (C, Size, Size)
    if len(dummy_map_tensor.shape) != 3:
        raise RuntimeError(f"Expected map tensor shape (C, H, W), but got {dummy_map_tensor.shape}")

    CHANNELS, MAP_SIZE, _ = dummy_map_tensor.shape
    if MAP_SIZE != dummy_map_tensor.shape[2]:
         print(f"Warning: Map tensor is not square. Assuming MAP_SIZE = {MAP_SIZE}")

    # Assuming Action_DIM is 4 for movement (Up, Down, Left, Right)
    ACTION_DIM = 4
    print(f"Detected Map size: {MAP_SIZE}, Channels: {CHANNELS}, Action Dim: {ACTION_DIM}")


    # Initialize PPO Model and Optimizer
    model = PPOActorCritic(
        action_dim=ACTION_DIM,
        map_size=MAP_SIZE,
        channels=CHANNELS,
        encoder_type="small" # Using "small" encoder
    ).to(device) # Move model to device

    # Select Optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        # This should not happen due to argparse choices, but as a fallback
        print(f"Warning: Unknown optimizer '{args.optim}'. Using Adam.")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create learning rate scheduler
    # Pass total_steps=args.timesteps to schedule over the entire training duration
    scheduler = create_scheduler(optimizer, args, total_steps=args.timesteps)

    # Resume training if requested
    start_timesteps = 0
    if args.resume:
        checkpoint_path = os.path.join(args.save_dir, 'latest.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Load scheduler state if it exists in the checkpoint
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 print("Loaded scheduler state.")
            start_timesteps = checkpoint['timesteps_elapsed']
            print(f"Resumed training from {checkpoint_path} at timestep {start_timesteps}")
        else:
            print(f"Warning: Resume requested but checkpoint not found at {checkpoint_path}. Starting from scratch.")


    # Experience collection buffer
    buffer = ExperienceBuffer()
    last_scout_step_info = None # Stores {S_t tensors, A_t, log_prob_t, V_t}


    # Outer loop for total timesteps
    timesteps_elapsed = start_timesteps
    while timesteps_elapsed < args.timesteps:
        # print(f"Timesteps elapsed: {timesteps_elapsed}/{args.timesteps}") # Too verbose


        # Reset environment if starting a new batch/episode, or if previous episode ended
        # In this per-episode training loop, we reset at the start of each episode
        print("Resetting environment.")
        env.reset(seed=args.seed + timesteps_elapsed) # Vary seed for new episodes
        # The first observation will be available via env.last() in the first agent's turn.
        last_scout_step_info = None # Clear previous step info
        # Re-initialize scout's map for the new episode
        scout_map = Map()

        # Clear the buffer at the start of each episode since we train per episode
        buffer.clear()


        episode_ended = False
        # Inner loop for one episode
        for agent in env.agent_iter():
            # Get observation, reward, etc. for the current agent's turn
            # This observation is the state *before* this agent acts.
            # reward/term/trunc are for this agent from the *previous* step.
            observation, reward, termination, truncation, info = env.last()

            # Check if the current agent is done. If so, episode ends for all in this simplified setup.
            if termination or truncation:
                episode_ended = True
                # If the scout had an unfinished step when another agent terminated or episode truncated,
                # finalize the last scout step with assumed reward/done.
                if last_scout_step_info is not None:
                    buffer.add(
                        map_input=last_scout_step_info['map_input'],
                        step_input=last_scout_step_info['step_input'],
                        action=last_scout_step_info['action'],
                        log_prob=last_scout_step_info['log_prob'],
                        value=last_scout_step_info['value'],
                        reward=0.0, # Assuming 0 reward on early termination/truncation
                        done=True # Episode ended
                    )
                    last_scout_step_info = None # Clear for next episode
                break # End agent_iter loop (episode)


            # Only process for the scout agent
            if agent == env.scout:
                # --- Finalize previous scout step (if exists) using info from env.last() ---
                # observation is S_t
                # reward, termination, truncation are R_{t-1}, done_{t-1} for the scout's previous step
                if last_scout_step_info is not None:
                    # S_{t-1}, A_{t-1}, log_prob_{t-1}, V_{t-1} are in last_scout_step_info
                    # R_{t-1}, done_{t-1} are available now from env.last()
                    buffer.add(
                        map_input=last_scout_step_info['map_input'],
                        step_input=last_scout_step_info['step_input'],
                        action=last_scout_step_info['action'],
                        log_prob=last_scout_step_info['log_prob'],
                        value=last_scout_step_info['value'],
                        reward=float(reward), # Reward for the step that ended just before this turn
                        done=termination or truncation # Done for the step that ended just before this turn
                    )
                    last_scout_step_info = None # Clear as step is finalized

                    # Check if the scout just finished its previous step
                    if termination or truncation:
                        episode_ended = True
                        break # End agent_iter loop


                # --- Collect state, action, log_prob, value for the current step (S_t, A_t, log_prob_t, V_t) ---
                # The current 'observation' is S_t
                # Use scout's map to get tensor representation
                scout_map(observation) # Update map with observation
                map_input = scout_map.get_tensor().unsqueeze(0) # Get tensor and add batch dim
                step_input = torch.tensor([observation['step']], dtype=torch.float32).unsqueeze(0) # Get step and add batch dim

                # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
                if args.bfloat16:
                    map_input = map_input.to(torch.bfloat16)
                    step_input = step_input.to(torch.bfloat16)


                # Get action (A_t), log_prob, value (V(S_t)) from the model
                with torch.no_grad():
                    action, log_prob, entropy, value = model.get_action_and_value(map_input, step_input, greedy=False)

                # Store info for this step to be finalized in the next scout turn
                last_scout_step_info = {
                    'map_input': map_input, # Store the tensor directly
                    'step_input': step_input, # Store the tensor directly
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                }

                # Perform the action A_t
                env.step(action.item())
                timesteps_elapsed += 1

                # Check if we have collected enough steps for a training batch
                # For simplicity, let's train after each episode for now.
                # The PPO update will happen after the agent_iter loop breaks.
                if timesteps_elapsed >= args.timesteps:
                    break # End outer loop if total timesteps reached mid-episode


                elif agent in guard_maps: # It's a guard agent
                    # Update the guard's map with its observation
                    # print(f"Updating map for guard {agent}") # Debugging
                    guard_maps[agent](observation)

                    # Get location and direction from observation
                    location = observation.get('location')
                    direction = observation.get('direction')

                    # Use the guard's pathfinder to determine action
                    action = 0 # Default action if pathfinder fails

                    if location is not None and direction is not None:
                        try:
                            # Pass location and direction as Point and Direction enums
                            action = int(guard_pathfinders[agent].get_optimal_action(
                                Point(location[0], location[1]),
                                Direction(direction)
                                # No destination needed for default guard behavior? Or should they patrol?
                                # Assuming default pathfinder logic finds a valid move
                            ))
                        except Exception as e:
                             print(f"Error getting action for guard {agent}: {e}")
                             action = env.action_space(agent).sample() # Fallback to random

                    # Perform the action
                    env.step(action)


                else: # Other agents (if any, not scout or controlled guards)
                    # Take a random action for other agents not explicitly controlled
                    random_action = env.action_space(agent).sample()
                    env.step(random_action)

                # Check if outer loop limit reached after any agent's step
                if timesteps_elapsed >= args.timesteps:
                    break # End agent_iter loop


            # After agent_iter loop finishes (episode ended or truncated)
            # Finalize the very last step taken by the scout if it wasn't finalized by termination within the loop.
            if last_scout_step_info is not None:
                buffer.add(
                    map_input=last_scout_step_info['map_input'],
                    step_input=last_scout_step_info['step_input'],
                    action=last_scout_step_info['action'],
                    log_prob=last_scout_step_info['log_prob'],
                    value=last_scout_step_info['value'],
                    reward=0.0, # Assuming 0 reward for the last step at episode end
                    done=True # Episode ended
                )
                last_scout_step_info = None # Clear for next episode


            # --- PPO Update ---
            # Perform training update if we have collected any scout steps in this episode
            if not buffer.is_empty(): # Check if buffer has data
                print(f"Collected {len(buffer)} scout steps in this episode. Performing PPO update.")

                # Get batch from buffer
                training_data = buffer.get_batch()

                assert training_data, "buffer empty? how?"

                # Move training data to device
                for k, v in training_data.items():
                    if isinstance(v, torch.Tensor):
                        training_data[k] = v.to(device)

                # Perform PPO update
                # Pass necessary arguments: model, optimizer, training_data, args
                # The ppo_update function will handle GAE, returns, epochs, minibatches, loss calculation, and optimization steps.
                update_losses = ppo_update(model, optimizer, training_data, args)

                # Step the learning rate scheduler if it exists
                if scheduler:
                    # Assuming stepping based on the number of environment steps processed in this update
                    scheduler.step(len(buffer)) # Use buffer length BEFORE clearing

                # Log losses
                print(f"Timestep {timesteps_elapsed}: Policy Loss = {update_losses['policy_loss']:.4f}, Value Loss = {update_losses['value_loss']:.4f}, Entropy = {update_losses['entropy_bonus']:.4f}, Total Loss = {update_losses['total_loss']:.4f}")

                # Clear batch after training (or collect multiple episodes before training)
                buffer.clear()

                # Perform evaluation periodically
                # Evaluate every X timesteps or at the end of training
                EVAL_INTERVAL = 50000 # Evaluate every 50,000 timesteps (example interval)
                if timesteps_elapsed > start_timesteps and timesteps_elapsed % EVAL_INTERVAL < len(training_data) or timesteps_elapsed >= args.timesteps: # Check interval non-overlappingly and at the end
                    evaluate_policy(model, env) # Run evaluation


                # Save checkpoint
                if timesteps_elapsed % 10000 == 0 or timesteps_elapsed >= args.timesteps: # Save periodically or at the end
                    checkpoint_path = os.path.join(args.save_dir, 'latest.pt')
                    torch.save({
                        'timesteps_elapsed': timesteps_elapsed,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, # Save scheduler state
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")


            # Check if total timesteps reached (redundant check due to break inside loop, but harmless)
            # if timesteps_elapsed >= args.timesteps:
            #     pass # Exit main loop


    print("Training finished.")
    env.close()

def evaluate_policy(model: torch.nn.Module, env, num_episodes: int = 10, seed: Optional[int] = None):
    """
    Evaluates the policy for a given number of episodes.

    Args:
        model: The PPOActorCritic model.
        env: The environment.
        num_episodes: Number of episodes to run for evaluation.
        seed: Optional seed for evaluation episodes.

    Returns:
        Average cumulative reward over the evaluation episodes.
    """
    # Set model to evaluation mode
    model.eval()

    total_rewards = []

    print(f"\nStarting policy evaluation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Reset environment for evaluation episode
        # Use a separate seed or offset the training seed
        current_seed = seed if seed is not None else 1000000 + episode # Use high seed to avoid overlap
        env.reset(seed=current_seed)

        scout_map = Map()
        guard_maps = {}
        guard_pathfinders = {}

        guards = [a for a in env.agents if a != env.scout]

        for guard_agent in guards:
            guard_maps[guard_agent] = Map()
            guard_pathfinders[guard_agent] = Pathfinder(
                guard_maps[guard_agent],
                PathfinderConfig(
                    use_viewcone=False,
                    use_path_density=False
                )
            )

        episode_reward = 0
        episode_ended = False

        # Iterate through agents in the episode
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                episode_ended = True
                break

            # Only get action for scout or guards
            if agent == env.scout:
                # Update scout's map and get tensor
                scout_map(observation)
                map_input = scout_map.get_tensor().unsqueeze(0)
                step_input = torch.tensor([observation['step']], dtype=torch.float32).unsqueeze(0)

                # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
                # Assuming model.dtype is the correct dtype for inputs
                model_dtype = next(model.parameters()).dtype
                if map_input.dtype != model_dtype:
                     map_input = map_input.to(model_dtype)
                if step_input.dtype != model_dtype:
                     step_input = step_input.to(model_dtype)


                # Get action from the model using greedy policy (no gradients)
                with torch.no_grad():
                     # Use greedy=True for evaluation
                     action, _, _, _ = model.get_action_and_value(map_input, step_input, greedy=True)

                # Step environment with scout action
                env.step(action.item())
                # Accumulate reward received by the scout (reward is from previous step)
                # The reward here is for the action taken in the *previous* scout turn
                # Need to correctly attribute reward to the step that earned it.
                # For simplicity in evaluation, let's accumulate the reward from env.last()
                # This might not be perfectly aligned with the action that earned it,
                # but gives a general sense of performance.
                # A more precise way involves storing reward with the state it resulted from.
                # Let's use the reward from env.last() for the agent whose turn it is.
                # We need the reward *given to the scout* after *its* action.
                # This reward appears in env.last() *on the next agent's turn*.
                # A simpler approach for evaluation is to just sum up all rewards received by the scout
                # throughout the episode. However, the pettingzoo API makes this tricky per agent within the loop.
                # The total episode reward is probably the most straightforward metric.
                # Let's assume the 'reward' from env.last() when it's the scout's turn is the reward
                # the scout received for its *previous* action.
                episode_reward += reward # Accumulate reward for the scout


            elif agent in guard_maps: # It's a guard agent
                # Update the guard's map with its observation
                guard_maps[agent](observation)

                # Get location and direction from observation
                location = observation.get('location')
                direction = observation.get('direction')

                # Use the guard's pathfinder to determine action
                action = 0 # Default action if pathfinder fails

                if location is not None and direction is not None:
                    try:
                        action = int(guard_pathfinders[agent].get_optimal_action(
                            Point(location[0], location[1]),
                            Direction(direction)
                        ))
                    except Exception as e:
                        print(f"Error getting action for guard {agent} during evaluation: {e}")
                        action = env.action_space(agent).sample() # Fallback to random

                # Perform the action
                env.step(action)

            else: # Other agents (if any)
                # Take a random action for other agents
                random_action = env.action_space(agent).sample()
                env.step(random_action)

            # Note: Cumulative reward calculation might be tricky with multi-agent turns.
            # The pettingzoo docs mention `env.rewards` to get total rewards per agent at the end.
            # Let's use the total episode reward for the scout from `env.rewards` after the loop.


        # After the episode loop, get the scout's total reward for this episode
        if env.scout in env.rewards:
            episode_reward = env.rewards[env.scout]
            total_rewards.append(episode_reward)
        else:
            # This might happen if the scout never got a turn or if rewards are not tracked this way
            print(f"Warning: Could not retrieve total reward for scout in episode {episode}.")
            # If no specific scout reward is available, maybe sum up all rewards? Or skip this episode?
            # Let's assume for now env.rewards[env.scout] is the correct way to get it.
            # If not, the total_rewards list might be empty or incorrect.
            pass # Skip adding if scout reward not found


        print(f" Episode {episode + 1}/{num_episodes} finished with reward: {episode_reward}")


    # Calculate average reward
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

    print(f"Evaluation finished. Average reward over {num_episodes} episodes: {avg_reward:.4f}\n")

    # Set model back to training mode
    model.train()

    return avg_reward


if __name__ == "__main__":
    args = parse_args()
    main(args)
