import os
import sys
import pathlib
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from networks.ppo import PPOActorCritic
from grid.map import Map
from grid.utils import Point
from til_environment import gridworld
from til_environment.types import RewardNames

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


NEW_REWARDS_DICT = {
    RewardNames.GUARD_CAPTURES: 1,
    RewardNames.SCOUT_CAPTURED: -1,
    RewardNames.SCOUT_RECON: 0.02,
    RewardNames.SCOUT_MISSION: 0.1,
    RewardNames.WALL_COLLISION: -0.05,
    RewardNames.SCOUT_TRUNCATION: 1,
    RewardNames.STATIONARY_PENALTY: -0.05
}


def init_agents(env, num_guards):
    scout_map = Map()
    guard_maps = {}
    guard_pathfinders = {}

    # Identify guard agents
    guards = [a for a in env.agents if a != env.scout]
    random.shuffle(guards)
    guards = guards[:num_guards]
    
    # Initialize maps and pathfinders for guards
    for agent in guards:
        guard_maps[agent] = Map()
        guard_maps[agent].create_trajectory_tree(Point(0, 0))
        guard_pathfinders[agent] = Pathfinder(
            guard_maps[agent],
            PathfinderConfig(
                use_viewcone=False,
                use_path_density=False
            )
        )

    return {
        'names': {
            'scout': env.scout,
            'guards': guards,
        },
        'maps': {
            'scout': scout_map,
            'guards': guard_maps,
        },
        'pathfinders': {
            'guards': guard_pathfinders
        }
    }


def main(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    # Set default tensor type if bfloat16 is requested
    if args.bfloat16:
        torch.set_default_dtype(torch.bfloat16)
        print("Using bfloat16 for training.")
    
    env = gridworld.env(
        env_wrappers=[],  # clear out default env wrappers
        render_mode="human" if args.render else None,  # Render the map if requested
        debug=False,  # Enable debug mode
        novice=False,  # Use same map layout every time (for Novice teams only)
        rewards_dict=NEW_REWARDS_DICT
    )
    # Reset the environment with seed
    env.reset(seed=args.seed)

    agents = init_agents(env, args.num_guards)

    scout_initial_observation = None
    # Iterate agents until we get scout's first observation after reset
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if agent == env.scout:
            scout_initial_observation = obs
            break # Found scout's first observation

        # Step other agents to get to scout's turn
        env.step(env.action_space(agent).sample()) # Step other agents randomly

    if scout_initial_observation is None:
        raise RuntimeError("Could not get initial observation for the scout after reset.")

    # Update scout's map with the initial observation to determine tensor shape
    agents['maps']['scout'](scout_initial_observation)
    dummy_map_tensor = agents['maps']['scout'].get_tensor()
    print(f"Scout map tensor shape: {dummy_map_tensor.shape}")

    # Determine MAP_SIZE and CHANNELS from the map tensor shape
    # The tensor shape is (C, H, W)
    if len(dummy_map_tensor.shape) != 3:
        raise RuntimeError(f"Expected map tensor shape (C, H, W), but got {dummy_map_tensor.shape}")

    CHANNELS, MAP_SIZE, _ = dummy_map_tensor.shape
    if MAP_SIZE != dummy_map_tensor.shape[2]:
         print(f"Warning: Map tensor is not square. Assuming MAP_SIZE = {MAP_SIZE}")

    # Assuming Action_DIM is 5 for movement (Foward, Backward, Turn left, Turn right, Stay)
    ACTION_DIM = 5
    print(f"Detected Map size: {MAP_SIZE}, Channels: {CHANNELS}, Action Dim: {ACTION_DIM}")


    # Initialize PPO Model and Optimizer
    model = PPOActorCritic(
        action_dim=ACTION_DIM,
        map_size=MAP_SIZE,
        channels=CHANNELS,
        encoder_type="large"
    )

    model.cuda()

    if args.bfloat16:
        model.to(torch.bfloat16)

    # Select Optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
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
        # Reset environment if episode ended
        print("Resetting environment.")
        env.reset(seed=args.seed + timesteps_elapsed) # Vary seed for new episodes
        last_scout_step_info = None # Clear previous step
        agents = init_agents(env, args.num_guards)

        # Clear the buffer at the start of each episode since we train per episode
        buffer.clear()
        scout_reward = 0


        episode_ended = False
        # Inner loop for one episode
        for agent in env.agent_iter():
            # Get observation, reward, etc. for the current agent's turn
            # This observation is the state *before* this agent acts.
            # reward/term/trunc are for this agent from the *previous* step.
            observation, reward, termination, truncation, info = env.last()
            
            # Check if the current agent is done. If so, episode ends for all in this simplified setup.
            if termination or truncation:
                reward = env.rewards[env.scout]
                scout_reward += reward
                episode_ended = True
                # If the scout had an unfinished step when another agent terminated or episode truncated,
                # finalize the last scout step with assumed reward/done.
                if last_scout_step_info is not None:
                    scout_reward += scout_reward
                    buffer.add(
                        map_input=last_scout_step_info['map_input'],
                        action=last_scout_step_info['action'],
                        log_prob=last_scout_step_info['log_prob'],
                        value=last_scout_step_info['value'],
                        reward=reward,
                        done=True # Episode ended
                    )
                    last_scout_step_info = None # Clear for next episode
                break # End agent_iter loop (episode)

            # Only process for the scout agent
            if agent == env.scout:
                scout_reward += reward
                # --- Finalize previous scout step (if exists) using info from env.last() ---
                # observation is S_t
                # reward, termination, truncation are R_{t-1}, done_{t-1} for the scout's previous step
                if last_scout_step_info is not None:
                    # S_{t-1}, A_{t-1}, log_prob_{t-1}, V_{t-1} are in last_scout_step_info
                    # R_{t-1}, done_{t-1} are available now from env.last()
                    buffer.add(
                        map_input=last_scout_step_info['map_input'],
                        action=last_scout_step_info['action'],
                        log_prob=last_scout_step_info['log_prob'],
                        value=last_scout_step_info['value'],
                        reward=reward, # Reward for the step that ended just before this turn
                        done=termination or truncation # Done for the step that ended just before this turn
                    )
                    last_scout_step_info = None # Clear as step is finalized

                    if termination or truncation:
                        episode_ended = True
                        break # End agent_iter loop

                # --- Collect state, action, log_prob, value for the current step (S_t, A_t, log_prob_t, V_t) ---
                # The current 'observation' is S_t
                # Use scout's map to get tensor representation
                agents['maps']['scout'](observation) # Update map with observation
                map_input = agents['maps']['scout'].get_tensor().unsqueeze(0) # Get tensor and add batch dim

                # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
                if args.bfloat16:
                    map_input = map_input.to(torch.bfloat16)

                map_input = map_input.cuda()

                # Get action (A_t), log_prob, value (V(S_t)) from the model
                with torch.no_grad():
                    action, log_prob, entropy, value = model.get_action_and_value(map_input, greedy=False)

                # Store info for this step to be finalized in the next scout turn
                last_scout_step_info = {
                    'map_input': map_input, # Store the tensor directly
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                }

                # Perform the action A_t
                env.step(action.item())
                timesteps_elapsed += 1

                # The PPO update will happen after the agent_iter loop breaks.
                if timesteps_elapsed >= args.timesteps:
                    break # End outer loop if total timesteps reached mid-episode


            elif agent in agents['names']['guards']: # It's a guard agent
                # Update the guard's map with its observation
                agents['maps']['guards'][agent](observation)

                # Get location and direction from observation
                location = observation.get('location')
                direction = observation.get('direction')

                # Use the guard's pathfinder to determine action
                action = 0 # Default action if pathfinder fails

                if location is not None and direction is not None:
                    try:
                        # Pass location and direction as Point and Direction enums
                        action = int(agents['pathfinders']['guards'][agent].get_optimal_action(
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
            # if last_scout_step_info is not None:
            #     buffer.add(
            #         map_input=last_scout_step_info['map_input'],
            #         action=last_scout_step_info['action'],
            #         log_prob=last_scout_step_info['log_prob'],
            #         value=last_scout_step_info['value'],
            #         reward=reward,
            #         done=True # Episode ended
            #     )
            #     last_scout_step_info = None # Clear for next episode


        # --- PPO Update ---
        # Perform training update if we have collected any scout steps in this episode
        if not buffer.is_empty(): # Check if buffer has data
            print(f"Collected {len(buffer)} scout steps in this episode with a reward of {scout_reward}. Performing PPO update.")

            # Get batch from buffer
            training_data = buffer.get_batch()

            assert training_data, "buffer empty? how?"

            # Move training data to device
            for k, v in training_data.items():
                if isinstance(v, torch.Tensor):
                    training_data[k] = v.to(device)

            # Perform PPO update
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
            EVAL_INTERVAL = 50000 # Evaluate every 50,000 timesteps (example interval)
            if timesteps_elapsed > start_timesteps and timesteps_elapsed % EVAL_INTERVAL < len(training_data) or timesteps_elapsed >= args.timesteps:
                evaluate_policy(model, env) # Run evaluation


            # Save checkpoint
            if timesteps_elapsed % 5000 == 0 or timesteps_elapsed >= args.timesteps:
                checkpoint_path = os.path.join(args.save_dir, 'latest.pt')
                torch.save({
                    'timesteps_elapsed': timesteps_elapsed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None, # Save scheduler state
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

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

        agents = init_agents(env)

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
                agents['maps']['scout'](observation)
                map_input = agents['maps']['scout'].get_tensor().unsqueeze(0)

                # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
                # Assuming model.dtype is the correct dtype for inputs
                model_dtype = next(model.parameters()).dtype
                if map_input.dtype != model_dtype:
                     map_input = map_input.to(model_dtype)


                # Get action from the model using greedy policy (no gradients)
                with torch.no_grad():
                     # Use greedy=True for evaluation
                     action, _, _, _ = model.get_action_and_value(map_input, greedy=True)

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
            # Warning if scout reward not tracked or available.
            print(f"Warning: Could not retrieve total reward for scout in episode {episode}.")
            pass


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
