import random

import torch
import numpy as np

from grid.map import Map, map_to_tiles, tiles_to_tensor
from grid.node import DirectionalNode
from grid.utils import Point, Action, Direction
from grid.pathfinder import Pathfinder, PathfinderConfig


def init_agents(env, num_guards):
    """
    Initialize scout and guard agents with maps and pathfinders

    Args:
        env: Gridworld environment
        num_guards: Number of guard agents to initialize

    Returns:
        dict: Dictionary containing agent information, maps, and pathfinders
    """
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

def process_scout_step(
    agent,
    observation,
    reward,
    termination,
    truncation,
    agents,
    model,
    device,
    buffer,
    last_scout_step_info,
    env,
    args,
):
    """
    Process a step for the scout agent

    Args:
        agent: Agent name/id
        observation: Current observation
        reward: Current reward
        termination: Whether episode terminated
        truncation: Whether episode truncated
        agents: Dictionary containing agent information
        model: PPO model
        device: Compute device (CPU/CUDA)
        buffer: Experience buffer
        last_scout_step_info: Info about the last scout step
        env: train env
        args: Command line arguments

    Returns:
        tuple: (updated last_scout_step_info, timestep increment)
    """
    # --- Finalize previous scout step (if exists) using info from env.last() ---
    # observation is S_t
    # reward, termination, truncation are R_{t-1}, done_{t-1} for the scout's previous step
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

    # --- Collect state, action, log_prob, value for the current step (S_t, A_t, log_prob_t, V_t) ---
    # The current 'observation' is S_t
    # Use scout's map to get tensor representation
    agents['maps']['scout'](observation)  # Update map with observation

    location = observation["location"]
    position = Point(int(location[0]), int(location[1]))
    direction = Direction(observation["direction"])
    
    map_input = agents['maps']['scout'].get_tensor().unsqueeze(0)  # Get tensor and add batch dim
    global_input = tiles_to_tensor(
        map_to_tiles(env.state().transpose()),
        location,
        direction,
        16,
        np.zeros((16, 16)),
        agents['maps']['scout'].step_counter
    ).unsqueeze(0)

    node: DirectionalNode = agents['maps']['scout'].get_node(position, direction)
    valid_actions = set(node.children.keys())

    if args.global_critic:
        critic_input = global_input
    else:
        critic_input = map_input

    # Ensure tensor dtype matches model dtype (bfloat16 if enabled)
    if args.bfloat16:
        map_input = map_input.to(torch.bfloat16)
        critic_input = critic_input.to(torch.bfloat16)

    map_input = map_input.to(device)
    critic_input = critic_input.to(device)

    # Get action (A_t), log_prob, value (V(S_t)) from the model
    action = None
    new_last_scout_step_info = {}

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

            action, log_prob, entropy, value = model.get_action_and_value(map_input, critic_input, deterministic=False)

        # Store info for this step to be finalized in the next scout turn
        new_last_scout_step_info = {
            'actor_input': map_input,
            'critic_input': critic_input,
            'action': action.item(),
            'log_prob': log_prob.item(),
            'value': value.item(),
        }

        action = action.item()

        tries += 1

        if tries >= max_retries:
            break

    assert new_last_scout_step_info

    return new_last_scout_step_info, action

def process_guard_step(agent, observation, agents, env, args):
    """
    Process a step for a guard agent

    Args:
        agent: Agent name/id
        observation: Current observation
        agents: Dictionary containing agent information
        env: Gridworld environment

    Returns:
        int: Action to take
    """
    # Update the guard's map with its observation
    agents['maps']['guards'][agent](observation)

    # Get location and direction from observation
    location = observation.get('location')
    direction = observation.get('direction')

    # Use the guard's pathfinder to determine action
    action = 0  # Default action if pathfinder fails

    if location is not None and direction is not None:
        try:
            if random.random() < args.guards_difficulty:
                # Pass location and direction as Point and Direction enums
                action = int(agents['pathfinders']['guards'][agent].get_optimal_action(
                    Point(location[0], location[1]),
                    Direction(direction)
                ))
            else:
                action = env.action_space(agent).sample()  # Fallback to random
        except Exception as e:
            print(f"Error getting action for guard {agent}: {e}")
            action = env.action_space(agent).sample()  # Fallback to random

    return action

def process_other_agents(agent, env):
    """
    Process a step for non-scout, non-guard agents

    Args:
        agent: Agent name/id
        env: Gridworld environment

    Returns:
        int: Action to take
    """
    # Take a random action for other agents not explicitly controlled
    return env.action_space(agent).sample()
