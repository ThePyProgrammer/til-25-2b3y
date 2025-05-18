import os
import sys
import pathlib
import random
import argparse
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld
from grid.map import Map
from grid.pathfinder import Pathfinder, PathfinderConfig
from utils.profiler import start_profiling, stop_profiling
from grid.utils import Point
from grid.map import Direction
from grid.viz import MapVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a simulation demo with optional recording')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999),
                        help='Random seed for environment initialization')
    parser.add_argument('--guard', type=int, default=0,
                        help='Guard number (0 for first guard, 1 for second guard, etc.)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Maximum number of steps to simulate')
    parser.add_argument('--human', action='store_true',
                        help='Show human view. Either --human or --record only.')
    parser.add_argument('--record', action='store_true',
                        help='Record simulation and create videos')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second in output videos (only used with --record)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling of the code')
    parser.add_argument('--path_density', action='store_true',
                        help='Use path density instead of (random walk) probability density')
    parser.add_argument('--control', type=str, default='all',
                        help='Comma-separated list of agent indices to control (0-based) or "all" for all agents or "none" for no agents')
    parser.add_argument('--scout_target', type=str, default=None,
                        help='Target coordinates for the scout in format "x,y" (e.g., "10,15")')
    return parser.parse_args()

def create_output_dirs():
    dirs = {
        'base': '_output',
        'frames': '_output/frames',
        'frames_oracle': '_output/frames/oracle',
        'frames_map': '_output/frames/map',
        'frames_proba': '_output/frames/proba',
        'frames_combined': '_output/frames/combined',
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

        # Clear existing files in frame directories
        if 'frames' in dir_path:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    return dirs

def normalize_proba_density(proba_density, output_dir):
    """Normalize probability density for visualization and return BGR image with consistent size"""
    # Define a consistent output size
    output_size = (600, 600)

    if proba_density is None or np.max(proba_density) == 0:
        return np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

    # Normalize the probability density to [0, 1]
    normalized = proba_density / np.max(proba_density)

    # Create a colormap
    cmap = plt.get_cmap('hot')

    # Apply colormap to the normalized data (returns RGBA)
    colored_data = cmap(normalized)

    # Convert from RGBA to RGB and then to 8-bit
    rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

    # Resize to consistent dimensions using INTER_NEAREST
    resized_bgr = cv2.resize(bgr_data, output_size, interpolation=cv2.INTER_NEAREST)

    # Draw a border around the image to make it more visible
    border_size = 2
    cv2.rectangle(resized_bgr, (0, 0), (output_size[0]-1, output_size[1]-1), (255, 255, 255), border_size)

    return resized_bgr

def resize_preserve_aspect_ratio(image, target_height):
    """Resize image to target height while preserving aspect ratio"""
    h, w = image.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

def combine_views(views, labels):
    """Combine multiple views horizontally with labels, preserving aspect ratio"""
    # Convert all views to BGR if they're in RGB
    processed_views = []
    for view in views:
        if len(view.shape) == 3 and view.shape[2] == 3:
            # Check if it's in RGB format (a heuristic)
            processed_view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        else:
            processed_view = view
        processed_views.append(processed_view)

    # Find common height (use the smallest height)
    target_height = max(view.shape[0] for view in processed_views)

    # Resize all views to have the same height while preserving aspect ratio
    resized_views = [resize_preserve_aspect_ratio(view, target_height) for view in processed_views]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    labeled_views = []
    for i, view in enumerate(resized_views):
        labeled_view = view.copy()
        cv2.putText(labeled_view, labels[i], (10, 20), font, font_scale, color, thickness)
        labeled_views.append(labeled_view)

    # Combine horizontally
    combined = np.hstack(labeled_views)
    return combined

def combine_views_grid(agent_data, oracle_view):
    """
    Combine views into a layout with:
    - Oracle view in the center (2 wide, 2 tall)
    - Agent 0 and 2 stacked on the left (each with map and probability side by side)
    - Agent 1 and 3 stacked on the right (each with map and probability side by side)

    Args:
        agent_data: Dictionary mapping agent indices (0-3) to their respective data
                   (map and probability density views)
        oracle_view: The global oracle view of the environment

    Returns:
        Combined frame with oracle view in center and agent views in quadrants
    """
    map_size = (400, 400)  # Size for map/probability views
    oracle_size = (800, 800)  # Size for oracle view (2x2)

    oracle_view = cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR)

    # Add label to oracle view
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 1

    # Process agent views (map and probability side by side)
    agent_rows = {}  # Will hold rows for agents 0&2 and 1&3

    for agent_idx in range(4):
        # Get data for this agent
        data = agent_data.get(agent_idx, {})
        map_view = data.get('map')
        proba_view = data.get('proba')

        # Create blank images if views are missing
        if map_view is None:
            map_view = np.zeros((400, 400, 3), dtype=np.uint8)
        if proba_view is None:
            proba_view = np.zeros((400, 400, 3), dtype=np.uint8)

        # Convert to BGR if in RGB format
        # if len(map_view.shape) == 3 and map_view.shape[2] == 3:
        #     if np.mean(map_view[:,:,0]) < np.mean(map_view[:,:,2]):  # Heuristic RGB check
        map_view = cv2.cvtColor(map_view, cv2.COLOR_RGB2BGR)

        # if len(proba_view.shape) == 3 and proba_view.shape[2] == 3:
        #     if np.mean(proba_view[:,:,0]) < np.mean(proba_view[:,:,2]):  # Heuristic RGB check
        proba_view = cv2.cvtColor(proba_view, cv2.COLOR_RGB2BGR)

        # Resize views to consistent size
        map_view_resized = cv2.resize(map_view, map_size, interpolation=cv2.INTER_NEAREST)
        proba_view_resized = cv2.resize(proba_view, map_size, interpolation=cv2.INTER_NEAREST)

        # Add labels to views
        cv2.putText(map_view_resized, f"Agent {agent_idx} - Map", (10, 30), font, font_scale*0.8, color, thickness)
        cv2.putText(proba_view_resized, f"Agent {agent_idx} - Probability", (10, 30), font, font_scale*0.8, color, thickness)

        # Put map and probability side by side
        agent_row = np.hstack((map_view_resized, proba_view_resized))

        # Group by column (agents 0,2 on left, agents 1,3 on right)
        col = agent_idx % 2  # 0 for left, 1 for right
        row_key = f"col_{col}"

        if row_key not in agent_rows:
            agent_rows[row_key] = []
        agent_rows[row_key].append(agent_row)

    # Stack agents 0&2 vertically, and 1&3 vertically
    left_column = np.vstack(agent_rows.get("col_0", [np.zeros((800, 800, 3), dtype=np.uint8)]))
    right_column = np.vstack(agent_rows.get("col_1", [np.zeros((800, 800, 3), dtype=np.uint8)]))

    # Ensure columns are the correct height (same as oracle)
    if left_column.shape[0] != oracle_size[1]:
        padding = np.zeros((oracle_size[1] - left_column.shape[0], left_column.shape[1], 3), dtype=np.uint8)
        left_column = np.vstack([left_column, padding])
    if right_column.shape[0] != oracle_size[1]:
        padding = np.zeros((oracle_size[1] - right_column.shape[0], right_column.shape[1], 3), dtype=np.uint8)
        right_column = np.vstack([right_column, padding])

    h, w = oracle_view.shape[:2]
    aspect = w / h
    target_height = right_column.shape[0]
    target_width = int(target_height * aspect)

    oracle_view_resized = cv2.resize(oracle_view, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    cv2.putText(oracle_view_resized, "Oracle View", (20, 40), font, font_scale, color, thickness)

    combined = np.hstack((left_column, oracle_view_resized, right_column))

    return combined

def create_video(frames, output_path, fps=5):
    """Create a video from a list of frames prioritizing MoviePy"""
    if not frames:
        print(f"No frames to create video at {output_path}")
        return

    height, width = frames[0].shape[:2]

    # Ensure all frames have the same size
    processed_frames = []
    for frame in frames:
        if frame is not None:
            # If frame size doesn't match the expected size, resize it
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            processed_frames.append(frame)

    if not processed_frames:
        print(f"No valid frames to create video at {output_path}")
        return

    # Make sure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = f"{output_path}.mp4"

    # Convert BGR (OpenCV) to RGB (MoviePy) if needed
    rgb_frames = []
    for frame in processed_frames:
        if frame is not None:
            # Assume BGR format from OpenCV, convert to RGB for MoviePy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)

    # Create clip from frames
    clip = ImageSequenceClip(rgb_frames, fps=fps)

    # Write video file
    clip.write_videofile(output_path, codec="libx264", fps=fps)
    print(f"Video saved to {output_path} using MoviePy")

def main(args):
    if args.profile:
        start_profiling()

    # Set all seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories if recording
    dirs = create_output_dirs() if args.record else None

    # Initialize maps and pathfinders for all agents
    agent_maps = {}
    agent_pathfinders = {}

    # Initialize environment with specified seed
    env = gridworld.env(
        env_wrappers=[],
        render_mode="rgb_array" if args.record else "human" if args.human else None,  # Only need rendering if recording
        debug=True,
        novice=False,
    )
    # Reset the environment with seed
    env.reset(seed=seed)

    # Get guard agent based on guard number
    guards = [a for a in env.agents if a != env.scout]
    if args.guard >= len(guards):
        print(f"Guard number {args.guard} is out of range. Using guard 0 instead.")
        guard = guards[0]
    else:
        guard = guards[args.guard]

    # Determine which agents to control
    controlled_agents = set()
    if args.control.lower() == 'all':
        # Control all agents (excluding scout)
        controlled_agents = set(guards)
    elif args.control.lower() != 'none':
        # Parse the comma-separated list of agent indices
        try:
            control_indices = [int(idx.strip()) for idx in args.control.split(',')]
            for idx in control_indices:
                if 0 <= idx < len(guards):
                    controlled_agents.add(guards[idx])
                else:
                    print(f"Warning: Invalid agent index: {idx}, ignoring.")
        except ValueError:
            print(f"Warning: Invalid control format: {args.control}, defaulting to controlling all agents.")
            controlled_agents = set(guards)

    # Parse scout target if provided
    scout_target = None
    if args.scout_target:
        try:
            x, y = map(int, args.scout_target.split(','))
            scout_target = Point(x, y)
            print(f"Scout target: {scout_target}")
        except ValueError:
            print(f"Warning: Invalid scout target format: {args.scout_target}, scout will move randomly")

    print(f"Controlled agents: {controlled_agents}")

    # Initialize maps and pathfinders for controlled guards only
    for agent in env.agents:
        if agent in controlled_agents:
            agent_maps[agent] = Map()
            agent_maps[agent].create_trajectory_tree(Point(0, 0))
            agent_pathfinders[agent] = Pathfinder(
                agent_maps[agent],
                PathfinderConfig(
                    use_viewcone=False,
                    use_path_density=args.path_density
                )
            )

    # Create a map and pathfinder for the scout if a target is provided
    if scout_target is not None:
        scout_agent = env.scout
        agent_maps[scout_agent] = Map()
        # Scout doesn't need a trajectory tree, just basic pathfinding
        agent_pathfinders[scout_agent] = Pathfinder(
            agent_maps[scout_agent],
            PathfinderConfig(
                use_viewcone=False,
                use_path_density=False
            )
        )

    print(f"Using seed: {args.seed}")
    print(f"Selected guard: {guard}")
    print(f"Created maps and pathfinders for {len(controlled_agents)} controlled guards: {controlled_agents}")
    print(f"Random movement agents: {[agent for agent in env.agents if agent not in controlled_agents and agent != env.scout]}")
    print(f"Scout agent: {env.scout}" + (f" (targeting {scout_target})" if scout_target else " (random movement)"))

    # Variables for recording if enabled
    # Store frames for each agent in separate dictionaries
    agent_frames = {agent: {'oracle': [], 'map': [], 'proba': []} for agent in env.agents}
    # Create a mapping from agent to grid position index (0-3)
    agent_grid_positions = {}
    for i, agent in enumerate(env.agents[:4]):  # Limit to 4 agents for the 2x2 grid
        agent_grid_positions[agent] = i

    frames_combined = []  # Combined view of all agents in 2x2 grid

    # Run simulation
    step = 0
    execution_times = []

    print("Starting simulation..." + (" and collecting frames..." if args.record else ""))

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation or step >= args.steps:
            break

        # Record execution time
        start_time = time.time()

        # Get next action
        location = observation['location']
        direction = observation['direction']

        if agent == "player_2":
            target_node = agent_maps["player_2"].registry.get_or_create_node(Point(9, 5), Direction.RIGHT)
            print(target_node)
            print(target_node.children)
            if len([traj for traj in agent_maps["player_2"].trees[0].trajectories if target_node in traj]) < 10:
                print("\n".join([str(traj._inherits_to) for traj in agent_maps["player_2"].trees[0].trajectories if target_node in traj]))

        if termination or truncation:
            action = None
        elif agent in controlled_agents:
            # Update the controlled agent's map with its observation
            print(f"reconstructing and predicting for {agent}")
            agent_maps[agent](observation)

            # Use the agent's pathfinder to determine action
            action = int(agent_pathfinders[agent].get_optimal_action(
                Point(location[0], location[1]),
                Direction(direction),
                0
            ))
        elif agent == env.scout and scout_target is not None:
            # Update scout's map with observation
            if agent in agent_maps:
                agent_maps[agent](observation)

            # Use pathfinder to navigate to target
            action = int(agent_pathfinders[agent].get_optimal_action(
                Point(location[0], location[1]),
                Direction(direction),
                destination=scout_target  # Pass the target coordinates
            ))
        else:
            # Random action for non-controlled agents
            action = random.choice([0, 1, 2, 3])

        elapsed_time = time.time() - start_time

        # Only count execution time for the selected guard
        if agent == guard:
            execution_times.append(elapsed_time)
            step += 1
            print(f"Step {step}/{args.steps} - Processing time: {elapsed_time:.6f} seconds - Agent: {agent}")

            # Recording logic if enabled - record for all agents on each step
            if args.record and dirs:
                # Get the oracle view (environment state)
                oracle_view = env.render()  # RGB format

                # Create dictionary to store views for each agent position (0-3)
                grid_agent_data = {}

                # For combined frames tracking
                combined_frames_dir = os.path.join(dirs['frames_combined'])
                os.makedirs(combined_frames_dir, exist_ok=True)

                # For each agent (both scout and guards), create their view for the grid
                for current_agent in env.agents:
                    # Skip if agent is not one of the first 4 (for 2x2 grid)
                    if current_agent not in agent_grid_positions:
                        continue

                    grid_position = agent_grid_positions[current_agent]
                    grid_agent_data[grid_position] = {'map': None, 'proba': None}

                    # For scout with target, show its map and blank probability
                    if current_agent == env.scout and scout_target is not None and current_agent in agent_maps:
                        # Get the map view
                        map_view = MapVisualizer(agent_maps[current_agent]).render(human_mode=False)  # BGR format

                        # Store frames for scout
                        agent_frames[current_agent]['oracle'].append(cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR))
                        agent_frames[current_agent]['map'].append(map_view)

                        # Add views to grid
                        grid_agent_data[grid_position]['map'] = map_view
                        grid_agent_data[grid_position]['proba'] = np.zeros_like(map_view)  # Blank probability density

                    # For non-controlled agents, we only have the oracle view
                    elif current_agent not in controlled_agents:
                        agent_view = cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR)
                        agent_frames[current_agent]['oracle'].append(agent_view)

                        # Add agent view to grid (oracle for both slots)
                        grid_agent_data[grid_position]['map'] = agent_view
                        grid_agent_data[grid_position]['proba'] = np.zeros_like(agent_view)  # Blank probability density

                    # For controlled agents, we use their maps and probability for the grid
                    elif current_agent in agent_maps:
                        # Get the map view
                        map_view = MapVisualizer(agent_maps[current_agent]).render(human_mode=False)  # BGR format

                        # Get probability density
                        proba_density = agent_maps[current_agent].trees[0].probability_density
                        proba_density_viz = normalize_proba_density(proba_density, dirs['frames_proba'])

                        # Store frames for this agent
                        agent_frames[current_agent]['oracle'].append(cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR))
                        agent_frames[current_agent]['map'].append(map_view)
                        agent_frames[current_agent]['proba'].append(proba_density_viz)

                        # Add both views to grid
                        grid_agent_data[grid_position]['map'] = map_view
                        grid_agent_data[grid_position]['proba'] = proba_density_viz

                # Create and save combined view with oracle in center and agents in quadrants
                combined_frame = combine_views_grid(grid_agent_data, oracle_view)
                frames_combined.append(combined_frame)
                cv2.imwrite(os.path.join(combined_frames_dir, f"frame_{step:04d}.png"), combined_frame)

        env.step(action)

    env.close()

    if args.profile:
        stop_profiling(sort_by=1, lines=50)

    # Create videos if recording was enabled
    if args.record and dirs:
        print("Creating videos from frames...")

        # Create combined video of all agents' perspectives
        # Create combined video with proper directory path
        create_video(frames_combined, os.path.join(dirs['base'], "all_agents_combined.mp4"), args.fps)

        # Create individual videos for each agent
        # Create agent-specific directories for individual videos
        for agent in env.agents:
            agent_dir = os.path.join(dirs['base'], f"agent_{agent}")
            os.makedirs(agent_dir, exist_ok=True)

            # Create oracle view video for all agents
            if 'oracle' in agent_frames[agent] and agent_frames[agent]['oracle']:
                create_video(agent_frames[agent]['oracle'],
                            os.path.join(agent_dir, f"{agent}_oracle.mp4"), args.fps)

            # Create additional videos for controlled agents and scout with target
            if agent in agent_maps:
                if 'map' in agent_frames[agent] and agent_frames[agent]['map']:
                    create_video(agent_frames[agent]['map'],
                                os.path.join(agent_dir, f"{agent}_map.mp4"), args.fps)

                # Only create probability videos for guards, not scout
                if agent in controlled_agents and 'proba' in agent_frames[agent] and agent_frames[agent]['proba']:
                    create_video(agent_frames[agent]['proba'],
                                os.path.join(agent_dir, f"{agent}_probability.mp4"), args.fps)

        print("Videos created:")
        print(f"  - Combined view of all agents: {os.path.join(dirs['base'], 'all_agents_combined.mp4')}")
        print("  - Individual agent videos in directories:")
        for agent in env.agents:
            agent_dir = os.path.join(dirs['base'], f'agent_{agent}')
            if os.path.exists(agent_dir):
                print(f"    - {agent}: {agent_dir}")

    # Report average execution time
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"Average processing time: {avg_time:.6f} seconds over {len(execution_times)} steps")
        print(f"Total steps completed: {step}")

if __name__ == "__main__":
    args = parse_arguments()

    try:
        main(args)
    except KeyboardInterrupt:
        pass

    print(f"Seed {args.seed}")
