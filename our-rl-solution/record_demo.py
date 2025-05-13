import os
import sys
import pathlib
import random
import argparse
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Add paths for imports
sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld
from utils.state import encode_viewcone, encode_observation
from networks.encoder import StateEncoder
from grid.map import Map
from grid.viz import MapVisualizer
from grid.trajectory import TrajectoryTree
from grid.utils import Point, Action
from grid.node import NodeRegistry, DirectionalNode
from grid.map import Direction

def parse_arguments():
    parser = argparse.ArgumentParser(description='Record simulation demo and create videos')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999),
                        help='Random seed for environment initialization')
    parser.add_argument('--guard', type=int, default=0,
                        help='Guard number (0 for first guard, 1 for second guard, etc.)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Maximum number of steps to simulate')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second in output videos')
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

    return dirs

def normalize_proba_density(proba_density, output_dir):
    """Normalize probability density for visualization and return BGR image"""
    if proba_density is None or np.max(proba_density) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Normalize the probability density to [0, 1]
    normalized = proba_density / np.max(proba_density)

    # Create heatmap using a colormap
    plt.figure(figsize=(5, 5))
    plt.imshow(normalized, cmap='hot')
    plt.axis('off')

    # Save to a temporary file and read it back as an image
    temp_path = os.path.join(output_dir, "temp_heatmap.png")
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the image (OpenCV reads in BGR format)
    heatmap = cv2.imread(temp_path)
    if heatmap is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return heatmap

def resize_preserve_aspect_ratio(image, target_height):
    """Resize image to target height while preserving aspect ratio"""
    h, w = image.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    return cv2.resize(image, (target_width, target_height))

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
    target_height = min(view.shape[0] for view in processed_views)

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

def create_video(frames, output_path, fps=5):
    """Create a video from a list of frames"""
    if not frames:
        print(f"No frames to create video at {output_path}")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

def main():
    # Parse arguments
    args = parse_arguments()

    # Set all seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create output directories
    dirs = create_output_dirs()

    # Initialize map and create trajectory tree
    recon_map = Map()
    recon_map.create_trajectory_tree(Point(0, 0))

    # Initialize environment with specified seed
    env = gridworld.env(
        env_wrappers=[],
        render_mode="rgb_array",
        debug=True,
        novice=False,
    )
    # Reset the environment with seed
    env.reset(seed=seed)

    # Create a numpy random generator with the same seed for action sampling
    np_random = np.random.RandomState(seed)

    # Seed the environment's action space once
    try:
        env.action_space.seed(seed)
    except (AttributeError, TypeError):
        # Some environments might not have this method or it might be structured differently
        pass

    # Get guard agent based on guard number
    guards = [a for a in env.agents if a != env.scout]
    if args.guard >= len(guards):
        print(f"Guard number {args.guard} is out of range. Using guard 0 instead.")
        guard = guards[0]
    else:
        guard = guards[args.guard]

    print(f"Using seed: {args.seed}")
    print(f"Selected guard: {guard}")

    # Run simulation and collect frames
    step = 0
    execution_times = []

    # Create separate lists for each type of frame
    frames_oracle = []
    frames_map = []
    frames_proba = []
    frames_combined = []

    print("Starting simulation and collecting frames...")

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation or step >= args.steps:
            break

        if agent == guard:
            # Record execution time
            start_time = time.time()
            recon_map(observation)

            elapsed_time = time.time() - start_time
            execution_times.append(elapsed_time)

            step += 1
            print(f"Step {step}/{args.steps} - Processing time: {elapsed_time:.6f} seconds")

            # Get the three views
            oracle_view = env.render()  # RGB format
            reconstructed_map = MapVisualizer(recon_map).render(human_mode=False)  # BGR format
            proba_density = recon_map.trees[0].probability_density

            # Normalize and visualize probability density (returns BGR format)
            proba_density_viz = normalize_proba_density(proba_density, dirs['frames_proba'])

            # Save individual frames
            cv2.imwrite(os.path.join(dirs['frames_oracle'], f"frame_{step:04d}.png"),
                        cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dirs['frames_map'], f"frame_{step:04d}.png"),
                        reconstructed_map)
            cv2.imwrite(os.path.join(dirs['frames_proba'], f"frame_{step:04d}.png"),
                        proba_density_viz)

            # Store frames for video creation
            frames_oracle.append(cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR))
            frames_map.append(reconstructed_map)
            frames_proba.append(proba_density_viz)

            # Create and save combined view
            views = [oracle_view, reconstructed_map, proba_density_viz]
            labels = ['Oracle View', 'Reconstructed Map', 'Probability Density']
            combined_frame = combine_views(views, labels)
            frames_combined.append(combined_frame)

            cv2.imwrite(os.path.join(dirs['frames_combined'], f"frame_{step:04d}.png"), combined_frame)

            # Get next action
            location = observation['location']
            direction = observation['direction']
            action = int(recon_map.get_optimal_action(Point(location[0], location[1]), Direction(direction), 0))
        else:
            # Use the seeded numpy random generator for deterministic sampling
            action_space = env.action_space(agent)
            # Sample using the consistent random state without re-seeding
            try:
                # Try with random_state parameter (newer gym versions)
                action = action_space.sample(random_state=np_random)
            except TypeError:
                # For older gym versions, use the already seeded action space
                action = action_space.sample()

        env.step(action)

    env.close()

    # Create videos from collected frames
    print("Creating videos from frames...")

    create_video(frames_oracle, os.path.join(dirs['base'], "oracle_video.mp4"), args.fps)
    create_video(frames_map, os.path.join(dirs['base'], "map_video.mp4"), args.fps)
    create_video(frames_proba, os.path.join(dirs['base'], "probability_video.mp4"), args.fps)
    create_video(frames_combined, os.path.join(dirs['base'], "combined_video.mp4"), args.fps)

    # Report average execution time
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"Average processing time: {avg_time:.6f} seconds over {len(execution_times)} steps")
        print(f"Total frames collected: {step}")
        print("Videos created:")
        print(f"  - Oracle view: {os.path.join(dirs['base'], 'oracle_video.mp4')}")
        print(f"  - Reconstructed map: {os.path.join(dirs['base'], 'map_video.mp4')}")
        print(f"  - Probability density: {os.path.join(dirs['base'], 'probability_video.mp4')}")
        print(f"  - Combined view: {os.path.join(dirs['base'], 'combined_video.mp4')}")

if __name__ == "__main__":
    main()
