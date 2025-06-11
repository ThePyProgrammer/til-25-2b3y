import os
import sys
import pathlib
import random
import time
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import numpy as np
import cv2

sys.path.append(
    str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment")
)
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld
from grid.map import Map
from grid.pathfinder import Pathfinder, PathfinderConfig
from utils.profiler import start_profiling, stop_profiling
from grid.utils import Point
from grid.map import Direction
from grid.viz import MapVisualizer

from utils.demo.visualization import normalize_proba_density, combine_views
from utils.demo.recording import create_output_dirs, create_video, save_frame


class BaseDemo:
    """Base class for demo scripts with common functionality."""

    def __init__(self, args):
        """Initialize the demo with command-line arguments."""
        self.args = args
        self.seed = args.seed
        self.guard_idx = args.guard
        self.max_steps = args.steps
        self.record = args.record
        self.human = args.human
        self.fps = args.fps if hasattr(args, "fps") else 5
        self.profile = args.profile

        # Will be set during initialization
        self.env = None
        self.guard = None
        self.guards = []
        self.scout = None
        self.controlled_agents = set()
        self.agent_maps = {}
        self.agent_pathfinders = {}
        self.dirs = None

        # For recording
        self.frames = {"oracle": [], "map": [], "proba": [], "combined": []}
        self.agent_frames = {}
        self.step = 0
        self.execution_times = []

    def setup(self):
        """Set up the environment and agents."""
        if self.profile:
            start_profiling()

        # Set all seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create output directories if recording
        self.dirs = create_output_dirs() if self.record else None

        # Initialize environment with specified seed
        self.env = gridworld.env(
            env_wrappers=[],
            render_mode="rgb_array" if self.record else "human" if self.human else None,
            debug=True,
            novice=False,
        )
        # Reset the environment with seed
        self.env.reset(seed=self.seed)

        # Get guard agent based on guard number
        self.guards = [a for a in self.env.agents if a != self.env.scout]
        if self.guard_idx >= len(self.guards):
            print(
                f"Guard number {self.guard_idx} is out of range. Using guard 0 instead."
            )
            self.guard = self.guards[0]
        else:
            self.guard = self.guards[self.guard_idx]

        self.scout = self.env.scout
        self.controlled_agents = {
            self.guard
        }  # By default, control only the specified guard

        print(f"Using seed: {self.seed}")
        print(f"Selected guard: {self.guard}")

    def initialize_agent_map(self, agent):
        """Initialize map and pathfinder for an agent."""
        self.agent_maps[agent] = Map()
        self.agent_maps[agent].create_trajectory_tree(Point(0, 0))
        # self.agent_maps[agent].create_particle_filter(Point(0, 0))
        self.agent_pathfinders[agent] = Pathfinder(
            self.agent_maps[agent], PathfinderConfig(use_viewcone=False)
        )

    def process_agent(self, agent, observation):
        """Process an agent's observation and decide on an action."""
        if agent not in self.controlled_agents:
            # Random movement for uncontrolled agents
            return random.choice([0, 1, 2, 3, 4])

        # Record execution time
        start_time = time.time()

        # Update the agent's map with the observation
        self.agent_maps[agent](observation)

        # Get next action
        location = observation["location"]
        direction = observation["direction"]
        action = int(
            self.agent_pathfinders[agent].get_optimal_action(
                Point(location[0], location[1]), Direction(direction), 0
            )
        )

        elapsed_time = time.time() - start_time
        self.execution_times.append(elapsed_time)

        self.step += 1
        print(
            f"Step {self.step}/{self.max_steps} - Processing time: {elapsed_time:.6f} seconds"
        )

        return action

    def record_frame(self, agent):
        """Record frames for visualization if recording is enabled."""
        if not self.record or agent != self.guard:
            return

        # Get the three views
        oracle_view = self.env.render()  # RGB format
        reconstructed_map = MapVisualizer(self.agent_maps[agent]).render(
            human_mode=False
        )  # BGR format
        proba_density = self.agent_maps[agent].trees[0].probability_density

        # Normalize and visualize probability density (returns BGR format)
        proba_density_viz = normalize_proba_density(
            proba_density, self.dirs["frames_proba"]
        )

        # Save individual frames
        if self.dirs:
            save_frame(
                cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR),
                self.dirs["frames_oracle"],
                self.step,
            )
            save_frame(reconstructed_map, self.dirs["frames_map"], self.step)
            save_frame(proba_density_viz, self.dirs["frames_proba"], self.step)

        # Store frames for video creation
        self.frames["oracle"].append(cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR))
        self.frames["map"].append(reconstructed_map)
        self.frames["proba"].append(proba_density_viz)

        # Create and save combined view
        views = [oracle_view, reconstructed_map, proba_density_viz]
        labels = ["Oracle View", "Reconstructed Map", "Probability Density"]
        combined_frame = combine_views(views, labels)
        self.frames["combined"].append(combined_frame)

        if self.dirs:
            save_frame(combined_frame, self.dirs["frames_combined"], self.step)

    def run(self):
        """Run the simulation."""
        print(
            "Starting simulation..."
            + (" and collecting frames..." if self.record else "")
        )

        try:
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                if termination or truncation or self.step >= self.max_steps:
                    break

                if agent in self.controlled_agents:
                    action = self.process_agent(agent, observation)
                    self.record_frame(agent)
                else:
                    # Use random actions for uncontrolled agents
                    action = random.choice([0, 1, 2, 3])

                self.env.step(action)

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        finally:
            self.env.close()

        if self.profile:
            stop_profiling(sort_by=1, lines=50)

        self.create_videos()
        self.report_statistics()

    def create_videos(self):
        """Create videos from recorded frames."""
        if not self.record or not self.dirs:
            return

        print("Creating videos from frames...")

        create_video(
            self.frames["oracle"],
            os.path.join(self.dirs["base"], "oracle_video.mp4"),
            self.fps,
        )
        create_video(
            self.frames["map"],
            os.path.join(self.dirs["base"], "map_video.mp4"),
            self.fps,
        )
        create_video(
            self.frames["proba"],
            os.path.join(self.dirs["base"], "probability_video.mp4"),
            self.fps,
        )
        create_video(
            self.frames["combined"],
            os.path.join(self.dirs["base"], "combined_video.mp4"),
            self.fps,
        )

        print("Videos created:")
        print(f"  - Oracle view: {os.path.join(self.dirs['base'], 'oracle_video.mp4')}")
        print(
            f"  - Reconstructed map: {os.path.join(self.dirs['base'], 'map_video.mp4')}"
        )
        print(
            f"  - Probability density: {os.path.join(self.dirs['base'], 'probability_video.mp4')}"
        )
        print(
            f"  - Combined view: {os.path.join(self.dirs['base'], 'combined_video.mp4')}"
        )

    def report_statistics(self):
        """Report statistics about the simulation."""
        if self.execution_times:
            avg_time = sum(self.execution_times) / len(self.execution_times)
            print(
                f"Average processing time: {avg_time:.6f} seconds over {len(self.execution_times)} steps"
            )
            print(f"Total steps completed: {self.step}")

        print(f"Seed {self.seed}")
