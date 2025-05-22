import os
import random
from typing import Dict, List, Set, Optional

import numpy as np
import cv2

from grid.utils import Point
from grid.map import Direction, Map
from grid.pathfinder import Pathfinder, PathfinderConfig
from utils.demo.base import BaseDemo
from utils.demo.visualization import combine_views_grid, normalize_proba_density
from utils.demo.recording import save_frame
from utils.demo.rl_agent import RLScoutAgent
from grid.viz import MapVisualizer


class AdvancedDemo(BaseDemo):
    """Advanced demo with multi-agent support and complex visualization."""

    def __init__(self, args):
        """Initialize the advanced demo with command-line arguments."""
        super().__init__(args)

        # Additional advanced options
        self.path_density = getattr(args, 'path_density', False)
        self.control_spec = getattr(args, 'control', 'all')
        self.scout_target_spec = getattr(args, 'scout_target', None)
        self.use_rl_scout = getattr(args, 'rl_scout', False)
        self.model_path = getattr(args, 'model_path', './models/scout.pt')

        # Advanced state
        self.scout_target = None
        self.agent_grid_positions = {}
        self.rl_scout_agent = None

        # Advanced recording
        self.agent_frames = {}
        self.grid_frames = []

    def setup(self):
        """Set up the environment and agents with advanced options."""
        super().setup()

        # Determine which agents to control
        self.setup_controlled_agents()

        # Parse scout target if provided
        self.setup_scout_target()

        # Initialize RL scout if enabled
        if self.use_rl_scout:
            self.setup_rl_scout()

        # Initialize maps and pathfinders for controlled agents
        for agent in self.env.agents:
            if agent in self.controlled_agents:
                self.initialize_agent_map_advanced(agent)

        # Create a map and pathfinder for the scout if a target is provided
        if self.scout_target is not None and not self.use_rl_scout:
            self.initialize_scout_pathfinder()

        # Create a mapping from agent to grid position index (0-3)
        for i, agent in enumerate(self.env.agents[:4]):  # Limit to 4 agents for the 2x2 grid
            self.agent_grid_positions[agent] = i
            self.agent_frames[agent] = {'oracle': [], 'map': [], 'proba': []}

        print(f"Created maps and pathfinders for {len(self.controlled_agents)} controlled guards: {self.controlled_agents}")
        print(f"Random movement agents: {[agent for agent in self.env.agents if agent not in self.controlled_agents and agent != self.env.scout]}")

        scout_mode = "using RL model" if self.use_rl_scout else f"targeting {self.scout_target}" if self.scout_target else "random movement"
        print(f"Scout agent: {self.env.scout} ({scout_mode})")

    def setup_controlled_agents(self):
        """Set up the set of controlled agents based on command-line args."""
        if self.control_spec.lower() == 'all':
            # Control all agents (excluding scout)
            self.controlled_agents = set(self.guards)
        elif self.control_spec.lower() != 'none':
            # Parse the comma-separated list of agent indices
            try:
                control_indices = [int(idx.strip()) for idx in self.control_spec.split(',')]
                for idx in control_indices:
                    if 0 <= idx < len(self.guards):
                        self.controlled_agents.add(self.guards[idx])
                    else:
                        print(f"Warning: Invalid agent index: {idx}, ignoring.")
            except ValueError:
                print(f"Warning: Invalid control format: {self.control_spec}, defaulting to controlling all agents.")
                self.controlled_agents = set(self.guards)
        else:
            # 'none' - don't control any agents
            self.controlled_agents = set()

    def setup_scout_target(self):
        """Set up the scout target based on command-line args."""
        if self.scout_target_spec:
            try:
                x, y = map(int, self.scout_target_spec.split(','))
                self.scout_target = Point(x, y)
                print(f"Scout target: {self.scout_target}")
            except ValueError:
                print(f"Warning: Invalid scout target format: {self.scout_target_spec}, scout will move randomly")

    def initialize_agent_map_advanced(self, agent):
        """Initialize map and pathfinder for an agent with advanced options."""
        self.agent_maps[agent] = Map()
        self.agent_maps[agent].create_trajectory_tree(Point(0, 0))
        self.agent_pathfinders[agent] = Pathfinder(
            self.agent_maps[agent],
            PathfinderConfig(
                use_viewcone=False,
                use_path_density=self.path_density
            )
        )

    def initialize_scout_pathfinder(self):
        """Initialize pathfinder for the scout agent."""
        scout_agent = self.env.scout
        self.agent_maps[scout_agent] = Map()
        # Scout doesn't need a trajectory tree, just basic pathfinding
        self.agent_pathfinders[scout_agent] = Pathfinder(
            self.agent_maps[scout_agent],
            PathfinderConfig(
                use_viewcone=False,
                use_path_density=False
            )
        )
        # Add scout to controlled agents
        self.controlled_agents.add(scout_agent)

    def setup_rl_scout(self):
        """Initialize the RL scout agent."""
        self.rl_scout_agent = RLScoutAgent(model_path=self.model_path)

        # Add scout to controlled agents
        self.controlled_agents.add(self.scout)

        # Initialize a map for the scout if not already done
        if self.scout not in self.agent_maps:
            self.agent_maps[self.scout] = Map()
            self.agent_maps[self.scout].create_trajectory_tree(Point(0, 0))

    def process_agent(self, agent, observation):
        """Process an agent's observation and decide on an action."""
        # Handle RL scout
        if agent == self.scout and self.use_rl_scout and self.rl_scout_agent is not None:
            return self.process_rl_scout_agent(observation)

        # Handle target-based scout
        if agent == self.scout and self.scout_target is not None:
            return self.process_scout_agent(observation)

        return super().process_agent(agent, observation)

    def process_rl_scout_agent(self, observation):
        """Process the scout agent using the RL model."""
        try:
            # Use the RL scout agent to get an action
            action = self.rl_scout_agent.get_action(observation, self.agent_maps[self.scout])
            return action
        except Exception as e:
            print(f"Error in RL scout processing: {e}")
            # Fallback to random action
            return random.choice([0, 1, 2, 3, 4])

    def process_scout_agent(self, observation):
        """Process the scout agent and make it move toward its target."""
        location = observation['location']
        direction = observation['direction']

        # Simple pathfinding toward target
        current = Point(location[0], location[1])
        if current.distance_to(self.scout_target) < 1.5:
            # We've reached the target, just wait
            return 0  # Stand still

        # Use the pathfinder to navigate to the target
        self.agent_maps[self.scout].set_target(self.scout_target)
        return int(self.agent_pathfinders[self.scout].get_optimal_action(
            current, Direction(direction), 0
        ))

    def record_frame(self, agent):
        """Record frames for visualization with advanced grid layout."""
        if not self.record:
            return

        # Get the views
        oracle_view = self.env.render()  # RGB format
        agent_idx = self.agent_grid_positions.get(agent)

        if agent in self.agent_maps:
            reconstructed_map = MapVisualizer(self.agent_maps[agent]).render(human_mode=False)  # BGR format
            proba_density = self.agent_maps[agent].trees[0].probability_density if hasattr(self.agent_maps[agent], 'trees') else None
            proba_density_viz = normalize_proba_density(proba_density)

            # Save the frames for this agent
            self.agent_frames[agent]['oracle'].append(oracle_view)
            self.agent_frames[agent]['map'].append(reconstructed_map)
            self.agent_frames[agent]['proba'].append(proba_density_viz)

            # For the guard we're specifically tracking, also save the simple combined view
            if agent == self.guard:
                views = [oracle_view, reconstructed_map, proba_density_viz]
                labels = ['Oracle View', 'Reconstructed Map', 'Probability Density']
                combined_frame = combine_views(views, labels)
                self.frames['combined'].append(combined_frame)

                if self.dirs:
                    save_frame(combined_frame, self.dirs['frames_combined'], self.step)

        # Create a grid view of all agents after each step
        if agent == self.env.agents[-1] or len(self.env.agents) == 1:
            # This is the last agent in the iteration, create the grid view
            self.create_grid_view(oracle_view)

    def create_grid_view(self, oracle_view):
        """Create a grid view of all agents."""
        # Prepare data for the grid view
        agent_data = {}
        for agent, pos in self.agent_grid_positions.items():
            if agent in self.agent_maps:
                map_view = MapVisualizer(self.agent_maps[agent]).render(human_mode=False)
                proba_density = self.agent_maps[agent].trees[0].probability_density if hasattr(self.agent_maps[agent], 'trees') else None
                proba_view = normalize_proba_density(proba_density)
                agent_data[pos] = {
                    'map': map_view,
                    'proba': proba_view
                }

        # Create the grid view
        grid_frame = combine_views_grid(agent_data, oracle_view)
        self.grid_frames.append(grid_frame)

        if self.dirs:
            grid_dir = os.path.join(self.dirs['base'], 'grid')
            os.makedirs(grid_dir, exist_ok=True)
            save_frame(grid_frame, grid_dir, self.step, name='grid')

    def create_videos(self):
        """Create videos from recorded frames."""
        super().create_videos()

        if not self.record or not self.dirs:
            return

        # Create grid video
        if self.grid_frames:
            from utils.demo.recording import create_video
            grid_video_path = os.path.join(self.dirs['base'], "grid_video.mp4")
            create_video(self.grid_frames, grid_video_path, self.fps)
            print(f"  - Grid view: {grid_video_path}")
