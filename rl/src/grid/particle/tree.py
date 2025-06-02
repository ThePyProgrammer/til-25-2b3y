from typing import Optional
import numpy as np

from .particle import Particles
from ..utils import Point, Direction, Action, Tile
from ..node import NodeRegistry, DirectionalNode

from ..trajectory.constraints import Constraints, TrajectoryConstraints, TemporalTrajectoryConstraints


class ParticleTree:
    def __init__(
        self,
        map,
        init_position: Point,
        init_direction: Optional[Direction] = None,
        min_total_particles: int = 1000,
        size: int = 16,
        registry: Optional[NodeRegistry] = None,
    ):
        self.map = map
        self.min_total_particles = min_total_particles
        self.size = size

        self.num_step = 0

        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        self.temporal_constraints: TemporalTrajectoryConstraints = TemporalTrajectoryConstraints()

        self.roots: list[DirectionalNode] = []
        self.particles: dict[DirectionalNode, Particles] = {}

        self.init(init_position, init_direction)

    def init(
        self,
        init_position: Point,
        init_direction: Optional[Direction] = None,
    ):
        self.roots: list[DirectionalNode] = []
        possible_init_directions: list[Direction] = []
        if init_direction is not None:
            possible_init_directions.append(init_direction)
        else:
            possible_init_directions.extend([d for d in Direction])

        for direction in possible_init_directions:
            # This will either create a new node or use an existing one
            root = self.registry.get_or_create_node(init_position, direction)
            self.roots.append(root)

            self.particles: dict[DirectionalNode, Particles] = {
                root: Particles(1000, 1/1000, root)
            }

        self.resample()

    def step(self):
        self.num_step += 1

        new_particles: dict[DirectionalNode, Particles] = {}

        for _ in range(self.num_step):
            for node, particles in self.particles.items():
                pass
                # for action, child_node in node.children.items()
                # add/update to new_particles with equal proba.

    def prune(self, information: list[tuple[Point, Tile]]):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)
        4. when I encounter a tile and it has no agent, I remove all trajectories ending with that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
        """

        route_constraints = Constraints([], [])
        tail_constraints = Constraints([], [])

        constraints = TrajectoryConstraints(
            route_constraints,
            tail_constraints
        )

        for position, tile in information:
            # Skip any tiles that are in our ambiguous set
            # if position in self.ambiguous_tiles:
            #     continue

            # Case 1: agent detected - only keep trajectories containing this position
            if tile.has_scout:
                constraints.tail.contains.append(position)
            # Case 2: not visited - remove trajectories containing this position
            if tile.is_visible and (tile.is_recon or tile.is_mission):
                constraints.route.excludes.append(position)
            # Case 3: visited - only keep trajectories containing this position
            if tile.is_empty:
                constraints.route.contains.append(position)
            # Case 4: no agent - remove trajectories ending at this position
            if not tile.has_scout:
                constraints.tail.excludes.append(position)

        # Use the spatial index for efficient filtering
        if not constraints:
            return  # No filtering needed

        # bugged? scout doesn't collect point at spawn location
        if Point(0, 0) in constraints.route.excludes:
            constraints.route.excludes.remove(Point(0, 0))

        if Point(0, 0) in constraints.tail.excludes:
            constraints.tail.excludes.remove(Point(0, 0))

    def resample(self):
        if not self.particles:
            return

        # Get all particles and their probabilities
        nodes = list(self.particles.keys())
        probabilities = np.array([self.particles[node].probability for node in nodes])

        # Calculate total probability
        total_prob = np.sum(probabilities)
        if total_prob == 0:
            return

        # Normalize probabilities
        normalized_probs = probabilities / total_prob

        # Resample using multinomial sampling
        new_counts = np.random.multinomial(self.min_total_particles, normalized_probs)

        # Update particles dictionary with new counts
        new_particles = {}
        total_new_count = np.sum(new_counts)

        for i, node in enumerate(nodes):
            if new_counts[i] > 0:
                # Calculate new individual probability to maintain total probability
                new_particles[node] = Particles(
                    count=int(new_counts[i]),
                    individual_probability=1/total_new_count,
                    node=node
                )

        self.particles = new_particles
