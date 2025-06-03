from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .particle import NodeParticles
from ..utils import Point, Direction, Tile
from ..node import NodeRegistry, DirectionalNode

from .constraints import Constraints, ParticleConstraints, TemporalConstraints


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

        self.last_seen_step = 0
        self.num_step = 0

        # Create a node registry for this trajectory tree
        self.registry = registry if registry is not None else NodeRegistry(size)

        self.temporal_constraints: TemporalConstraints = TemporalConstraints()

        constraints = ParticleConstraints()
        constraints.tail.contains.add(Point(0, 0))

        self.temporal_constraints.update(constraints)

        self.roots: list[DirectionalNode] = []
        self.particles: dict[DirectionalNode, NodeParticles] = {}

        # set roots
        self.set_roots(init_position, init_direction)

    def set_roots(self, position: Point, direction: Optional[Direction]):
        self.roots: list[DirectionalNode] = []

        possible_init_directions: list[Direction] = []
        if direction is not None:
            possible_init_directions.append(direction)
        else:
            possible_init_directions.extend([d for d in Direction])

        for direction in possible_init_directions:
            # This will either create a new node or use an existing one
            root = self.registry.get_or_create_node(position, direction)
            self.roots.append(root)

    def step(self):
        self.num_step += 1

        particles = {root: NodeParticles(1, 1, root) for root in self.roots}
        particles = resample_particles(particles, min_total_particles=self.min_total_particles)

        for step in range(self.last_seen_step, self.num_step):
            particles = propagate_particles(particles)
            particles = apply_particle_reweigh(particles, self.temporal_constraints[step])
            particles = resample_particles(particles, min_total_particles=self.min_total_particles)

        self.particles = particles

        self.resample()

    def prune(self, information: list[tuple[Point, Tile]]):
        """
        1. when I encounter the agent, I can destroy all trajectories since the agent's position is known
        2. when I encounter a tile that has not been visited, I remove all trajectories containing that tile (when seeking scout)
        3. when I encounter a tile that has been visited, I remove all trajectories not containing that tile (when seeking scout)
        4. when I encounter a tile and it has no agent, I remove all trajectories ending with that tile (when seeking scout)

        Args:
            information (list[tuple[Point, Tile]]): recently updated/observed tiles
        """
        constraints = ParticleConstraints()

        scout_seen = None

        for position, tile in information:

            # Case 1: agent detected - only keep trajectories containing this position
            if tile.has_scout:
                constraints.tail.contains.add(position)
                scout_seen = position
            # Case 2: not visited - remove trajectories containing this position
            if tile.is_visible and (tile.is_recon or tile.is_mission):
                constraints.route.excludes.add(position)
            # Case 3: visited - only keep trajectories containing this position
            if tile.is_empty:
                constraints.route.contains.add(position)
            # Case 4: no agent - remove trajectories ending at this position
            if not tile.has_scout:
                constraints.tail.excludes.add(position)

        # Use the spatial index for efficient filtering
        if not constraints:
            return  # No filtering needed

        # bugged? scout doesn't collect point at spawn location
        if Point(0, 0) in constraints.route.excludes:
            constraints.route.excludes.remove(Point(0, 0))

        self.temporal_constraints.update(constraints)

        if scout_seen is not None:
            # set roots to the last seen loc.
            self.last_seen_step = self.num_step
            self.set_roots(scout_seen, None)

            self.particles = {root: NodeParticles(1, 1, root) for root in self.roots}
        else:
            self.particles = apply_particle_reweigh(self.particles, constraints)

        self.resample()

    def resample(self):
        self.particles = resample_particles(self.particles, min_total_particles=self.min_total_particles)

    @property
    def probability_density(self) -> NDArray[np.float32]:
        """
        Calculate a probability density over all grid positions.

        This method computes how likely each position is to contain an agent,
        based on the number of valid trajectories passing through it.

        Returns:
            numpy.ndarray: 2D array with probability for each position
        """
        probas = np.zeros((self.size, self.size), dtype=np.float32)

        if not self.particles:
            return probas

        # Accumulate probabilities for each position
        for node, node_particles in self.particles.items():
            position = node.position
            x, y = position.x, position.y

            if 0 <= x < self.size and 0 <= y < self.size:
                probas[y, x] += node_particles.count * node_particles.individual_probability

        return probas / probas.sum()

    def check_wall_trajectories(self, *args, **kwargs):
        pass

def resample_particles(
    particles: dict[DirectionalNode, NodeParticles],
    min_total_particles: int = 1000
) -> dict[DirectionalNode, NodeParticles]:
    if not particles:
        return {}

    probabilities = np.array([particles.probability for particles in particles.values()])

    # Calculate total probability
    total_prob = np.sum(probabilities)
    if total_prob == 0:
        return {}

    # Normalize probabilities
    normalized_probs = probabilities / total_prob

    # Resample using multinomial sampling
    new_counts = np.random.multinomial(min_total_particles, normalized_probs)

    # Update particles dictionary with new counts
    new_particles = {}
    total_new_count = np.sum(new_counts)

    for i, (node, node_particles) in enumerate(particles.items()):
        if new_counts[i] > 0:
            node_particles.count = new_counts[i].item()
            node_particles.individual_probability = 1 / total_new_count.item()

            new_particles[node] = node_particles

    return new_particles

def propagate_particles(particles) -> dict[DirectionalNode, NodeParticles]:
    """Propagate particles from current nodes to their children based on available actions."""
    new_particles: dict[DirectionalNode, NodeParticles] = {}

    for node, node_particles in particles.items():
        if node.children:
            distribute_particles_to_children(node, node_particles, new_particles)
        else:
            # Leaf node - particles stay at current position
            new_particles[node] = node_particles

    return new_particles

def distribute_particles_to_children(
    node: DirectionalNode,
    node_particles: NodeParticles,
    new_particles: dict[DirectionalNode, NodeParticles]
) -> None:
    """Distribute particles from a parent node to its children based on available actions."""
    num_actions = len(node.children)
    particles_per_action = node_particles.count // num_actions
    remaining_particles = node_particles.count % num_actions

    for i, (action, child_node) in enumerate(node.children.items()):
        action_particles = particles_per_action
        if i < remaining_particles:
            action_particles += 1

        if action_particles > 0:
            if child_node in new_particles:
                new_particles[child_node].count += action_particles
            else:
                new_particles[child_node] = NodeParticles(
                    count=action_particles,
                    individual_probability=node_particles.individual_probability,
                    node=child_node
                )

            new_particles[child_node].incoming_action_counts[action] = action_particles

def apply_particle_reweigh(
    particles: dict[DirectionalNode, NodeParticles],
    constraints: ParticleConstraints
) -> dict[DirectionalNode, NodeParticles]:
    """
    Apply proba updates to the particles based on constraints.

    Route contains: Increase proba since particle should have been here
    Route excludes: Decrease proba since particle should not have been here
    Tail contains: Zero proba everywhere else since particle must be here. (should be handled by prune() when scout seen.)
    Tail excludes: Zero proba since particle cannot must not be here.
    """
    if not constraints:
        return particles

    updated_particles = {}

    for node, node_particles in particles.items():
        current_position = node.position

        # Apply tail constraints (most restrictive)

        # Tail excludes: Zero probability if particle is at excluded tail position
        if current_position in constraints.tail.excludes:
            node_particles.individual_probability = 0.0
            updated_particles[node] = node_particles
            continue

        # Tail contains: Zero probability if particle is not at required tail position
        if constraints.tail.contains and current_position not in constraints.tail.contains:
            node_particles.individual_probability = 0.0
            updated_particles[node] = node_particles
            continue

        # Apply route constraints (modify probability but don't zero out)
        probability_modifier = 1.0

        # Route excludes: Decrease probability if particle is at excluded route position
        if current_position in constraints.route.excludes:
            probability_modifier *= 0.1  # Significant penalty

        # Route contains: Increase probability if particle is at required route position
        elif current_position in constraints.route.contains:
            probability_modifier *= 2.0  # Boost probability

        # Apply the probability modifier
        node_particles.individual_probability *= probability_modifier

        # Only keep particles with non-zero probability
        if node_particles.individual_probability > 0:
            updated_particles[node] = node_particles

    return updated_particles
