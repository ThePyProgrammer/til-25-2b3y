import os
import sys
import pathlib
import random
import argparse
import time
import numpy as np

# Add paths for imports
sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld
from grid.map import Map
from grid.utils import start_profiling, stop_profiling
from grid.utils import Point
from grid.map import Direction

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a simplified simulation demo')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999),
                        help='Random seed for environment initialization')
    parser.add_argument('--guard', type=int, default=0,
                        help='Guard number (0 for first guard, 1 for second guard, etc.)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Maximum number of steps to simulate')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    start_profiling()

    # Set all seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # Initialize map and create trajectory tree
    recon_map = Map()
    recon_map.create_trajectory_tree(Point(0, 0))

    # Initialize environment with specified seed
    env = gridworld.env(
        env_wrappers=[],
        render_mode=None,  # No rendering needed
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

    # Run simulation
    step = 0
    execution_times = []

    print("Starting simulation...")

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

            # Get next action
            location = observation['location']
            direction = observation['direction']
            action = int(recon_map.get_optimal_action(Point(location[0], location[1]), Direction(direction), 0))
        else:
            # Use the seeded numpy random generator for deterministic sampling
            action_space = env.action_space(agent)
            action = random.choice([0, 1, 2, 3])

        env.step(action)

    env.close()

    stop_profiling(sort_by=1, lines=50)

    # Report average execution time
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"Average processing time: {avg_time:.6f} seconds over {len(execution_times)} steps")
        print(f"Total steps completed: {step}")

if __name__ == "__main__":
    main()
