import argparse
import random


def get_base_parser():
    """Create a base argument parser with common options for demo scripts.

    Returns:
        ArgumentParser: Base parser with common arguments
    """
    parser = argparse.ArgumentParser(description='Run a simulation demo with optional recording')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999),
                        help='Random seed for environment initialization')
    parser.add_argument('--guard', type=int, default=0,
                        help='Guard number (0 for first guard, 1 for second guard, etc.)')
    parser.add_argument('--steps', type=int, default=400,
                        help='Maximum number of steps to simulate')
    parser.add_argument('--human', action='store_true',
                        help='Show human view. Either --human or --record only.')
    parser.add_argument('--record', action='store_true',
                        help='Record simulation and create videos')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second in output videos (only used with --record)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling of the code')

    return parser


def parse_simple_arguments():
    """Parse arguments for the simple demo script.

    Returns:
        Namespace: Parsed arguments
    """
    parser = get_base_parser()
    return parser.parse_args()


def parse_advanced_arguments():
    """Parse arguments for the advanced demo script.

    Returns:
        Namespace: Parsed arguments
    """
    parser = get_base_parser()
    parser.add_argument('--path_density', action='store_true',
                        help='Use path density instead of (random walk) probability density')
    parser.add_argument('--control', type=str, default='all',
                        help='Comma-separated list of agent indices to control (0-based) or '
                             '"all" for all agents or "none" for no agents')
    parser.add_argument('--scout_target', type=str, default=None,
                        help='Target coordinates for the scout in format "x,y" (e.g., "10,15")')
    parser.add_argument('--rl_scout', action='store_true',
                        help='Use the trained RL model for scout agent')
    parser.add_argument('--model_path', type=str, default='./models/scout.pt',
                        help='Path to the trained model file for RL scout')
    return parser.parse_args()
