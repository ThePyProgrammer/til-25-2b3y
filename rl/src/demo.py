import os
import sys
import pathlib

# Add project paths to sys.path
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from utils.demo.args import parse_advanced_arguments
from utils.demo.advanced import AdvancedDemo
from utils.demo.rl_agent import RLScoutAgent


def main():
    """Main entry point for the advanced demo."""
    args = parse_advanced_arguments()
    
    # Print whether RL scout is enabled
    if args.rl_scout:
        print(f"RL scout mode enabled. Model path: {args.model_path}")
    
    demo = AdvancedDemo(args)
    demo.setup()
    demo.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")