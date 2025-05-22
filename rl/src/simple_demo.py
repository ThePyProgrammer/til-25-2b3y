import os
import sys
import pathlib

# Add project paths to sys.path
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from utils.demo.args import parse_simple_arguments
from utils.demo.base import BaseDemo


class SimpleDemo(BaseDemo):
    """Simple demo focusing on a single guard agent."""
    
    def setup(self):
        """Set up the environment and a single guard agent."""
        super().setup()
        
        # Initialize map and pathfinder for the selected guard
        self.initialize_agent_map(self.guard)
        
        # Only control the selected guard
        self.controlled_agents = {self.guard}
        
        print(f"Controlling single guard agent: {self.guard}")


def main():
    """Main entry point for the simple demo."""
    args = parse_simple_arguments()
    demo = SimpleDemo(args)
    demo.setup()
    demo.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass