# Demo Scripts for TIL-25 RL

This directory contains demo scripts for visualizing and testing agents in the TIL-25 RL environment.

## Available Scripts

### simple_demo.py

A basic demo focused on a single guard agent. This script demonstrates the fundamental functionality of the environment and agent behavior.

```bash
python src/simple_demo.py --seed 42 --record
```

### demo.py

An advanced demo with multi-agent support, complex visualization layout, and additional features.

```bash
python src/demo.py --seed 42 --record --control 0,1
```

## Command-line Arguments

Both scripts support the following common arguments:

- `--seed INT`: Random seed for environment initialization (default: random)
- `--guard INT`: Guard number to focus on (0 for first guard, 1 for second, etc.)
- `--steps INT`: Maximum number of steps to simulate (default: 100)
- `--human`: Show human view (cannot be used with `--record`)
- `--record`: Record simulation and create videos
- `--fps INT`: Frames per second in output videos (default: 5)
- `--profile`: Enable profiling of the code

### Advanced Arguments (demo.py only)

- `--path_density`: Use path density instead of probability density
- `--control STR`: Comma-separated list of agent indices to control (e.g., "0,1"), or "all" or "none"
- `--scout_target STR`: Target coordinates for the scout in format "x,y" (e.g., "10,15")
- `--rl_scout`: Use the trained RL model for scout agent
- `--model_path STR`: Path to the trained model file for RL scout (default: "./models/scout.pt")

## RL Scout Functionality

The advanced demo script supports using a trained PPO model for controlling the scout agent. 

### Prerequisites

To use the RL scout, you need a trained model. Place your model checkpoint at:

```
./models/scout.pt
```

Or specify a custom path with the `--model_path` argument.

### Example Usage

```bash
# Run demo with RL scout
python src/demo.py --seed 42 --record --rl_scout

# Run demo with RL scout using a custom model
python src/demo.py --seed 42 --record --rl_scout --model_path ./models/my_custom_scout.pt
```

## Visualization Output

When using the `--record` flag, the scripts generate several video outputs:

### simple_demo.py outputs:

- Oracle view: Global view of the environment
- Reconstructed map: Agent's internal map representation
- Probability density: Visualization of the agent's belief about the environment
- Combined view: All three views side by side

### demo.py outputs:

- Combined view of all agents in a grid layout
- Individual videos for each agent showing their perspective, reconstructed map, and probability density

The videos are saved in the `_output` directory.

## Project Structure

The demo scripts are built on a modular framework:

- `utils/demo/args.py`: Command-line argument parsing
- `utils/demo/base.py`: Base demo class with common functionality
- `utils/demo/advanced.py`: Advanced demo class extending the base
- `utils/demo/recording.py`: Utilities for recording and saving videos
- `utils/demo/visualization.py`: Utilities for visualization
- `utils/demo/rl_agent.py`: Integration with trained RL models

## Contributing

To extend the demos with new functionality:
1. Add new utility functions to the appropriate module in `utils/demo/`
2. Update the demo classes to use the new functionality
3. Add command-line arguments to support the new features