import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld

env = gridworld.env(
    env_wrappers=[],  # clear out default env wrappers
    render_mode="human",  # Render the map
    debug=True,  # Enable debug mode
    novice=True,  # Use same map layout every time (for Novice teams only)
)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break
    else:
        # Insert your policy here
        action = env.action_space(agent).sample()

    env.step(action)

env.close()
