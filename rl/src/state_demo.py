import os
import sys
import pathlib
import time
from collections import defaultdict
import random
import numpy as np

sys.path.append(str(pathlib.Path(os.getcwd()).parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from til_environment import gridworld
from agent.episode import State, StateConfig
from agent.experience import Experience

# Create the environment
env = gridworld.env(
    env_wrappers=[],  # clear out default env wrappers
    render_mode="human",  # Render the map
    debug=True,  # Enable debug mode
    novice=False,  # Use same map layout every time (for Novice teams only)
)

def reset_env():
    env.reset(random.randint(0, 10**10))

# Get initial environment information
env_reset = reset_env()
all_agents = list(env.agents)
print(f"Environment initialized with {len(all_agents)} agents: {all_agents}")

# Create state config with max history of 5
state_config = StateConfig(max_history=5)

# Create separate replay buffers for each agent
agent_buffers = {agent: Experience(capacity=10000, episode_length=100, state_config=state_config)
                 for agent in all_agents}

# Create states for each agent
agent_states = {agent: State(config=state_config) for agent in all_agents}

# Track metrics for each agent
agent_steps = {agent: 0 for agent in all_agents}
agent_episodes = {agent: 1 for agent in all_agents}
agent_rewards = {agent: 0 for agent in all_agents}

# Define team-based colors for console output
AGENT_COLORS = {
    "scout": "\033[94m",     # Blue for scout
    "guard": "\033[91m",     # Red for guards
    "END_COLOR": "\033[0m"   # Reset color
}

def get_agent_color(agent_name):
    if "scout" in agent_name:
        return AGENT_COLORS["scout"]
    elif "guard" in agent_name:
        return AGENT_COLORS["guard"]
    return ""

def color_print(agent, message):
    color = get_agent_color(agent)
    print(f"{color}{message}{AGENT_COLORS['END_COLOR']}")

print("Running environment simulation...")
total_steps = 0
simulation_limit = 1000  # Maximum simulation steps to avoid infinite loops

while total_steps < simulation_limit:
    total_steps += 1

    dones = 0

    # Iterate through agents
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        # Update agent metrics
        agent_steps[agent] += 1
        agent_rewards[agent] += reward

        # Get the agent's state
        current_state = agent_states[agent]

        # Update state with new observation
        current_state.update(observation)

        # Check if episode is done for this agent
        done = termination or truncation

        if done:
            color_print(agent, f"Episode {agent_episodes[agent]} ended for {agent} after {agent_steps[agent]} steps with total reward: {agent_rewards[agent]:.2f}")

            # Force end episode in this agent's buffer
            agent_buffers[agent].force_end_episode()

            # Reset agent state and metrics for new episode
            agent_states[agent] = State(config=state_config)
            agent_episodes[agent] += 1
            agent_steps[agent] = 0
            agent_rewards[agent] = 0

            next_observation = None
            dones += 1
        else:
            # Sample a random action
            action = env.action_space(agent).sample()

            # Take a step in the environment
            env.step(action)

            # Get the next observation
            next_observation, *_ = env.last()

        # Add experience to the buffer (if we have observations)
        if len(current_state) > 0 and not current_state.is_empty():
            agent_buffers[agent].push(
                observation=observation,
                action=action if not done else 0,
                reward=reward,
                next_observation=next_observation,
                done=done
            )

        # Print information about the current step (less frequently to reduce output)
        if total_steps % 10 == 0 or done:
            color_print(
                agent,
                f"Step {total_steps}, Agent: {agent}, Steps: {agent_steps[agent]}, "
                f"Episode: {agent_episodes[agent]}, Reward: {reward:.2f}, "
                f"Total Reward: {agent_rewards[agent]:.2f}, Done: {done}"
            )

        if dones == 1:
            break

    print("\nResetting environment...\n")
    reset_env()

# Print summary for each agent's experience buffer
print("\n" + "="*50)
print("EXPERIENCE BUFFER SUMMARY")
print("="*50)

for agent, buffer in agent_buffers.items():
    color_print(
        agent,
        f"Agent: {agent} | Buffer size: {len(buffer)} episodes"
    )

# Sample from each agent's buffer
print("\n" + "="*50)
print("SAMPLING FROM EXPERIENCE BUFFERS")
print("="*50)

for agent, buffer in agent_buffers.items():
    if len(buffer) > 0:
        color_print(agent, f"\nSampling transitions for {agent}:")
        batch_size = min(2, len(buffer))
        transitions = buffer.sample(batch_size)

        for i, t in enumerate(transitions):
            color_print(
                agent,
                f"Transition {i+1}:\n"
                f"  State length: {len(t.state)}\n"
                f"  Action: {t.action}\n"
                f"  Reward: {t.reward}\n"
                f"  Done: {t.done}"
            )

        # Sample a batch for the encoder
        color_print(agent, f"\nSampling encoder batch for {agent}:")
        encoder_batch = buffer.get_batch(batch_size)
        if encoder_batch:
            states, actions, rewards, next_states, dones = encoder_batch
            color_print(
                agent,
                f"Batch size: {len(states)}\n"
                f"Actions shape: {actions.shape}\n"
                f"Rewards shape: {rewards.shape}\n"
                f"Dones shape: {dones.shape}"
            )

            if states and states[0]:
                color_print(agent, f"Example observation type: {type(states[0][0])}")

# Calculate overall statistics
print("\n" + "="*50)
print("AGENT PERFORMANCE METRICS")
print("="*50)

team_episodes = defaultdict(int)
for agent, episodes in agent_episodes.items():
    team = "scout" if "scout" in agent else "guard"
    team_episodes[team] += episodes - 1  # Subtract 1 as we start from episode 1

for team, episodes in team_episodes.items():
    print(f"{AGENT_COLORS[team]}{team.capitalize()} team completed {episodes} episodes{AGENT_COLORS['END_COLOR']}")

env.close()
