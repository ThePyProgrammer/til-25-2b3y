import os
import sys
import pathlib
from typing import Any, Optional

import torch
import numpy as np

# Add paths for imports - make sure these paths are accessible
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent.resolve() / "til-25-environment"))
sys.path.append(str(pathlib.Path(os.getcwd()).resolve()))

from grid.map import Map
from grid.utils import Point, Direction, Action
from grid.node import DirectionalNode
from networks.v1.ppo import PPOActorCritic

# Constants for the scout model
CHANNELS, MAP_SIZE, ACTION_DIM = 10, 31, 5

class RLScoutAgent:
    """Integration with trained RL model for scout agent behavior."""

    def __init__(self, model_path: str = "./models/scout.pt"):
        """Initialize the RL scout agent.

        Args:
            model_path: Path to the trained model checkpoint
        """
        self.model_path = model_path
        self.scout_policy = None
        self.initialized = False
        self.recon_map = Map()

        # Load the model if the file exists
        self._load_model()

    def _load_model(self) -> bool:
        """Load the model from the checkpoint file.

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Create the model architecture
            self.scout_policy = PPOActorCritic(
                action_dim=ACTION_DIM,
                map_size=MAP_SIZE,
                channels=CHANNELS,
                encoder_type="small",
                shared_encoder=False,
                embedding_dim=32,
                actor_hidden_dims=[32, 32],
                critic_hidden_dims=[32, 32],
                encoder_kwargs={
                    "use_center_only": True
                }
            )

            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Warning: Model file not found at {self.model_path}")
                print("The RL scout will use random actions until a valid model is loaded.")
                return False

            # Load the model weights
            checkpoint = torch.load(self.model_path)
            self.scout_policy.load_state_dict(checkpoint['model_state_dict'])
            self.scout_policy.eval()

            self.initialized = True
            print(f"Successfully loaded RL scout model from {self.model_path}")
            return True

        except Exception as e:
            print(f"Error loading RL scout model: {e}")
            return False

    def get_action(self, observation: dict[str, Any], recon_map: Optional[Map] = None) -> int:
        """Get the next action for the scout agent based on observation.

        Args:
            observation: The observation dictionary from the environment
            recon_map: Optional reconstructed map to use instead of the internal one

        Returns:
            int: Action to take (0-4)
        """
        # Use provided map if available, otherwise use internal map
        map_to_use = recon_map if recon_map is not None else self.recon_map

        # Ensure observation has the right types
        observation['viewcone'] = np.array(observation['viewcone'], dtype=np.uint8)
        observation['location'] = np.array(observation['location'], dtype=np.uint8)
        observation['direction'] = int(observation['direction'])

        # Update the map with the observation
        map_to_use(observation)

        location = observation["location"]
        position = Point(int(location[0]), int(location[1]))
        direction = Direction(observation["direction"])

        node: DirectionalNode = map_to_use.get_node(position, direction)
        valid_actions = set(node.children.keys())

        # If model is loaded, use it to determine action
        if self.initialized and self.scout_policy is not None:
            try:
                with torch.no_grad():
                    # Get tensor representation of the map
                    map_tensor = map_to_use.get_tensor().unsqueeze(0)
                    embedding = self.scout_policy.actor_encoder(map_tensor)

                    # Get action distribution from policy
                    dist = self.scout_policy.actor.get_distribution(embedding)
                    action_probs = dist.probs.squeeze(0)  # Remove batch dimension

                    print(f"Action probabilities: {action_probs}")
                    print(f"Valid actions: {valid_actions}")

                    # Find the most probable valid action
                    best_action = None
                    best_prob = -1.0

                    for action in range(len(action_probs)):
                        if action in valid_actions and action_probs[action] > best_prob:
                            best_action = action
                            best_prob = action_probs[action]

                    # If we found a valid action, return it
                    if best_action is not None:
                        print(f"Selected action: {Action(best_action)} with probability: {best_prob:.4f}")
                        return best_action
                    else:
                        # Fallback: if no valid actions found in probabilities, pick random valid action
                        print("No valid actions found in model output, selecting random valid action")
                        return np.random.choice(list(valid_actions))

            except Exception as e:
                print(f"Error getting action from RL model: {e}")
                # Fall back to random valid action if error occurs
                if valid_actions:
                    return np.random.choice(list(valid_actions))
                else:
                    return np.random.randint(0, 5)
        else:
            # If model not loaded, use random valid action
            if valid_actions:
                return np.random.choice(list(valid_actions))
            else:
                return np.random.randint(0, 5)

    def reset(self):
        """Reset the agent state for a new episode."""
        self.recon_map = Map()
