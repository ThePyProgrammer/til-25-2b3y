from typing import Optional

import torch
import torch.nn.functional as F

class ExperienceBuffer:
    """
    A buffer for collecting training experiences (state, action, reward, etc.)
    for Proximal Policy Optimization (PPO).
    """
    def __init__(self):
        self.map_inputs = []
        self.step_inputs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        map_input: torch.Tensor,
        step_input: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ):
        """
        Adds a single step's experience to the buffer.

        Args:
            map_input: The map observation tensor for the state *before* the action.
                       Expected shape (C, H, W).
            step_input: The step tensor for the state *before* the action.
                        Expected shape (1,).
            action: The action taken (integer).
            log_prob: The log probability of the action taken under the policy *before* the update.
            value: The value estimate of the state *before* the action, under the value function *before* the update.
            reward: The reward received *after* taking the action.
            done: The done signal *after* taking the action (True if episode terminated or truncated).
        """
        # Store unbatched tensors/values
        self.map_inputs.append(map_input.squeeze(0) if map_input.ndim == 4 else map_input) # Ensure (C, H, W)
        self.step_inputs.append(step_input.squeeze(0) if step_input.ndim == 2 else step_input) # Ensure (1,)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done)) # Store done as float (0.0 or 1.0)

    def get_batch(self) -> Optional[dict]:
        """
        Collects all stored experiences into batch tensors.

        Returns:
            A dictionary containing batched tensors for map_inputs, step_inputs,
            actions, log_probs, values, rewards, and dones.
            Returns None if the buffer is empty.
        """
        if not self.map_inputs:
            return None # Buffer is empty

        # Stack collected tensors/values into batch tensors
        # Add batch dimension at dim 0
        b_map_inputs = torch.stack(self.map_inputs, dim=0)
        b_step_inputs = torch.stack(self.step_inputs, dim=0)
        b_actions = torch.tensor(self.actions, dtype=torch.long)
        b_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        b_values = torch.tensor(self.values, dtype=torch.float32)
        b_rewards = torch.tensor(self.rewards, dtype=torch.float32)
        b_dones = torch.tensor(self.dones, dtype=torch.float32) # Still float (T,)

        return {
            'map_inputs': b_map_inputs, # Shape (T, C, H, W)
            'step_inputs': b_step_inputs, # Shape (T, 1)
            'actions': b_actions, # Shape (T,)
            'log_probs': b_log_probs, # Shape (T,)
            'values': b_values, # Shape (T,)
            'rewards': b_rewards, # Shape (T,)
            'dones': b_dones # Shape (T,)
        }

    def clear(self):
        """
        Clears all stored experiences from the buffer.
        """
        self.map_inputs = []
        self.step_inputs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        """
        Returns the number of steps currently stored in the buffer.
        """
        return len(self.map_inputs)

    def is_empty(self):
        """
        Checks if the buffer is empty.
        """
        return len(self) == 0
