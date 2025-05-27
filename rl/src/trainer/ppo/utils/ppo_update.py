from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensordict.tensordict import TensorDict


class RunningStatistics:
    """
    Running statistics class that tracks mean and standard deviation
    as described in Algorithm 1.
    """
    def __init__(self):
        # Initialize statistics trackers
        self.count = 0
        self.mean = 0.0
        self.var_sum = 0.0

    def add(self, value):
        """Add a new value to the running statistics"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.var_sum += delta * (value - self.mean)

    def standard_deviation(self):
        """Calculate the standard deviation of recorded values"""
        if self.count < 2:
            return torch.tensor(1.0)  # Default value when insufficient data
        return torch.sqrt(self.var_sum / self.count)

class RewardScaling(nn.Module):
    """
    Implementation of PPO reward scaling as described in Algorithm 1.

    This module tracks a running sum of discounted rewards R_t = γR_{t-1} + r_t
    and scales rewards by dividing by the standard deviation of this sum.
    """

    def __init__(self, gamma=0.99, epsilon=1e-8):
        """
        Initialize the reward scaling module.

        Args:
            gamma (float): Discount factor γ for calculating the running sum R_t
            epsilon (float): Small constant to avoid division by zero
        """
        super(RewardScaling, self).__init__()

        # Initialize as per Algorithm 1
        self.R_t = torch.tensor(0.0)  # R_0 ← 0
        self.RS = RunningStatistics()  # RS ← RunningStatistics()
        self.gamma = gamma  # γ is the reward discount
        self.epsilon = epsilon

    def forward(self, rewards):
        """
        Scale observation (rewards) as per Algorithm 1.
        Handles both single rewards and batched rewards of shape [batch_size].

        Args:
            rewards (torch.Tensor): Rewards r_t to be scaled, can be shape [] or [batch_size]

        Returns:
            torch.Tensor: Scaled rewards with same shape as input
        """
        # Convert input to tensor if needed
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        # Process each reward in the batch
        scaled_rewards = torch.zeros_like(rewards)

        # Flatten rewards to iterate through them
        flat_rewards = rewards.flatten()

        for i, r in enumerate(flat_rewards):
            # R_t ← γR_{t-1} + r_t
            self.R_t = self.gamma * self.R_t + r

            # Add(RS, R_t)
            self.RS.add(self.R_t)

            # r_t / StandardDeviation(RS)
            std = self.RS.standard_deviation()
            std = torch.max(std, torch.tensor(self.epsilon))  # Prevent division by zero

            # Store the scaled reward
            scaled_rewards.flatten()[i] = r / std

        # Return with original shape
        return scaled_rewards

    def reset(self):
        """Reset the internal statistics"""
        self.R_t = torch.tensor(0.0)
        self.RS = RunningStatistics()

def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: TensorDict,
    reward_scaler: nn.Module,
    args: Any
):
    """
    Performs one PPO training update on a batch of experience data.

    Args:
        model: The PPOActorCritic model.
        optimizer: The optimizer for the model.
        data: A dictionary containing batched tensors for the update
        args: Parsed arguments containing hyperparameters (gamma, gae, epochs, batch_size, clip_eps, bfloat16).

    Returns:
        A dictionary containing mean loss values for the update:
        - policy_loss
        - value_loss
        - entropy_bonus
        - total_loss
    """
    if args.bfloat16:
        data = data.to(torch.bfloat16)

    advantages = data['advantage']
    data['advantage'] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if args.normalize_returns:
        data['return'] = reward_scaler(data['return'])

    # --- PPO Epochs ---
    # Create indices for mini-batching
    batch_size = len(data)
    mini_batch_size = args.batch_size
    batch_indices = torch.arange(batch_size)

    # Store epoch losses for logging
    epoch_policy_losses = []
    epoch_value_losses = []
    epoch_entropy_bonuses = []
    epoch_total_losses = []


    for epoch in range(args.epochs):
        # Shuffle indices for random mini-batches
        shuffled_indices = batch_indices[torch.randperm(batch_size)]

        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            mini_batch_indices = shuffled_indices[start_idx:end_idx]

            mb = data[mini_batch_indices]

            # 4. Compute Loss
            # Get new log probs and values from the current model
            # Ensure inputs match model dtype (already handled during collection if bfloat16)
            log_probs_new, entropy, values_new = model.evaluate_actions(mb['actor_input'], mb['critic_input'], mb['action'])

            # Policy Loss (Clipped Surrogate Objective)
            ratio = torch.exp(log_probs_new - mb['log_prob'].detach()) # detach old log_probs, cast to float for ratio stability
            clip_adv = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb['advantage'].detach()
            policy_loss = -torch.min(ratio * mb['advantage'].detach(), clip_adv).mean()

            # Clipped Value Loss
            # Clip the new value estimate if it's too far from the old one
            values_clipped = mb['value'] + torch.clamp(values_new - mb['value'], -args.clip_eps, args.clip_eps)
            value_loss_unclipped = F.mse_loss(values_new, mb['return'])
            value_loss_clipped = F.mse_loss(values_clipped, mb['return'])
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped) # Use the maximum of clipped and unclipped loss

            # Entropy Bonus
            entropy_bonus = entropy.mean() # Maximize entropy, so add negative entropy to loss

            # Total Loss
            total_loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_bonus

            # 5. Backpropagate and Update
            optimizer.zero_grad()
            total_loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            # Store mini-batch losses for averaging
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_entropy_bonuses.append(entropy_bonus.item())
            epoch_total_losses.append(total_loss.item())


    # Calculate mean losses over all epochs and mini-batches
    mean_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses) if epoch_policy_losses else 0
    mean_value_loss = sum(epoch_value_losses) / len(epoch_value_losses) if epoch_value_losses else 0
    mean_entropy_bonus = sum(epoch_entropy_bonuses) / len(epoch_entropy_bonuses) if epoch_entropy_bonuses else 0
    mean_total_loss = sum(epoch_total_losses) / len(epoch_total_losses) if epoch_total_losses else 0

    return {
        'policy_loss': mean_policy_loss,
        'value_loss': mean_value_loss,
        'entropy_bonus': mean_entropy_bonus,
        'total_loss': mean_total_loss
    }
