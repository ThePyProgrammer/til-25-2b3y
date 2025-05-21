from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        
        # Store original shape for return value
        original_shape = rewards.shape
        
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
        
def calculate_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates Generalized Advantage Estimation (GAE) and returns (value targets).

    Args:
        rewards: Tensor of rewards (T,).
        values: Tensor of value estimates (T,).
        dones: Tensor of done flags (0.0 or 1.0) (T,).
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        A tuple containing:
        - advantages: Tensor of calculated advantages (T,).
        - returns: Tensor of calculated returns (T,).
    """
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    # Iterate backwards to calculate GAE
    for i in reversed(range(len(rewards))):
        # next_value: Value of the state *after* step i.
        # If not the last step in the batch, and episode didn't end, use value of next state.
        # If last step in batch, or episode ended, next_value is 0.
        # For a single episode batch:
        # next_value for step i (0 to T-2) is values[i+1] if not done[i].
        # next_value for step T-1 (last step) is 0 if done[T-1] is True.
        # If done[T-1] is False (truncated), commonly bootstrap from V(S_T) i.e., values[T-1].
        # Let's use the standard GAE approach which handles this: next_value is V(S_{t+1}) if not done_t, 0 otherwise.
        # And the recursive calculation naturally uses the bootstrapped value (or 0) from the last step.

        # If not the last step
        if i < len(rewards) - 1:
            next_value = values[i+1]
            next_non_terminal = 1.0 - dones[i]
        else:
            # Last step in the batch (end of episode)
            # If done[i] is True (termination/truncation), next_value is 0.
            # If done[i] is False (truncated episode, and batch ended here),
            # we need V(S_{T+1}), but PPO usually bootstraps from V(S_T).
            # Standard GAE formula uses V(S_{t+1}). For the last step T-1, V(S_T).
            # If done[T-1]=True, V(S_T)=0. If done[T-1]=False, V(S_T) is the predicted value.
            # The formula requires V(S_{t+1}), which for t=T-1 is V(S_T).
            # So, next_value for i=len-1 should be values[i] if not done[i], and 0 if done[i].
            next_value = values[i] if not dones[i] else 0.0 # Bootstrapped value for truncated episode end
            next_non_terminal = 1.0 - dones[i] # will be 0 if done[i] is True


        delta = rewards[i] + gamma * next_non_terminal * next_value - values[i]
        advantages[i] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

    # Calculate returns (Value Targets) V_target = Advantage + V_old
    # Or, using backward calculation: Returns_t = R_t + gamma * (1-done_t) * Returns_{t+1}
    # Let's calculate returns directly from values and advantages: V_target = A + V_old
    returns = advantages + values

    # Normalize advantages (optional but common)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns

def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: Dict[str, torch.Tensor],
    reward_scaler: nn.Module,
    args: Any
):
    """
    Performs one PPO training update on a batch of experience data.

    Args:
        model: The PPOActorCritic model.
        optimizer: The optimizer for the model.
        data: A dictionary containing batched tensors for the update:
              'b_critic_inputs': (T, C, H, W)
              'actions': (T,)
              'log_probs': (T,) (old log probabilities)
              'values': (T,) (old value estimates)
              'rewards': (T,)
              'dones': (T,)
        args: Parsed arguments containing hyperparameters (gamma, gae, epochs, batch_size, clip_eps, bfloat16).

    Returns:
        A dictionary containing mean loss values for the update:
        - policy_loss
        - value_loss
        - entropy_bonus
        - total_loss
    """
    b_critic_inputs = data['critic_inputs']
    b_actions = data['actions']
    b_log_probs_old = data['log_probs']
    b_values_old = data['values']
    b_rewards = data['rewards']
    b_dones = data['dones']

    # Ensure tensors have the correct dtype if bfloat16 is enabled
    # Inputs (map, step) are assumed to be already in the correct dtype from buffer
    # Convert other tensors needed for calculations
    if args.bfloat16:
        b_log_probs_old = b_log_probs_old.to(torch.bfloat16)
        b_values_old = b_values_old.to(torch.bfloat16)
        b_rewards = b_rewards.to(torch.bfloat16)
        b_dones = b_dones.to(torch.bfloat16) # Use bfloat16 for done flags in calculations

    # Calculate advantages and returns
    advantages, returns = calculate_gae_returns(
        b_rewards, b_values_old, b_dones, args.gamma, args.gae
    )

    # Normalize returns using running statistics if enabled
    if args.normalize_returns:
        returns = reward_scaler(returns)

    # --- PPO Epochs ---
    # Create indices for mini-batching
    batch_size = len(b_actions) # Use the size of the collected data as the initial batch size
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

            mb_critic_inputs = b_critic_inputs[mini_batch_indices]
            mb_actions = b_actions[mini_batch_indices]
            mb_log_probs_old = b_log_probs_old[mini_batch_indices]
            mb_values_old = b_values_old[mini_batch_indices]
            mb_advantages = advantages[mini_batch_indices]
            mb_returns = returns[mini_batch_indices]

            # 4. Compute Loss
            # Get new log probs and values from the current model
            # Ensure inputs match model dtype (already handled during collection if bfloat16)
            log_probs_new, entropy, values_new = model.evaluate_actions(mb_critic_inputs, mb_actions)
            
            # Policy Loss (Clipped Surrogate Objective)
            # Ensure ratio calculation is done in floating point (bfloat16 or float32)
            ratio = torch.exp(log_probs_new.float() - mb_log_probs_old.float().detach()) # detach old log_probs, cast to float for ratio stability
            clip_adv = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_advantages.detach() # detach advantages
            policy_loss = -torch.min(ratio * mb_advantages.detach(), clip_adv).mean() # detach advantages

            # Value Loss (Clipped Value Loss - optional but common)
            # Clip the new value estimate if it's too far from the old one
            # Ensure calculations are done in floating point
            values_new_float = values_new.float()
            mb_values_old_float = mb_values_old.float().detach()
            mb_returns_float = mb_returns.float().detach()

            values_clipped = mb_values_old_float + torch.clamp(values_new_float - mb_values_old_float, -args.clip_eps, args.clip_eps)
            value_loss_unclipped = F.mse_loss(values_new_float, mb_returns_float)
            value_loss_clipped = F.mse_loss(values_clipped, mb_returns_float)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped) # Use the maximum of clipped and unclipped loss


            # Entropy Bonus (to encourage exploration)
            # Ensure entropy calculation is done in floating point
            entropy_bonus = entropy.float().mean() # Maximize entropy, so add negative entropy to loss

            # Total Loss
            # Coefficients for balancing policy, value, and entropy losses
            total_loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_bonus


            # 5. Backpropagate and Update
            optimizer.zero_grad()
            # Ensure backward pass uses correct precision, usually float32 is safe
            # If using bfloat16 model, autograd should handle gradients correctly
            # total_loss.backward()
            # If autocast is used, backward should be called on the loss in the autocast context
            # With bfloat16 default dtype, backward might happen in bfloat16.
            # Let's just call backward on the loss directly.
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
