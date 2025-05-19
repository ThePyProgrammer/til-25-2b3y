import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any

# Assuming PPOActorCritic, etc. are available via sys.path or passed in if needed
# from networks.ppo import PPOActorCritic

def calculate_gae_returns(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
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

def ppo_update(model: torch.nn.Module, optimizer: optim.Optimizer, data: Dict[str, torch.Tensor], args: Any):
    """
    Performs one PPO training update on a batch of experience data.

    Args:
        model: The PPOActorCritic model.
        optimizer: The optimizer for the model.
        data: A dictionary containing batched tensors for the update:
              'map_inputs': (T, C, H, W)
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
    b_map_inputs = data['map_inputs']
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

            mb_map_inputs = b_map_inputs[mini_batch_indices]
            mb_actions = b_actions[mini_batch_indices]
            mb_log_probs_old = b_log_probs_old[mini_batch_indices]
            mb_values_old = b_values_old[mini_batch_indices]
            mb_advantages = advantages[mini_batch_indices]
            mb_returns = returns[mini_batch_indices]

            # 4. Compute Loss
            # Get new log probs and values from the current model
            # Ensure inputs match model dtype (already handled during collection if bfloat16)
            log_probs_new, entropy, values_new = model.evaluate_actions(mb_map_inputs, mb_actions)

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
            VALUE_LOSS_COEF = 0.5 # Common value
            ENTROPY_COEF = 0.01 # Common value
            total_loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy_bonus


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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

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

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This part is just for demonstrating the function
    # In the actual training script, this would be called from the main loop
    print("Testing ppo_update function (dummy data)...")

    # Dummy data
    batch_size = 128
    map_size = 16
    channels = 14
    action_dim = 4

    dummy_map_inputs = torch.randn(batch_size, channels, map_size, map_size, dtype=torch.float32)
    dummy_actions = torch.randint(0, action_dim, (batch_size,), dtype=torch.long)
    dummy_log_probs_old = torch.randn(batch_size, dtype=torch.float32) # Log probs from a distribution
    dummy_values_old = torch.randn(batch_size, dtype=torch.float32)
    dummy_rewards = torch.randn(batch_size, dtype=torch.float32)
    dummy_dones = torch.randint(0, 2, (batch_size,), dtype=torch.float32) # 0.0 or 1.0

    dummy_data = {
        'map_inputs': dummy_map_inputs,
        'actions': dummy_actions,
        'log_probs': dummy_log_probs_old,
        'values': dummy_values_old,
        'rewards': dummy_rewards,
        'dones': dummy_dones
    }

    # Dummy model and optimizer (requires importing PPOActorCritic)
    # from networks.ppo import PPOActorCritic
    # dummy_model = PPOActorCritic(action_dim=action_dim, map_size=map_size, channels=channels, encoder_type="small")
    # dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=3e-4)
    #
    # # Dummy args
    # class DummyArgs:
    #     def __init__(self):
    #         self.gamma = 0.99
    #         self.gae = 0.95
    #         self.epochs = 3
    #         self.batch_size = 32
    #         self.clip_eps = 0.2
    #         self.bfloat16 = False # Set to True to test bfloat16 path
    #
    # dummy_args = DummyArgs()
    #
    # # Perform update
    # # losses = ppo_update(dummy_model, dummy_optimizer, dummy_data, dummy_args)
    # # print("Losses:", losses)

    print("Calculate GAE and Returns (dummy data)...")
    advantages, returns = calculate_gae_returns(dummy_rewards, dummy_values_old, dummy_dones, 0.99, 0.95)
    print(f"Advantages shape: {advantages.shape}")
    print(f"Returns shape: {returns.shape}")
