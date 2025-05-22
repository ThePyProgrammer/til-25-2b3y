import torch
import torch.nn.functional as F


def validate_sampling_params(k: int, temperature: float, action_dim: int = 5) -> tuple[int, float]:
    k = min(max(1, k), action_dim)  # Clamp between 1 and action_dim
    temperature = max(0.01, temperature)  # Minimum temperature to avoid division by zero
    return k, temperature

def top_k_sampling(logits: torch.Tensor, k: int = 3, temperature: float = 1.0) -> int:
    """
    Perform top-k sampling on action logits.

    Args:
        logits: Raw logits from the policy network
        k: Number of top actions to consider
        temperature: Temperature for softmax scaling

    Returns:
        Selected action index
    """
    scaled_logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_values, dim=-1)
    selected_idx = torch.multinomial(top_k_probs, 1)
    action = top_k_indices[selected_idx].item()

    return int(action)