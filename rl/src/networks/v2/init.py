import torch.nn as nn

def _orthogonal_init(scale=1.0):
    def _init(tensor):
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")

        nn.init.orthogonal_(tensor)
    return _init

def orthogonal_init(module):
    """
    Applies orthogonal initialization to nn.Linear layers' weights
    and initializes biases to zero.
    """
    orthogonal_init_fn = _orthogonal_init(scale=1.4)

    if isinstance(module, nn.Linear):
        orthogonal_init_fn(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
