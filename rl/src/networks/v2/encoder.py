import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Any
from dataclasses import dataclass, field

from ..layers import ConvBlock

@dataclass
class MapEncoderConfig:
    """Configuration for map encoders."""
    map_size: int = 16
    channels: int = 12
    output_dim: int = 64
    conv_layers: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    use_center_only: bool = True

class MapEncoder(nn.Module):
    def __init__(
        self,
        config: MapEncoderConfig
    ):
        super().__init__()

        self.config = config

        # Build CNN layers
        assert len(self.config.conv_layers) == len(self.config.kernel_sizes) == len(self.config.strides), "Conv params must have same length"

        self.conv_blocks = nn.ModuleList()
        in_channels = self.config.channels
        current_size = self.config.map_size

        # Add convolutional blocks
        for out_channels, kernel_size, stride in zip(self.config.conv_layers, self.config.kernel_sizes, self.config.strides):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    use_batch_norm=self.config.use_batch_norm
                )
            )
            in_channels = out_channels
            current_size -= 2
            current_size = current_size // stride if stride > 1 else current_size

        # Record final spatial size for center extraction if needed
        self.final_spatial_size = current_size

        # Calculate flattened size after convolutions
        if self.config.use_center_only:
            self.flattened_size = self.config.conv_layers[-1]  # Only center point features
        else:
            self.flattened_size = self.config.conv_layers[-1] * current_size * current_size

        # Build fully connected layers
        self.fc_blocks = nn.ModuleList()
        in_features = self.flattened_size

        # Final output layer
        self.output_layer = nn.Linear(in_features, self.config.output_dim)
        if self.config.use_layer_norm:
            self.output_norm = nn.LayerNorm(self.config.output_dim)

    def get_config(self):
        """Return the configuration of this encoder."""
        return self._config

    def forward(self, map_input):
        """
        Forward pass through the encoder.

        Args:
            map_input: Tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of shape [batch_size, embedding_dim] containing the encoded state
        """
        # Process map through CNN blocks
        x = map_input
        for block in self.conv_blocks:
            x = block(x)

        # Process spatial features
        if self.config.use_center_only:
            # Only take the center feature
            center_idx = self.final_spatial_size // 2
            x = x[:, :, center_idx, center_idx]  # Shape: [batch_size, channels]
        else:
            # Flatten all spatial features
            x = x.view(-1, self.flattened_size)

        # Final output layer
        x = self.output_layer(x)
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

        return x
