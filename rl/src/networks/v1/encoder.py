import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class ConvBlock(nn.Module):
    """A configurable convolutional block with optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = False,
        activation: nn.Module = nn.ReLU
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_batch_norm = use_batch_norm
        self.activation = activation()
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.bn(x)
        return x


class MapEncoder(nn.Module):
    """
    Configurable encoder network for processing grid maps for PPO.

    Designed to be:
    1. Extensible - easy to modify architecture
    2. CPU-friendly - balanced complexity for inference
    3. Configurable - adjust capacity based on requirements

    Map Channels:
    - no_vision (0/1)
    - empty (0/1)
    - recon (0/1)
    - mission (0/1)
    - scout (0/1)
    - guard (0/1)
    - top_wall (0/1)
    - bottom_wall (0/1)
    - left_wall (0/1)
    - right_wall (0/1)
    - last_updated (0/1)
    - is_here (0/1/2/3) for directions
    """

    def __init__(
        self,
        map_size: int = 16,
        channels: int = 12,
        embedding_dim: int = 256,
        conv_layers: list[int] = [32, 64, 128, 256],
        kernel_sizes: list[int] = [3, 3, 3, 3],
        strides: list[int] = [2, 2, 2, 2],
        fc_layers: list[int] = [512],
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        use_center_only: bool = False
    ):
        super().__init__()

        # Input parameters
        self.map_size = map_size
        self.channels = channels

        # Build CNN layers
        assert len(conv_layers) == len(kernel_sizes) == len(strides), "Conv params must have same length"

        # Initialize layers list and current channels count
        self.conv_blocks = nn.ModuleList()
        in_channels = channels

        # Current spatial dimension
        current_size = map_size

        # Add convolutional blocks
        for out_channels, kernel_size, stride in zip(conv_layers, kernel_sizes, strides):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,  # No padding
                    use_batch_norm=use_batch_norm
                )
            )
            in_channels = out_channels
            current_size -= 2
            current_size = current_size // stride if stride > 1 else current_size

        # Record final spatial size for center extraction if needed
        self.final_spatial_size = current_size

        # Whether to use only the center feature instead of flattening
        self.use_center_only = use_center_only

        # Calculate flattened size after convolutions
        if use_center_only:
            self.flattened_size = conv_layers[-1]  # Only center point features
        else:
            self.flattened_size = conv_layers[-1] * current_size * current_size

        # Build fully connected layers
        self.fc_blocks = nn.ModuleList()
        in_features = self.flattened_size

        for out_features in fc_layers:
            self.fc_blocks.append(nn.Linear(in_features, out_features))
            if use_layer_norm:
                self.fc_blocks.append(nn.LayerNorm(out_features))
            self.fc_blocks.append(nn.ReLU())
            if dropout_rate > 0:
                self.fc_blocks.append(nn.Dropout(dropout_rate))
            in_features = out_features

        # Final output layer
        self.output_layer = nn.Linear(in_features, embedding_dim)
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(embedding_dim)

        # Record architecture for debugging
        self._config = {
            'map_size': map_size,
            'channels': channels,
            'embedding_dim': embedding_dim,
            'conv_layers': conv_layers,
            'kernel_sizes': kernel_sizes,
            'strides': strides,
            'fc_layers': fc_layers,
            'final_spatial_size': current_size,
            'use_batch_norm': use_batch_norm,
            'use_layer_norm': use_layer_norm,
            'dropout_rate': dropout_rate,
            'use_center_only': use_center_only
        }

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
        if self.use_center_only:
            # Only take the center feature
            center_idx = self.final_spatial_size // 2
            x = x[:, :, center_idx, center_idx]  # Shape: [batch_size, channels]
        else:
            # Flatten all spatial features
            x = x.view(-1, self.flattened_size)

        # Process through FC blocks
        for layer in self.fc_blocks:
            x = layer(x)

        # Final output layer
        x = self.output_layer(x)
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

        return x


class TinyMapEncoder(MapEncoder):
    """Smaller variant optimized for CPU inference."""

    def __init__(self, map_size=16, channels=14, embedding_dim=32, use_center_only=False):
        super().__init__(
            map_size=map_size,
            channels=channels,
            embedding_dim=embedding_dim,
            conv_layers=[8, 16, 16, 32],
            kernel_sizes=[7, 3, 3, 3],  # Larger initial kernel to capture more context
            strides=[1, 2, 2, 1],       # More aggressive downsampling
            fc_layers=[],          # Smaller FC layer
            use_batch_norm=True,     # Skip batch norm for CPU efficiency
            dropout_rate=0.1,        # Skip dropout for inference speed
            use_layer_norm=False,     # Keep layer norm for stability
            use_center_only=use_center_only
        )


class SmallMapEncoder(MapEncoder):
    """Smaller variant optimized for CPU inference."""

    def __init__(self, map_size=16, channels=14, embedding_dim=64, use_center_only=False):
        super().__init__(
            map_size=map_size,
            channels=channels,
            embedding_dim=embedding_dim,
            conv_layers=[32, 32, 32, 32],
            kernel_sizes=[7, 3, 3, 3],  # Larger initial kernel to capture more context
            strides=[1, 1, 1, 1],       # More aggressive downsampling
            fc_layers=[],          # Smaller FC layer
            use_batch_norm=True,     # Skip batch norm for CPU efficiency
            dropout_rate=0.1,        # Skip dropout for inference speed
            use_layer_norm=True,     # Keep layer norm for stability
            use_center_only=use_center_only
        )


class LargeMapEncoder(MapEncoder):
    """Larger variant for more complex environments."""

    def __init__(self, map_size=16, channels=12, embedding_dim=256, use_center_only=False):
        super().__init__(
            map_size=map_size,
            channels=channels,
            embedding_dim=embedding_dim,
            conv_layers=[64, 64, 128, 256, 256, 512],
            kernel_sizes=[3, 3, 3, 3, 3, 3],
            strides=[1, 1, 1, 1, 1, 1],
            fc_layers=[512],
            use_batch_norm=True,
            dropout_rate=0.1,
            use_layer_norm=True,
            use_center_only=use_center_only
        )


def create_encoder(encoder_type="standard", **kwargs):
    """Factory function to create different encoder variants."""
    if encoder_type == "tiny":
        return TinyMapEncoder(**kwargs)
    elif encoder_type == "small":
        return SmallMapEncoder(**kwargs)
    elif encoder_type == "large":
        return LargeMapEncoder(**kwargs)
    else:  # "standard" or any other value
        return MapEncoder(**kwargs)

if __name__ == "__main__":
    # Example usage and benchmarking
    import time

    # Sample data
    batch_size = 1
    map_input = torch.rand(batch_size, 12, 16, 16)

    # Test different encoder configurations
    encoders = {
        "standard": create_encoder(),
        "small": create_encoder("small"),
        "large": create_encoder("large"),
        "custom": create_encoder(
            conv_layers=[16, 32, 64],
            kernel_sizes=[5, 3, 3],
            strides=[2, 2, 2],
            fc_layers=[128],
            use_batch_norm=False
        ),
        "center_only": create_encoder(use_center_only=True),
        "small_center_only": create_encoder("small", use_center_only=True)
    }

    # Simple benchmark
    print("Encoder benchmarks (CPU):")
    for name, encoder in encoders.items():
        # Warm up
        for _ in range(5):
            _ = encoder(map_input)

        # Benchmark
        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            output = encoder(map_input)

        avg_time = (time.time() - start_time) / iterations * 1000  # ms
        param_count = sum(p.numel() for p in encoder.parameters())

        print(f"  {name}: {avg_time:.2f}ms per batch, params: {param_count:,}, "
              f"output shape: {output.shape}")

        # Print architecture details
        print(f"  Config: {encoder.get_config()}")
        print()
