import torch.nn as nn
from dataclasses import dataclass, field

from ..layers import ConvBlock, Conv3DBlock

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
        if self.config.use_layer_norm:
            x = self.output_norm(x)

        return x

@dataclass
class TemporalMapEncoderConfig:
    """Configuration for temporal map encoders."""
    map_size: int = 16
    channels: int = 12
    output_dim: int = 32
    frames: int = 3

    # Spatial-only 3D convolution parameters (preserving temporal dimension)
    conv3d_channels: list[int] = field(default_factory=lambda: [32, 32, 32])
    conv3d_kernel_sizes: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(1, 3, 3), (1, 3, 3), (3, 3, 3)]
    )
    conv3d_strides: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
    )
    conv3d_paddings: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    )

    # 2D convolution parameters (after temporal dimension is processed)
    conv_layers: list[int] = field(default_factory=lambda: [32, 32])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    strides: list[int] = field(default_factory=lambda: [1, 1])
    paddings: list[int] = field(default_factory=lambda: [0, 0])

    # Other parameters
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    use_center_only: bool = True

class TemporalMapEncoder(nn.Module):
    def __init__(
        self,
        config: TemporalMapEncoderConfig
    ):
        super().__init__()

        self.config = config

        current_size = self.config.map_size

        # Build spatial-only 3D CNN layers (depthwise convs with kernel (1,3,3))
        assert len(self.config.conv3d_channels) == len(self.config.conv3d_kernel_sizes) == len(self.config.conv3d_strides) == len(self.config.conv3d_paddings), "Spatial conv params must have same length"

        self.conv3d_blocks = nn.ModuleList()
        in_channels = self.config.channels

        # Add spatial-only 3D convolutional blocks
        for out_channels, kernel_size, stride, padding in zip(
            self.config.conv3d_channels,
            self.config.conv3d_kernel_sizes,
            self.config.conv3d_strides,
            self.config.conv3d_paddings
        ):
            self.conv3d_blocks.append(
                Conv3DBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    use_batch_norm=self.config.use_batch_norm
                )
            )
            in_channels = out_channels
            # Update spatial dimensions
            k_t, k_h, k_w = kernel_size
            s_t, s_h, s_w = stride
            p_t, p_h, p_w = padding

            # Height and width calculations
            current_size = ((current_size + 2 * p_h - k_h) // s_h) + 1

        # After temporal convolution, we have shape [batch, channels, 1, height, width]
        # Build regular 2D CNN layers
        assert len(self.config.conv_layers) == len(self.config.kernel_sizes) == len(self.config.strides) == len(self.config.paddings), "Conv params must have same length"

        self.conv_blocks = nn.ModuleList()

        # Add 2D convolutional blocks
        for out_channels, kernel_size, stride, padding in zip(
            self.config.conv_layers,
            self.config.kernel_sizes,
            self.config.strides,
            self.config.paddings
        ):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    use_batch_norm=self.config.use_batch_norm
                )
            )
            in_channels = out_channels
            # Update spatial dimensions
            current_size = ((current_size + 2 * padding - kernel_size) // stride) + 1

        # Record final spatial size for center extraction if needed
        self.final_spatial_size = current_size

        # Calculate flattened size after convolutions
        if self.config.use_center_only:
            self.flattened_size = self.config.conv_layers[-1]  # Only center point features
        else:
            self.flattened_size = self.config.conv_layers[-1] * current_size * current_size

        # Final output layer
        self.output_layer = nn.Linear(self.flattened_size, self.config.output_dim)
        if self.config.use_layer_norm:
            self.output_norm = nn.LayerNorm(self.config.output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def forward(self, map_input):
        """
        Forward pass through the encoder.

        Args:
            map_input: Tensor of shape [batch_size, channels, frames, height, width]

        Returns:
            Tensor of shape [batch_size, embedding_dim] containing the encoded state
        """
        # Process through spatial 3D convs (preserving temporal dimension)
        x = map_input
        for block in self.conv3d_blocks:
            x = block(x)

        # Squeeze the temporal dimension since it should now be 1
        # Shape becomes [batch_size, channels, height, width]
        x = x.squeeze(2)

        # Process through 2D convs
        for block in self.conv_blocks:
            x = block(x)

        # Process spatial features
        if self.config.use_center_only:
            # Only take the center feature
            center_idx = self.final_spatial_size // 2
            x = x[:, :, center_idx, center_idx]  # Shape: [batch_size, channels]
        else:
            # Flatten all spatial features
            x = x.reshape(x.size(0), -1)

        # Apply dropout
        x = self.dropout(x)

        # Final output layer
        x = self.output_layer(x)
        if self.config.use_layer_norm:
            x = self.output_norm(x)

        return x
