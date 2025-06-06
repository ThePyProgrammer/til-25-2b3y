from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from ..v2.encoder import MapEncoder as V2MapEncoder, MapEncoderConfig
from ..layers import ConvBlock, Conv3DBlock


class ViewconeEncoder(nn.Module):
    def __init__(self, n_frames):
        super().__init__()

        kernels = [
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (n_frames, 3, 3),
        ]

        paddings = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 0),
            (0, 0, 0),
        ]

        channels = [32, 64, 64, 128]
        batch_norms = [True, True, True, False]

        in_channel = 10

        self.conv_blocks = nn.ModuleList()

        for kernel, padding, out_channel, batch_norm in zip(kernels, paddings, channels, batch_norms):
            self.conv_blocks.append(
                Conv3DBlock(
                    in_channel,
                    out_channel,
                    kernel_size=kernel,
                    padding=padding,
                    use_batch_norm=batch_norm
                )
            )
            in_channel = out_channel

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = x.flatten(1)
        return x

class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        use_batch_norm = True

        kernels = [3, 3, 3, 3, 3, 3, 3]
        paddings = [0, 0, 0, 0, 0, 0, 0]
        channels = [32, 32, 48, 48, 64, 64, 64]

        in_channel = 10

        self.conv_blocks = nn.ModuleList()

        for kernel, padding, out_channel in zip(kernels, paddings, channels):
            self.conv_blocks.append(
                ConvBlock(
                    in_channel,
                    out_channel,
                    kernel_size=kernel,
                    padding=padding,
                    use_batch_norm=use_batch_norm
                )
            )
            in_channel = out_channel

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = x.flatten(1)
        return x

@dataclass
class StateEncoderConfig:
    n_frames: int = 4

class StateEncoder(nn.Module):
    def __init__(self, config: StateEncoderConfig):
        super().__init__()

        self.config = config

        self.viewcone_encoder = ViewconeEncoder(self.config.n_frames)
        self.map_encoder = MapEncoder()
        self.info_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

    def forward(self, state: TensorDict):
        info = torch.cat(
            [
                state['location'],
                state['direction'].unsqueeze(-1),
                state['step'].unsqueeze(-1),
            ],
            dim=-1
        )

        info_embedding = self.info_encoder(info)
        viewcone_embedding = self.viewcone_encoder(state['viewcone'])
        map_embedding = self.map_encoder(state['map'])

        embedding = torch.cat(
            [
                info_embedding,
                viewcone_embedding,
                map_embedding
            ],
            dim=-1
        )

        return embedding

class MapStateEncoder(nn.Module):
    def __init__(self, config: MapEncoderConfig):
        super().__init__()

        self.config = config

        self.map_encoder = V2MapEncoder(config)

        self.info_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

    def forward(self, state: TensorDict):
        info = torch.cat(
            [
                state['location'],
                state['direction'].unsqueeze(-1),
                state['step'].unsqueeze(-1),
            ],
            dim=-1
        )

        info_embedding = self.info_encoder(info)
        map_embedding = self.map_encoder(state['map'])

        embedding = torch.cat(
            [
                info_embedding,
                map_embedding
            ],
            dim=-1
        )

        return embedding


@dataclass
class RecurrentMapStateEncoderConfig:
    map_encoder_config: MapEncoderConfig = field(default_factory=MapEncoderConfig)
    recurrent_type: Literal["gru", "lstm", "rnn"] = "gru"  # "gru", "lstm", or "rnn"
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0

class RecurrentMapStateEncoder(nn.Module):
    def __init__(self, config: RecurrentMapStateEncoderConfig):
        super().__init__()

        self.config = config

        self.map_encoder = V2MapEncoder(config.map_encoder_config)

        self.info_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        dummy_map_input = torch.zeros(1, 1, 12, 31, 31)  # Assuming typical map dimensions
        dummy_info_input = torch.zeros(1, 1, 4)

        with torch.no_grad():
            dummy_map_out = self.map_encoder(dummy_map_input.squeeze(1))
            dummy_info_out = self.info_encoder(dummy_info_input.squeeze(1))
            combined_dim = dummy_map_out.shape[-1] + dummy_info_out.shape[-1]

        recurrent_kwargs = {
            "input_size": combined_dim,
            "hidden_size": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout if config.num_layers > 1 else 0,
            "bidirectional": False,
            "batch_first": True
        }

        recurrent_type = config.recurrent_type.lower()

        if recurrent_type == "gru":
            recurrent_cls = nn.GRU
        elif recurrent_type == "lstm":
            recurrent_cls = nn.LSTM
        elif recurrent_type == "rnn":
            recurrent_cls = nn.RNN
        else:
            raise ValueError(f"Unsupported recurrent type: {config.recurrent_type}")

        self.recurrent = recurrent_cls(**recurrent_kwargs)

    def forward(
        self,
        state: TensorDict,
        hidden=None,
        return_hidden: bool = False
    ):
        """
        Args:
            state (TensorDict):
                {
                    "map": (batch_size, seq_len, channels, height, width),
                    "location": (batch_size, seq_len, 2),
                    "direction": (batch_size, seq_len, 1),
                    "step": (batch_size, seq_len, 1)
                }
            hidden: Optional hidden state for recurrent layer

        Returns:
            output: (batch_size, seq_len, hidden_dim * num_directions)
            hidden: Updated hidden state
        """
        b, s, c, h, w = state['map'].shape

        # Reshape to process all timesteps at once
        map_reshaped = state['map'].reshape(b * s, c, h, w)

        info = torch.cat(
            [
                state['location'],
                state['direction'],
                state['step'],
            ],
            dim=-1
        )
        info_reshaped = info.reshape(b * s, -1)

        # Process through encoders
        info_embedding = self.info_encoder(info_reshaped)  # (b*s, 16)
        map_embedding = self.map_encoder(map_reshaped)  # (b*s, map_embed_dim)

        # Combine embeddings
        combined_embedding = torch.cat(
            [
                info_embedding,
                map_embedding
            ],
            dim=-1
        )  # (b*s, combined_dim)

        # Reshape back to sequence format
        combined_embedding = combined_embedding.reshape(b, s, -1)  # (b, s, combined_dim)

        seq_len = state['seq_len'].cpu().long()
        packed_input = nn.utils.rnn.pack_padded_sequence(combined_embedding, seq_len, batch_first=True, enforce_sorted=False)

        # Pass through recurrent layer
        packed_output, hidden_new = self.recurrent(packed_input, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output[torch.arange(b), seq_len-1].squeeze(1)
