from dataclasses import dataclass

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
