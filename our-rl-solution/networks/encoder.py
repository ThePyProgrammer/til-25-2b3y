from typing import Optional

import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(
        self,
        act_cls: type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.LeakyReLU] = nn.ReLU,
        dropout: float = 0.1,
        lstm_dropout: float = 0.1,
        lstm_layers: int = 3
    ):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=1),
            act_cls(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3),
            act_cls(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3),
            act_cls(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        self.static = nn.Sequential(
            nn.Linear(3, 32),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            act_cls(),
            nn.Dropout(dropout),
            nn.LayerNorm(32),
        )

        self.joint = nn.Sequential(
            nn.Linear(224, 128),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            act_cls(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
        )

        self.recurrent = nn.LSTM(128, 128, num_layers=lstm_layers, dropout=lstm_dropout)

        self.h_n: Optional[torch.Tensor] = None # cached final hidden state
        self.c_n: Optional[torch.Tensor] = None # cached final cell state

    def forward(
        self,
        states: list[list[tuple[torch.Tensor, torch.Tensor]]],
        h_n: Optional[torch.Tensor] = None,
        c_n: Optional[torch.Tensor] = None,
        use_cached_states: bool = False
    ):
        # Process each sequence in the batch
        all_embeddings = []
        seq_lengths = []

        # Process each sequence separately first
        for seq in states:
            seq_spatial = []
            seq_static = []

            # Extract spatial and static components
            for spatial, static in seq:
                seq_spatial.append(spatial)
                seq_static.append(static)

            # Skip if sequence is empty
            if not seq_spatial:
                seq_lengths.append(0)
                continue

            # Batch process all spatial tensors in this sequence
            seq_spatial = torch.stack(seq_spatial)
            seq_static = torch.stack(seq_static)

            # Forward through respective networks
            spatial_embeddings = self.spatial(seq_spatial)
            static_embeddings = self.static(seq_static)

            # Concatenate and process through joint network
            combined = torch.cat([spatial_embeddings, static_embeddings], dim=1)
            seq_embeddings = self.joint(combined)

            all_embeddings.append(seq_embeddings)
            seq_lengths.append(len(seq))

        # Handle case with no data or all empty sequences
        if not all_embeddings:
            batch_size = len(states)
            output = torch.zeros(0, batch_size, 128)  # [seq_len, batch_size, hidden_size]
            h_out = torch.zeros(self.recurrent.num_layers, batch_size, 128) if h_n is None else h_n
            c_out = torch.zeros(self.recurrent.num_layers, batch_size, 128) if c_n is None else c_n
            self.h_n, self.c_n = h_out, c_out
            return output, (self.h_n, self.c_n)

        # Pack sequences for LSTM processing
        # Filter out empty sequences
        non_empty_embeddings = [emb for emb, length in zip(all_embeddings, seq_lengths) if length > 0]
        packed_sequences = nn.utils.rnn.pack_sequence(non_empty_embeddings, enforce_sorted=False)

        # Set up LSTM initial states
        lstm_kwargs = {}
        if use_cached_states:
            if self.h_n is not None and self.c_n is not None:
                lstm_kwargs["hx"] = (self.h_n, self.c_n)
        elif h_n is not None and c_n is not None:
            lstm_kwargs["hx"] = (h_n, c_n)

        # Process through LSTM
        packed_output, (self.h_n, self.c_n) = self.recurrent(packed_sequences, **lstm_kwargs)

        # Unpack sequences
        output = nn.utils.rnn.unpack_sequence(packed_output)

        return output, (self.h_n, self.c_n)
