import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def orthogonal_init(scale=1.0):
    def _init(tensor):
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        nn.init.orthogonal_(tensor)
        tensor.mul_(scale)
    return _init


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        orthogonal_init(2.0)(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        orthogonal_init(0.01)(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        orthogonal_init(0.01)(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DiscreteActor(nn.Module):
    def __init__(self, input_size, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        orthogonal_init(2.0)(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_size, action_dim)
        orthogonal_init(0.01)(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        orthogonal_init(2.0)(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_size, 1)
        orthogonal_init(1.0)(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ScannedRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

    def initialize_carry(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, carry, inputs):
        """
        carry: (batch_size, hidden_size)
        inputs: tuple of (seq_inputs, seq_resets)
            - seq_inputs: (seq_len, batch_size, hidden_size)
            - seq_resets: (seq_len, batch_size)
        Returns:
            final hidden state and outputs: (seq_len, batch_size, hidden_size)
        """
        seq_inputs, seq_resets = inputs
        seq_len, batch_size, _ = seq_inputs.shape

        outputs = []
        h = carry

        for t in range(seq_len):
            x_t = seq_inputs[t]                   # (batch_size, hidden_size)
            reset_t = seq_resets[t].unsqueeze(1)  # (batch_size, 1)

            reset_mask = reset_t.bool()
            h = torch.where(reset_mask, self.initialize_carry(batch_size).to(h.device), h)

            h = self.gru_cell(x_t, h)  # Apply GRUCell step
            outputs.append(h.unsqueeze(0))

        return h, torch.cat(outputs, dim=0) 


class DiscreteActorCritic(nn.Module):
    def __init__(self, input_size, action_dim, hidden_size=128, double_critic=False):
        super().__init__()
        self.embedding_net = SimpleNN(input_size, hidden_size)
        self.actor = DiscreteActor(input_size, action_dim, hidden_size)
        self.double_critic = double_critic
        if double_critic:
            self.critic = nn.ModuleList([Critic(input_size, hidden_size), Critic(input_size, hidden_size)])
        else:
            self.critic = Critic(input_size, hidden_size)

    def forward(self, _, inputs):
        obs, dones = inputs
        embedding = self.embedding_net(obs)
        pi = self.actor(embedding)
        if self.double_critic:
            v = torch.stack([critic(embedding) for critic in self.critic], dim=2)
        else:
            v = self.critic(embedding).unsqueeze(2)
        return _, pi, v.squeeze(-1)


class DiscreteActorCriticRNN(nn.Module):
    def __init__(self, input_size, action_dim, hidden_size=128, double_critic=False):
        super().__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        orthogonal_init(scale=math.sqrt(2))(self.embed.weight)
        nn.init.constant_(self.embed.bias, 0.0)

        self.rnn = ScannedRNN(hidden_size, hidden_size)
        self.actor = DiscreteActor(action_dim, hidden_size)
        self.double_critic = double_critic
        if double_critic:
            self.critic = nn.ModuleList([Critic(hidden_size), Critic(hidden_size)])
        else:
            self.critic = Critic(hidden_size)

    def forward(self, hidden, inputs):
        obs, dones = inputs
        x = F.relu(self.embed(obs))
        _, embedding = self.rnn(hidden, (x, dones))
        # rnn_out: (seq_len, batch_size, hidden_size)

        # embedding = rnn_out.squeeze(1)
        pi = self.actor(embedding)

        if self.double_critic:
            v = torch.stack([critic(embedding) for critic in self.critic], dim=-1)
        else:
            v = self.critic(embedding).unsqueeze(2)

        return hidden, pi, v.squeeze(-1)