import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueModel(nn.Module):
    def __init__(
        self, belief_size, state_size, hidden_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        x = self.act_fn(self.ln1(self.fc1(x)))
        x = self.act_fn(self.ln2(self.fc2(x)))
        x = self.act_fn(self.ln3(self.fc3(x)))
        reward = self.fc4(x).squeeze(dim=1)
        return reward
