"""Recurrent pixel actor-critic; shared unchanged by every training condition."""
import torch
from torch import nn


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(1024, 512), nn.ReLU())
        self.gru = nn.GRUCell(512, 512)
        self.actor = nn.Linear(512, 17)
        self.critic = nn.Linear(512, 1)

    def forward(self, images, hidden):
        x = images.permute(0, 3, 1, 2).float() / 255.0
        hidden = self.gru(self.encoder(x), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden
