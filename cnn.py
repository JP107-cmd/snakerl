import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, state_shape, action_dim, hidden_layer_dim=256):
        super().__init__()

        channels, height, width = state_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            flattened_size = self.features(sample).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flattened_size, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, action_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)