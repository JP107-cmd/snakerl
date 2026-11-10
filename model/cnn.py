import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, state_shape, board_size, action_dim, hidden_layer_dim=64):
        super().__init__()

        self.board_size = board_size
        self.grid_size = 3 * self.board_size * self.board_size




        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Preserve coarse spatial layout without flattening the full
            # 15x15 feature map into a very large linear layer.
            # The preceding strided convolutions produce a 4x4 map for the
            # 15x15 board. A 2x2 target divides evenly and is supported by
            # Apple's MPS backend.
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        with torch.no_grad():
            sample = torch.zeros(1, 3, self.board_size, self.board_size)
            flattened_size = self.features(sample).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flattened_size + 4, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, action_dim),
        )

    def forward(self, x):
        grid = x[:, :self.grid_size]
        direction = x[:, self.grid_size:]

        grid = grid.reshape(-1, 3, self.board_size, self.board_size)
        features = self.features(grid)
        features = torch.flatten(features, start_dim=1)
        combined = torch.cat([features, direction], dim=1)
        return self.head(combined)
