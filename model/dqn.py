import torch 
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layer_dim=256):

        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = nn.Linear(hidden_layer_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
