import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        input_tensor = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.fc1(input_tensor), inplace=True)
        q1 = F.relu(self.fc2(q1), inplace=True)
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(input_tensor), inplace=True)
        q2 = F.relu(self.fc5(q2), inplace=True)
        q2 = self.fc6(q2)

        return q1, q2
