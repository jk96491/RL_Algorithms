import torch
import torch.nn as nn
import Pytorch.Utils as Utils


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 256),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(256, 2 * action_dim))

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x).chunk(2, dim=-1)[0]
        x = torch.tanh(x)

        return x

    def sample(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        means, log_stds = self.fc3(x).chunk(2, dim=-1)

        return Utils.reparameterize(means, log_stds.clamp_(-20, 2))
