import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Sequential(nn.Linear(state_size, 128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(130, 128),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, 128),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, state, action):
        x = self.layer1(state)

        x = torch.cat([x, action], dim=-1)

        x = self.layer2(x)
        x = self.layer3(x)
        q_val = self.layer4(x)

        return q_val