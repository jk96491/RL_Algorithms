import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(state_size, 128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 128),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, action_size),
                                    nn.Tanh())

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        action = self.layer3(x)

        return action
