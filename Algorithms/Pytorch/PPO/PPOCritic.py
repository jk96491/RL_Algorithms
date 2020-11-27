import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = lr

        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16),
                                 nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(16, action_dim),
                                 nn.ReLU())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def predict(self, state):
        v = self.forward(state)
        return v

    def Learn(self, states, td_target):
        td_target = torch.FloatTensor(td_target)
        predict = self.forward(states)

        loss = torch.mean((predict - td_target) **2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
