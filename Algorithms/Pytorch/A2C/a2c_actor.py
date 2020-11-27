import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        self.std_bound = [1e-2, 1.0]

        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16),
                                 nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(16, action_dim),
                                 nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(16, action_dim),
                                 nn.Softplus())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        out_mu = self.fc4(x)
        std_output = self.fc5(x)

        return out_mu * self.action_bound, std_output

    def log_pdf(self, mu, std, action):
        std = std.clamp(self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = - 0.5 * (((action - mu) ** 2 / var) + (torch.log(var * 2 * np.pi)))  # 가우시안 분포
        return log_policy_pdf

    def get_action(self, state):
        mu_a, std_a = self.forward(state)

        mu_a = mu_a.item()
        std_a = std_a.item()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)

        return action

    def predict(self, state):
        mu_a, std_a = self.forward(state)
        return mu_a

    def Learn(self, states, actions, advantages):
        actions = torch.FloatTensor(actions).view(states.shape[0], 1)
        advantages = torch.FloatTensor(advantages).view(states.shape[0], 1)

        mu, std = self.forward(states)
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss = torch.sum(-log_policy_pdf * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
