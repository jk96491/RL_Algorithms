import numpy as np
import numpy.random as nr

mu = 0
theta = 0.15
sigma = 0.3

class OU_noise:
    def __init__(self, action_size):
        self.action_dimension = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state