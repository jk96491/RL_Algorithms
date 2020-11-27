import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    env = gym.make('CartPole-v1')
    policy = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        state = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = policy(torch.from_numpy(state).float())
            m = Categorical(prob)
            action = m.sample()
            next_state, reward, done, info = env.step(action.item())
            policy.put_data((reward, prob[action]))
            state = next_state
            score += reward
            env.render()

        policy.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
