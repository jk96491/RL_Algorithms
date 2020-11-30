import gym
import torch
from torch.distributions import Categorical
from Pytorch.REINFORCE.PolicyNet import Policy

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

def main():
    env = gym.make('CartPole-v1')
    policy = Policy(learning_rate, gamma)
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
