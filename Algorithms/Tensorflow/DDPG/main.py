import gym
from Practice.DDPG.DDPG import DDPGAgent
import numpy as np
from collections import deque

env = gym.make("CartPole-v0")

train_mode = True
load_model = False

num_epochs = 5000
discount_factor = 0.99
steps_per_epoch = 100
step_count = 0
step = 0

start_train_episode = 50

print_interval = 1

loss = None

if __name__ == '__main__':
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DDPGAgent(state_size, action_size, train_mode, load_model)

    rewards = deque(maxlen=print_interval)

    for episode in range(num_epochs):

        state = env.reset()
        done = False
        step = 0

        while not done:
            env.render()
            step += 1

            action = agent.get_action([state])[0]
            next_state, reward, done, _ = env.step(np.argmax(action))

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)

            state = next_state

            if episode > start_train_episode and train_mode:
                loss = agent.train_model()

        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / loss: {} ".format(step, episode, loss))

