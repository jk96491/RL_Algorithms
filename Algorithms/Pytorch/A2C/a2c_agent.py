import numpy as np
import matplotlib.pyplot as plt
from Pytorch.Utils import convertToTensorInput

from Pytorch.A2C.a2c_critic import Critic
from Pytorch.A2C.a2c_actor import Actor


class A2Cagnet(object):
    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = v_value
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value

        return advantage, y_k

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack

    def train(self, max_episode_num):
        for ep in range(max_episode_num):
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

            time, episode_reward, done = 0, 0, False

            state = self.env.reset()

            while not done:
                self.env.render()

                action = self.actor.get_action(convertToTensorInput(state, self.state_dim))
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])

                v_value = self.critic.predict(convertToTensorInput(state, self.state_dim)).item()
                next_v_value = self.critic.predict(convertToTensorInput(next_state, self.state_dim)).item()

                train_reward = (reward + 8) / 8
                advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)

                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)

                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                advantages = self.unpack_batch(batch_advantage)

                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                self.critic.Learn(convertToTensorInput(states, self.state_dim, states.shape[0]), td_targets)
                self.actor.Learn(convertToTensorInput(states, self.state_dim, states.shape[0]), actions, advantages)

                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights('pendulum_actor.th')
                self.critic.save_weights('pendulum_critic.th')

        np.savetxt('save_weights/pendulum_epi_reward.txt', self.save_epi_reward)

    def plot_Result(self):
        plt.plot(self.save_epi_reward)
        plt.show()



























