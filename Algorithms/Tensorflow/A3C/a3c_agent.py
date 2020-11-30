import gym

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

import threading
import multiprocessing

from Tensorflow.A3C.a3c_actor import Global_Actor, Worker_Actor
from Tensorflow.A3C.a3c_critic import Global_Critic, Worker_Critic

# shared global parameters across all workers
global_episode_count = 0
global_step = 0
global_episode_reward = []  # save the results


class A3Cagent(object):
    def __init__(self, env_name):

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.env_name = env_name
        self.WORKERS_NUM = multiprocessing.cpu_count()

        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        self.global_actor = Global_Actor(state_dim, action_dim, action_bound)
        self.global_critic = Global_Critic(state_dim)

    def train(self, max_episode_num):

        workers = []

        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(A3Cworker(worker_name, self.env_name, self.sess, self.global_actor,
                                     self.global_critic, max_episode_num))

        self.sess.run(tf.global_variables_initializer())

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        np.savetxt('pendulum_epi_reward.txt', global_episode_reward)
        print(global_episode_reward)

    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()


class A3Cworker(threading.Thread):
    def __init__(self, worker_name, env_name, sess, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        self.GAMMA = 0.95
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.ENTROPY_BETA = 0.01
        self.t_MAX = 4 # t-step prediction

        self.max_episode_num = max_episode_num

        self.env = gym.make(env_name)
        self.worker_name = worker_name
        self.sess = sess

        self.global_actor = global_actor
        self.global_critic = global_critic

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.worker_actor = Worker_Actor(self.sess, self.state_dim, self.action_dim, self.action_bound,
                                         self.ACTOR_LEARNING_RATE, self.ENTROPY_BETA, self.global_actor)
        self.worker_critic = Worker_Critic(self.sess, self.state_dim, self.action_dim,
                                           self.CRITIC_LEARNING_RATE, self.global_critic)

        self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
        self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack

    def run(self):

        global global_episode_count, global_step
        global global_episode_reward  # total episode across all workers

        print(self.worker_name, "starts ---")

        while global_episode_count <= int(self.max_episode_num):

            batch_state, batch_action, batch_reward = [], [], []

            step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self.worker_actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                action = np.reshape(action, [1, self.action_dim])

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward+8)/8)

                # update state and step
                state = next_state
                step += 1
                episode_reward += reward[0]

                if len(batch_state) == self.t_MAX or done:
                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward)

                    batch_state, batch_action, batch_reward = [], [], []

                    next_state = np.reshape(next_state, [1, self.state_dim])
                    next_v_value = self.worker_critic.model.predict(next_state)
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    v_values = self.worker_critic.model.predict(states)
                    advantages = n_step_td_targets - v_values

                    self.worker_critic.train(states, n_step_td_targets)
                    self.worker_actor.train(states, actions, advantages)

                    self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
                    self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

                    global_step += 1

                if done:
                    global_episode_count += 1
                    print('Worker name:', self.worker_name, ', Episode: ', global_episode_count,
                          ', Step: ', step, ', Reward: ', episode_reward)

                    global_episode_reward.append(episode_reward)

                    if global_episode_count % 10 == 0:
                        self.global_actor.save_weights("pendulum_actor.h5")
                        self.global_critic.save_weights("pendulum_critic.h5")
