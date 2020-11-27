import random
import tensorflow as tf
import numpy as np
from Practice.DDPG.ACModels import Actor
from Practice.DDPG.ACModels import Critic
from Practice.DDPG.Noise import OU_noise
from collections import deque

actor_lr = 1e-4
critic_lr = 5e-4

batch_size = 128

mem_maxlen = 5000

tau = 1e-3

discount_factor = 0.99


class DDPGAgent():
    def __init__(self, state_size, action_size, train_mode_, load_model_):
        self.train_mode = train_mode_
        self.load_model = load_model_

        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor('actor', state_size, action_size)
        self.critic = Critic('critic', state_size, action_size)
        self.target_actor = Actor('target_actor', state_size, action_size)
        self.target_critic = Critic('target_critic', state_size, action_size)

        self.target_q = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.losses.mean_squared_error(self.target_q, self.critic.predict_q)
        self.train_critic = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

        action_gradient = tf.gradients(tf.squeeze(self.critic.predict_q), self.critic.action)
        policy_gradient = tf.gradients(self.actor.action, self.actor.trainable_var, action_gradient)
        for idx, gradients in enumerate(policy_gradient):
            policy_gradient[idx] -= gradients / batch_size

        self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(policy_gradient, self.actor.trainable_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.OU = OU_noise(action_size)
        self.memory = deque(maxlen=mem_maxlen)

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            soft_tau = tau * self.actor.trainable_var[idx].value() + (1 - tau) * self.target_actor.trainable_var[idx].value()
            self.target_actor.trainable_var[idx].assign(soft_tau)
            self.soft_update_target.append(self.target_actor.trainable_var[idx])

        for idx in range(len(self.critic.trainable_var)):
            soft_tau = tau * self.critic.trainable_var[idx].value() + (1 - tau) * self.target_critic.trainable_var[idx].value()
            self.target_critic.trainable_var[idx].assign(soft_tau)
            self.soft_update_target.append(self.target_critic.trainable_var[idx])

        init_update_target = []

        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(self.critic.trainable_var[idx]))

        self.sess.run(init_update_target)

    def get_action(self, state):
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})
        noise = self.OU.sample()

        if self.train_mode:
            return action + noise
        else:
            return action

    def append_sample(self, state, action, rewards, next_state, done):
        self.memory.append((state, action, rewards, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)

        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions = self.sess.run(self.target_actor.action, feed_dict={self.target_actor.state: next_states})
        target_critic_predict_qs = self.sess.run(self.target_critic.predict_q, feed_dict={self.target_critic.state: next_states,
                                                                                          self.target_critic.action: target_actor_actions})

        target_qs = np.asarray([reward + discount_factor * (1 - done) *
                                target_critic_predict_q for reward, target_critic_predict_q, done in
                                zip(rewards, target_critic_predict_qs, dones)])

        self.sess.run(self.train_critic, feed_dict={self.critic.state: states, self.critic.action: actions,
                                                    self.target_q: target_qs})

        actions_for_train = self.sess.run(self.actor.action, feed_dict={self.actor.state: states})
        self.sess.run(self.train_actor, feed_dict={self.actor.state: states, self.critic.state: states,
                                                   self.critic.action: actions_for_train})

        self.sess.run(self.soft_update_target)
















