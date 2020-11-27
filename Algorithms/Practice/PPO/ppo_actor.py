import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Lambda

import tensorflow as tf


class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, ratio_clipping):
        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.ratio_clipping = ratio_clipping

        self.std_bound = [1e-2, 1.0]

        self.model, self.theta, self.states = self.build_network()

        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
        self.advantages = tf.placeholder(tf.float32, [None, 1])

        # 이전 policy 확률밀도함수
        self.log_old_policy_pdf = tf.placeholder(tf.float32, [None, 1])

        # 현재 policy 확률밀도함수
        mu_a, std_a = self.model.output
        log_policy_pdf = self.log_pdf(mu_a, std_a, self.actions)

        # 이전 policy 와  현재 policy 비율
        ratio = tf.exp(log_policy_pdf - self.log_old_policy_pdf)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.ratio_clipping, 1.0 + self.ratio_clipping)
        surrogate = -tf.minimum(ratio * self.advantages, clipped_ratio * self.advantages)
        loss = tf.reduce_mean(surrogate)
        dj_dtheta = tf.gradients(loss, self.theta)
        grads = zip(dj_dtheta, self.theta)
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out_mu = Dense(self.action_dim, activation='tanh')(h3)
        std_output = Dense(self.action_dim, activation='softplus')(h3)

        # Scale output to [-action_bound, action_bound]
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        model = Model(state_input, [mu_output, std_output])
        model.summary()
        return model, model.trainable_weights, state_input

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_policy_action(self, state):
        mu_a, std_a = self.model.predict(np.reshape(state, [1, self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return mu_a, std_a, action

    def predict(self, state):
        mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    def train(self, log_old_policy_pdf, states, actions, advantages):
        self.sess.run(self.actor_optimizer, feed_dict={
            self.log_old_policy_pdf: log_old_policy_pdf,
            self.states: states,
            self.actions: actions,
            self.advantages: advantages
        })

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_actor.h5')