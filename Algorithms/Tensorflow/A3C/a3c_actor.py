import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Lambda

import tensorflow as tf


def build_network(state_dim, action_dim, action_bound):
    state_input = Input((state_dim,))
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    out_mu = Dense(action_dim, activation='tanh')(h3)
    std_output = Dense(action_dim, activation='softplus')(h3)

    mu_output = Lambda(lambda x: x * action_bound)(out_mu)
    model = Model(state_input, [mu_output, std_output])
    model.summary()
   # model._make_predict_function()  # class 안에서 def가 정의되면 필요없음
    return model, model.trainable_weights, state_input


class Global_Actor(object):
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

    # actor prediction
    def predict(self, state):
        mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_actor.h5')


class Worker_Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, entropy_beta, global_actor):
        self.sess = sess
        self.global_actor = global_actor

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        self.std_bound = [1e-2, 1]  # std bound

        self.model, self.theta, self.states = build_network(self.state_dim,
                                                           self.action_dim,
                                                           self.action_bound)

        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
        self.advantages = tf.placeholder(tf.float32, [None, 1])

        # policy pdf
        mu_a, std_a = self.model.output
        log_policy_pdf, entropy = self.log_pdf(mu_a, std_a, self.actions)

        # calculate local gradients
        loss_policy = log_policy_pdf * self.advantages
        loss = tf.reduce_sum(-loss_policy - entropy_beta * entropy)
        dj_dtheta = tf.gradients(loss, self.theta)

        # gradient clipping
        dj_dtheta, _ = tf.clip_by_global_norm(dj_dtheta, 40)  # 40

        grads = zip(dj_dtheta, self.global_actor.theta)
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.log(var * 2 * np.pi)
        entropy = 0.5 * (tf.log(2 * np.pi * std ** 2) + 1.0)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True), tf.reduce_sum(entropy, 1, keepdims=True)

    def get_action(self, state):
        mu_a, std_a = self.model.predict(np.reshape(state, [1, self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)

        return action

    def train(self, states, actions, advantages):
        self.sess.run(self.actor_optimizer, feed_dict={
            self.states: states,
            self.actions: actions,
            self.advantages: advantages
        })