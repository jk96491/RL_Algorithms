from keras.models import Model
from keras.layers import Dense, Input

import tensorflow as tf


def build_network(state_dim):
    state_input = Input((state_dim,))
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    v_output = Dense(1, activation='linear')(h3)
    model = Model(state_input, v_output)
    #model.summary()
    model._make_predict_function()  # class 안에서 def가 정의되면 필요없음
    return model, model.trainable_weights, state_input


class Global_Critic(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim

        self.model, self.phi, _ = build_network(state_dim)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5')


class Worker_Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, global_critic):
        self.sess = sess
        self.global_critic = global_critic

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.model, self.phi, self.states = build_network(self.state_dim)

        self.td_targets = tf.placeholder(tf.float32, [None, 1])

        v_values = self.model.output
        loss = tf.reduce_sum(tf.square(self.td_targets - v_values))
        dj_dphi = tf.gradients(loss, self.phi)

        # gradient clipping
        dj_dphi, _ = tf.clip_by_global_norm(dj_dphi, 40)  # 40

        grads = zip(dj_dphi, self.global_critic.phi)
        self.critic_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def train(self, states, td_targets):
        self.sess.run(self.critic_optimizer, feed_dict={
            self.states: states,
            self.td_targets: td_targets
        })