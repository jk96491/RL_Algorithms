import tensorflow as tf
import numpy as np
import random
from collections import deque

mem_maxlen = 5000
learning_rate = 0.001
batch_size = 32
epsilon_min = 0.1
discount_factor = 0.99

save_path = ''


class Model():
    def __init__(self, state_size, action_size, model_name):
        self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.fc1 = tf.layers.dense(self.input, 512, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 512, activation=tf.nn.relu)

            self.Q_Out = tf.layers.dense(self.fc2, action_size, activation=tf.nn.relu)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_Out))
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DQNAgent():
    def __init__(self, epsilon_, state_size, action_size):
        self.action_size = action_size
        self.epsilon = epsilon_
        self.state_size = state_size

        self.model = Model(self.state_size, self.action_size, 'Q')
        self.targetModel = Model(self.state_size, self.action_size, 'Target-Q')

        self.replayMemory = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.Saver = tf.train.Saver()

    def GetAction(self, state):
        if self.epsilon >= np.random.rand():
            action = np.random.randint(0, self.action_size)
        else:
            reshape_state = np.reshape(state, [1, self.state_size])
            action = np.argmax(self.sess.run(self.model.Q_Out, feed_dict={self.model.input: reshape_state}))
        return action

    def AppendReplayMemory(self, state, action, reward, next_state, done):
        self.replayMemory.append((state, action, reward, next_state, done))

    def Save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    def Train(self, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon = 1. / ((1/10)/ + 1)

        mini_batch = random.sample(self.replayMemory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input: states})
        target_val = self.sess.run(self.targetModel.Q_Out, feed_dict={self.targetModel.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        self.sess.run(self.model.UpdateModel, feed_dict={self.model.input: states, self.model.target_Q: target})
        loss = self.sess.run(self.model.loss, feed_dict={self.model.input: states, self.model.target_Q: target})

        return loss

    def update_Target(self):
        for i in range(len(self.model.trainable_var)):
            self.sess.run(self.targetModel.trainable_var.assign(self.model.trainable_var[i]))








