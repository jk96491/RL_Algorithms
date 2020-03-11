import tensorflow as tf


class Critic():
    def __init__(self, name, state_size, action_size):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.fc1, self.action], axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)


class Actor():
    def __init__(self, name, state_size, action_size):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc2, action_size, activation=tf.nn.tanh)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
