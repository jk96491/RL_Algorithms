import tensorflow as tf
import numpy as np


def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
        return tf.layers.dense(x, units=output_size, activation=last_activation)


class REINFORCE():
    def __init__(self, obs_dim, act_dim, learning_rate):
        self.obs_input = tf.placeholder(shape=[None, obs_dim[0]], dtype=tf.float32, name='obs')
        self.act_input = tf.placeholder(shape=[None, ], dtype=tf.int32, name='act')
        self.return_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='return')

        # policy
        self.p_logits = mlp(self.obs_input, [64], act_dim, activation=tf.tanh)
        self. act_multn = tf.squeeze(tf.random.multinomial(self.p_logits, 1))
        self.action_mask = tf.one_hot(self.act_input, depth=act_dim)

        self.p_log = tf.reduce_sum(self.action_mask * tf.nn.log_softmax(self.p_logits), axis=1)

        self.p_loss = -tf.reduce_mean(self.p_log * self.return_input)

        self.p_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.p_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def GetAction(self, obs):
        action = self.sess.run(self.act_multn, feed_dict={self.obs_input: obs})
        return action

    def Train(self, obs_batch, act_batch, ret_batch):
        self.sess.run(self.p_opt, feed_dict={self.obs_input: obs_batch, self.act_input: act_batch, self.return_input: ret_batch})


class REINFORCE_BASELINE(REINFORCE):
    def __init__(self, obs_dim, act_dim, p_learning_rate, v_learning_rate):
        REINFORCE.__init__(self, obs_dim, act_dim, p_learning_rate)

        self.rtg_ph = tf.placeholder(shape=(None, ), dtype=tf.float32, name='rtg')

        self.s_values = tf.squeeze(mlp(self.obs_input, [64], 1, activation=tf.nn.tanh))
        self.v_loss = tf.reduce_mean((self.rtg_ph - self.s_values) ** 2)
        self.v_opt = tf.train.AdamOptimizer(v_learning_rate).minimize(self.v_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def GetAction(self, obs):
        act = REINFORCE.GetAction(self, obs)
        val = self.sess.run(self.s_values, feed_dict={self.obs_input: obs})
        return act, val

    def Train(self, obs_batch, act_batch, ret_batch, rtg_batch):
        REINFORCE.Train(self, obs_batch, act_batch, ret_batch)
        self.sess.run(self.v_opt, feed_dict={self.obs_input: obs_batch, self.rtg_ph: rtg_batch})



def discounted_reward(rews, gamma):
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1]
    for i in reversed(range(len(rews) - 1)):
        rtg[i] = rews[i] + gamma * rtg[i + 1]
    return rtg


class Buffer():
    def __init__(self, gamma, use_baseLine_):
        self.gamma = gamma
        self.obs = []
        self.act = []
        self.ret = []
        self.rtg = []
        self.use_baseLine = use_baseLine_

    def store(self, temp_traj):
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:, 0])
            rtg = discounted_reward(temp_traj[:, 1], self.gamma)
            self.act.extend(temp_traj[:, 2])

            if self.use_baseLine:
                self.ret.extend(rtg - temp_traj[:, 3])
                self.rtg.extend(rtg)
            else:
                self.ret.extend(rtg)

    def get_batch(self):
        b_ret = self.ret
        return self.obs, self.act, b_ret, self.rtg;

    def __len__(self):
        if self.use_baseLine:
            assert (len(self.obs) == len(self.act) == len(self.ret) == len(self.rtg))
        else:
            assert (len(self.obs) == len(self.act) == len(self.ret))
        return len(self.obs)

