import gym
import tensorflow as tf
from Practice.REINFORCE.REINFORCE_MODELS import REINFORCE
from Practice.REINFORCE.REINFORCE_MODELS import REINFORCE_BASELINE
from Practice.REINFORCE.REINFORCE_MODELS import Buffer
import numpy as np

num_epochs = 500
discount_factor = 0.99
steps_per_epoch = 100
step_count = 0

p_learning_rate = 5e-3
v_learning_rate = 8e-3

use_baseLine = True

env = gym.make("CartPole-v0")

if __name__ == '__main__':

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    if use_baseLine:
        agent = REINFORCE_BASELINE(obs_dim, act_dim, p_learning_rate, v_learning_rate)
    else:
        agent = REINFORCE(obs_dim, act_dim, p_learning_rate)

    train_rewards = []
    train_ep_len = []
    step_count = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for episode in range(num_epochs):
        obs = env.reset()

        buffer = Buffer(discount_factor, use_baseLine)
        env_buf = []
        episode_rewards = []

        while len(buffer) < steps_per_epoch:
            env.render()

            if use_baseLine:
                action, val = agent.GetAction([obs])
                obs2, reward, done, _ = env.step(action)
                env_buf.append([obs.copy(), reward, action, np.squeeze(val)])
            else:
                action = agent.GetAction([obs])
                obs2, reward, done, _ = env.step(action)
                env_buf.append([obs.copy(), reward, action])

            obs = obs2.copy()
            step_count += 1
            episode_rewards.append(reward)

            if done:
                buffer.store(np.array(env_buf))
                env_buf = []
                train_rewards.append(np.sum(episode_rewards))
                train_ep_len.append(len(episode_rewards))
                obs = env.reset()
                episode_rewards = []

            obs_batch, act_batch, ret_batch, rtg_batch = buffer.get_batch()

        if use_baseLine:
            agent.Train(obs_batch, act_batch, ret_batch, rtg_batch)
        else:
            agent.Train(obs_batch, act_batch, ret_batch)

        if episode % 10 == 0:
            print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d' % (
            episode, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count))

    env.close()