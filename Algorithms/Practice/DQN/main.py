import gym
from Practice.DQN.DQN import DQNAgent
import numpy as np

env = gym.make("CartPole-v0")
env.reset()

state_size = 4
action_size = 2

num_episode = 20000

start_train_episode = 50

target_update_step = 100

rList = []

if __name__ == '__main__':
    agent = DQNAgent(0.7, state_size, action_size)

    for episode in range(num_episode):
        state = env.reset()
        step_count = 0
        done = False
        loss = None

        while not done:
            env.render()
            step_count += 1

            action = agent.GetAction(state)
            next_state, reward, done, _ = env.step(action)

            agent.AppendReplayMemory(state, action, reward, next_state, done)
            state = next_state

            if episode > start_train_episode:
                loss = agent.Train(done)

                if step_count % target_update_step == 0:
                    agent.update_Target()

        rList.append(step_count)

        print("Episode: {} steps: {} loss: {}".format(episode, step_count, loss))

        if len(rList) < 10 and np.mean(rList[-10:]) > 500:
            break

    env.close()