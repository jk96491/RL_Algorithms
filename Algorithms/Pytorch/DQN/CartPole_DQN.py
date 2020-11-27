import numpy as np
import random
from collections import deque
import gym
import torch.nn as nn
import torch
from Pytorch.Utils import convertToTensorInput

env = gym.make('CartPole-v0')

input_size = 4
output_size = 2

dis = 0.9
REPLAY_MEMORY = 50000


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, state):
        x = self.fc1(state)
        q_val = self.fc2(x)

        return q_val


def simple_replay_train(mainDQN, targetDQN, train_batch, optimizer):
    Q_val_List = []
    Q_target_val_List = []

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN(convertToTensorInput(state, input_size))
        Q1 = targetDQN(convertToTensorInput(next_state, input_size))
        maxQ1 = torch.max(Q1.data)

        if done:
            Q1[0, action] = reward
        else:
            Q1[0, action] = reward + torch.mul(maxQ1, dis)

        Q_val_List.append(Q)
        Q_target_val_List.append(Q1)

    Q_val_List = torch.stack(Q_val_List).squeeze(1)
    Q_target_val_List = torch.stack(Q_target_val_List).squeeze(1)
    loss = torch.mean((Q_val_List - Q_target_val_List) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0

    while True:
        env.render()
        a = np.argmax(mainDQN(s))
        s, reward, done, _ = env.step(a)

        reward_sum += reward
        if done:
            print("Total score:{}".format(reward_sum))
            break


def update_target(mainDQN, targetDQN):
    targetDQN.load_state_dict(mainDQN.state_dict())


def running():
    max_episode = 5000
    replay_buffer = deque()

    mainDQN = DQN(input_size, output_size)
    targetDQN = DQN(input_size, output_size)

    update_target(mainDQN, targetDQN)

    optimizer = torch.optim.Adam(mainDQN.parameters(), lr=1e-1)

    for episode in range(max_episode):
        e = 1/((episode / 10) + 1)
        done = False
        step_count = 0

        state = env.reset()

        while not done:
            # e-greedy 기법
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                q_val = mainDQN(convertToTensorInput(state, input_size))
                _, action = torch.max(q_val, 1)
                action = action.data[0].item()

            # gym으로 부터 정보 받아옴
            next_state, reward, done, _ = env.step(action)

            # 정보를 쌓아 둠
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state
            step_count += 1

            if step_count > 10000:
                break
            env.render()

        print("Episode: {} steps: {}".format(episode, step_count))
        if step_count > 10000:
            pass

        # 10회 주기로 미니배칭하여 타켓 신경망 업데이트
        if episode % 30 == 1:
            for _ in range(50):
                minibatch = random.sample(replay_buffer, 30)
                loss = simple_replay_train(mainDQN, targetDQN, minibatch, optimizer)
            print("Loss", loss.item())
            update_target(mainDQN, targetDQN)

    bot_play(mainDQN)

    env.close()

running()