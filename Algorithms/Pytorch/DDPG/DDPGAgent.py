import torch
from Pytorch.DDPG.Actor import Actor
from Pytorch.DDPG.Critic import Critic
import copy
from Pytorch.DDPG.Noise import OU_noise
from collections import deque
import random
from Pytorch.Utils import convertToTensorInput

batch_size = 128
mem_maxlen = 5000
discount_factor = 0.99

actor_lr = 1e-4
critic_lr = 5e-4

tau = 1e-3


class Agent():
    def __init__(self, state_size, action_size, train_mode_, load_model_):
        self.train_mode = train_mode_
        self.load_model = load_model_

        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size, self.action_size)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.OU = OU_noise(action_size)
        self.memory = deque(maxlen=mem_maxlen)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state):
        action = self.actor(convertToTensorInput(state, self.state_size)).detach().numpy()

        noise = self.OU.sample()

        if self.train_mode:
            return action + noise
        else:
            return action

    def append_sample(self, state, action, rewards, next_state, done):
        self.memory.append((state, action, rewards, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([sample[0] for sample in mini_batch])
        actions = torch.FloatTensor([sample[1] for sample in mini_batch])
        rewards = torch.FloatTensor([sample[2] for sample in mini_batch])
        next_states = torch.FloatTensor([sample[3] for sample in mini_batch])
        dones = torch.FloatTensor([sample[4] for sample in mini_batch])

        target_actor_actions = self.target_actor(next_states)
        target_critic_predict_qs = self.target_critic(next_states, target_actor_actions)

        target_qs = [reward + discount_factor * (1 - done) *
                                target_critic_predict_q for reward, target_critic_predict_q, done in
                                zip(rewards, target_critic_predict_qs, dones)]
        target_qs = torch.FloatTensor([target_qs])

        q_val = self.critic(states, actions)

        critic_loss = torch.mean((q_val - target_qs)**2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target(self.target_critic, self.critic)
        self.soft_update_target(self.target_actor, self.actor)

        return actor_loss

    def soft_update_target(self, target, orign):
        for target_param, orign_param in zip(target.parameters(), orign.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * orign_param.data)





