import numpy as np
import torch
import torch.optim as optim
from Pytorch.SAC.Actor import Actor
from Pytorch.SAC.Critic import Critic
from Pytorch.SAC.memory import ReplayMemory


class SAC:
    def __init__(self, env, alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4, reward_scale=1.0, entropy_tune=True, writer=None):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Seed
        np.random.seed(0)
        torch.manual_seed(0)

        # models
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device).eval()

        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay Memory
        self.memory = ReplayMemory(self.state_dim, self.action_dim)

        # const
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        # Entropy Tune
        self.entropy_tune = entropy_tune
        if entropy_tune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape)).item()
            self.log_alpha = torch.tensor([0.2], requires_grad=True, device='cpu')
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        if writer is not None:
            self.writer = writer

    def store_transition(self, state, action, next_state, reward, done):
        return self.memory.store_transition(state, action, next_state, reward, done)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def get_deterministic_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def update(self):
        states, actions, next_states, rewards, terminals = self.memory.sample(batch_size=256)
        loss_critic = self.update_critic(states, actions, next_states, rewards, terminals)
        loss_actor, loss_entropy = self.update_actor(states)
        self.update_target()
        return loss_critic, loss_actor, loss_entropy

    def update_critic(self, states, actions, next_states, rewards, terminals):
        q1, q2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pis
        target_q = rewards * self.reward_scale + terminals * self.gamma * next_q

        loss_q1 = (q1 - target_q).pow_(2).mean()
        loss_q2 = (q2 - target_q).pow_(2).mean()

        self.critic_optimizer.zero_grad()
        (loss_q1 + loss_q2).backward(retain_graph=False)
        self.critic_optimizer.step()

        return loss_q1.item() + loss_q2.item()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)

        # entropy tuning
        loss_entropy = 0
        if self.entropy_tune:
            loss_entropy = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()

        loss_actor = (self.alpha * log_pis - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.actor_optimizer.step()

        if self.entropy_tune:
            self.alpha_optimizer.zero_grad()
            loss_entropy.backward(retain_graph=False)
            self.alpha_optimizer.step()
            return loss_actor.item(), loss_entropy.item()

        return loss_actor.item(), loss_entropy

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)