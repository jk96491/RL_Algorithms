import numpy as np
import torch


class ReplayMemory:
    def __init__(self, state_dim, action_dim, device='cpu', max_size=int(32)):
        self.max_size = max_size
        self.state_memory = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)
        self.action_memory = torch.zeros((self.max_size, action_dim), dtype=torch.float, device=device)
        self.next_state_memory = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)
        self.reward_memory = torch.zeros(self.max_size, dtype=torch.float, device=device)
        self.terminal_memory = torch.zeros(self.max_size, dtype=torch.uint8, device=device)
        self.mem_ctrl = 0

    def store_transition(self, state, action, next_state, reward, done):
        index = self.mem_ctrl % self.max_size
        self.state_memory[index] = torch.from_numpy(state)
        self.action_memory[index] = torch.from_numpy(action)
        self.next_state_memory[index] = torch.from_numpy(next_state)
        self.reward_memory[index] = torch.from_numpy(np.array([reward]).astype(np.float))
        self.terminal_memory[index] = torch.from_numpy(np.array([1. - done]).astype(np.uint8))
        self.mem_ctrl += 1

        need_reset = False
        if self.mem_ctrl >= self.max_size:
            need_reset = True
            self.mem_ctrl = 0

        return need_reset

    def sample(self, batch_size=256):
        batch_idx = np.random.choice(1, batch_size)

        states = self.state_memory[batch_idx]
        actions = self.action_memory[batch_idx]
        next_states = self.next_state_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        terminals = self.terminal_memory[batch_idx]

        return states, actions, next_states, rewards, terminals

    def __len__(self):
        return self.mem_ctrl