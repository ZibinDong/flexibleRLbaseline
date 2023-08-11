import numpy as np
import torch

import utils


class BasicReplayBuffer():
    def __init__(
        self,
        
        observation_dim: int,
        action_dim: int,
        max_replay_buffer_size: int,
        
        device: str = "cpu",
    ):
        self.is_action_discrete = (action_dim == 0)
        
        self._obs = torch.empty((max_replay_buffer_size, observation_dim), \
            dtype=torch.float32, device=device)
        self._next_obs = torch.empty((max_replay_buffer_size, observation_dim), \
            dtype=torch.float32, device=device)
        
        if self.is_action_discrete:
            self._act = torch.empty((max_replay_buffer_size,), \
                dtype=torch.long, device=device)
        else:
            self._act = torch.empty((max_replay_buffer_size, action_dim), \
                dtype=torch.float32, device=device)
        self._rew = torch.empty((max_replay_buffer_size,), \
            dtype=torch.float32, device=device)
        self._done = torch.empty((max_replay_buffer_size,), \
            dtype=torch.float32, device=device)
        
        self.device = device
        self.max_replay_buffer_size = max_replay_buffer_size
        
        self.ptr, self.size = 0, 0
        
    def __len__(self):
        return self.size
    
    def __add__(self, *args):
        self.add(*args)
        
    def add(self, obs, act, rew, done, next_obs):
        self._obs[self.ptr] = utils.to_torch(obs, device=self.device)
        self._next_obs[self.ptr] = utils.to_torch(next_obs, device=self.device)
        self._act[self.ptr] = utils.to_torch(act, device=self.device)
        self._rew[self.ptr] = float(rew)
        self._done[self.ptr] = float(done)
        self.ptr = (self.ptr+1) % self.max_replay_buffer_size
        self.size = min(self.size+1, self.max_replay_buffer_size)
        
    def sample(self, batch_size):
        indeces = torch.randint(0, self.size, (batch_size,))
        return (
            self._obs[indeces],
            self._act[indeces],
            self._rew[indeces, None],
            self._done[indeces, None],
            self._next_obs[indeces],
        )