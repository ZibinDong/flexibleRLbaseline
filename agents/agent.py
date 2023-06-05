from pathlib import Path

import numpy as np
import torch

import models
import replaybuffers


class Agent():
    def __init__(self,
        observation_dim: int,
        action_dim: int,
    ):
        self.obersevation_dim = observation_dim
        self.action_dim = action_dim
        
        self.actor: models.Actor
        self.critic: models.Critic
        self.buffer: replaybuffers.BasicReplayBuffer
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        raise NotImplementedError
    
    def update(self):
        log = {}
        return log
    
    def save(self, path: Path, step: int):
        raise NotImplementedError
    def load(self, path: Path, filename: str):
        raise NotImplementedError

class ContinuousAgent(Agent):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        act = np.random.uniform(-1, 1, size=(self.action_dim,), dtype=np.float32)
        return act


class DiscreteAgent(Agent):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        act = np.random.randint(0, self.action_dim, size=(1,), dtype=np.int64)
        return act