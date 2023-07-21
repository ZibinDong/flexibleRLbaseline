from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import models
import replaybuffers
import utils

from .agent import DiscreteAgent


class DQNAgent(DiscreteAgent):
    def __init__(self, 
                 
        observation_dim: int, 
        action_dim: int,
        
        critic_hidden_dims: list = [512, 512],
        activation: Union[torch.nn.Module, str] = 'relu',
        
        critic_learning_rate: float = 3e-4,
        batch_size: int = 512,
        
        max_grad_norm: float = 10.,
        
        gamma: float = 0.99,
        tau: float = 0.005,
        n_q_nets: int = 2,
        
        max_replay_buffer_size: int = 500_000,
        
        exploration_scheme: str = "epsilon-greedy",
        exploration_noise: float = 0.1,
        
        device: str = 'cpu',
    ):
        super().__init__(observation_dim, action_dim)
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.exploration_scheme = exploration_scheme
        self.exploration_noise = exploration_noise
        
        activation = utils.str2activation(activation)
        
        self.critic = models.DiscreteCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            tau=tau,
            n_q_nets=n_q_nets,
        ).to(device)
        
        self.buffer = replaybuffers.BasicReplayBuffer(
            observation_dim=observation_dim,
            action_dim=0, # discrete action
            max_replay_buffer_size=max_replay_buffer_size,
            device=device,
        )
    
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        with utils.EvalModules([self.critic]):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)[None,]
            q_values = self.critic(obs)
            if deterministic:
                act = q_values.argmax(dim=1).cpu().numpy()
            else:
                if self.exploration_scheme == "epsilon-greedy":
                    if np.random.rand() < self.exploration_noise:
                        act = np.random.randint(self.action_dim)
                    else:
                        act = q_values.argmax(dim=1).cpu().numpy()[0]
                elif self.exploration_scheme == "boltzmann":
                    act = torch.multinomial(F.softmax(q_values, dim=1), 1).cpu().numpy()[0]
                else:
                    raise NotImplementedError(f"Unknown exploration scheme: {self.exploration_scheme}")
            return act
            
    def update(self):
        log = {}
        (obs, act, rew, done, next_obs) = self.buffer.sample(self.batch_size)
        self.update_critic(obs, act, rew, done, next_obs, log)
        return log

    def update_critic(self, obs, act, rew, done, next_obs, log):
        
        with torch.no_grad():
            next_q = self.critic(next_obs, use_target_net=True)
            target_q = rew + self.gamma * (1 - done) * next_q.max(-1, keepdim=True).values
        
        q = self.critic(obs, return_q_min=False)
        q = torch.gather(
            q, -1, act[:,None,None].repeat(1, q.shape[1], 1))
 
        loss_critic = ((q - target_q.unsqueeze(1))**2).mean()
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        
        log['loss_critic'] = loss_critic.item()
    
    def save(self, path: Path, step: int):
        torch.save({
            'step': step,
            'critic': self.critic.state_dict(),
        }, path / f"dqn_{utils.abbreviate_number(step)}.pt")
        
    def load(self, path: Path, filename: str):
        checkpoint = torch.load(path / filename)
        self.critic.load_state_dict(checkpoint['critic'])
        
        
class HighwayDQNAgent(DiscreteAgent):
    def __init__(self, 
                 
        n_vehicles: int,
        n_features: int, 
        action_dim: int,
        
        critic_learning_rate: float = 3e-4,
        batch_size: int = 512,

        max_grad_norm: float = 10.,
        
        gamma: float = 0.99,
        tau: float = 0.005,
        n_q_nets: int = 2,
        
        max_replay_buffer_size: int = 500_000,
        
        exploration_scheme: str = "epsilon-greedy",
        exploration_noise: float = 0.1,
        
        device: str = 'cpu',
    ):
        super().__init__(n_features, action_dim)
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.exploration_scheme = exploration_scheme
        self.exploration_noise = exploration_noise

        self.critic = models.HighwayDiscreteCritic(
            n_features = n_features,
            action_dim = action_dim,
            tau = tau,
            n_q_nets = n_q_nets
        ).to(device)
        
        self.buffer = replaybuffers.BasicReplayBuffer(
            observation_dim=n_vehicles * n_features,
            action_dim=0, # discrete action
            max_replay_buffer_size=max_replay_buffer_size,
            device=device,
        )
    
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def _auto_decision(self, obs):
        
        be_stucked = False
        
        for i in range(obs.shape[0]):
            # 车辆存在 且 在同一车道 且 距离太近
            if i != 0 and obs[i, 0] == 1. and obs[i, 2] == 0. and obs[i, 1] > 0. and obs[i, 1]*200. < 4.0 * 5.:
                be_stucked = True
                break
        
        if be_stucked:
            left_available = obs[0, 2] > 0
            right_available = obs[0, 2] < 0.75
            left_count = 0
            right_count = 0
            for i in range(obs.shape[0]):
                if i != 0 and obs[i, 0] == 1. and obs[i, 2] == 0.25:
                    right_count += 1
                    if obs[i, 1]*200. < 4.0 * 5. and obs[i, 1] > 0:
                        right_available = False
                if i != 0 and obs[i, 0] == 1. and obs[i, 2] == -0.25:
                    left_count += 1
                    if obs[i, 1]*200. < 4.0 * 5. and obs[i, 1] > 0:
                        left_available = False
        
        if not be_stucked: return None
        else:
            if left_available and right_available:
                if left_count < right_count: return 0
                else: return 2
            elif left_available: return 0
            elif right_available: return 2
            else: return None
        
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        act = self._auto_decision(obs.reshape(-1, 5))
        if act is not None: return act
        with utils.EvalModules([self.critic]):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)[None,].reshape(1, -1)
            q_values = self.critic(obs)
            if deterministic:
                act = q_values.argmax(dim=1).cpu().numpy()
            else:
                if self.exploration_scheme == "epsilon-greedy":
                    if np.random.rand() < self.exploration_noise:
                        act = np.random.randint(self.action_dim)
                    else:
                        act = q_values.argmax(dim=1).cpu().numpy()[0]
                elif self.exploration_scheme == "boltzmann":
                    act = torch.multinomial(F.softmax(q_values, dim=1), 1).cpu().numpy()[0]
                else:
                    raise NotImplementedError(f"Unknown exploration scheme: {self.exploration_scheme}")
            return act
            
    def update(self):
        log = {}
        (obs, act, rew, done, next_obs) = self.buffer.sample(self.batch_size)
        self.update_critic(obs, act, rew, done, next_obs, log)
        return log

    def update_critic(self, obs, act, rew, done, next_obs, log):
        
        with torch.no_grad():
            next_q = self.critic(next_obs, use_target_net=True)
            target_q = rew + self.gamma * (1 - done) * next_q.max(-1, keepdim=True).values
        
        q = self.critic(obs, return_q_min=False)
        q = torch.gather(
            q, -1, act[:,None,None].repeat(1, q.shape[1], 1))
 
        loss_critic = ((q - target_q.unsqueeze(1))**2).mean()
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        
        log['loss_critic'] = loss_critic.item()
        
    def save(self, path: Path, step: int):
        torch.save({
            'step': step,
            'critic': self.critic.state_dict(),
        }, path / f"highway_dqn_{utils.abbreviate_number(step)}.pt")
        
    def load(self, path: Path, filename: str):
        checkpoint = torch.load(path / filename)
        self.critic.load_state_dict(checkpoint['critic'])