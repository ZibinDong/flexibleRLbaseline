import numpy as np
import torch
import torch.distributions as D
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

import utils

from .components import MLP, Reshape, HighwayPerceptionLayer

Actor = nn.Module

class ContinuousStochesticActor(nn.Module):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU(),
        
        use_truncated_action: bool = True,
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.use_truncated_action = use_truncated_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.policy = MLP(
            observation_dim, action_dim * 2,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        
    def log_abs_det_jacobian(self, u):
        return 2. * (np.log(2) - u - torch.nn.functional.softplus(-2. * u))
        
    def forward(self, obs, deterministic=False):
        mean, logstd = torch.chunk(self.policy(obs), 2, dim=-1)
        if deterministic:
            if self.use_truncated_action: return torch.tanh(mean), None
            else: return mean, None
            
        logstd = torch.tanh(logstd)
        logstd = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (logstd + 1)
        std = logstd.exp()
        
        u = mean + std * torch.randn_like(mean)
        
        logp_u = - logstd - 0.5 * torch.pow((u - mean) / std, 2) - 0.5 * np.log(2 * np.pi)
        
        if self.use_truncated_action:
            a = torch.tanh(u)
            log_prob = logp_u - self.log_abs_det_jacobian(u)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            a = u
            log_prob = logp_u.sum(dim=-1, keepdim=True)
        return a, log_prob
        
class DiscreteStochesticActor(nn.Module):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        
        self.policy = MLP(
            observation_dim, action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        
    def forward(self, obs, deterministic=False):
        
        logits = self.policy(obs)
        if deterministic:
            return logits.argmax(dim=-1), None
        
        pi = dist.Categorical(logits=logits)
        a = pi.sample()
        log_prob = pi.log_prob(a)
        return a, log_prob

class HighwayDiscreteStochesticActor(nn.Module):
    def __init__(self,
        n_features: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        
        self.policy = nn.Sequential(
            Reshape((-1, n_features)),
            HighwayPerceptionLayer(n_features),
            MLP(
                128, action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )
        )
        
    def forward(self, obs, deterministic=False):
        
        logits = self.policy(obs)
        if deterministic:
            return logits.argmax(dim=-1), None
        
        pi = dist.Categorical(logits=logits)
        a = pi.sample()
        log_prob = pi.log_prob(a)
        return a, log_prob