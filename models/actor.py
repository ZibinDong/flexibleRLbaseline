import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import utils

from .components import MLP

Actor = nn.Module

class ContinuousStochesticActor(nn.Module):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU(),
        
        use_truncated_action: bool = True,
        log_std_min: float = -20,
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
        
    def forward(self, obs, deterministic=False):
        mean, logstd = torch.chunk(self.policy(obs), 2, dim=-1)
        if deterministic:
            if self.use_truncated_action: return torch.tanh(mean), None
            else: return mean, None
        logstd = self.log_std_max - F.softplus(self.log_std_max - logstd)
        logstd = self.log_std_min + F.softplus(logstd - self.log_std_min)
        
        dist_u = D.Normal(mean, torch.exp(logstd))
        u = dist_u.rsample()
        if self.use_truncated_action:
            a = torch.tanh(u)
            log_prob = dist_u.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            a = u
            log_prob = dist_u.log_prob(u).sum(dim=-1, keepdim=True)
        return a, log_prob
        