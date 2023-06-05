from copy import deepcopy

import torch
import torch.nn as nn

import utils

from .components import MLP


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
    @torch.no_grad()
    def update_target_parameters(self):
        raise NotImplementedError

class ContinuousCritic(Critic):
    def __init__(self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU(),
        tau: float = 0.005,
        n_q_nets: int = 2,
    ):
        super().__init__()
        self.tau = tau
        
        self.q_nets = nn.ModuleList([
            MLP(
                observation_dim + action_dim, 1,
                hidden_dims=hidden_dims,
                activation=activation,
            ) for _ in range(n_q_nets)])
    
        self.target_q_nets = nn.ModuleList([
            MLP(
                observation_dim + action_dim, 1,
                hidden_dims=hidden_dims,
                activation=activation,
            ) for _ in range(n_q_nets)])
    
        utils.hard_update(self.target_q_nets, self.q_nets)
        utils.freeze_params(self.target_q_nets)
        
    @torch.no_grad()
    def update_target_parameters(self):
        utils.soft_update(self.target_q_nets, self.q_nets, self.tau)
        
    def forward(self, obs, act, return_q_min=True, use_target_net=False):
        chosen_q_nets = self.target_q_nets if use_target_net else self.q_nets
        q_values = torch.cat([q_net(torch.cat([obs, act], dim=-1)) \
            for q_net in chosen_q_nets], dim=-1)
        if return_q_min:
            return q_values.min(dim=-1, keepdim=True).values
        else:
            return q_values