from copy import deepcopy

import torch
import torch.nn as nn

import utils

from .components import MLP, HighwayPerceptionLayer


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
                activation=activation, use_ln=True,
            ) for _ in range(n_q_nets)])
    
        self.target_q_nets = nn.ModuleList([
            MLP(
                observation_dim + action_dim, 1,
                hidden_dims=hidden_dims,
                activation=activation, use_ln=True,
            ) for _ in range(n_q_nets)])
    
        self.apply(utils.orthogonal_init)
        utils.freeze_params(self.target_q_nets)
        self.target_q_nets.load_state_dict(self.q_nets.state_dict())
        
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
        

class DiscreteCritic(Critic):
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
                observation_dim, action_dim,
                hidden_dims=hidden_dims,
                activation=activation
            ) for _ in range(n_q_nets)])
    
        self.target_q_nets = nn.ModuleList([
            MLP(
                observation_dim, action_dim,
                hidden_dims=hidden_dims,
                activation=activation
            ) for _ in range(n_q_nets)])
    
        utils.hard_update(self.target_q_nets, self.q_nets)
        utils.freeze_params(self.target_q_nets)
    
    @torch.no_grad()
    def update_target_parameters(self):
        utils.soft_update(self.target_q_nets, self.q_nets, self.tau)
        
    def forward(self, obs, return_q_min=True, use_target_net=False):
        chosen_q_nets = self.target_q_nets if use_target_net else self.q_nets
        q_values = torch.stack([q_net(obs) \
            for q_net in chosen_q_nets], dim=1)
        if return_q_min:
            return q_values.min(dim=1).values
        else:
            return q_values
        
        
class HighwayDiscreteCritic(Critic):
    def __init__(self,
        n_features: int = 5,
        action_dim: int = 5,
        tau: float = 0.005,
        n_q_nets: int = 2,
    ):
        super().__init__()
        self.tau = tau
        self.n_features = n_features
            
        self.input_layer = HighwayPerceptionLayer(n_features)
        self.input_layer_target = HighwayPerceptionLayer(n_features)
        
        self.q_nets = nn.ModuleList([
            MLP(
                128, action_dim + 1,
                hidden_dims=[256, 256],
                activation=nn.ELU(),
            ) for _ in range(n_q_nets)])
    
        self.target_q_nets = nn.ModuleList([
            MLP(
                128, action_dim + 1,
                hidden_dims=[256, 256],
                activation=nn.ELU(),
            ) for _ in range(n_q_nets)])
    
        utils.hard_update(self.target_q_nets, self.q_nets)
        utils.hard_update(self.input_layer_target, self.input_layer)
        utils.freeze_params(self.target_q_nets)
        utils.freeze_params(self.input_layer_target)
    
    @torch.no_grad()
    def update_target_parameters(self):
        utils.soft_update(self.target_q_nets, self.q_nets, self.tau)
        utils.soft_update(self.input_layer_target, self.input_layer, self.tau)
        
    def forward(self, obs, return_q_min=True, use_target_net=False):
        chosen_q_nets = self.target_q_nets if use_target_net else self.q_nets
        # obs: (batch_size, n_vehicles * n_features)
        obs = obs.reshape(obs.shape[0], -1, self.n_features)   
        h = self.input_layer_target(obs) if use_target_net else self.input_layer(obs)
        
        values = torch.stack([q_net(h) \
            for q_net in chosen_q_nets], dim=1)
        a_values = values[:, :, :-1]
        a_values = a_values - a_values.mean(-1, keepdim=True)
        v_values = values[:, :, -1:]
        q_values = a_values + v_values
        
        if return_q_min:
            return q_values.min(dim=1).values
        else:
            return q_values