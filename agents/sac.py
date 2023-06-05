from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

import models
import replaybuffers
import utils

from .agent import ContinuousAgent


class SACAgent(ContinuousAgent):
    def __init__(self,

        observation_dim: int,
        action_dim: int,

        actor_hidden_dims: list = [256, 256],
        critic_hidden_dims: list = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU(),
        
        init_temperature: float = 0.2,
        target_entropy: Optional[float] = None,
        
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
        batch_size: int = 256,
        
        max_grad_norm: float = 100.,
        
        use_truncated_action: bool = True,
        log_std_min: float = -20,
        log_std_max: float = 2,
        
        gamma: float = 0.99,
        tau: float = 0.005,
        n_q_nets: int = 2,
        
        max_replay_buffer_size: int = 500_000,
        
        seed: int = 0,
        device: str = 'cpu',
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
        utils.set_seed(seed)
        self.device = device

        self.target_entropy = - action_dim if target_entropy is None else target_entropy
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        self.actor = models.ContinuousStochesticActor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_truncated_action=use_truncated_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)
        
        self.critic = models.ContinuousCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            tau=tau,
            n_q_nets=n_q_nets,
        ).to(device)
        
        self.alpha = torch.nn.Parameter(
            torch.tensor(init_temperature, device=device), requires_grad=True)
        
        self.buffer = replaybuffers.BasicReplayBuffer(
            observation_dim=observation_dim,
            action_dim=action_dim,
            max_replay_buffer_size=max_replay_buffer_size,
            device=device,
        )
        
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.optim_alpha = torch.optim.Adam([self.alpha], lr=temperature_learning_rate)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device)[None,]
        action, _ = self.actor(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()
    
    def update(self):
        log = {}
        (obs, act, rew, done, next_obs) = self.buffer.sample(self.batch_size)
        self.update_critic(obs, act, rew, done, next_obs, log)
        self.update_actor(obs, log)
        return log
    
    def update_critic(self, obs, act, rew, done, next_obs, log):
        
        with torch.no_grad():
            next_act, next_log_prob = self.actor(next_obs)
            next_q = self.critic(next_obs, next_act, use_target_net=True)
            next_v = next_q - self.alpha * next_log_prob
            target_q = rew + self.gamma * (1 - done) * next_v
        
        q = self.critic(obs, act, return_q_min=False)
        
        loss_critic = ((q - target_q)**2).mean()
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        
        log['loss_critic'] = loss_critic.item()

    
    def update_actor(self, obs, log):
        
        with utils.FreezeParameters([self.critic, self.alpha]):
            act, log_prob = self.actor(obs)
            q = self.critic(obs, act, return_q_min=True)
            loss_actor = (self.alpha * log_prob - q).mean()
            
            self.optim_alpha.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm)
            self.optim_actor.step()
            
        loss_alpha = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.alpha], self.max_grad_norm)
        self.optim_alpha.step()
        self.alpha.data.clamp_(0, None)
        
        log['loss_actor'] = loss_actor.item()
        log['loss_alpha'] = loss_alpha.item()
        log['alpha'] = self.alpha.item()
        
    def save(self, path: Path, step: int):
        torch.save({
            'step': step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'alpha': self.alpha.state_dict(),
        }, path / f"sac_{utils.abbreviate_number(step)}.pt")
        
    def load(self, path: Path, filename: str):
        checkpoint = torch.load(path / filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.alpha.load_state_dict(checkpoint['alpha'])