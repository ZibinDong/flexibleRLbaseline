import os
from pathlib import Path

import gym
import numpy as np
import utils
import wandb
from agents import Agent


class HighwayTrainer:
    def __init__(self,
        agent: Agent,
        env: gym.Env,
        env_eval: gym.Env,
        
        exploration_steps: int = 2000,
        eval_interval: int = 10_000,
        log_interval: int = 5_000,
        save_interval: int = 100_000,
        target_update_interval: int = 1,
        gradient_step_interval: int = 1,
        n_gradient_steps: int = 1,
        
        seed: int = 1,
        wandb_log: bool = True,
        
        save_video: bool = True
    ):
        utils.set_seed(seed)
        self.agent = agent
        self.env = env
        self.env_eval = env_eval
        
        self.exploration_steps = exploration_steps
        self.target_update_interval = target_update_interval
        self.gradient_step_interval = gradient_step_interval
        self.n_gradient_steps = n_gradient_steps
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_video = save_video
        
        self.step = 0
        self.wandb_log = wandb_log
        
        self._init_train_info()
        
    def _init_train_info(self):
        time = utils.get_time_str()
        self.save_path = Path(__file__).parents[1]/"results"/time
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def _explore(self):
        obs, done = self.env.reset(), False
        for _ in range(self.exploration_steps):
            act = self.env.action_space.sample()
            next_obs, rew, done, info = self.env.step(act)
            if info["truncated"]: done = False
            self.agent.buffer.add(obs, act, rew, done)
            if done or info["truncated"]:
                obs, done = self.env.reset(), False
            else:
                obs = next_obs

    def _eval(self, n_eval_episodes: int = 1):
        print("evaluating...")
        ep_mean_rew, ep_mean_len = 0., 0.
        
        if self.save_video: 
            frames = []

        for e in range(n_eval_episodes):
            obs, done, truncated = self.env_eval.reset(), False, False
            while not (done or truncated):
                act = self.agent.act(obs, deterministic=True)
                obs, rew, done, info = self.env_eval.step(act)
                truncated = info["truncated"]
                ep_mean_rew += rew
                ep_mean_len += 1
                
                if self.save_video and len(frames) < 200:
                    frames.append(self.env_eval.render())
                
        print("uploading...")
        log = {
            'eval_ep_rew': ep_mean_rew / n_eval_episodes,
            'eval_ep_len': ep_mean_len / n_eval_episodes,
        }
        if self.save_video: log["eval_video"] = wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=20, format="mp4")
        if self.wandb_log: wandb.log(log, step=self.step)
        else: print(log)

    def run(
        self,
        total_steps: int = 100_000,
    ):
        self._explore()
        self._eval()
        
        obs, done = self.env.reset(), False
        ep_rew, ep_len, log = 0., 0., {}
        for t in range(total_steps):
            self.agent.exploration_noise = max(0.1, 1. - t / 10_000)
            act = self.agent.act(obs, deterministic=False)
            next_obs, rew, done, info = self.env.step(act)
            if info["truncated"]: done = False
            ep_rew += rew
            ep_len += 1
            self.agent.buffer.add(obs, act, rew, done or self.env.raw_env.vehicle.crashed)
            if done or info["truncated"]:
                log = {
                    "ep_rew": ep_rew,
                    "ep_len": ep_len,
                }
                if self.wandb_log: wandb.log(log, step=self.step)
                else: print(log)
                obs, done = self.env.reset(), False
                ep_rew, ep_len = 0., 0.
            else:
                obs = next_obs
                
            self.step += 1
            
            if self.step % self.gradient_step_interval == 0:
                for _ in range(self.n_gradient_steps):
                    log = self.agent.update()
                    
            if self.step % self.log_interval == 0:
                if self.wandb_log: wandb.log(log, step=self.step)
                else: print(log)
                
            if self.step % self.target_update_interval == 0:
                self.agent.critic.update_target_parameters()
                
            if self.step % self.eval_interval == 0:
                self._eval()
                
            if self.step % self.save_interval == 0:
                self.agent.save(self.save_path, self.step)
