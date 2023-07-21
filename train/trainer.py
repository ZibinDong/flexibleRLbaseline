import os
from pathlib import Path

import gym
import numpy as np
import utils
import wandb
from agents import Agent


class Trainer:
    def __init__(self,
        agent: Agent,
        env: gym.Env,
        env_eval: gym.Env,
        
        no_terminal: bool = True,
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
        self.no_terminal = no_terminal
        
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
        ep_len = 0
        for _ in range(self.exploration_steps):
            act = self.env.action_space.sample()
            next_obs, rew, done, _ = self.env.step(act)
            ep_len += 1
            self.agent.buffer.add(obs, act, rew, done and (1-self.no_terminal), next_obs)
            if done:
                obs, done = self.env.reset(), False
            else:
                obs = next_obs

    def _eval(self, n_eval_episodes: int = 5):
        ep_mean_rew, ep_mean_len = 0., 0.
        
        ep_mean_vel = 0.
        ep_mean_height = 0.
        log = {}
        
        if self.save_video: 
            frames, ptr = np.empty((200, 3, 100, 100), dtype=np.uint8), 0
            
        for e in range(n_eval_episodes):
            obs, done = self.env_eval.reset(), False
            while not done:
                act = self.agent.act(obs, deterministic=True)
                obs, rew, done, info = self.env_eval.step(act)
                ep_mean_rew += rew
                ep_mean_len += 1
                
                if "wandb_log_info" in info.keys():
                    for k, v in info["wandb_log_info"].items():
                        if k in log.keys(): log[k] += v
                        else: log[k] = v
                
                ep_mean_vel += np.linalg.norm(self.env_eval.physics.center_of_mass_velocity()[[0, 1]])
                ep_mean_height += self.env_eval.physics.head_height()
                
                if self.save_video and ptr < 200:
                    frames[ptr] = np.transpose(self.env_eval.render(mode='rgb_array', camera_id=1, height=100, width=100), (2, 0, 1))
                    ptr += 1
        
        for k, v in log.items():
            log[k] = v / n_eval_episodes / 1000.
        
        log['eval_ep_rew'] = ep_mean_rew / n_eval_episodes
        log['eval_ep_len'] = ep_mean_len / n_eval_episodes

        if self.save_video: log["eval_video"] = wandb.Video(frames, fps=30, format="gif")
        if self.wandb_log: wandb.log(log, step=self.step)
        else: print(log)
            
    def run(
        self,
        total_steps: int = 100_000,
    ):
        self._explore()
        
        obs, done = self.env.reset(), False
        ep_rew, ep_len, log = 0., 0., {}
        for _ in range(total_steps):
            act = self.agent.act(obs, deterministic=False)
            next_obs, rew, done, info = self.env.step(act)
            ep_rew += rew
            ep_len += 1
            
            if "wandb_log_info" in info.keys():
                for k, v in info["wandb_log_info"].items():
                    if k in log.keys(): log[k] += v
                    else: log[k] = v
            
            # ！ set done=False when the episode reaches max length
            self.agent.buffer.add(obs, act, rew, done and (1-self.no_terminal), next_obs)
            if done:
                for k, v in log.items():
                    log[k] = v / 1000.
                log["ep_rew"] = ep_rew
                log["ep_len"] = ep_len
                if self.wandb_log: wandb.log(log, step=self.step)
                else: print(log)
                obs, done = self.env.reset(), False
                ep_rew, ep_len, log = 0., 0., {}
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
