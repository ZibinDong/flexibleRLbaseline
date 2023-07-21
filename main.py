import dmc2gym
import gym
import hydra
import wandb
import utils

from agents import SACAgent
from train.trainer import Trainer

import utils
import matplotlib.pyplot as plt
env = utils.HumanoidDIYEnv(seed=1)
o = env.reset()
plt.imshow(env.render())
o2, r, d, info = env.step(env.action_space.sample())

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    
    # prepare environment 
    
    # cfg.Trainer.save_video *= (cfg.env.benchmark == 'dmc')
    cfg.Trainer.save_video *= (cfg.env.benchmark in ['dmc', 'diy'])
    if cfg.env.benchmark == 'dmc':
        env = dmc2gym.make(seed=cfg.seed, **cfg.env.param)
        env_eval = dmc2gym.make(seed=cfg.seed, **cfg.env.param)
    elif cfg.env.benchmark == 'diy':
        env = utils.HumanoidDIYEnv(seed=cfg.seed)
        env_eval = utils.HumanoidDIYEnv(seed=cfg.seed)
    else:
        raise NotImplementedError(f"benchmark {cfg.env.benchmark} is not implemented")
    
    observation_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_type = "continuous"
    else:
        action_dim = env.action_space.n
        action_type = "discrete"
        
    # instantiate agent
    
    if action_type == "continuous":
        if cfg.agent.name == "sac":
            agent = hydra.utils.instantiate(
                cfg.agent.param,
                observation_dim=observation_dim,
                action_dim=action_dim,
                device=cfg.device,
            )
        else:
            raise NotImplementedError(f"agent {cfg.agent.name} is not implemented")
    else:
        raise NotImplementedError(f"action type {action_type} is not implemented")
    
    # instantiate trainer
    
    trainer = hydra.utils.instantiate(
        cfg.Trainer,
        agent=agent,
        env=env,
        env_eval=env_eval,
        wandb_log=cfg.wandb_cfg.use_wandb_log,
    )
    
    if cfg.wandb_cfg.use_wandb_log:
        wandb_project = 'flexibleRL_' + cfg.env.benchmark
        wandb_name = cfg.agent.name + '_seed' + str(cfg.seed)
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity=cfg.wandb_cfg.entity,
            config=dict(cfg),
        )
    trainer.run(cfg.total_steps)
    
    wandb.finish()


if __name__ == "__main__":
    main()