import dmc2gym

from agents import SACAgent
from train.trainer import Trainer


if __name__ == "__main__":
    
    device = "cuda"
    seed = 1
    
    env = dmc2gym.make(domain_name='walker', task_name='walk', seed=seed)
    env_eval = dmc2gym.make(domain_name='walker', task_name='walk', seed=seed)
    
    agent = SACAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device
    )
    
    trainer = Trainer(
        agent,
        env,
        env_eval,
        wandb_log=False,
    )
    
    trainer.run(10_000)