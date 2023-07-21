import gym
import gymnasium
import numpy as np

class DangerousHighway(gym.Env):
    def __init__(self, env: gymnasium.Env,
        x_safe_zone = [1.5, 1.5],
        r_safe_zone = 1.2,
    ):
        super().__init__()
        self.x_safe_zone = x_safe_zone
        self.r_safe_zone = r_safe_zone
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.raw_env = env
        self.n_vehicles, self.n_features = env.observation_space.shape
        
    def reset(self):
        return self.raw_env.reset()[0].reshape(-1)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.raw_env.step(action)
        info["truncated"] = truncated
        reward = self._reward1(obs)
        done = done or self.raw_env.vehicle.crashed
        return obs.reshape(-1), reward, done, info
    
    def render(self):
        return self.raw_env.render()
    
    def _reward1(self, obs):
        if self.raw_env.vehicle.crashed:
            return -1.
        forward_speed = self.raw_env.vehicle.speed * np.cos(self.raw_env.vehicle.heading)
        scaled_speed = (np.clip(forward_speed, 10, 30)-10.)/20.
        
        # for i in range(obs.shape[0]):
        #     if i == 0: continue
        #     if obs[i, 2] == 0:
        #         if obs[i, 1] < 0 and (obs[i, 1]*200.) > -self.x_safe_zone[0] * 5.:
        #             dist = (obs[i, 1]*200.)+self.x_safe_zone[0]*5.
        #             decay = ((dist-2.5)/(self.x_safe_zone[0]*5-2.5))**2
        #             scaled_speed *= decay
        #         elif obs[i, 1] > 0 and (obs[i, 1]*200.) < self.x_safe_zone[1] * 5.:
        #             dist = self.x_safe_zone[1]*5.-obs[i, 1]*200.
        #             decay = ((dist-2.5)/(self.x_safe_zone[1]*5-2.5))**2
        #             scaled_speed *= decay
        
        return scaled_speed