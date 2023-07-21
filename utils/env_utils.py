from gym import core, spaces, Env
from dm_control import suite
from dm_env import specs
import numpy as np
from collections import OrderedDict
from dm_control.utils import rewards


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class HumanoidDIYEnv(core.Env):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        
        self._height = 200
        self._width = 200
        self._camera_id = 1
        self._frame_skip = 1
        self._channels_first = True

        self._env = suite.load(
            domain_name="humanoid",
            task_name="run",
            task_kwargs={"random": seed},
        )
        
        self._init_qpos = self._env.physics.data.qpos.ravel().copy()
        self._init_qvel = self._env.physics.data.qvel.ravel().copy()
        self._init_noise = 0.1
        
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )
        self._observation_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )
        self._state_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )
        self.current_state = None
        
        self.seed(seed=seed)
        self.init_reward_target()
        
        
    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)
    
    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        reward, reward_info = self.cal_reward()
        extra['wandb_log_info'] = reward_info
        return obs, reward, done, extra
    
    def reset(self):
        time_step = self._env.reset()
        qpos = self._init_qpos + np.random.uniform(
            -self._init_noise, self._init_noise, self._env.physics.model.nq
        )
        qvel = self._init_qvel + self._init_noise * np.random.randn(self._env.physics.model.nv)
        self._env.physics.set_state(np.concatenate([qpos, qvel]))
        self._env.physics.forward()
        
        obs = OrderedDict()
        obs['joint_angles'] = self._env.physics.joint_angles()
        obs['head_height'] = self._env.physics.head_height()
        obs['extremities'] = self._env.physics.extremities()
        obs['torso_vertical'] = self._env.physics.torso_vertical_orientation()
        obs['com_velocity'] = self._env.physics.center_of_mass_velocity()
        obs['velocity'] = self._env.physics.velocity()
        
        self.current_state = _flatten_obs(obs)
        obs = _flatten_obs(obs)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
        
    def init_reward_target(self):
        self._target_torso_upright = np.sin(np.random.rand()*np.pi/3 + np.pi/6) # 30~90 degree
        self._target_waist_height = np.random.rand()*0.4 + 0.7 # 0.7~1.1
        self._target_velocity = np.random.rand()*6.5 + 0.5 # 0.5~1.0
        
    def cal_reward(self):
        physics = self._env.physics
        
        # torso_upright
        upright = rewards.tolerance(physics.torso_upright(),
                                bounds=(self._target_torso_upright, float('inf')), sigmoid='linear',
                                margin=1.+self._target_torso_upright, value_at_margin=0)
    
        # waist_height
        height = rewards.tolerance(physics.named.data.xpos["lower_waist", "z"],
                                 bounds=(self._target_waist_height, float('inf')),
                                 margin=self._target_waist_height/4)
        
        # move
        com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
        move = rewards.tolerance(com_velocity,
                               bounds=(self._target_velocity, float('inf')),
                               margin=self._target_velocity, value_at_margin=0,
                               sigmoid='linear')
        move = (5*move + 1) / 6
        
        # orientation
        vel = self._env.physics.center_of_mass_velocity()
        theta = np.arctan2(vel[1], vel[0])
        ori = rewards.tolerance(
            theta, margin=2.,
            value_at_margin=0.3,
            sigmoid='gaussian'
        )
        
        return upright * height * move * ori, {
            "upright_reward": upright,
            "height_reward": height,
            "move_reward": move,
            "orientation_reward": ori,
            
            "upright_distaince": self._target_torso_upright - physics.torso_upright(),
            "height_distaince": self._target_waist_height - physics.named.data.xpos["lower_waist", "z"],
            "move_distaince": self._target_velocity - com_velocity,
            "orientation": theta,
        }
