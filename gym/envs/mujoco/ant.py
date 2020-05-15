import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        # xposbefore = self.get_body_com("torso")[0]
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[13]
        reward_height = -3.0 * np.square(old_ob[0] - 0.57)
        reward = reward_run + reward_ctrl + reward_height + 1.0  # 1.0 alive
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
