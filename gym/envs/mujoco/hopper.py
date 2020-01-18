import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        # posbefore = self.sim.data.qpos[0]
        # self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        # ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[5]
        reward_height = -3.0 * np.square(old_ob[0] - 1.3)
        reward = reward_run + reward_ctrl + reward_height + 1

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.model.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
