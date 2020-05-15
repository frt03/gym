import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]

        if getattr(self, 'action_space', None):
            action = np.clip(
                action, self.action_space.low,
                self.action_space.high
            )
            lb, ub = self.action_space.low, self.action_space.high
        else:
            lb, ub = -1, 1
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.

        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        ob = self._get_obs()
        notdone = np.isfinite(ob[:29]).all() and ob[2] >= 0.2 and ob[2] <= 1.0
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 15
            self.model.data.qvel.flat,  # 14
            # np.clip(self.model.data.cfrc_ext, -1, 1).flat,  # 84
            self.get_body_xmat("torso").flat,  # 9
            self.get_body_com("torso"),  # 3
            self.get_body_comvel("torso"),  # 3
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
