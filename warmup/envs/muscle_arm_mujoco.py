import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box

from .muscle_arm import MuscleArm


class MuscleArmMuJoCo(MuscleArm):
    def __init__(self):
        self.model_type = "muscle_arm_mujoco"
        self.tracking_str = "endeffector"
        self.nq = 2
        self.ball_attached = False
        super(MuscleArm, self).__init__()
        self.set_gravity([9.81, 0, 0])
        self.has_init = True

    def reset_model(self):
        self.randomise_init_state()
        if self.random_goals:
            self.target = self.sample_rectangular_goal()
        self.sim.data.qpos[-2:] = self.target[:2]
        return self._get_obs()

    def sample_rectangular_goal(self):
        return np.random.uniform([-0.2, 0.5, 0.0], [0.15, 0.65, 0.0])

    def sample_circular_goal(self):
        rho = np.random.uniform(-1.0, 0)
        phi = np.random.uniform(0.5 * np.pi, np.pi)
        # no typo here, MuJoCo coordinates are rotated wrt the distribution
        y = rho * np.cos(phi)
        x = rho * np.sin(phi)
        return np.array([x, y, 0.0])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.4
        self.viewer.cam.lookat[:] = [0.15, -0.0, 1.35]
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 180

    @property
    def xml_path(self):
        return "xml_files/muscle_arm_mujoco.xml"

    def activate_ball(self):
        self.ball_attached = 1
        self.reinitialise(self.args)
        self.model.opt.gravity[0] = 9.81

    def reinitialise(self, args):
        """if we want to load from specific xml, not the creator"""
        self.need_reinit = 0
        while True:
            path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                self.xml_path,
            )
            try:
                mujoco_env.MujocoEnv.__init__(self, 
                                              model_path = path, 
                                              frame_skip = self.frame_skip, 
                                              observation_space = self.observation_space,
                                              render_mode = self.render_mode)
                break
            except FileNotFoundError:
                print("xml file not found, reentering loop.")
        utils.EzPickle.__init__(self)
        self._set_action_space()

    def _get_obs(self):
        """Creates observation for MDP.
        The choice here is to either use normalized com_vel in state and reward or just in reward. I could
        imagine it leading to faster learning when normalized, as larger velocities don't constitute "new"
        state space regions. But it also shifts the learning target.
        Removed adaptive scaling."""
        return np.concatenate(
            [
                self.sim.data.qpos[: self.nq],
                self.sim.data.qvel[: self.nq],
                self.muscle_length(),
                self.muscle_velocity(),
                self.muscle_force(),
                self.muscle_activity(),
                self.target,
                self.data.site_xpos[self.tracking_id],
            ]
        )
