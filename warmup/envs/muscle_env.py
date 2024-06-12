import os
from abc import ABC, abstractmethod

import numpy as np
from math import inf
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box
from mujoco import mj_name2id, mjtObj

from ..utils.env_utils import load_default_params


class AbstractMuscleEnv(mujoco_env.MujocoEnv, utils.EzPickle, ABC):
    @abstractmethod
    def reset_model(self):
        pass

    @abstractmethod
    def viewer_setup(self):
        pass

    @abstractmethod
    def set_target(self):
        pass


class MuscleEnv(AbstractMuscleEnv):
    def __init__(self):
        # initialise this with large number for env creation
        self.manually_set_action_space = 0
        self.render_substep_bool = 0
        default_path = self.get_default_params_path()
        _, args = load_default_params(path=default_path)
        self.observation_space = Box(low = float(args.observation_space["low"]),
                                     high = float(args.observation_space["high"]),
                                     shape = args.observation_space["shape"],
                                     dtype = getattr(np, args.observation_space["dtype"]))
        self.args = args
        self.frame_skip = args.frameskip
        self.quick_settings(args)
        super().__init__(model_path = self.xml_path,
                         frame_skip = self.frame_skip,
                         observation_space = self.observation_space,
                         render_mode = self.render_mode)
        self.tracking_id = mj_name2id(self.model, mjtObj.mjOBJ_SITE, self.tracking_str)
        self.reset()

    def render_substep(self):
        self.render_substep_bool = 1

    def do_simulation(self, ctrl, n_frames):
        # if not hasattr(self, "action_multiplier"):
        #     if np.array(ctrl).shape != self.action_space.shape:
        #         raise ValueError("Action dimension mismatch")

        if self.render_substep_bool:
            raise NotImplementedError("Please implement intermediate rendering")
        self.data.ctrl[:] = ctrl
        self._step_mujoco_simulation(ctrl=ctrl, n_frames=n_frames)

        # for _ in range(n_frames):
        #     if self.render_substep_bool:
        #         # self.render('rgb_array')
        #         self.render("human")
        #     self.sim.step()

    def get_default_params_path(self):
        default_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
        return os.path.join(
            default_path, f"param_files/{self.model_type}.yaml"
        )

    def merge_args(self, args):
        if args is not None:
            for k, v in args.items():
                setattr(self.args, k, v)

    def apply_args(self):
        self.quick_settings(self.args)

    def quick_settings(self, args):
        """Applies correct settings from args.
        Don't change the order if you don't know what you are doing.
        Some settings are used by the mujoco_env initilization."""
        self.set_target(args.target)
        self.set_random_goals(args.random_goals)
        self.set_termination(args.termination)
        self.set_termination_distance(args.termination_distance)
        self.set_sparse_reward(self.sparse_reward)
        self.reinitialise(args)
        self.set_gravity(args.gravity)
        if hasattr(args, "force_maximum"):
            self.set_force_maximum(args.force_maximum)

    def set_sparse_reward(self, sparse_reward):
        self.sparse_reward = sparse_reward

    def set_random_goals(self, random_goals=False):
        """Should the goals be randomly sampled."""
        self.random_goals = random_goals

    def set_termination_distance(self, distance=0.08):
        """Set endeffector to goal distance at which the episode is considered to be solved."""
        self.termination_distance = distance

    def set_termination(self, termination=False):
        """Decide wether the episode will be prematurely terminated when achieving the goal.
        A <done> signal will be emitted."""
        self.termination = termination

    def set_gravity(self, value):
        """
        Set gravity, orientation depends on envrionment.
        """
        for idx, val in enumerate(value):
            self.model.opt.gravity[idx] = val

    def randomise_init_state(self, diff=0.01):
        """
        Randomises initial joint positions slightly.
        """
        qpos = self.init_qpos
        qvel = self.init_qvel
        qvel = np.random.normal(0.0, diff, size=(self.model.nq))
        self.set_state(qpos, qvel)

    def seed(self, seed=None):
        self._seed = seed
        seed = 0 if seed is None else seed
        super().seed(seed)
        np.random.seed(seed)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def process_state(self):
        return (self.full_state, self.joint_state)

    @property
    def full_state(self):
        return self._get_obs()

    @property
    def joint_state(self):
        return self.data.qpos[: self.unwrapped.nq]

    def reset(self, *args, **kwargs):
        if self.render_substep_bool:
            if hasattr(self, "frames") and len(self.frames) != 0:
                self.display_video()
            self.frames = []
        return super().reset(*args, **kwargs)

    def _get_obs(self):
        """get_obs() implicitly calls this function to get the
        MDP state in the default case."""
        act = self.muscle_activations()
        if act is None:
            act = np.zeros_like(self.muscle_lengths())
        return np.concatenate(
            [
                self.data.qpos[: self.nq],
                self.data.qvel[: self.nq],
                self.muscle_lengths(),
                self.muscle_forces(),
                self.muscle_velocities(),
                act,
                self.target,
                self.data.site_xpos[self.tracking_str],
            ]
        )

    def muscle_length(self):
        if (
            not hasattr(self, "action_multiplier")
            or self.action_multiplier == 1
        ):
            return self.data.actuator_length.copy()
        return np.repeat(
            self.data.actuator_length.copy(), self.action_multiplier
        )

    def muscle_velocity(self):
        if (
            not hasattr(self, "action_multiplier")
            or self.action_multiplier == 1
        ):
            return np.clip(self.data.actuator_velocity, -100, 100).copy()
        return np.repeat(
            np.clip(self.data.actuator_velocity, -100, 100).copy(),
            self.action_multiplier,
        )

    def muscle_activity(self):
        if (
            not hasattr(self, "action_multiplier")
            or self.action_multiplier == 1
        ):
            return np.clip(self.data.act, -100, 100).copy()
        return np.repeat(
            np.clip(self.data.act, -100, 100).copy(), self.action_multiplier
        )

    def muscle_force(self):
        if (
            not hasattr(self, "action_multiplier")
            or self.action_multiplier == 1
        ):
            return np.clip(self.data.actuator_force / 1000, -100, 100).copy()
        return np.repeat(
            np.clip(self.data.actuator_force / 1000, -100, 100).copy(),
            self.action_multiplier,
        )

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, val):
        if not self.manually_set_action_space:
            self._action_space = val
        else:
            pass

    def set_force_maximum(self, force):
        if self.model.actuator_gaintype[0] == 2:
            self.model.actuator_gainprm[:, 2] = force
        else:
            self.model.actuator_gear[:, 0] = force
