import numpy as np

from .muscle_env import MuscleEnv


class MuscleArm(MuscleEnv):
    def __init__(self, *args, **kwargs):
        self.tracking_str = "endeffector"
        super(MuscleArm, self).__init__(*args, **kwargs)

    def set_target(self, target):
        """
        Target that we have to reach, reward is based on distance between this target and endeffector position
        """
        self.target = np.array(target, dtype=np.float32)

    def step(self, a):
        assert a.shape == self.action_space.shape
        if self.need_reinit:
            raise Exception("Need to call self.reinitialise before stepping")
        # mujoco-py checks step() before __init__() is called
        if not hasattr(self, "target"):
            self.target = np.array([10.0, 10.0, 10.0])
        if hasattr(self, "action_multiplier") and self.action_multiplier != 1:
            a = self.redistribute_action(a)
        self.do_simulation(a, self.frame_skip)
        ee_pos = self.ee_pos
        act = self.data.act
        reward = self._get_reward(ee_pos, a)
        done = self._get_done(ee_pos)
        if done:
            reward += 10.0
        return (
            self._get_obs(),
            reward,
            done,
            False, # TODO: implement truncation
            {"tracking": ee_pos, "activity": act},
        )

    def _get_reward(self, ee_pos, action):
        lamb = 1e-4  # 1e-4
        epsilon = 1e-4
        log_weight = 1.0
        rew_weight = 0.1

        d = np.mean(np.square(ee_pos - self.target))
        activ_cost = lamb * np.mean(np.square(action))
        if self.sparse_reward:
            return -1.0
        return (
            -rew_weight * (d + log_weight * np.log(d + epsilon**2))
            - activ_cost
            - 2
        )

    def _get_done(self, ee_pos):
        if (
            (not self.termination) or
            (np.linalg.norm(self.target - ee_pos) >= self.termination_distance)
        ):
            return False
        else:
            return True

    def _get_extended_done(self, ee_pos):
        """Emit termination if endeffector is stationary at goal for several
        time steps. Not used atm."""
        cdt = [
            1
            if np.linalg.norm(self.target - ee_pos) < self.termination_distance
            else 0
        ][0]
        if not cdt:
            self._done_steps = 0
        if cdt and self._done_steps > 0:
            self._done_steps += 1
        return [1 if self._done_steps >= 10 else 0][0]

    @property
    def ee_pos(self):
        return self.data.site_xpos[self.tracking_id]

    @property
    def goal(self):
        return self.target

    def set_goal_manually(self, target):
        self.target = target
        self.data.qpos[-2:] = self.target[:2]
