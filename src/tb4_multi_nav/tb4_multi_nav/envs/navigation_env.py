"""Task A: point-to-point navigation. Each robot drives to the diagonally opposite corner.

Rewards (paper Table 1): +100 at the goal, +2.0 x (decrease in goal distance),
-50 collision, -0.1 per step. The episode ends when every robot has reached its
goal or after 200 steps. A robot that reached its goal stops and stays there.
"""
from __future__ import annotations

import numpy as np

from tb4_multi_nav.envs.base_env import MultiRobotEnv


class NavigationEnv(MultiRobotEnv):
    default_reward = {"goal": 100.0, "progress": 2.0, "collision": -50.0, "time": -0.1}

    def __init__(self, *args, goal_radius: float = 0.5, **kw):
        super().__init__(*args, **kw)
        self.goal_radius = goal_radius
        self.goals = np.array(self.layout.goals[:self.n], dtype=np.float64)

    def _goal_dist(self) -> np.ndarray:
        return np.hypot(*(self.goals - self.sim.poses[:, :2]).T)

    def _task_reset(self) -> None:
        self.reached = np.zeros(self.n, dtype=bool)
        self.prev_dist = self._goal_dist()

    def _goal_for(self, i: int):
        return tuple(self.goals[i])

    def _task_step(self, contact_start, rew):
        dist = self._goal_dist()
        newly = (~self.reached) & (dist < self.goal_radius)
        rew += self.rw["progress"] * (self.prev_dist - dist) * (~self.reached)
        rew[newly] += self.rw["goal"]
        self.reached |= newly
        self.stopped |= newly
        self.prev_dist = dist
        return newly, bool(self.reached.all())

    def _task_info(self) -> dict:
        return {"reached": self.reached.copy(), "success_rate": float(self.reached.mean())}
