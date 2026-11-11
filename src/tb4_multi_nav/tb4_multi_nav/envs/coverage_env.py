"""Task B: area coverage. The fleet tries to visit as much free space as it can.

The free space is split into 0.5 m cells. A cell counts as covered once any robot
has been within `cover_radius` of its centre (walls are ignored, a choice of this
reference code). Rewards (paper Table 1): +5.0 per new cell, -2.0 when a robot
spends a step in a cell that was already covered and finds nothing new, +10 when
90 % of the free cells are covered, -50 collision, -0.1 per step.
The episode ends at 90 % coverage or after 200 steps.

Metrics: coverage = covered free cells / free cells. Redundant overlap = share of
robot steps spent in a cell that was already covered at the start of the step.
"""
from __future__ import annotations

import numpy as np

from tb4_multi_nav.envs.base_env import MultiRobotEnv


class CoverageEnv(MultiRobotEnv):
    default_reward = {"success": 10.0, "new_cell": 5.0, "revisit": -2.0,
                      "collision": -50.0, "time": -0.1}

    def __init__(self, *args, cell: float = 0.5, cover_radius: float = 2.0,
                 coverage_target: float = 0.9, **kw):
        super().__init__(*args, **kw)
        self.cell, self.cover_radius, self.coverage_target = cell, cover_radius, coverage_target
        h = self.layout.half
        self.gn = int(round(self.layout.size / cell))
        centres = (np.arange(self.gn) + 0.5) * cell - h
        X, Y = np.meshgrid(centres, centres, indexing="ij")
        free = np.ones((self.gn, self.gn), dtype=bool)
        for b in self.layout.all_boxes():
            x0, y0, x1, y1 = b.bounds
            free &= ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
        self.free, self.X, self.Y, self.n_free = free, X, Y, int(free.sum())

    def _task_reset(self) -> None:
        self.covered = np.zeros((self.gn, self.gn), dtype=bool)
        self.redundant_steps, self.robot_steps = 0, 0
        self.coverage = 0.0
        self._mark(first=True)

    def _cell_of(self, x: float, y: float):
        h, c = self.layout.half, self.cell
        return (min(max(int((x + h) // c), 0), self.gn - 1),
                min(max(int((y + h) // c), 0), self.gn - 1))

    def _mark(self, first: bool = False) -> np.ndarray:
        """Mark cells near each robot. Return new-cell counts per robot."""
        new = np.zeros(self.n)
        before = self.covered.copy()
        for i in range(self.n):
            x, y = self.sim.poses[i, :2]
            near = (np.hypot(self.X - x, self.Y - y) <= self.cover_radius) & self.free
            new[i] = int((near & ~self.covered).sum())
            self.covered |= near
        self._before = before
        self.coverage = float((self.covered & self.free).sum() / self.n_free)
        return new

    def _task_step(self, contact_start, rew):
        before = self.covered.copy()
        new = self._mark()
        for i in range(self.n):
            ix, iy = self._cell_of(*self.sim.poses[i, :2])
            was_covered = bool(before[ix, iy])
            self.robot_steps += 1
            self.redundant_steps += int(was_covered)
            rew[i] += self.rw["new_cell"] * new[i]
            if was_covered and new[i] == 0:
                rew[i] += self.rw["revisit"]
        done = self.coverage >= self.coverage_target
        if done:
            rew += self.rw["success"]
        return np.full(self.n, done), done

    def _task_info(self) -> dict:
        return {"coverage": self.coverage,
                "overlap": self.redundant_steps / max(self.robot_steps, 1)}
