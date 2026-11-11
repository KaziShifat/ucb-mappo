"""Fast 2D kinematic simulator (no Gazebo, no ROS).

Differential drive discs, axis aligned box walls, a 360 beam lidar with 12 m
range. Used so that every algorithm can be trained on CPU for a quick check.
It is NOT the Gazebo simulation of the paper: contacts, wheel slip and sensor
noise are not modelled, so numbers trained here are not comparable with Gazebo.

Interface shared with gazebo_backend.GazeboBackend:
    reset(poses)         poses [n, 3] (x, y, yaw)
    step(cmds, dt)       cmds [n, 2] (v, w) -> hit [n] bool, pairs set of (i, j)
    scan360(i)           360 ranges, beam k points at yaw + k degrees
    poses                [n, 3]
"""
from __future__ import annotations

import numpy as np

from tb4_multi_nav.layouts import Layout
from tb4_multi_nav.observation import LIDAR_RANGE

N_BEAMS = 360


class KinematicSim:
    def __init__(self, layout: Layout, n_agents: int, lidar_range: float = LIDAR_RANGE,
                 robot_radius: float = 0.17):
        self.layout, self.n, self.lidar_range, self.r = layout, n_agents, lidar_range, robot_radius
        b = np.array([bx.bounds for bx in layout.all_boxes()], dtype=np.float64)
        self._bmin, self._bmax = b[:, :2], b[:, 2:]
        self.beam_angles = np.deg2rad(np.arange(N_BEAMS, dtype=np.float64))
        self.poses = np.zeros((n_agents, 3))

    def reset(self, poses) -> None:
        self.poses = np.array(poses, dtype=np.float64).reshape(self.n, 3)

    def step(self, cmds: np.ndarray, dt: float):
        hit = np.zeros(self.n, dtype=bool)
        pairs = set()
        for i in range(self.n):
            v, w = float(cmds[i, 0]), float(cmds[i, 1])
            x, y, yaw = self.poses[i]
            yaw_n = (yaw + w * dt + np.pi) % (2 * np.pi) - np.pi
            x_n, y_n = x + v * np.cos(yaw_n) * dt, y + v * np.sin(yaw_n) * dt
            blocked, other = self._blocked(i, x_n, y_n)
            if blocked:
                hit[i] = True
                if other is not None:
                    pairs.add((min(i, other), max(i, other)))
                self.poses[i, 2] = yaw_n                    # may turn in place, may not translate
            else:
                self.poses[i] = (x_n, y_n, yaw_n)
        return hit, pairs

    def _blocked(self, i: int, x: float, y: float):
        p = np.array([x, y])
        closest = np.clip(p, self._bmin, self._bmax)
        if np.any(np.sum((closest - p) ** 2, axis=1) < self.r ** 2):
            return True, None
        for j in range(self.n):
            if j != i and np.hypot(*(self.poses[j, :2] - p)) < 2 * self.r:
                return True, j
        return False, None

    def scan360(self, i: int) -> np.ndarray:
        ox, oy, yaw = self.poses[i]
        ang = yaw + self.beam_angles
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        o = np.array([ox, oy])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (self._bmin[None] - o) * inv[:, None, :]
            t2 = (self._bmax[None] - o) * inv[:, None, :]
        tnear = np.nanmax(np.minimum(t1, t2), axis=2)
        tfar = np.nanmin(np.maximum(t1, t2), axis=2)
        hit = (tfar >= np.maximum(tnear, 0.0)) & (tnear > 0.0)
        best = np.minimum(self.lidar_range, np.where(hit, tnear, np.inf).min(axis=1))
        for j in range(self.n):
            if j == i:
                continue
            c = self.poses[j, :2] - o
            proj = d @ c
            perp2 = c @ c - proj ** 2
            ok = (proj > 0) & (perp2 < self.r ** 2)
            t_c = proj - np.sqrt(np.maximum(self.r ** 2 - perp2, 0.0))
            best = np.where(ok, np.minimum(best, t_c), best)
        return best
