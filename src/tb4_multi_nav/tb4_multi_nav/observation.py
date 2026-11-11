"""The 38-value observation, shared by the simulator and the real robots.

    36 values  RPLidar scan split into 36 sectors of 10 degrees, shortest range
               in each sector, readings beyond 12 m or invalid set to 12 m,
               divided by 12
     1 value   distance to the goal divided by the diagonal of the environment
     1 value   signed angle between heading and goal direction divided by pi

The observation holds no map and nothing about the other robots.
Coverage task: there is no goal, so the two goal values are zero (a choice of
this reference code, see README).
"""
from __future__ import annotations

import math

import numpy as np

LIDAR_RANGE = 12.0
N_SECTORS = 36
OBS_DIM = N_SECTORS + 2


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scan_to_sectors(ranges, angles, max_range: float = LIDAR_RANGE) -> np.ndarray:
    """ranges [m] and their angles [rad, 0 = heading, counter-clockwise] -> 36 values in [0, 1].

    Works for any number of beams (360 in simulation, 720 to 800 on the RPLidar A1).
    """
    r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=max_range,
                      posinf=max_range, neginf=max_range)
    r = np.where((r <= 0.0) | (r > max_range), max_range, r)
    a = np.mod(np.asarray(angles, dtype=np.float64), 2 * np.pi)
    sector = np.minimum((a / (2 * np.pi) * N_SECTORS).astype(int), N_SECTORS - 1)
    out = np.full(N_SECTORS, max_range)
    np.minimum.at(out, sector, r)
    return (out / max_range).astype(np.float32)


def build_obs(ranges, angles, pose, goal, arena_diagonal: float,
              max_range: float = LIDAR_RANGE) -> np.ndarray:
    """pose = (x, y, yaw) in the frame the goal is given in. goal=None gives zeros."""
    lidar = scan_to_sectors(ranges, angles, max_range)
    if goal is None:
        tail = np.zeros(2, dtype=np.float32)
    else:
        dx, dy = goal[0] - pose[0], goal[1] - pose[1]
        bearing = wrap_angle(math.atan2(dy, dx) - pose[2])
        tail = np.array([min(math.hypot(dx, dy) / arena_diagonal, 1.0), bearing / math.pi],
                        dtype=np.float32)
    return np.concatenate([lidar, tail])
