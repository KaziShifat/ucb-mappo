"""Shared machinery of the two tasks. Use NavigationEnv or CoverageEnv.

One control step lasts 0.5 s (five 0.1 s physics sub-steps) and an episode lasts
at most 200 steps. Actions are physical (v, w) commands, clipped to
v in [-0.3, 0.3] m/s and w in [-1.0, 1.0] rad/s.

Collisions do not end an episode. The -50 collision penalty is charged once when
a contact starts, not on every step a robot stays pressed against an obstacle
(charging every step made early random policies collapse to standing still).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from tb4_multi_nav.layouts import Layout, get_layout
from tb4_multi_nav.observation import LIDAR_RANGE, OBS_DIM, build_obs

V_MAX, W_MAX = 0.3, 1.0


class MultiRobotEnv:
    obs_dim = OBS_DIM
    act_dim = 2
    v_max, w_max = V_MAX, W_MAX
    default_reward: Dict[str, float] = {"collision": -50.0, "time": -0.1}

    def __init__(self, layout: str = "maze", n_agents: int = 1, backend: str = "kinematic",
                 max_steps: int = 200, control_dt: float = 0.5, sim_dt: float = 0.1,
                 lidar_range: float = LIDAR_RANGE, robot_radius: float = 0.17,
                 start_jitter: float = 0.3, reward: Optional[Dict[str, float]] = None,
                 seed: int = 0, **backend_kwargs):
        assert 1 <= n_agents <= 4
        self.layout: Layout = get_layout(layout)
        self.n, self.max_steps = n_agents, max_steps
        self.control_dt, self.sim_dt = control_dt, sim_dt
        self.n_sub = max(int(round(control_dt / sim_dt)), 1)
        self.lidar_range, self.start_jitter = lidar_range, start_jitter
        self.rw = {**self.default_reward, **(reward or {})}
        self.rng = np.random.default_rng(seed)
        self.angles = np.deg2rad(np.arange(360))
        if backend == "kinematic":
            from tb4_multi_nav.envs.kinematic_sim import KinematicSim
            self.sim = KinematicSim(self.layout, n_agents, lidar_range, robot_radius)
        elif backend == "gazebo":
            from tb4_multi_nav.envs.gazebo_backend import GazeboBackend
            self.sim = GazeboBackend(self.layout, n_agents, lidar_range, **backend_kwargs)
        else:
            raise ValueError(backend)

    # ------------------------------------------------------------- task hooks
    def _task_reset(self) -> None: ...

    def _goal_for(self, i: int):
        return None

    def _task_step(self, contact_start: np.ndarray, rew: np.ndarray):
        """Add task rewards in place. Return (terminated[n], episode_done)."""
        raise NotImplementedError

    def _task_info(self) -> dict:
        return {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        poses = np.zeros((self.n, 3))
        for i in range(self.n):
            x, y, yaw = self.layout.start_pose(i)
            j = self.start_jitter
            poses[i] = (x + self.rng.uniform(-j, j), y + self.rng.uniform(-j, j),
                        yaw + self.rng.uniform(-0.3, 0.3))
        self.sim.reset(poses)
        self.t = 0
        self.stopped = np.zeros(self.n, dtype=bool)        # robots parked at their goal
        self.in_contact = np.zeros(self.n, dtype=bool)
        self.active_pairs: set = set()
        self.robot_collisions = 0                          # contacts between robots
        self.path_len = np.zeros(self.n)
        self._task_reset()
        return self._observe(), {}

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        cmds = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.n, 2),
                       [-self.v_max, -self.w_max], [self.v_max, self.w_max])
        cmds[self.stopped] = 0.0
        before = self.sim.poses[:, :2].copy()
        hit_any = np.zeros(self.n, dtype=bool)
        pairs_now: set = set()
        for _ in range(self.n_sub):
            h, p = self.sim.step(cmds, self.sim_dt)
            hit_any |= h
            pairs_now |= p
        self.t += 1
        self.path_len += np.hypot(*(self.sim.poses[:, :2] - before).T)

        contact_start = hit_any & ~self.in_contact
        self.in_contact = hit_any
        new_pairs = pairs_now - self.active_pairs
        self.active_pairs = pairs_now
        self.robot_collisions += len(new_pairs)
        rew = np.full(self.n, self.rw["time"])
        rew[contact_start] += self.rw["collision"]
        term, done_all = self._task_step(contact_start, rew)
        truncated = (self.t >= self.max_steps) and not done_all
        info = {"t": self.t, "robot_collisions": self.robot_collisions,
                "path_len": self.path_len.copy(), **self._task_info()}
        return self._observe(), rew.astype(np.float32), term, truncated, info

    # ------------------------------------------------------------ observation
    def _observe(self) -> np.ndarray:
        out = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        for i in range(self.n):
            out[i] = build_obs(self.sim.scan360(i), self.angles, self.sim.poses[i],
                               self._goal_for(i), self.layout.diagonal, self.lidar_range)
        return out
