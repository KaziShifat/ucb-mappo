"""Gazebo Harmonic backend (ROS 2 Jazzy). NOT TESTED: written without Gazebo installed.

Same interface as KinematicSim, so NavigationEnv / CoverageEnv work unchanged:
    make_env("navigation", "maze", 4, backend="gazebo")
Needs the world from launch/sim_*.launch.py running and un-paused, the bridge
topics of config/bridge.yaml, and ros_gz_interfaces for teleporting robots at reset.
One control step waits control_dt of wall time, so run Gazebo at real-time factor 1.
Contacts are inferred from the lidar (a beam shorter than the robot radius); robot
to robot pairs cannot be told apart from walls here, so `pairs` stays empty and the
collision rate must be taken from your own contact sensor or from distances.
"""
from __future__ import annotations

import math
import time

import numpy as np

from tb4_multi_nav.layouts import Layout
from tb4_multi_nav.observation import LIDAR_RANGE, wrap_angle

N_BEAMS = 360


class GazeboBackend:
    def __init__(self, layout: Layout, n_agents: int, lidar_range: float = LIDAR_RANGE,
                 world_name: str | None = None, robot_radius: float = 0.17):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan

        if not rclpy.ok():
            rclpy.init()
        self.rclpy, self.Twist = rclpy, Twist
        self.node = rclpy.create_node("ucb_gazebo_backend")
        self.layout, self.n, self.lidar_range, self.r = layout, n_agents, lidar_range, robot_radius
        self.world = world_name or f"{layout.name}_4robots"
        self.poses = np.zeros((n_agents, 3))
        self._start = np.zeros((n_agents, 3))
        self._scan = [np.full(N_BEAMS, lidar_range, dtype=np.float32) for _ in range(n_agents)]
        self.pubs = []
        for i in range(n_agents):
            ns = f"/robot{i + 1}"
            self.pubs.append(self.node.create_publisher(Twist, f"{ns}/cmd_vel", 10))
            self.node.create_subscription(Odometry, f"{ns}/odom", lambda m, i=i: self._on_odom(i, m), 10)
            self.node.create_subscription(LaserScan, f"{ns}/scan", lambda m, i=i: self._on_scan(i, m), 10)
        try:
            from ros_gz_interfaces.srv import SetEntityPose
            self._Req = SetEntityPose
            self._cli = self.node.create_client(SetEntityPose, f"/world/{self.world}/set_pose")
        except ImportError:
            self._cli = None

    def _on_odom(self, i, msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x0, y0, th0 = self._start[i]
        c, s = math.cos(th0), math.sin(th0)
        self.poses[i] = (x0 + c * p.x - s * p.y, y0 + s * p.x + c * p.y, wrap_angle(th0 + yaw))

    def _on_scan(self, i, msg) -> None:
        r = np.nan_to_num(np.asarray(msg.ranges), nan=self.lidar_range, posinf=self.lidar_range)
        ang = wrap_angle(msg.angle_min + msg.angle_increment * np.arange(len(r)))
        order = np.argsort(ang)
        grid = wrap_angle(np.deg2rad(np.arange(N_BEAMS)))
        a, rr = ang[order], r[order]
        self._scan[i] = np.interp(grid, np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi]),
                                  np.concatenate([rr, rr, rr])).astype(np.float32)

    def reset(self, poses: np.ndarray) -> None:
        poses = np.asarray(poses, dtype=np.float64).reshape(self.n, 3)
        if self._cli is None:
            raise RuntimeError("ros_gz_interfaces not found: cannot teleport robots")
        for i in range(self.n):
            self._publish(i, 0.0, 0.0)
            req = self._Req.Request()
            req.entity.name, req.entity.type = f"robot{i + 1}", 2
            req.pose.position.x, req.pose.position.y = float(poses[i, 0]), float(poses[i, 1])
            req.pose.orientation.z, req.pose.orientation.w = math.sin(poses[i, 2] / 2), math.cos(poses[i, 2] / 2)
            self._cli.wait_for_service(timeout_sec=5.0)
            self.rclpy.spin_until_future_complete(self.node, self._cli.call_async(req), timeout_sec=2.0)
        self._start, self.poses = poses.copy(), poses.copy()
        self._spin(0.3)

    def _publish(self, i: int, v: float, w: float) -> None:
        m = self.Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pubs[i].publish(m)

    def _spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.005)

    def step(self, cmds: np.ndarray, dt: float):
        for i in range(self.n):
            self._publish(i, cmds[i, 0], cmds[i, 1])
        self._spin(dt)
        return np.array([self._scan[i].min() < self.r + 0.02 for i in range(self.n)]), set()

    def scan360(self, i: int) -> np.ndarray:
        return self._scan[i]
