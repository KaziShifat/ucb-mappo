#!/usr/bin/env python3
"""Base station recorder. NOT TESTED on hardware. It listens and sends nothing back.

Writes one trial folder:
    poses.csv   t, robot, x, y, yaw        (odometry composed with each robot's start pose)
    cmd.csv     t, robot, v, w             (velocity commands)
    scans.npz   per robot: times and raw ranges
    meta.json   written by you or by the launch file (fleet size, task, trial, model files)

    python3 hardware/logger_node.py --ros-args -p out:=hardware/trials/coverage/n2/trial1 \
        -p robots:=2 -p start_poses:="[-3.0,-4.0,0.0, 3.0,4.0,3.14]"
"""
import csv
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class Logger(Node):
    def __init__(self):
        super().__init__("trial_logger")
        self.declare_parameter("out", "trial")
        self.declare_parameter("robots", 1)
        self.declare_parameter("start_poses", [0.0, 0.0, 0.0])
        out = Path(self.get_parameter("out").value)
        out.mkdir(parents=True, exist_ok=True)
        self.out, n = out, int(self.get_parameter("robots").value)
        sp = list(self.get_parameter("start_poses").value)
        self.start = [sp[3 * i:3 * i + 3] for i in range(n)]
        self.t0 = self.get_clock().now()
        self.poses_f, self.cmd_f = open(out / "poses.csv", "w", newline=""), open(out / "cmd.csv", "w", newline="")
        self.poses, self.cmd = csv.writer(self.poses_f), csv.writer(self.cmd_f)
        self.poses.writerow(["t", "robot", "x", "y", "yaw"])
        self.cmd.writerow(["t", "robot", "v", "w"])
        self.scans = {i: ([], []) for i in range(n)}
        for i in range(n):
            ns = f"/robot{i + 1}"
            self.create_subscription(Odometry, f"{ns}/odom", lambda m, i=i: self.on_odom(i, m), qos_profile_sensor_data)
            self.create_subscription(Twist, f"{ns}/cmd_vel", lambda m, i=i: self.on_cmd(i, m), 10)
            self.create_subscription(LaserScan, f"{ns}/scan", lambda m, i=i: self.on_scan(i, m), qos_profile_sensor_data)

    def now(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def on_odom(self, i, m):
        q, p = m.pose.pose.orientation, m.pose.pose.position
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x0, y0, th0 = self.start[i]
        c, s = math.cos(th0), math.sin(th0)
        self.poses.writerow([f"{self.now():.2f}", i + 1, f"{x0 + c * p.x - s * p.y:.3f}",
                             f"{y0 + s * p.x + c * p.y:.3f}", f"{th0 + yaw:.3f}"])

    def on_cmd(self, i, m):
        self.cmd.writerow([f"{self.now():.2f}", i + 1, f"{m.linear.x:.3f}", f"{m.angular.z:.3f}"])

    def on_scan(self, i, m):
        self.scans[i][0].append(self.now())
        self.scans[i][1].append(np.asarray(m.ranges, dtype=np.float32))

    def close(self):
        self.poses_f.close()
        self.cmd_f.close()
        arrays = {}
        for i, (times, ranges) in self.scans.items():
            width = max((len(r) for r in ranges), default=0)
            padded = np.full((len(ranges), width), np.nan, dtype=np.float32)
            for k, r in enumerate(ranges):
                padded[k, :len(r)] = r
            arrays[f"robot{i + 1}_t"] = np.array(times)
            arrays[f"robot{i + 1}_ranges"] = padded
        np.savez_compressed(self.out / "scans.npz", **arrays)


def main():
    rclpy.init()
    node = Logger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
