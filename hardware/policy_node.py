#!/usr/bin/env python3
"""Runs one trained actor on one TurtleBot4's Raspberry Pi 4. NOT TESTED on hardware.

One instance per robot, inside that robot's namespace. Needs only rclpy, numpy and
ONNX Runtime. One forward pass with dropout off, well inside the 0.5 s control step.
The observation is built by tb4_multi_nav.observation, the same function the simulator
uses: the raw RPLidar scan (720 to 800 points) is reduced to 36 sectors with the shortest
range kept, readings beyond 12 m or invalid are set to 12 m, and the goal distance and
angle come from odometry composed with the declared start pose.

    python3 hardware/policy_node.py --ros-args -p model:=hardware/models/navigation_n2/robot1.onnx \
        -p namespace:=robot1 -p task:=navigation -p goal_x:=7.0 -p goal_y:=7.0 \
        -p start_x:=-7.0 -p start_y:=-7.0 -p start_yaw:=0.0

There is no localisation: pose drift from odometry adds up over a trial. For the coverage
task the two goal values are zero (same as in simulation). Set stamped:=true if the robot
image expects TwistStamped on cmd_vel.
"""
import math

import numpy as np
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from tb4_multi_nav.observation import LIDAR_RANGE, build_obs, wrap_angle


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        for name, default in (("model", ""), ("namespace", "robot1"), ("task", "navigation"),
                              ("stamped", False), ("goal_x", 0.0), ("goal_y", 0.0), ("start_x", 0.0),
                              ("start_y", 0.0), ("start_yaw", 0.0), ("arena_diagonal", 25.46),
                              ("goal_radius", 0.5), ("period", 0.5)):
            self.declare_parameter(name, default)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        ns = g("namespace")
        self.sess = ort.InferenceSession(g("model"), providers=["CPUExecutionProvider"])
        self.task, self.stamped = g("task"), g("stamped")
        self.goal = (g("goal_x"), g("goal_y")) if self.task == "navigation" else None
        self.start = (g("start_x"), g("start_y"), g("start_yaw"))
        self.diag, self.goal_radius = g("arena_diagonal"), g("goal_radius")
        self.pose = self.scan = None
        msg = TwistStamped if self.stamped else Twist
        self.pub = self.create_publisher(msg, f"/{ns}/cmd_vel", 10)
        self.create_subscription(Odometry, f"/{ns}/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, f"/{ns}/scan", self.on_scan, qos_profile_sensor_data)
        self.create_timer(g("period"), self.tick)

    def on_odom(self, m):
        q, p = m.pose.pose.orientation, m.pose.pose.position
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x0, y0, th0 = self.start
        c, s = math.cos(th0), math.sin(th0)
        self.pose = (x0 + c * p.x - s * p.y, y0 + s * p.x + c * p.y, wrap_angle(th0 + yaw))

    def on_scan(self, m):
        n = len(m.ranges)
        self.scan = (np.asarray(m.ranges, dtype=np.float64), m.angle_min + m.angle_increment * np.arange(n))

    def send(self, v, w):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def tick(self):
        if self.pose is None or self.scan is None:
            return
        if self.goal and math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]) < self.goal_radius:
            self.send(0.0, 0.0)
            return
        obs = build_obs(*self.scan, self.pose, self.goal, self.diag, LIDAR_RANGE)[None].astype(np.float32)
        v, w = self.sess.run(None, {"obs": obs})[0][0]
        self.send(v, w)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.send(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
