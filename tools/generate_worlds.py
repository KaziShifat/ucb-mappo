"""Generate the Gazebo worlds, robot models and bridge config from layouts.py.

    python tools/generate_worlds.py            # run from the repo root

Writes  src/tb4_multi_nav/worlds/{maze,warehouse}_4robots.sdf
        src/tb4_multi_nav/models/robot{1..4}_model.sdf
        src/tb4_multi_nav/config/bridge.yaml

If you use your own hand-made SDF files, skip this script and copy them in.
Gazebo files were NOT opened in Gazebo when this repo was written.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tb4_multi_nav"))

from tb4_multi_nav.layouts import Box, Layout, get_layout  # noqa: E402

PKG = ROOT / "src" / "tb4_multi_nav"
COLORS = {"robot1": ("red", (0.9, 0.1, 0.1)), "robot2": ("cyan", (0.0, 0.8, 0.8)),
          "robot3": ("orange", (1.0, 0.55, 0.0)), "robot4": ("green", (0.5, 0.9, 0.1))}
BODY_R, BODY_H, WHEEL_R, WHEEL_SEP, LIDAR_RANGE = 0.17, 0.20, 0.036, 0.233, 12.0

WORLD_HEAD = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{name}">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <gravity>0 0 -9.8</gravity>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.8 0.8 0.85 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>{g} {g}</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{g} {g}</size></plane></geometry>
        <material><ambient>0.75 0.75 0.75 1</ambient><diffuse>0.75 0.75 0.75 1</diffuse></material></visual>
    </link></model>
"""


def robot_model(name: str, x: float, y: float, yaw: float) -> str:
    rgb = COLORS[name][1]
    col = f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
    wheel = lambda side, sign: f"""
      <link name="{side}_wheel"><pose>0 {sign * WHEEL_SEP / 2} {WHEEL_R} -1.5708 0 0</pose>
        <inertial><mass>0.2</mass><inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>{WHEEL_R}</radius><length>0.02</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><cylinder><radius>{WHEEL_R}</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material></visual></link>"""
    return f"""
    <model name="{name}">
      <pose>{x} {y} 0.0 0 0 {yaw}</pose>
      <link name="base_link"><pose>0 0 {WHEEL_R + 0.02} 0 0 0</pose>
        <inertial><mass>3.0</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.05</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="body_collision"><pose>0 0 0.08 0 0 0</pose><geometry><cylinder><radius>{BODY_R}</radius><length>{BODY_H}</length></cylinder></geometry></collision>
        <visual name="body_visual"><pose>0 0 0.08 0 0 0</pose><geometry><cylinder><radius>{BODY_R}</radius><length>{BODY_H}</length></cylinder></geometry>
          <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.08 0.08 0.08 1</diffuse></material></visual>
        <visual name="bumper_ring"><pose>0 0 0.02 0 0 0</pose><geometry><cylinder><radius>{BODY_R + 0.008}</radius><length>0.035</length></cylinder></geometry>
          <material><ambient>{col}</ambient><diffuse>{col}</diffuse></material></visual>
        <collision name="caster_front"><pose>0.1 0 -0.02 0 0 0</pose><geometry><sphere><radius>0.016</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <collision name="caster_rear"><pose>-0.1 0 -0.02 0 0 0</pose><geometry><sphere><radius>0.016</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <sensor name="lidar" type="gpu_lidar"><pose>0 0 0.22 0 0 0</pose><topic>/{name}/scan</topic>
          <update_rate>10</update_rate><always_on>true</always_on><visualize>false</visualize>
          <lidar><scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
            <range><min>0.15</min><max>{LIDAR_RANGE}</max><resolution>0.01</resolution></range>
            <noise><type>gaussian</type><mean>0.0</mean><stddev>0.01</stddev></noise></lidar></sensor>
        <sensor name="imu" type="imu"><topic>/{name}/imu</topic><update_rate>50</update_rate><always_on>true</always_on></sensor>
      </link>{wheel("left", 1)}{wheel("right", -1)}
      <joint name="left_wheel_joint" type="revolute"><parent>base_link</parent><child>left_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>
      <joint name="right_wheel_joint" type="revolute"><parent>base_link</parent><child>right_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>
      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
        <wheel_separation>{WHEEL_SEP}</wheel_separation><wheel_radius>{WHEEL_R}</wheel_radius>
        <topic>/{name}/cmd_vel</topic><odom_topic>/{name}/odom</odom_topic>
        <frame_id>{name}/odom</frame_id><child_frame_id>{name}/base_link</child_frame_id>
        <odom_publisher_frequency>30</odom_publisher_frequency>
        <max_linear_velocity>0.46</max_linear_velocity><max_angular_velocity>1.9</max_angular_velocity>
      </plugin>
    </model>"""


def box_model(name: str, b: Box, height: float, rgb) -> str:
    c = f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
    return f"""
    <model name="{name}"><static>true</static><pose>{b.cx} {b.cy} {height / 2} 0 0 0</pose>
      <link name="link"><collision name="c"><geometry><box><size>{b.sx} {b.sy} {height}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{b.sx} {b.sy} {height}</size></box></geometry>
          <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material></visual></link></model>"""


def world_sdf(layout: Layout, n_robots: int = 4) -> str:
    wall = (0.35, 0.35, 0.40)
    inner = (0.55, 0.45, 0.35) if layout.name == "warehouse" else (0.50, 0.50, 0.55)
    label = "shelf" if layout.name == "warehouse" else "wall"
    parts = [WORLD_HEAD.format(name=f"{layout.name}_{n_robots}robots", g=layout.size * 3)]
    parts += [box_model(f"outer_wall_{k}", b, layout.wall_height, wall) for k, b in enumerate(layout.outer_walls())]
    parts += [box_model(f"{label}_{k}", b, layout.interior_height, inner) for k, b in enumerate(layout.interior)]
    for i in range(n_robots):
        x, y, yaw = layout.start_pose(i)
        parts.append(robot_model(f"robot{i + 1}", x, y, round(yaw, 4)))
    parts.append("\n  </world>\n</sdf>\n")
    return "".join(parts)


def bridge_yaml(n_robots: int = 4) -> str:
    out = ["# Gazebo <-> ROS 2 topics for ros_gz_bridge (parameter_bridge, config_file:=...).",
           "# Written without a Gazebo install; check topic names against your own setup.",
           "- ros_topic_name: /clock\n  gz_topic_name: /clock\n  ros_type_name: rosgraph_msgs/msg/Clock\n"
           "  gz_type_name: gz.msgs.Clock\n  direction: GZ_TO_ROS"]
    for i in range(1, n_robots + 1):
        for topic, ros, gz, d in (("cmd_vel", "geometry_msgs/msg/Twist", "gz.msgs.Twist", "ROS_TO_GZ"),
                                  ("odom", "nav_msgs/msg/Odometry", "gz.msgs.Odometry", "GZ_TO_ROS"),
                                  ("scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan", "GZ_TO_ROS"),
                                  ("imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU", "GZ_TO_ROS")):
            out.append(f"- ros_topic_name: /robot{i}/{topic}\n  gz_topic_name: /robot{i}/{topic}\n"
                       f"  ros_type_name: {ros}\n  gz_type_name: {gz}\n  direction: {d}")
    return "\n".join(out) + "\n"


def main() -> None:
    for sub in ("worlds", "models", "config"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    for env in ("maze", "warehouse"):
        (PKG / "worlds" / f"{env}_4robots.sdf").write_text(world_sdf(get_layout(env)))
    for name in COLORS:
        (PKG / "models" / f"{name}_model.sdf").write_text(
            f'<?xml version="1.0"?>\n<sdf version="1.9">{robot_model(name, 0, 0, 0)}\n</sdf>\n')
    (PKG / "config" / "bridge.yaml").write_text(bridge_yaml())
    print("wrote worlds, robot models and bridge.yaml under", PKG)


if __name__ == "__main__":
    main()
