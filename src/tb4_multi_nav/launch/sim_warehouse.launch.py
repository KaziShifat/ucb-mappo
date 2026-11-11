"""Start Gazebo with the warehouse and bridge the robot topics.

    ros2 launch tb4_multi_nav sim_warehouse.launch.py

Not tested: written without a Gazebo install. The world starts un-paused (-r).
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

WORLD = "warehouse"


def generate_launch_description():
    pkg = get_package_share_directory("tb4_multi_nav")
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"),
                                                   "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-r " + os.path.join(pkg, "worlds", f"{WORLD}_4robots.sdf")}.items())
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  parameters=[{"config_file": os.path.join(pkg, "config", "bridge.yaml")}])
    return LaunchDescription([gz, bridge])
