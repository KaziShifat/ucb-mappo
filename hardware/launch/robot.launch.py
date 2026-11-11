"""Start the policy node of one robot. NOT TESTED on hardware.

    ros2 launch hardware/launch/robot.launch.py namespace:=robot1 \
        model:=hardware/models/navigation_n2/robot1.onnx task:=navigation \
        goal_x:=7.0 goal_y:=7.0 start_x:=-7.0 start_y:=-7.0 start_yaw:=0.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

ARGS = {"namespace": "robot1", "model": "", "task": "navigation", "goal_x": "0.0", "goal_y": "0.0",
        "start_x": "0.0", "start_y": "0.0", "start_yaw": "0.0", "stamped": "false"}


def generate_launch_description():
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
    node = Node(executable="python3", arguments=["hardware/policy_node.py"], output="screen",
                parameters=[{k: LC(k) for k in ARGS}])
    return LaunchDescription(decl + [node])
