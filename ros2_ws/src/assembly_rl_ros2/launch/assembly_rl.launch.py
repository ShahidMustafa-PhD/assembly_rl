"""Launches the full perception -> policy -> actuation pipeline.

    ros2 launch assembly_rl_ros2 assembly_rl.launch.py policy_path:=/path/to/sac_final.zip

Assumes a UR5e driver (ur_robot_driver or similar) is already publishing
/joint_states and /wrench, a wrist camera is publishing /camera/wrist/image_raw,
and a ros2_control effort controller is active and subscribed to
/forward_effort_controller/commands. See README.md "Sim-to-real reuse" for
the full hardware bring-up checklist -- this launch file only starts the
three nodes in this package.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy_path_arg = DeclareLaunchArgument(
        "policy_path", default_value="",
        description="Path to a trained SB3 .zip policy (SAC or PPO). Empty -> zero-action stub.",
    )
    use_stiff_gains_arg = DeclareLaunchArgument(
        "use_stiff_gains", default_value="false",
        description="true -> classical-baseline gains; false -> softer RL-safe gains.",
    )

    return LaunchDescription([
        policy_path_arg,
        use_stiff_gains_arg,
        Node(package="assembly_rl_ros2", executable="perception_node", name="perception_node", output="screen"),
        Node(package="assembly_rl_ros2", executable="policy_node", name="policy_node", output="screen",
             parameters=[{"policy_path": LaunchConfiguration("policy_path")}]),
        Node(package="assembly_rl_ros2", executable="impedance_controller_node", name="impedance_controller_node",
             output="screen", parameters=[{"use_stiff_gains": LaunchConfiguration("use_stiff_gains")}]),
    ])
