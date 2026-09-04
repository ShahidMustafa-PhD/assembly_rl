"""
policy_node -- second stage of the perception -> policy -> actuation loop.

Subscribes to the processed camera image (perception_node), joint states,
and the wrist F/T sensor; assembles the *exact same* observation dict that
AssemblyEnv hands the policy in simulation (envs/assembly_env.py:_get_obs),
runs the trained SAC/PPO policy, and publishes a target end-effector pose
for impedance_controller_node to track.

This is the crux of "trained policy interface is directly reusable on
physical UR5e hardware": the observation assembly and action-to-target-pose
mapping here are copy-consistent with the sim env by construction (both call
UR5eModel.forward_kinematics_offset with the same PEG_TOOL_OFFSET), so a
policy trained in MuJoCo does not need a translation layer to run here --
only the image source, joint-state source, and F/T source change from
simulated to real sensors.

NOTE: importing `control`/`envs` from the main project assumes this ROS2
workspace is colcon-built with the project root on PYTHONPATH (e.g. via a
symlink or `pip install -e` of the main project) -- see README.md
"Sim-to-real reuse" for the exact setup. This is a deliberate scaffold
choice: duplicating the kinematics/controller code into the ROS2 package
would be exactly the sim/real drift this architecture is designed to avoid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from cv_bridge import CvBridge

# See module docstring NOTE above.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # -> assembly_rl/ project root
from control.pinocchio_model import UR5eModel  # noqa: E402
import pinocchio as pin  # noqa: E402

PEG_TOOL_OFFSET = np.array([0.0, 0.0, 0.196])
POLICY_HZ = 20.0
MAX_POS_DELTA = 0.008
MAX_ROT_DELTA = 0.10

UR5E_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("policy_path", "")
        policy_path = self.get_parameter("policy_path").value

        self.robot = UR5eModel()
        self.bridge = CvBridge()
        self.model = self._load_policy(policy_path)

        self._image = np.zeros((84, 84, 3), dtype=np.uint8)
        self._q = np.zeros(6)
        self._qdot = np.zeros(6)
        self._wrench = np.zeros(6)
        self._have_joint_state = False

        self.create_subscription(Image, "/assembly/obs/image", self._on_image, QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(WrenchStamped, "/wrench", self._on_wrench, QoSPresetProfiles.SENSOR_DATA.value)
        self.pose_pub = self.create_publisher(PoseStamped, "/assembly/policy/target_pose", 10)

        self._x_ref: pin.SE3 | None = None
        self.timer = self.create_timer(1.0 / POLICY_HZ, self._on_tick)
        self.get_logger().info(f"policy_node running at {POLICY_HZ} Hz, policy={policy_path or '(none loaded)'}")

    def _load_policy(self, policy_path: str):
        if not policy_path:
            self.get_logger().warn("No policy_path parameter given -- running with a zero-action stub policy.")
            return None
        from stable_baselines3 import SAC, PPO
        loader = SAC if "sac" in policy_path.lower() else PPO
        return loader.load(policy_path)

    def _on_image(self, msg: Image) -> None:
        self._image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in name_to_idx for j in UR5E_JOINT_ORDER):
            return  # message doesn't (yet) carry all arm joints, e.g. gripper-only state
        self._q = np.array([msg.position[name_to_idx[j]] for j in UR5E_JOINT_ORDER])
        self._qdot = np.array([msg.velocity[name_to_idx[j]] if msg.velocity else 0.0 for j in UR5E_JOINT_ORDER])
        self._have_joint_state = True

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # Assumed already expressed in world axes by the driver/TF; if not, rotate by
        # the wrist F/T frame's orientation the same way scripts/smoke_test.py does.
        w = msg.wrench
        self._wrench = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])

    def _on_tick(self) -> None:
        if not self._have_joint_state:
            return
        tip = self.robot.forward_kinematics_offset(self._q, PEG_TOOL_OFFSET)
        if self._x_ref is None:
            self._x_ref = tip

        obs = self._build_obs(tip)
        action = self._infer(obs)

        pos_delta = action[:3] * MAX_POS_DELTA
        rot_delta = action[3:6] * MAX_ROT_DELTA
        new_pos = self._x_ref.translation + pos_delta
        new_rot = pin.exp3(rot_delta) @ self._x_ref.rotation
        self._x_ref = pin.SE3(new_rot, new_pos)

        self._publish_target(self._x_ref)

    def _build_obs(self, tip: pin.SE3) -> dict:
        quat = pin.Quaternion(tip.rotation)
        ee_pose = np.concatenate([tip.translation, [quat.x, quat.y, quat.z, quat.w]]).astype(np.float32)
        return {
            "image": self._image[None].astype(np.uint8),
            "proprio": np.concatenate([self._q, self._qdot])[None].astype(np.float32),
            "force_torque": np.clip(self._wrench, -100, 100)[None].astype(np.float32),
            "ee_pose": ee_pose[None],
        }

    def _infer(self, obs: dict) -> np.ndarray:
        if self.model is None:
            return np.zeros(6)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action).flatten()[:6]

    def _publish_target(self, pose: pin.SE3) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose.translation
        q = pin.Quaternion(pose.rotation)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q.x, q.y, q.z, q.w
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
