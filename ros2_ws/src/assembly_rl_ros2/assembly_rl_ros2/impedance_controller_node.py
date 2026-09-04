"""
impedance_controller_node -- third stage of the perception -> policy ->
actuation loop, and the actual sim-to-real reuse point.

Subscribes to the target pose (policy_node, or a hand-coded state machine --
see run_classical_baseline.py for the latter), joint states, and wrist F/T;
runs control/impedance_controller.CartesianImpedanceController.step() --
*the exact same class used in MuJoCo simulation* -- and publishes joint
torques to a ros2_control effort controller. This is what makes the
benchmark comparisons (benchmark/evaluate.py) meaningful for real hardware
too: the low-level control law is not reimplemented per-target, only the
target source changes between sim, the classical baseline, and the learned
policy.

Target real controller: a ros2_control `effort_controllers/JointGroupEffortController`
(or the UR driver's torque-passthrough mode where available) publishing to
`/forward_effort_controller/commands`. UR5e's factory firmware does not
expose raw joint torque control without ur_robot_driver's external control
interface configured for that mode -- see README.md "Sim-to-real reuse" for
the caveats before pointing this at real hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # -> assembly_rl/ project root
from control.pinocchio_model import UR5eModel  # noqa: E402
from control.impedance_controller import CartesianImpedanceController, ImpedanceGains  # noqa: E402
import pinocchio as pin  # noqa: E402

PEG_TOOL_OFFSET = np.array([0.0, 0.0, 0.196])
CONTROL_HZ = 250.0  # a real torque loop should run much faster than the ~20Hz policy tick

UR5E_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])  # matches assets/urdf/ur5e.urdf


class ImpedanceControllerNode(Node):
    def __init__(self):
        super().__init__("impedance_controller_node")
        self.robot = UR5eModel()
        # Same gain philosophy split as sim: start from the RL-safe (softer) gains by
        # default since this node also serves policy_node's targets; switch to the
        # classical ImpedanceGains() defaults via the `use_stiff_gains` param when
        # running run_classical_baseline.py against real hardware instead.
        self.declare_parameter("use_stiff_gains", False)
        gains = ImpedanceGains() if self.get_parameter("use_stiff_gains").value else ImpedanceGains(
            kp_pos=np.array([300.0, 300.0, 300.0]), kd_pos=np.array([30.0, 30.0, 30.0]),
            kp_rot=np.array([15.0, 15.0, 15.0]), kd_rot=np.array([2.0, 2.0, 2.0]),
        )
        self.controller = CartesianImpedanceController(self.robot, gains=gains, tool_offset=PEG_TOOL_OFFSET)

        self._q = np.zeros(6)
        self._qdot = np.zeros(6)
        self._wrench = np.zeros(6)
        self._target: pin.SE3 | None = None
        self._have_joint_state = False

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(WrenchStamped, "/wrench", self._on_wrench, QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(PoseStamped, "/assembly/policy/target_pose", self._on_target, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, "/forward_effort_controller/commands", 10)

        self.timer = self.create_timer(1.0 / CONTROL_HZ, self._on_tick)
        self.get_logger().info(f"impedance_controller_node running at {CONTROL_HZ} Hz "
                                f"(stiff_gains={self.get_parameter('use_stiff_gains').value})")

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in name_to_idx for j in UR5E_JOINT_ORDER):
            return
        self._q = np.array([msg.position[name_to_idx[j]] for j in UR5E_JOINT_ORDER])
        self._qdot = np.array([msg.velocity[name_to_idx[j]] if msg.velocity else 0.0 for j in UR5E_JOINT_ORDER])
        self._have_joint_state = True

    def _on_wrench(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        self._wrench = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])

    def _on_target(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        R = pin.Quaternion(o.w, o.x, o.y, o.z).toRotationMatrix()
        self._target = pin.SE3(R, np.array([p.x, p.y, p.z]))

    def _on_tick(self) -> None:
        if not self._have_joint_state or self._target is None:
            return
        tau = self.controller.step(
            self._q, self._qdot, self._target, f_des_z=0.0,
            pos_selection=np.array([1.0, 1.0, 1.0]), ft_wrench_world=self._wrench,
            dt=1.0 / CONTROL_HZ,
        )
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.torque_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImpedanceControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
