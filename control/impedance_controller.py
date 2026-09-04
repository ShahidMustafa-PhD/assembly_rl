"""
Classical hybrid position/force Cartesian impedance controller -- the
benchmark baseline that the learned SAC/PPO policy is compared against
(success rate, insertion time, peak contact force, robustness to pose
perturbation; see benchmark/evaluate.py).

Control law (per axis, expressed in the world frame at the EE):
    F_task = S_pos . (Kp*(x_des - x) - Kd*(J qdot)) + S_force . (Kf*(f_des - f_meas))
    tau    = J^T F_task + g(q) + tau_null

where S_pos + S_force = I (a diagonal 0/1 selection matrix per Cartesian
DOF) implements hybrid position/force control: XY and orientation stay
position-controlled while the insertion (Z) axis is handed to force control
once contact is detected. A small null-space posture term keeps the elbow
away from the workspace boundary without disturbing the task-space motion.

This is intentionally classical -- fixed gains, no learning -- so the
comparison against the RL policy in benchmark/evaluate.py is meaningful.
"""
from __future__ import annotations

import dataclasses
from enum import Enum, auto

import numpy as np
import pinocchio as pin

from control.pinocchio_model import UR5eModel


@dataclasses.dataclass
class ImpedanceGains:
    kp_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.array([800.0, 800.0, 800.0]))
    kd_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.array([60.0, 60.0, 60.0]))
    kp_rot: np.ndarray = dataclasses.field(default_factory=lambda: np.array([40.0, 40.0, 40.0]))
    kd_rot: np.ndarray = dataclasses.field(default_factory=lambda: np.array([4.0, 4.0, 4.0]))
    kf_force: float = 0.0008     # force-tracking gain (m per N of error, applied as velocity cmd)
    force_limit: float = 25.0    # N, safety clamp on commanded contact force
    # UR5e is exactly 6-DOF (task space == joint space), so there is no true null
    # space to regularize away from singular configurations -- and the pinv-based
    # projector below picks up spurious non-zero components near a wrist singularity
    # (wrist_2 ~ 0), where it was observed to fight the task-space term hard enough
    # to stall the arm well short of its target. Disabled by default; only enable
    # (and re-derive properly, e.g. with a damped/weighted projector) if you extend
    # this to a redundant arm.
    kp_null: float = 0.0
    kd_null: float = 0.0
    q_null_target: np.ndarray | None = None  # elbow-comfortable posture
    # The UR5e's joint torque limits (150Nm shoulder/elbow, 28Nm wrist) cannot supply
    # the instantaneous force that kp_pos*(a large setpoint jump) demands -- commanding
    # a distant x_des directly saturates the actuators and the arm stalls at a
    # steady-state offset instead of converging. Real impedance controllers avoid this
    # by moving the *reference* at a bounded Cartesian speed instead of jumping to the
    # target; the high stiffness then only ever has to correct a small tracking error.
    max_lin_vel: float = 0.25   # m/s
    max_ang_vel: float = 1.5    # rad/s


class Phase(Enum):
    APPROACH = auto()   # rigid, position-controlled move to a pose above the hole
    SEARCH = auto()      # compliant in XY, gentle downward force -- find the hole
    INSERT = auto()       # Z handed to force control, XY stays position-controlled
    DONE = auto()


class CartesianImpedanceController:
    """Stateless (per-call) Cartesian impedance law; call `step` every control tick.

    `tool_offset` (3,): the controlled point, expressed in tool0's local frame --
    e.g. the peg tip, which sits ~0.196m below the wrist flange once the gripper +
    rigidly-mounted peg are accounted for (measured in scripts/smoke_test.py from
    the compiled MuJoCo model). Controlling tool0 itself instead of the peg tip was
    an earlier bug here: an "80mm above the hole" tool0 target actually drove the
    peg tip ~116mm *into* the table. Torques are still referred correctly to tool0
    via the rigid-body Jacobian correction in UR5eModel.jacobian_offset.
    """

    def __init__(self, robot: UR5eModel, gains: ImpedanceGains | None = None,
                 tool_offset: np.ndarray | None = None):
        self.robot = robot
        self.gains = gains or ImpedanceGains()
        if self.gains.q_null_target is None:
            self.gains.q_null_target = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.tool_offset = tool_offset if tool_offset is not None else np.zeros(3)
        self._x_ref: pin.SE3 | None = None

    def reset_reference(self, x_cur: pin.SE3 | None = None) -> None:
        """Call on episode/phase reset so the rate-limited reference doesn't jump
        from a stale pose. If x_cur is None, the next step() seeds it from FK."""
        self._x_ref = x_cur

    def step(self, q: np.ndarray, qdot: np.ndarray, x_des: pin.SE3, f_des_z: float,
             pos_selection: np.ndarray, ft_wrench_world: np.ndarray, dt: float = 0.002) -> np.ndarray:
        """Return joint torques (nv,) for one control tick.

        pos_selection: length-3 array in {0,1} for XYZ -- 1 = position controlled,
            0 = that axis is handed to force control (only Z is typically relaxed).
        ft_wrench_world: 6-vector [Fx,Fy,Fz,Tx,Ty,Tz] measured at the EE, rotated
            into world axes (see AssemblyEnv._wrench_world for the MuJoCo-side
            sensor -> world-frame conversion).
        """
        g = self.gains
        x_cur = self.robot.forward_kinematics_offset(q, self.tool_offset)
        J = self.robot.jacobian_offset(q, self.tool_offset)  # 6xnv, [linear; angular], world-aligned
        v_task = J @ qdot            # 6, current controlled-point spatial velocity

        if self._x_ref is None:
            self._x_ref = x_cur
        x_ref = self._advance_reference(self._x_ref, x_des, dt)
        self._x_ref = x_ref

        pos_err = x_ref.translation - x_cur.translation
        rot_err = pin.log3(x_ref.rotation @ x_cur.rotation.T)

        f_meas = ft_wrench_world[:3]
        sel = np.asarray(pos_selection, dtype=float)

        # Position-controlled axes: standard Cartesian PD.
        f_pos_pd = g.kp_pos * pos_err - g.kd_pos * v_task[:3]
        # Force-controlled axes: track f_des_z via a compliant velocity-forming law
        # (force error -> stiffness-limited corrective force), clamped for safety.
        f_err_z = f_des_z - f_meas[2]
        f_force = np.array([0.0, 0.0, np.clip(g.kf_force * f_err_z * 1000.0, -g.force_limit, g.force_limit)])

        F_lin = sel * f_pos_pd + (1 - sel) * f_force
        F_rot = g.kp_rot * rot_err - g.kd_rot * v_task[3:]
        F_task = np.concatenate([F_lin, F_rot])

        tau_task = J.T @ F_task
        tau_grav = self.robot.gravity(q)

        # Null-space posture regularization -- see ImpedanceGains.kp_null docstring
        # for why this is disabled by default on a non-redundant 6-DOF arm.
        tau_null = np.zeros(self.robot.nv)
        if g.kp_null > 0.0 or g.kd_null > 0.0:
            posture_force = g.kp_null * (g.q_null_target - q) - g.kd_null * qdot
            Jt_pinv = np.linalg.pinv(J.T, rcond=1e-4)
            N = np.eye(self.robot.nv) - J.T @ Jt_pinv
            tau_null = N @ posture_force

        return tau_task + tau_grav + tau_null

    def _advance_reference(self, x_ref: pin.SE3, x_des: pin.SE3, dt: float) -> pin.SE3:
        g = self.gains
        pos_delta = x_des.translation - x_ref.translation
        max_step = g.max_lin_vel * dt
        pos_dist = np.linalg.norm(pos_delta)
        new_pos = x_ref.translation + pos_delta * (min(max_step, pos_dist) / pos_dist if pos_dist > 1e-9 else 0.0)

        rot_delta = pin.log3(x_des.rotation @ x_ref.rotation.T)  # world-frame axis-angle
        max_rot_step = g.max_ang_vel * dt
        rot_dist = np.linalg.norm(rot_delta)
        if rot_dist > 1e-9:
            rot_step = rot_delta * (min(max_rot_step, rot_dist) / rot_dist)
            new_rot = pin.exp3(rot_step) @ x_ref.rotation
        else:
            new_rot = x_ref.rotation
        return pin.SE3(new_rot, new_pos)


class HybridPegInHolePolicy:
    """State machine sequencing APPROACH -> SEARCH -> INSERT using the impedance
    controller above. This is the full classical baseline evaluated in
    benchmark/evaluate.py."""

    def __init__(self, robot: UR5eModel, hole_pos_world: np.ndarray, hole_quat_world: np.ndarray | None = None,
                 approach_height: float = 0.08, search_force_z: float = -3.0,
                 insert_force_z: float = -6.0, contact_force_thresh: float = 1.5,
                 tool_offset: np.ndarray | None = None):
        self.robot = robot
        # Default tool_offset: peg tip, ~0.196m below tool0 along its local Z once
        # the gripper + rigid peg mount are stacked on (see class docstring above).
        self.tool_offset = tool_offset if tool_offset is not None else np.array([0.0, 0.0, 0.196])
        self.controller = CartesianImpedanceController(robot, tool_offset=self.tool_offset)
        # hole_quat_world=None -> keep whatever EE orientation the arm starts in
        # (captured lazily on the first act() call). Commanding a specific target
        # orientation here matters -- an unreachable/awkward one (e.g. world identity,
        # which does NOT generally match a "peg pointing straight down" pose for this
        # arm) fights the position objective through the shared Jacobian and can stall
        # the controller well short of the target; see scripts/smoke_test.py history.
        self.hole_R = pin.Quaternion(*hole_quat_world).toRotationMatrix() if hole_quat_world is not None else None
        self.hole_pos = hole_pos_world
        self.approach_height = approach_height
        self.search_force_z = search_force_z
        self.insert_force_z = insert_force_z
        self.contact_force_thresh = contact_force_thresh
        self.phase = Phase.APPROACH
        self._search_t = 0.0

    def reset(self):
        self.phase = Phase.APPROACH
        self._search_t = 0.0
        self.controller.reset_reference(None)

    def act(self, q: np.ndarray, qdot: np.ndarray, ft_wrench_world: np.ndarray, dt: float) -> np.ndarray:
        x_cur = self.robot.forward_kinematics_offset(q, self.tool_offset)
        if self.hole_R is None:
            self.hole_R = x_cur.rotation.copy()
        contact = abs(ft_wrench_world[2]) > self.contact_force_thresh

        if self.phase is Phase.APPROACH:
            target = pin.SE3(self.hole_R, self.hole_pos + np.array([0, 0, self.approach_height]))
            sel = np.array([1.0, 1.0, 1.0])
            f_des_z = 0.0
            if np.linalg.norm(x_cur.translation - target.translation) < 0.005:
                self.phase = Phase.SEARCH
                self._search_t = 0.0
        elif self.phase is Phase.SEARCH:
            # Gentle spiral search in XY while pressing down lightly, compliant in Z.
            self._search_t += dt
            r = min(0.006, 0.0015 * self._search_t)
            xy = self.hole_pos[:2] + r * np.array([np.cos(4 * self._search_t), np.sin(4 * self._search_t)])
            target = pin.SE3(self.hole_R, np.array([xy[0], xy[1], self.hole_pos[2] + 0.01]))
            sel = np.array([1.0, 1.0, 0.0])
            f_des_z = self.search_force_z
            if contact and self._search_t > 0.3:
                self.phase = Phase.INSERT
        elif self.phase is Phase.INSERT:
            target = pin.SE3(self.hole_R, self.hole_pos)
            sel = np.array([1.0, 1.0, 0.0])
            f_des_z = self.insert_force_z
            if x_cur.translation[2] - self.hole_pos[2] < 0.003:
                self.phase = Phase.DONE
        else:  # DONE -- hold position
            target = pin.SE3(self.hole_R, self.hole_pos)
            sel = np.array([1.0, 1.0, 1.0])
            f_des_z = 0.0

        return self.controller.step(q, qdot, target, f_des_z, sel, ft_wrench_world, dt=dt)
