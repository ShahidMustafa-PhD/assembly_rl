"""
Pinocchio kinematics/dynamics wrapper for the UR5e arm.

Provides forward kinematics, task-space (geometric) Jacobians, the joint-space
mass matrix, Coriolis/gravity terms, and the operational-space (Cartesian)
mass matrix used by:
  * `control/impedance_controller.py` -- the classical baseline
  * `envs/assembly_env.py` -- for Jacobian-based observations / IK

The joint ordering here (shoulder_pan, shoulder_lift, elbow, wrist_1,
wrist_2, wrist_3) is identical to the MuJoCo UR5e model's joint order, so
`q`/`qdot` vectors can be passed between the two without remapping -- verified
in `scripts/smoke_test.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

ASSETS = Path(__file__).resolve().parent.parent / "assets"
URDF_PATH = ASSETS / "urdf" / "ur5e.urdf"
PACKAGE_DIRS = [str(ASSETS / "pinocchio_packages")]

EE_FRAME = "tool0"
FT_FRAME = "wrist_3_link-ft_frame"  # matches the wrist F/T sensor site in MuJoCo

# mujoco_menagerie's ur5e.xml mounts the arm with <body name="base" quat="0 0 0 -1">,
# a fixed 180deg rotation about Z relative to the URDF's base_link convention (verified
# in scripts/smoke_test.py: MuJoCo and Pinocchio EE positions agree exactly once this
# is applied). All world-frame outputs below are corrected by this fixed transform so
# downstream code (impedance controller, env) can treat "Pinocchio world" and "MuJoCo
# world" as the same frame.
_BASE_FIX = pin.SE3(pin.utils.rotate("z", np.pi), np.zeros(3))


def mujoco_world_from_pin(pose: pin.SE3) -> pin.SE3:
    return _BASE_FIX * pose


def _rotate_jacobian(J: np.ndarray) -> np.ndarray:
    R = _BASE_FIX.rotation
    Rblk = np.zeros((6, 6))
    Rblk[:3, :3] = R
    Rblk[3:, 3:] = R
    return Rblk @ J


class UR5eModel:
    def __init__(self, urdf_path: Path = URDF_PATH, package_dirs=PACKAGE_DIRS, ee_frame: str = EE_FRAME):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        try:
            self.geom_model = pin.buildGeomFromUrdf(
                self.model, str(urdf_path), pin.GeometryType.COLLISION, package_dirs=package_dirs
            )
            self.geom_data = self.geom_model.createData()
        except Exception:
            # Geometry is optional for our use (Jacobians/dynamics only need the
            # kinematic/inertial model); keep going if meshes are unavailable.
            self.geom_model = None
            self.geom_data = None

        if not self.model.existFrame(ee_frame):
            raise ValueError(f"Frame {ee_frame!r} not found in URDF. Available: "
                              f"{[f.name for f in self.model.frames]}")
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]

    # ---- kinematics ----------------------------------------------------
    def forward_kinematics(self, q: np.ndarray) -> pin.SE3:
        """End-effector pose (tool0), expressed in the MuJoCo world frame."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return mujoco_world_from_pin(self.data.oMf[self.ee_frame_id])

    def jacobian(self, q: np.ndarray, frame: str | None = None, local: bool = False) -> np.ndarray:
        """6xN task-space Jacobian [linear; angular] at `frame` (default EE).

        local=False -> expressed in the MuJoCo world frame with the frame's origin
        (LOCAL_WORLD_ALIGNED), which is what you want for Cartesian impedance
        control (force/velocity errors expressed in world axes). local=True returns
        Pinocchio's frame-local Jacobian unmodified (base-frame convention doesn't
        affect a purely local representation).
        """
        frame_id = self.model.getFrameId(frame) if frame else self.ee_frame_id
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        if local:
            return pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL)
        J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return _rotate_jacobian(J)

    def forward_kinematics_offset(self, q: np.ndarray, offset_local: np.ndarray) -> pin.SE3:
        """Pose of a point rigidly attached to the EE, `offset_local` away from tool0
        in tool0's own frame (e.g. the peg tip -- see scripts/smoke_test.py for how
        this offset is measured from the compiled MuJoCo model). Same orientation as
        tool0 (rigid, no extra rotation)."""
        x_ee = self.forward_kinematics(q)
        return pin.SE3(x_ee.rotation, x_ee.translation + x_ee.rotation @ offset_local)

    def jacobian_offset(self, q: np.ndarray, offset_local: np.ndarray) -> np.ndarray:
        """Jacobian of the same rigidly-attached offset point (world-aligned).
        Standard rigid-body point-translation Jacobian correction:
            J_p_linear  = J_ee_linear - skew(p_world - ee_world) @ J_ee_angular
            J_p_angular = J_ee_angular
        """
        x_ee = self.forward_kinematics(q)
        J = self.jacobian(q)
        p_world_offset = x_ee.rotation @ offset_local  # ee -> offset point, world-aligned
        skew = pin.skew(p_world_offset)
        J_p = J.copy()
        J_p[:3, :] = J[:3, :] - skew @ J[3:, :]
        return J_p

    def jacobian_dot_qdot(self, q: np.ndarray, v: np.ndarray, frame: str | None = None) -> np.ndarray:
        """Classical (Jdot * qdot) term, needed for operational-space accel control."""
        frame_id = self.model.getFrameId(frame) if frame else self.ee_frame_id
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.nv))
        acc = pin.getFrameClassicalAcceleration(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return _rotate_jacobian(acc.vector.reshape(6, 1)).flatten()  # q_ddot=0 -> this IS Jdot*qdot

    # ---- dynamics --------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return pin.crba(self.model, self.data, q)

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Coriolis + gravity (what RNEA returns with qdd=0)."""
        return pin.rnea(self.model, self.data, q, v, np.zeros(self.nv))

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def operational_space_mass_matrix(self, q: np.ndarray, frame: str | None = None,
                                        damping: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Lambda = (J M^-1 J^T)^-1, the Cartesian-space inertia used for impedance
        control, plus the Jacobian used to compute it (J-pinv needed downstream)."""
        M = self.mass_matrix(q)
        J = self.jacobian(q, frame)
        M_inv = np.linalg.inv(M + damping * np.eye(self.nv))
        JMJt = J @ M_inv @ J.T
        Lambda = np.linalg.pinv(JMJt, rcond=1e-4)
        return Lambda, J

    def inverse_kinematics(self, target: pin.SE3, q_init: np.ndarray, frame: str | None = None,
                            max_iters: int = 200, eps: float = 1e-4, damping: float = 1e-6) -> tuple[np.ndarray, bool]:
        """Damped-least-squares IK to a target EE pose given in the MuJoCo world
        frame (same convention as forward_kinematics). Returns (q, converged)."""
        frame_id = self.model.getFrameId(frame) if frame else self.ee_frame_id
        target = _BASE_FIX.inverse() * target  # -> Pinocchio's own base-frame convention
        q = q_init.copy()
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, frame_id)
            err = pin.log6(self.data.oMf[frame_id].actInv(target)).vector
            if np.linalg.norm(err) < eps:
                return q, True
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = pin.integrate(self.model, q, dq * 0.5)
        return q, False


if __name__ == "__main__":
    m = UR5eModel()
    q = pin.neutral(m.model)
    print("Joint names:", m.joint_names)
    print("EE pose at q=neutral:\n", m.forward_kinematics(q))
    J = m.jacobian(q)
    print("Jacobian shape:", J.shape)
    Lambda, _ = m.operational_space_mass_matrix(q)
    print("Operational-space mass matrix eigenvalues:", np.linalg.eigvalsh(Lambda))
