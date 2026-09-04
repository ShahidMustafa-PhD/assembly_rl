"""
Gymnasium environment for vision-guided contact-rich assembly (peg-in-hole /
connector mating / kitting) on a simulated UR5e, matching the observation
space, action interface and domain-randomization scheme described in the
project proposal.

Observation (Dict): hybrid vision + proprioceptive + force/torque -- the
policy is given the same information a real system would have (camera +
joint encoders + wrist F/T), NOT the ground-truth hole pose. Localizing the
hole from the wrist camera is part of what the policy has to learn; that's
the actual point of the "vision-guided" framing.
    image:        (84, 84, 3) uint8   -- wrist-mounted eye-in-hand RGB
    proprio:       (12,) float32       -- [q(6), qdot(6)]
    force_torque:   (6,) float32        -- wrist wrench, world-frame, clipped
    ee_pose:         (7,) float32        -- [xyz, quat] of the controlled tool
                                          point (peg tip / gripper), world frame
                                          -- exact from forward kinematics, NOT
                                          a vision estimate (matches what a real
                                          system gets for free from joint encoders)

Action (Box, peg_in_hole/connector_mating: 6D, kitting: 7D), all in [-1, 1]:
    [0:3]  delta position of the controlled tool point, world frame
    [3:6]  delta orientation (small-angle axis-angle), world frame
    [6]    (kitting only) gripper open<->close command

The action is a *reference delta* fed into a fixed-gain Cartesian impedance
controller (control/impedance_controller.py) running every physics substep --
deliberately softer-gain than the classical baseline's, so that a random/
early-training policy doesn't produce baseline-style torque saturation or
contact-force spikes (see scripts/smoke_test.py for how those gains were
tuned). Compliance and force limiting therefore come from the *fixed* low-
level controller, exactly like a real impedance-controlled arm; the policy
only ever sees/produces task-space deltas.

Domain randomization is split by cost (see scene_builder.py docstring):
  * every reset(), no recompile: fixture pose jitter, peg/table friction,
    table/peg color -- done directly on the compiled MjModel's arrays.
  * only when `difficulty` changes: hole clearance tier (recompiled model,
    cached per (task, difficulty) so switching back is instant).
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import pinocchio as pin
from gymnasium import spaces

import mujoco

from envs.scene_builder import SceneConfig, build_model
from control.pinocchio_model import UR5eModel
from control.impedance_controller import CartesianImpedanceController, ImpedanceGains

HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
PEG_TOOL_OFFSET = np.array([0.0, 0.0, 0.196])  # see impedance_controller.py docstring

# Softer than the classical baseline's gains (ImpedanceGains() defaults) -- keeps a
# random/early policy's action deltas from saturating torque limits or slamming the
# fixture; see module docstring.
RL_GAINS = ImpedanceGains(
    kp_pos=np.array([300.0, 300.0, 300.0]),
    kd_pos=np.array([30.0, 30.0, 30.0]),
    kp_rot=np.array([15.0, 15.0, 15.0]),
    kd_rot=np.array([2.0, 2.0, 2.0]),
    max_lin_vel=0.15,
    max_ang_vel=1.0,
)

_MODEL_CACHE: dict[tuple, mujoco.MjModel] = {}


def _get_model(task: str, difficulty: str, seed: int | None) -> mujoco.MjModel:
    """Structural changes (clearance tier / part count) require recompiling the
    MjSpec; cache per (task, difficulty, seed) so repeated resets at the same
    curriculum stage are cheap. Pose/friction/visual DR happens post-compile."""
    key = (task, difficulty, seed)
    if key not in _MODEL_CACHE:
        cfg = SceneConfig(task=task, difficulty=difficulty if task != "kitting" else "loose", seed=seed)
        _MODEL_CACHE[key] = build_model(cfg)
    return _MODEL_CACHE[key]


class AssemblyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str = "peg_in_hole", difficulty: str = "loose",
                 max_episode_steps: int = 400, control_decimation: int = 10,
                 image_size: int = 84, render_images: bool = True,
                 domain_randomize: bool = True, structural_seed: int | None = 0):
        super().__init__()
        assert task in ("peg_in_hole", "connector_mating", "kitting")
        assert difficulty in ("loose", "tight")
        self.task = task
        self.difficulty = difficulty
        self.max_episode_steps = max_episode_steps
        self.control_decimation = control_decimation
        self.image_size = image_size
        self.render_images = render_images
        self.domain_randomize = domain_randomize
        self._structural_seed = structural_seed

        self.model = _get_model(task, difficulty, structural_seed)
        self.data = mujoco.MjData(self.model)
        self.robot = UR5eModel()
        self.controller = CartesianImpedanceController(self.robot, gains=RL_GAINS, tool_offset=PEG_TOOL_OFFSET)

        self._renderer = None
        if self.render_images:
            self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)

        self._is_kitting = task == "kitting"
        action_dim = 7 if self._is_kitting else 6
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
            "force_torque": spaces.Box(-100.0, 100.0, shape=(6,), dtype=np.float32),
            "ee_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        })

        # Step-scale for actions (max motion commanded per env.step()).
        self.max_pos_delta = 0.008   # 8mm
        self.max_rot_delta = 0.10    # ~5.7deg
        self.force_penalty_thresh = 15.0
        self.force_fail_thresh = 60.0

        self._nominal = self._cache_nominal_dr_values()
        self._episode_step = 0
        self._prev_dist = None
        self._x_ref = None
        self._rng = np.random.default_rng(structural_seed)

    def _switch_model(self, new_difficulty: str) -> None:
        """Swap to the (cached, pre-compiled) MjModel for a new difficulty tier --
        used by rl/train.py's CurriculumCallback. Cheap: only recompiles the very
        first time a given (task, difficulty, seed) combination is requested (see
        _get_model's cache); every later call just swaps the object reference."""
        if new_difficulty == self.difficulty and self.model is not None:
            return
        self.difficulty = new_difficulty
        self.model = _get_model(self.task, new_difficulty, self._structural_seed)
        self.data = mujoco.MjData(self.model)
        if self.render_images:
            self._renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self._nominal = self._cache_nominal_dr_values()

    # ---- domain randomization (cheap, in-place, no recompile) --------------
    def _cache_nominal_dr_values(self):
        m = self.model
        cache = {}
        if "fixture" in [m.body(i).name for i in range(m.nbody)]:
            fid = m.body("fixture").id
            cache["fixture_pos"] = m.body_pos[fid].copy()
        peg_gid = m.geom("peg_geom").id if self._has_geom("peg_geom") else None
        cache["peg_geom_id"] = peg_gid
        if peg_gid is not None:
            cache["peg_friction"] = m.geom_friction[peg_gid].copy()
            cache["peg_rgba"] = m.geom_rgba[peg_gid].copy()
        table_gid = m.geom("table_top").id
        cache["table_gid"] = table_gid
        cache["table_rgba"] = m.geom_rgba[table_gid].copy()
        return cache

    def _has_geom(self, name: str) -> bool:
        try:
            self.model.geom(name)
            return True
        except KeyError:
            return False

    def _apply_domain_randomization(self):
        if not self.domain_randomize:
            return
        m = self.model
        rng = self._rng
        if "fixture_pos" in self._nominal:
            fid = m.body("fixture").id
            jitter = rng.uniform(-0.01, 0.01, size=3)
            jitter[2] = 0.0
            m.body_pos[fid] = self._nominal["fixture_pos"] + jitter
        pgid = self._nominal["peg_geom_id"]
        if pgid is not None:
            scale = rng.uniform(0.7, 1.3)
            m.geom_friction[pgid] = self._nominal["peg_friction"] * [scale, 1.0, 1.0]
            hue_jitter = rng.uniform(-0.08, 0.08, size=3)
            m.geom_rgba[pgid, :3] = np.clip(self._nominal["peg_rgba"][:3] + hue_jitter, 0, 1)
        tgid = self._nominal["table_gid"]
        hue_jitter = rng.uniform(-0.15, 0.15, size=3)
        m.geom_rgba[tgid, :3] = np.clip(self._nominal["table_rgba"][:3] + hue_jitter, 0, 1)

    # ---- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_Q + self._rng.uniform(-0.03, 0.03, size=6)
        self._apply_domain_randomization()
        mujoco.mj_forward(self.model, self.data)

        self.controller.reset_reference(None)
        self._episode_step = 0
        q = self.data.qpos[:6].copy()
        x_cur = self.robot.forward_kinematics_offset(q, PEG_TOOL_OFFSET)
        self._x_ref = x_cur
        self._prev_dist = self._goal_distance()

        obs = self._get_obs()
        info = {"goal_pos": self._goal_pos().copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        pos_delta = action[:3] * self.max_pos_delta
        rot_delta = action[3:6] * self.max_rot_delta
        gripper_cmd = float(action[6]) if self._is_kitting else None

        new_pos = self._x_ref.translation + pos_delta
        new_rot = pin.exp3(rot_delta) @ self._x_ref.rotation
        self._x_ref = pin.SE3(new_rot, new_pos)

        dt = self.model.opt.timestep
        max_force_seen = 0.0
        for _ in range(self.control_decimation):
            q = self.data.qpos[:6].copy()
            qdot = self.data.qvel[:6].copy()
            wrench = self._wrench_world()
            max_force_seen = max(max_force_seen, float(np.linalg.norm(wrench[:3])))
            tau = self.controller.step(q, qdot, self._x_ref, 0.0, np.array([1.0, 1.0, 1.0]), wrench, dt=dt)
            self.data.ctrl[:6] = np.clip(tau, self.model.actuator_ctrlrange[:6, 0], self.model.actuator_ctrlrange[:6, 1])
            if self._is_kitting:
                self.data.ctrl[6] = np.clip((gripper_cmd + 1.0) / 2.0 * 255.0, 0, 255)
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)):
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:6] = HOME_Q
                mujoco.mj_forward(self.model, self.data)
                break

        self._episode_step += 1
        obs = self._get_obs()
        reward, success, safety_violation = self._compute_reward(action, max_force_seen)
        terminated = bool(success or safety_violation)
        truncated = self._episode_step >= self.max_episode_steps
        info = {"success": success, "safety_violation": safety_violation, "goal_distance": self._goal_distance()}
        return obs, reward, terminated, truncated, info

    # ---- helpers -----------------------------------------------------------
    def _goal_pos(self) -> np.ndarray:
        if self.task == "kitting":
            return self.data.site("kit_slot_0").xpos.copy()
        return self.data.site("hole_target").xpos.copy()

    def _tip_pose(self) -> pin.SE3:
        q = self.data.qpos[:6].copy()
        return self.robot.forward_kinematics_offset(q, PEG_TOOL_OFFSET)

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self._tip_pose().translation - self._goal_pos()))

    def _wrench_world(self) -> np.ndarray:
        f_local = self.data.sensor("wrist_force").data.copy()
        t_local = self.data.sensor("wrist_torque").data.copy()
        site_id = self.model.site("wrist_ft_site").id
        R = self.data.site_xmat[site_id].reshape(3, 3)
        return np.concatenate([R @ f_local, R @ t_local])

    def _get_obs(self) -> dict:
        q = self.data.qpos[:6].astype(np.float32)
        qdot = self.data.qvel[:6].astype(np.float32)
        wrench = np.clip(self._wrench_world(), -100.0, 100.0).astype(np.float32)
        tip = self._tip_pose()
        quat = pin.Quaternion(tip.rotation)
        ee_pose = np.concatenate([tip.translation, [quat.x, quat.y, quat.z, quat.w]]).astype(np.float32)

        if self.render_images:
            self._renderer.update_scene(self.data, camera="wrist_cam")
            image = self._renderer.render()
            if image.shape[0] != self.image_size:
                image = image[: self.image_size, : self.image_size]
        else:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        return {
            "image": image.astype(np.uint8),
            "proprio": np.concatenate([q, qdot]).astype(np.float32),
            "force_torque": wrench,
            "ee_pose": ee_pose,
        }

    def _compute_reward(self, action: np.ndarray, max_force_seen: float) -> tuple[float, bool, bool]:
        dist = self._goal_distance()
        progress = self._prev_dist - dist
        self._prev_dist = dist

        reward = 2.0 * progress - 0.02 * dist - 0.01
        reward -= 0.001 * float(np.sum(action ** 2))  # smoothness

        force_excess = max(0.0, max_force_seen - self.force_penalty_thresh)
        reward -= 0.02 * force_excess

        depth_ok = self._tip_pose().translation[2] - self._goal_pos()[2] < 0.005
        success = bool(dist < 0.004 and depth_ok)
        if success:
            reward += 20.0

        safety_violation = bool(max_force_seen > self.force_fail_thresh)
        if safety_violation:
            reward -= 5.0
        return float(reward), success, safety_violation

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="scene_cam")
        return self._renderer.render()

    def close(self):
        self._renderer = None


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = AssemblyEnv(task="peg_in_hole", difficulty="loose")
    obs, info = env.reset(seed=0)
    print({k: v.shape for k, v in obs.items()})
    total_r = 0.0
    for _ in range(50):
        a = env.action_space.sample() * 0.3
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            break
    print("ran 50 random steps, total reward:", total_r, "info:", info)
