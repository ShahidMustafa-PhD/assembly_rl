"""Fast end-to-end smoke test: scene compiles, MuJoCo<->Pinocchio joint mapping
agrees, the impedance baseline runs stably and drives the peg toward the hole,
and the Gymnasium env resets/steps without error. Not a unit-test suite --
a single script that fails loudly if any layer of the stack is broken.

Run with: MUJOCO_GL=egl python3 scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pinocchio as pin
import mujoco

from envs.scene_builder import build_model, SceneConfig
from control.pinocchio_model import UR5eModel
from control.impedance_controller import HybridPegInHolePolicy

HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def check_fk_agreement():
    print("[1/4] Checking MuJoCo <-> Pinocchio FK agreement at home pose ...")
    model = build_model(SceneConfig(task="peg_in_hole", difficulty="loose"))
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME_Q
    mujoco.mj_forward(model, data)

    robot = UR5eModel()
    ee_pin = robot.forward_kinematics(HOME_Q)

    ee_body = model.body("wrist_3_link")
    site_id = model.site("attachment_site").id
    mj_pos = data.site_xpos[site_id]

    diff = np.linalg.norm(ee_pin.translation - mj_pos)
    print(f"    Pinocchio EE (tool0): {ee_pin.translation}")
    print(f"    MuJoCo attachment_site: {mj_pos}")
    print(f"    Position diff: {diff:.4f} m", "OK" if diff < 0.03 else "**MISMATCH**")
    assert diff < 0.05, "MuJoCo/Pinocchio end-effector frames disagree by more than 5cm"


def run_impedance_rollout(difficulty: str, n_steps: int = 1500, render_path: str | None = None):
    print(f"[2-3/4] Running hybrid impedance baseline rollout (difficulty={difficulty}) ...")
    cfg = SceneConfig(task="peg_in_hole", difficulty=difficulty, seed=0)
    model = build_model(cfg)
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME_Q
    mujoco.mj_forward(model, data)

    robot = UR5eModel()
    hole_site_id = model.site("hole_target").id
    hole_pos = data.site_xpos[hole_site_id].copy()

    # hole_quat_world=None -> hold the arm's starting (home) orientation throughout,
    # rather than commanding an arbitrary/unreachable target (see policy docstring).
    policy = HybridPegInHolePolicy(robot, hole_pos, None)

    dt = model.opt.timestep
    peg_geom_id = model.geom("peg_geom").id
    min_gap_to_hole = np.inf
    max_force = 0.0

    for i in range(n_steps):
        q = data.qpos[:6].copy()
        qdot = data.qvel[:6].copy()
        f_local = data.sensor("wrist_force").data.copy()
        t_local = data.sensor("wrist_torque").data.copy()
        # Rotate the wrist-local F/T reading into world axes using the sensor site.
        site_id = model.site("wrist_ft_site").id
        R = data.site_xmat[site_id].reshape(3, 3)
        wrench_world = np.concatenate([R @ f_local, R @ t_local])

        tau = policy.act(q, qdot, wrench_world, dt)
        data.ctrl[:6] = np.clip(tau, model.actuator_ctrlrange[:6, 0], model.actuator_ctrlrange[:6, 1])
        mujoco.mj_step(model, data)

        peg_pos = data.geom_xpos[peg_geom_id]
        gap = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        min_gap_to_hole = min(min_gap_to_hole, gap)
        max_force = max(max_force, abs(wrench_world[2]))

        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"Simulation diverged (NaN) at step {i}, phase={policy.phase}")

    print(f"    Final phase: {policy.phase.name}")
    print(f"    Min XY gap peg-vs-hole during rollout: {min_gap_to_hole * 1000:.2f} mm")
    print(f"    Max |Fz| observed: {max_force:.2f} N")
    final_peg_z = data.geom_xpos[peg_geom_id][2]
    print(f"    Final peg height above hole target: {(final_peg_z - hole_pos[2]) * 1000:.1f} mm")

    if render_path:
        import PIL.Image
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="scene_cam")
        PIL.Image.fromarray(renderer.render()).save(render_path)
        print(f"    Saved final-state render to {render_path}")

    print(f"    -> {'PASS (reached DONE)' if policy.phase.name == 'DONE' else 'incomplete (expected for tight clearance; see README)'}")
    return policy.phase.name, min_gap_to_hole, max_force


def check_gym_env():
    print("[4/4] Checking Gymnasium AssemblyEnv reset/step ...")
    from envs.assembly_env import AssemblyEnv

    env = AssemblyEnv(task="peg_in_hole", difficulty="loose", render_images=False)
    obs, info = env.reset(seed=0)
    print(f"    obs keys: {list(obs.keys())}, shapes: {[ (k, v.shape) for k, v in obs.items()]}")
    for _ in range(20):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("    Gym env OK")


if __name__ == "__main__":
    t0 = time.time()
    check_fk_agreement()
    phase, _, _ = run_impedance_rollout("loose", n_steps=3000, render_path=str(Path(__file__).parent / "impedance_loose_final.png"))
    assert phase == "DONE", "Expected the loose-clearance baseline to complete insertion"
    run_impedance_rollout("tight", n_steps=3000, render_path=str(Path(__file__).parent / "impedance_tight_final.png"))
    check_gym_env()
    print(f"\nAll smoke tests passed in {time.time() - t0:.1f}s")
