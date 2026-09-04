"""
Benchmark harness: classical hybrid impedance baseline vs. a trained SAC/PPO
policy, on identical MuJoCo episodes, reporting the metrics named in the
project proposal -- success rate, insertion time, peak contact force, and
robustness to part-pose perturbation -- across the difficulty ladder
(loose-clearance peg-in-hole -> tight-clearance insertion).

Usage:
    # Classical baseline only (works with no trained policy):
    MUJOCO_GL=egl python3 -m benchmark.evaluate --difficulties loose tight --episodes 20

    # Classical baseline vs. a trained policy:
    MUJOCO_GL=egl python3 -m benchmark.evaluate --policy_path runs/sac_loose/sac_peg_in_hole_final.zip \
        --algo sac --difficulties loose tight --episodes 20 --robustness

Output: a printed comparison table + benchmark/results_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import mujoco

from envs.scene_builder import SceneConfig, build_model
from envs.assembly_env import AssemblyEnv, PEG_TOOL_OFFSET, HOME_Q
from control.pinocchio_model import UR5eModel
from control.impedance_controller import HybridPegInHolePolicy

RESULTS_DIR = Path(__file__).resolve().parent


@dataclass
class EpisodeResult:
    success: bool
    steps_to_success: int | None
    time_to_success: float | None
    peak_force: float
    final_xy_gap_mm: float


@dataclass
class AggregateResult:
    controller: str
    task: str
    difficulty: str
    pose_perturbation_mm: float
    n_episodes: int
    success_rate: float
    mean_time_to_success: float | None
    mean_peak_force: float
    p95_peak_force: float


def _run_classical_episode(task: str, difficulty: str, seed: int, pose_perturbation_mm: float,
                            max_steps: int = 3000) -> EpisodeResult:
    cfg = SceneConfig(task=task, difficulty=difficulty, seed=seed)
    model = build_model(cfg)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    data.qpos[:6] = HOME_Q + rng.uniform(-0.02, 0.02, size=6)
    if pose_perturbation_mm > 0 and "fixture" in [model.body(i).name for i in range(model.nbody)]:
        fid = model.body("fixture").id
        jitter = rng.uniform(-1, 1, size=3) * (pose_perturbation_mm / 1000.0)
        jitter[2] = 0.0
        model.body_pos[fid] = model.body_pos[fid] + jitter
    mujoco.mj_forward(model, data)

    robot = UR5eModel()
    hole_pos = data.site("hole_target").xpos.copy()
    policy = HybridPegInHolePolicy(robot, hole_pos, None)

    dt = model.opt.timestep
    peak_force = 0.0
    success_step = None
    for step in range(max_steps):
        q, qdot = data.qpos[:6].copy(), data.qvel[:6].copy()
        f_local = data.sensor("wrist_force").data.copy()
        t_local = data.sensor("wrist_torque").data.copy()
        R = data.site_xmat[model.site("wrist_ft_site").id].reshape(3, 3)
        wrench = np.concatenate([R @ f_local, R @ t_local])
        peak_force = max(peak_force, float(np.linalg.norm(wrench[:3])))

        tau = policy.act(q, qdot, wrench, dt)
        data.ctrl[:6] = np.clip(tau, model.actuator_ctrlrange[:6, 0], model.actuator_ctrlrange[:6, 1])
        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)):
            break
        if policy.phase.name == "DONE" and success_step is None:
            success_step = step
            break

    tip = robot.forward_kinematics_offset(data.qpos[:6].copy(), PEG_TOOL_OFFSET)
    xy_gap = float(np.linalg.norm(tip.translation[:2] - hole_pos[:2]) * 1000)
    return EpisodeResult(
        success=success_step is not None, steps_to_success=success_step,
        time_to_success=(success_step * dt) if success_step is not None else None,
        peak_force=peak_force, final_xy_gap_mm=xy_gap,
    )


def _run_rl_episode(env: AssemblyEnv, model, seed: int, pose_perturbation_mm: float,
                     max_steps: int = 400) -> EpisodeResult:
    # Temporarily override the env's DR pose-jitter range for the robustness sweep.
    orig_dr = env.domain_randomize
    env.domain_randomize = pose_perturbation_mm > 0
    obs, info = env.reset(seed=seed)
    if pose_perturbation_mm > 0 and "fixture" in [env.model.body(i).name for i in range(env.model.nbody)]:
        fid = env.model.body("fixture").id
        rng = np.random.default_rng(seed + 10_000)
        jitter = rng.uniform(-1, 1, size=3) * (pose_perturbation_mm / 1000.0)
        jitter[2] = 0.0
        env.model.body_pos[fid] = env._nominal["fixture_pos"] + jitter
        mujoco.mj_forward(env.model, env.data)
    env.domain_randomize = orig_dr

    peak_force = 0.0
    success_step = None
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        peak_force = max(peak_force, float(np.linalg.norm(env._wrench_world()[:3])))
        if info.get("success"):
            success_step = step
            break
        if terminated or truncated:
            break

    dt = env.model.opt.timestep * env.control_decimation
    xy_gap = float(np.linalg.norm(env._tip_pose().translation[:2] - env._goal_pos()[:2]) * 1000)
    return EpisodeResult(
        success=success_step is not None, steps_to_success=success_step,
        time_to_success=(success_step * dt) if success_step is not None else None,
        peak_force=peak_force, final_xy_gap_mm=xy_gap,
    )


def _aggregate(results: list[EpisodeResult], controller: str, task: str, difficulty: str,
               pose_perturbation_mm: float) -> AggregateResult:
    n = len(results)
    successes = [r for r in results if r.success]
    times = [r.time_to_success for r in successes]
    forces = [r.peak_force for r in results]
    return AggregateResult(
        controller=controller, task=task, difficulty=difficulty, pose_perturbation_mm=pose_perturbation_mm,
        n_episodes=n, success_rate=len(successes) / n if n else 0.0,
        mean_time_to_success=float(np.mean(times)) if times else None,
        mean_peak_force=float(np.mean(forces)) if forces else 0.0,
        p95_peak_force=float(np.percentile(forces, 95)) if forces else 0.0,
    )


def run_benchmark(task: str, difficulties: list[str], episodes: int, policy_path: str | None,
                   algo: str, robustness_levels: list[float]) -> list[AggregateResult]:
    aggregates: list[AggregateResult] = []

    rl_model = None
    rl_env = None
    if policy_path:
        from stable_baselines3 import SAC, PPO
        loader = SAC if algo == "sac" else PPO
        rl_model = loader.load(policy_path)

    for difficulty in difficulties:
        for pert in robustness_levels:
            classical_results = [_run_classical_episode(task, difficulty, seed=i, pose_perturbation_mm=pert)
                                  for i in range(episodes)]
            aggregates.append(_aggregate(classical_results, "classical_impedance", task, difficulty, pert))
            print(f"[classical] task={task} difficulty={difficulty} perturbation={pert}mm -> "
                  f"success={aggregates[-1].success_rate:.0%}")

            if rl_model is not None:
                if rl_env is None or rl_env.difficulty != difficulty:
                    rl_env = AssemblyEnv(task=task, difficulty=difficulty, render_images=True, structural_seed=0)
                rl_results = [_run_rl_episode(rl_env, rl_model, seed=i, pose_perturbation_mm=pert)
                              for i in range(episodes)]
                aggregates.append(_aggregate(rl_results, f"rl_{algo}", task, difficulty, pert))
                print(f"[{algo}]       task={task} difficulty={difficulty} perturbation={pert}mm -> "
                      f"success={aggregates[-1].success_rate:.0%}")

    return aggregates


def print_table(aggregates: list[AggregateResult]) -> None:
    header = f"{'controller':<20}{'difficulty':<10}{'pert(mm)':<10}{'success':<10}{'mean_t(s)':<12}{'mean_Fpeak(N)':<15}{'p95_Fpeak(N)':<12}"
    print("\n" + header)
    print("-" * len(header))
    for a in aggregates:
        mt = f"{a.mean_time_to_success:.2f}" if a.mean_time_to_success is not None else "--"
        print(f"{a.controller:<20}{a.difficulty:<10}{a.pose_perturbation_mm:<10.1f}{a.success_rate:<10.0%}"
              f"{mt:<12}{a.mean_peak_force:<15.1f}{a.p95_peak_force:<12.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="peg_in_hole")
    p.add_argument("--difficulties", nargs="+", default=["loose", "tight"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--policy_path", default=None)
    p.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    p.add_argument("--robustness", action="store_true", help="Sweep pose-perturbation magnitude 0/5/10/15mm")
    args = p.parse_args()

    robustness_levels = [0.0, 5.0, 10.0, 15.0] if args.robustness else [0.0]
    aggregates = run_benchmark(args.task, args.difficulties, args.episodes, args.policy_path,
                                args.algo, robustness_levels)
    print_table(aggregates)

    out_path = RESULTS_DIR / f"results_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(a) for a in aggregates], f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
