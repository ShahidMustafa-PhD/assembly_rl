"""
Train SAC or PPO on the assembly-RL environment with identical network
architecture (rl/policies.py) and environment/curriculum setup, so the two
algorithms are compared fairly (see benchmark/evaluate.py for the actual
head-to-head against the classical baseline).

Usage:
    MUJOCO_GL=egl python3 -m rl.train --algo sac --task peg_in_hole --difficulty loose \
        --timesteps 200000 --n_envs 4 --logdir runs/sac_loose

    MUJOCO_GL=egl python3 -m rl.train --algo ppo --task peg_in_hole --difficulty loose \
        --timesteps 200000 --n_envs 8 --logdir runs/ppo_loose

Curriculum (--curriculum): starts on `--difficulty`, and once a rolling window
of episodes crosses `--curriculum_success` success rate, switches every
subsequent reset to the next tier (loose -> tight). Implemented as a
SB3 callback so it works identically for both algorithms.
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage

from envs.assembly_env import AssemblyEnv
from rl.policies import POLICY_KWARGS

DIFFICULTY_ORDER = ["loose", "tight"]


class CurriculumCallback(BaseCallback):
    """Advances env.difficulty (loose -> tight) once recent success rate clears a
    threshold. Difficulty is a *per-env* attribute read at the next reset(); the
    underlying MjModel is cached per-difficulty in envs/assembly_env.py so this
    switch does not stall training on a recompile."""

    def __init__(self, success_threshold: float = 0.6, window: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.window = window
        self._successes: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "success" in info and (info.get("TimeLimit.truncated", False) or info["success"]):
                self._successes.append(1.0 if info["success"] else 0.0)
        if len(self._successes) >= self.window and np.mean(self._successes) >= self.success_threshold:
            envs = self.training_env.get_attr("difficulty")
            idx = DIFFICULTY_ORDER.index(envs[0])
            if idx < len(DIFFICULTY_ORDER) - 1:
                new_diff = DIFFICULTY_ORDER[idx + 1]
                self.training_env.set_attr("difficulty", new_diff)
                self.training_env.env_method("_switch_model", new_diff)
                self._successes.clear()
                if self.verbose:
                    print(f"[curriculum] success rate cleared {self.success_threshold:.0%} -> advancing to {new_diff!r}")
        return True


def make_env(task: str, difficulty: str, render_images: bool, seed: int):
    def _init():
        env = AssemblyEnv(task=task, difficulty=difficulty, render_images=render_images, structural_seed=seed)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_vec_env(args):
    import stable_baselines3.common.vec_env
    env_fns = [make_env(args.task, args.difficulty, True, seed=i) for i in range(args.n_envs)]
    if args.n_envs == 1:
        vec_env = stable_baselines3.common.vec_env.DummyVecEnv(env_fns)
    else:
        # "fork" (Linux's default) duplicates the parent's already-loaded GPU/EGL
        # driver state into each child, which crashes those children before they
        # can respond -- surfacing here as SubprocVecEnv's handshake failing with
        # ConnectionResetError. "spawn" starts each worker as a genuinely fresh
        # interpreter instead, avoiding the inherited GPU state.
        vec_env = stable_baselines3.common.vec_env.SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo"], required=True)
    p.add_argument("--task", default="peg_in_hole", choices=["peg_in_hole", "connector_mating", "kitting"])
    p.add_argument("--difficulty", default="loose", choices=["loose", "tight"])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--logdir", default="runs/default")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(args)

    common_kwargs = dict(
        policy="MultiInputPolicy", env=vec_env, policy_kwargs=POLICY_KWARGS,
        verbose=1, tensorboard_log=args.logdir, seed=args.seed, device="cpu",
    )
    if args.algo == "sac":
        model = SAC(buffer_size=100_000, batch_size=256, learning_starts=1_000,
                     train_freq=1, gradient_steps=1, learning_rate=3e-4, **common_kwargs)
    else:
        model = PPO(n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4, **common_kwargs)

    callbacks = [CheckpointCallback(save_freq=max(10_000 // args.n_envs, 1), save_path=args.logdir,
                                     name_prefix=f"{args.algo}_{args.task}")]
    if args.curriculum:
        callbacks.append(CurriculumCallback())

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(str(Path(args.logdir) / f"{args.algo}_{args.task}_final"))
    print(f"Saved final model to {args.logdir}/{args.algo}_{args.task}_final.zip")


if __name__ == "__main__":
    main()
