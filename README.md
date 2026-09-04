# Vision-Guided Deep RL for Contact-Rich Robotic Assembly

A simulated UR5e learns robust peg-in-hole / connector-mating / kitting
policies from a hybrid vision + force/torque observation space, benchmarked
against a classical hybrid position/force impedance controller. Built around
MuJoCo (contact-rich physics), Pinocchio (kinematics/dynamics), SAC/PPO
(stable-baselines3), and a ROS2 deployment path that reuses the exact same
control code as simulation.

This extends the digital-twin/predictive-maintenance methodology of
[RoboticPreM](https://github.com/ShahidMustafa-PhD) (same UR5e/MuJoCo/
Pinocchio stack) into contact-rich manipulation RL.

## Architecture

```
                          ┌─────────────────────────┐
                          │   envs/scene_builder.py  │  MjSpec composition:
                          │   UR5e + 2F-85 + fixture │  loose/tight clearance,
                          └────────────┬─────────────┘  kitting tray, DR hooks
                                       │ compiles to
                                       ▼
   ┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
   │ control/          │◄────►│  MuJoCo MjModel/Data │◄────►│ envs/assembly_env.py  │
   │ pinocchio_model.py│      │  (physics + F/T sim) │      │ Gymnasium Dict-obs env│
   │ (FK/J/M/g, UR5e)  │      └─────────────────────┘      └──────────┬────────────┘
   └────────┬──────────┘                                              │
            │ reused by both                                          │ trains
            ▼                                                          ▼
   ┌────────────────────────────┐                          ┌───────────────────────┐
   │ control/                    │  classical baseline      │  rl/policies.py +      │
   │ impedance_controller.py     │  (benchmark/evaluate.py) │  rl/train.py           │
   │ Cartesian impedance +       │                          │  SAC / PPO, hybrid     │
   │ hybrid pos/force + FSM      │                          │  CNN+MLP extractor     │
   └────────┬─────────────────────                          └──────────┬────────────┘
            │                                                          │
            └─────────────────────────┬────────────────────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │        benchmark/evaluate.py        │
                     │  success rate, insertion time,      │
                     │  peak force, pose-perturbation       │
                     │  robustness -- classical vs. RL      │
                     └───────────────────────────────────┘

   ros2_ws/src/assembly_rl_ros2/  -- deployment path reusing control/* directly:
     perception_node -> policy_node -> impedance_controller_node -> real UR5e
```

## Setup

Install torch **first**, from the wheel index matching your hardware --
a bare `pip install torch` resolves to whatever CUDA build is newest on
PyPI (currently CUDA 13, requiring driver >=580), which fails with
`CUDA error: no kernel image is available for execution on the device`
or a driver-too-old error on anything with an older driver. `torch` is
deliberately left out of `requirements.txt` for this reason --
`stable-baselines3>=2.2` needs `torch>=2.8,<3.0`, and CUDA 12.6 wheels
cover that range while needing only driver >=560.28 (broadly available,
including Colab's T4 runtime):

```bash
# NVIDIA GPU (check your driver first: `nvidia-smi` -- if it reports
# driver <560, either update it or use an older CUDA index/torch combo):
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# No GPU / CPU only:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Verify before moving on:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

pip install -r requirements.txt
```

The UR5e + Robotiq 2F-85 MuJoCo models come from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(already copied into `assets/`, so no re-clone needed). The UR5e URDF used
by Pinocchio (`assets/urdf/ur5e.urdf`) was generated once via `xacro` from
[UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
with `ur_type:=ur5e` -- it's committed, so you don't need `xacro`/ROS2
installed just to use it; regenerate it with:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git /tmp/ur_desc
cd /tmp/ur_desc && git sparse-checkout set urdf meshes config
sed -i 's|\$(find ur_description)|/tmp/ur_desc|g' $(grep -rl '\$(find ur_description)' urdf config)
xacro urdf/ur.urdf.xacro ur_type:=ur5e name:=ur5e -o ur5e.urdf
```

MuJoCo needs a headless GL backend in a container/CI (no display): run
everything with `MUJOCO_GL=egl` in the environment (already the default
assumption throughout this README and in `envs/assembly_env.py`).

## Google Colab (GPU runtime)

Use a GPU runtime (Runtime -> Change runtime type -> GPU) -- MuJoCo's
`MUJOCO_GL=egl` headless rendering needs an actual GPU/EGL device, and
stable-baselines3 has no TPU/XLA backend, so a TPU runtime cannot run this
regardless of `--device`.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ShahidMustafa-PhD/assembly_rl.git /content/assembly_rl
%cd /content/assembly_rl

# Colab's GPU image ships the NVIDIA driver but not the EGL ICD vendor file
# MuJoCo's MUJOCO_GL=egl needs to find it -- without this you get
# "Cannot initialize EGL" / "an OpenGL platform library has not been loaded".
!mkdir -p /usr/share/glvnd/egl_vendor.d
!echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

!nvidia-smi -L   # confirm the GPU runtime actually attached (T4)

# Install torch FIRST, pinned to a CUDA 12.6 wheel -- plain `pip install
# torch` (or letting requirements.txt pull it in transitively) resolves to
# the newest PyPI build, currently CUDA 13 (needs driver >=580), which is
# newer than what Colab's T4 runtime typically ships and fails with a
# driver-mismatch/"no kernel image" CUDA error at training time, not at
# install time -- this is the error that kept recurring. cu126 satisfies
# stable-baselines3's torch>=2.8,<3.0 floor while only needing driver >=560.
!pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
!python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- check nvidia-smi driver version above against https://docs.nvidia.com/deploy/cuda-compatibility/'; print('torch', torch.__version__, '| CUDA OK:', torch.cuda.get_device_name(0))"

!pip install -q -r requirements.txt

!mkdir -p /content/drive/MyDrive/assembly_rl_runs
!MUJOCO_GL=egl python3 scripts/smoke_test.py   # ~3s end-to-end sanity check first

!MUJOCO_GL=egl python3 -m rl.train --algo sac --task peg_in_hole --difficulty loose \
    --timesteps 500000 --n_envs 2 --logdir /content/drive/MyDrive/assembly_rl_runs/sac_loose \
    --curriculum --device cuda
```

Notes:
- If the `torch.cuda.is_available()` assert fails: `nvidia-smi -L` still
  showing a T4 means the runtime attached fine, so the problem is the
  driver being older than CUDA 12.6 needs (<560). First try
  Runtime -> Disconnect and delete runtime, then reconnect -- this
  usually reprovisions a newer driver. If it's still too old, drop the
  torch install to `--index-url https://download.pytorch.org/whl/cu121`
  (needs only driver >=530), but that index doesn't publish a torch
  >=2.8, which `stable-baselines3>=2.2` requires -- so you'd also need to
  pin an older `stable-baselines3` compatible with a torch<2.8 (check
  https://pypi.org/project/stable-baselines3/#history for a version whose
  own `torch` requirement matches what cu121 offers before relying on
  this path).
- `--n_envs 2` matches Colab's usual 2-core CPU allocation for the free
  tier (env stepping is CPU-bound, per "Training budget" below) -- raise it
  if you're on a Colab Pro/Pro+ instance with more cores.
- `--device cuda` is pinned explicitly here (rather than left at the
  `auto` default) so a run fails fast if the GPU runtime didn't actually
  attach, instead of silently falling back to CPU.
- Checkpoints/tensorboard logs go straight to Drive (`--logdir`) so they
  survive a Colab disconnect; re-running the same cell after a disconnect
  does *not* resume from the last checkpoint -- `model.learn()` in
  `rl/train.py` always starts fresh, so load the latest `*.zip` from
  `--logdir` with `SAC.load(...)`/`PPO.load(...)` if you need to continue
  a run.

## Quick start

```bash
# 1. Verify the whole stack end-to-end (~3s): scene compiles, MuJoCo<->Pinocchio
#    frames agree, the classical baseline completes a loose-clearance insertion,
#    the Gym env resets/steps cleanly.
MUJOCO_GL=egl python3 scripts/smoke_test.py

# 2. Train (SAC and PPO share the same network -- rl/policies.py -- for a fair
#    comparison; see rl/train.py --help for all flags). A real run needs far
#    more than this to converge -- see "Training budget" below.
MUJOCO_GL=egl python3 -m rl.train --algo sac --task peg_in_hole --difficulty loose \
    --timesteps 500000 --n_envs 8 --logdir runs/sac_loose --curriculum

# 3. Benchmark the classical baseline against a trained policy across the
#    difficulty ladder and a pose-perturbation robustness sweep.
MUJOCO_GL=egl python3 -m benchmark.evaluate \
    --policy_path runs/sac_loose/sac_peg_in_hole_final.zip --algo sac \
    --difficulties loose tight --episodes 20 --robustness
```

## What's actually validated here

Everything up through a **working, physically-consistent simulation and
control stack** has been run and checked (`scripts/smoke_test.py`), not just
written:

- MuJoCo (UR5e + gripper + peg/hole fixture) and Pinocchio (UR5e URDF) agree
  on end-effector kinematics to <1mm once the MuJoCo model's baked-in 180°
  base rotation is accounted for (`control/pinocchio_model.py`'s `_BASE_FIX`).
- The classical hybrid impedance controller **completes** loose-clearance
  peg-in-hole insertion (APPROACH → SEARCH → INSERT → DONE, ~4s, XY alignment
  converges to <1mm) and **correctly fails to fully seat** the tight-clearance
  (0.3mm) tier -- this is the intended, useful benchmark behavior: it's what
  gives a learned policy room to demonstrate improvement, not a bug to fix.
- The Gymnasium env, SB3 SAC and PPO training loops, the curriculum
  callback, and the benchmark harness all run end-to-end (verified with
  short smoke runs, not full training -- see below).

**RL training was not run to convergence in this environment** -- contact-
rich insertion policies typically need 10^5-10^6+ environment steps, which
is a multi-hour-to-multi-day GPU job, not something to run inside a scaffold
session. What's here is the complete, tested infrastructure to do that
training; `benchmark/evaluate.py` works with any trained checkpoint you
point it at, including a partially-trained one (it'll just show a low
success rate, which is honest and useful for tracking progress).

## Design decisions worth knowing before you extend this

- **Rigid peg mount for peg-in-hole/connector-mating** (`SceneConfig.rigid_peg=True`
  in `envs/scene_builder.py`): the peg is fixed to the gripper rather than
  requiring a learned grasp, so the RL problem is centered on the proposal's
  actual target -- contact-rich *insertion* -- rather than also solving grasp
  stability. Kitting uses the real Robotiq 2F-85 with free parts, so grasping
  *is* part of that task. Set `rigid_peg=False` to require a real grasp once
  insertion is solved.
- **Controlled point is the peg tip, not the wrist flange**: `tool0` (the
  Pinocchio/URDF end-effector frame) sits ~196mm above the actual peg tip once
  the gripper and rigid peg mount are stacked on. `control/pinocchio_model.py`'s
  `forward_kinematics_offset`/`jacobian_offset` and the `tool_offset` on
  `CartesianImpedanceController` handle this. Getting this wrong was the
  single biggest bug during development (the controller would drive the wrist
  to a sensible-looking pose while the peg tip was actually rammed through
  the table) -- if you change the gripper or peg geometry, re-measure this
  offset (see `scripts/smoke_test.py`'s history / git log for the measurement
  method).
- **Rate-limited Cartesian reference, not raw setpoint tracking**: commanding
  a distant target directly saturates the UR5e's joint torque limits (150Nm
  shoulder/elbow, 28Nm wrist) and stalls short of the target instead of
  converging. `CartesianImpedanceController` moves an internal reference pose
  at a bounded Cartesian velocity toward whatever target it's given.
- **RL's low-level gains are softer than the classical baseline's**
  (`RL_GAINS` in `envs/assembly_env.py` vs. `ImpedanceGains()` defaults in
  `control/impedance_controller.py`): a random/early-training policy
  producing large action deltas against baseline-stiffness gains reproduces
  the same saturation/contact-spike problems noted above. The policy still
  sees true force/torque feedback in its observation and has to learn
  contact-aware behavior -- the softer gains just make random exploration
  safe rather than pre-solving compliance for it.
- **Pinocchio's UR5e URDF is UR5e-specific** (config-driven via
  `Universal_Robots_ROS2_Description`, not the generic UR5), matching the
  MuJoCo model's kinematics; joint names and order are identical between the
  two by construction, verified in the smoke test.
- **Fixture bore is a wedge-segment approximation of a cylinder** (MuJoCo has
  no native tapered-tube primitive) with a two-tier design: a wider chamfer
  over the top ~40% of the depth to make the opening findable, tight nominal
  clearance below. An earlier version's segments intruded ~2mm into the
  opening from a padding bug -- fixed and documented in `scene_builder.py`;
  worth a sanity render (see `scripts/scene_check_*` pattern in smoke_test.py)
  if you change segment count or clearance values.
- **Observations do not include ground-truth hole pose** -- only camera +
  proprioception + F/T, matching what a real system has. `ee_pose` in the
  observation is exact forward kinematics (free from joint encoders on real
  hardware too), not a vision estimate; localizing the hole is left to the
  learned policy via the wrist camera, which is the actual point of the
  "vision-guided" framing.

## Sim-to-real reuse (`ros2_ws/`)

`ros2_ws/src/assembly_rl_ros2/` is a real ament_python ROS2 package
(`perception_node` → `policy_node` → `impedance_controller_node`) that
imports `control/pinocchio_model.py` and `control/impedance_controller.py`
**directly** -- the same `CartesianImpedanceController` class used in
simulation runs the real robot, so there is no separate "real robot control
law" to keep in sync. This is deliberate: the whole point of validating the
controller in sim first is that it doesn't need to be reimplemented.

This was written but **not tested against ROS2** (not installed in this
environment) -- treat it as a structurally-correct, syntax-checked scaffold,
not a validated deployment. Before pointing it at real hardware:

1. Build a real ROS2 (Humble/Jazzy) workspace, put this package's parent
   directory on `PYTHONPATH` (or `pip install -e` the project root) so the
   `sys.path.insert` shim at the top of `policy_node.py`/
   `impedance_controller_node.py` resolves correctly -- see the NOTE in
   `policy_node.py`'s docstring.
2. UR5e's factory controller does not expose raw joint-torque control by
   default -- you need `ur_robot_driver`'s external control interface
   configured for torque/effort passthrough, with a `ros2_control`
   `JointGroupEffortController` publishing to `/forward_effort_controller/commands`.
   **Do not run the stiff (`use_stiff_gains:=true`) classical-baseline gains
   against real hardware without first validating on the loose-clearance
   tier at low speed** -- they were tuned in simulation only.
3. Wire your actual camera/F/T topic names into `config/topics.yaml` (see
   that file's header -- it's documentation of the wiring, not yet live
   config the nodes read).

## Training budget

Contact-rich insertion is sample-hungry. As a starting point:
- **loose clearance**: SAC typically needs low-10^5 steps to reach useful
  success rates on tasks like this; PPO usually needs several times more
  (it's on-policy).
- **tight clearance**: expect materially more, and strongly consider
  `--curriculum` (starts at `loose`, advances to `tight` once a rolling
  50-episode success rate clears 60% -- see `CurriculumCallback` in
  `rl/train.py`) rather than training tight-clearance from scratch.
- Scale `--n_envs` to your CPU count (physics-bound, not GPU-bound, for
  stepping; the CNN forward/backward pass is where a GPU helps).

## Repository layout

```
assets/                   MuJoCo models (UR5e, 2F-85), Pinocchio URDF + meshes
envs/scene_builder.py     Procedural scene composition (MjSpec), difficulty tiers
envs/assembly_env.py      Gymnasium Dict-obs environment
control/pinocchio_model.py    UR5e FK/Jacobians/dynamics wrapper
control/impedance_controller.py   Cartesian impedance + hybrid pos/force + FSM baseline
rl/policies.py             Hybrid CNN+MLP feature extractor (shared by SAC/PPO)
rl/train.py                 Training entrypoint + curriculum callback
benchmark/evaluate.py       Classical-vs-RL benchmark harness
scripts/smoke_test.py       End-to-end validation (run this first)
ros2_ws/src/assembly_rl_ros2/   Sim-to-real deployment package (see above)
```
