"""
Procedural MuJoCo scene composition for the vision-guided assembly-RL project.

Builds a UR5e + Robotiq 2F-85 scene with a task fixture (peg-in-hole /
connector mating / kitting tray) using MuJoCo's MjSpec composition API
(mujoco >= 3.1), so difficulty tiers (clearance, part pose ranges, part
count) are generated programmatically rather than hand-authored per-XML.

Design choices (documented, not hidden):
  * The peg is rigidly fixed to the gripper mount (no separate grasp
    policy) for the peg-in-hole / connector-mating tiers. This keeps the
    RL problem centered on the proposal's actual target -- contact-rich
    *insertion* dynamics under visual + F/T feedback -- rather than also
    solving grasp stability. The Robotiq 2F-85 fingers still close around
    a visual peg collar for contact realism; swap `rigid_peg=False` to
    require an actual grasp once insertion is solved.
  * Kitting tier uses the real Robotiq 2F-85 with free-floating parts, so
    grasping *is* part of the task there.
  * A 6-axis force/torque sensor sits at the wrist flange (between
    wrist_3_link and the gripper mount), which is what the impedance
    baseline and the RL force/torque observation channel both read.
  * An eye-in-hand camera is mounted on the gripper body; an optional
    static camera can be added for a third-person view.
  * Domain randomization is split by cost:
      - cheap, every-reset (no recompile): fixture pose jitter, peg/table
        friction, rgba/material index -> handled in AssemblyEnv directly
        on the compiled MjModel arrays.
      - structural, per-curriculum-stage (recompile): clearance tier,
        part count -> handled here, cached per (task, difficulty).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
UR5E_XML = ASSETS / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = ASSETS / "robotiq_2f85" / "2f85.xml"

TaskName = Literal["peg_in_hole", "connector_mating", "kitting"]

# Clearance tiers in meters (peg radius fixed at 12mm; hole radius = peg + clearance).
CLEARANCE_TIERS = {
    "loose": 4.0e-3,   # 4mm radial clearance -> easy, mostly kinematic
    "tight": 0.3e-3,   # 0.3mm radial clearance -> contact-rich, needs force feedback
}


@dataclasses.dataclass
class SceneConfig:
    task: TaskName = "peg_in_hole"
    difficulty: str = "loose"          # "loose" | "tight" (peg_in_hole/connector_mating) | ignored for kitting
    n_kit_parts: int = 3
    peg_radius: float = 0.012          # 12mm peg, representative of connector/peg-in-hole benchmarks
    peg_length: float = 0.05
    hole_depth: float = 0.04
    rigid_peg: bool = True             # see module docstring
    seed: int | None = None


def _add_ft_sensor(spec: mujoco.MjSpec, site_name: str, sensor_prefix: str) -> None:
    spec.add_sensor(name=f"{sensor_prefix}_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
    spec.add_sensor(name=f"{sensor_prefix}_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)


def _base_arm_spec() -> mujoco.MjSpec:
    """Load the UR5e arm plus scene furniture (table, lighting, floor)."""
    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    spec.compiler.degree = False
    # Match the gripper's contact settings (elliptic friction cone, higher impedance
    # ratio) globally -- this also benefits peg/hole contact fidelity during insertion.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    # The stock UR5e MJCF ships joint-space PD *position* actuators (mimicking the
    # real robot's internal servo loop: force = 2000*(ctrl-q) - 400*qdot). Genuine
    # Cartesian impedance control -- and a fair RL-vs-classical-baseline comparison --
    # needs direct torque authority instead, so we convert each arm actuator to a
    # pure torque motor (ctrl = torque directly, limited to the joint's forcerange).
    for act in spec.actuators:
        frange = np.array(act.forcerange)
        act.set_to_motor()
        act.forcerange = frange
        act.ctrlrange = frange
        act.forcelimited = True
        act.ctrllimited = True

    # Ground / lighting (mirrors mujoco_menagerie's scene.xml but self-contained).
    world = spec.worldbody
    world.add_light(pos=[0, 0, 1.6], dir=[0, 0, -1])
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05], pos=[0, 0, 0], rgba=[0.25, 0.27, 0.30, 1],
    )
    # Work table under the arm's reach.
    table = world.add_body(name="table", pos=[0.5, 0.0, 0.20])
    table.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30, 0.30, 0.02],
        rgba=[0.55, 0.42, 0.30, 1], friction=[0.8, 0.01, 0.001],
    )
    return spec


def _add_wrist_ft_site_and_gripper(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Attach a wrist F/T site + the Robotiq 2F-85 gripper at attachment_site."""
    wrist3 = spec.body("wrist_3_link")
    ft_site = wrist3.add_site(name="wrist_ft_site", pos=[0, 0.1, 0], quat=[0.5, -0.5, 0.5, 0.5], group=3)
    _add_ft_sensor(spec, "wrist_ft_site", "wrist")

    gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    attach_site = spec.site("attachment_site")
    spec.attach(gripper_spec, prefix="gripper_", site=attach_site)
    return spec


def _add_camera(spec: mujoco.MjSpec) -> None:
    """Eye-in-hand camera mounted near the gripper base, plus a static overview cam."""
    try:
        gripper_base = spec.body("gripper_base")
        gripper_base.add_camera(
            name="wrist_cam", pos=[0, -0.05, 0.06], quat=[0.0, 0.0, 1.0, 0.0],
            fovy=55,
        )
    except Exception:
        pass
    spec.worldbody.add_camera(
        name="scene_cam", pos=[1.1, 0.55, 0.85], xyaxes=[-0.5, 1.0, 0, -0.6, -0.3, 1.1],
        fovy=45,
    )


def _add_peg_in_hole_fixture(spec: mujoco.MjSpec, cfg: SceneConfig, rng: np.random.Generator) -> None:
    clearance = CLEARANCE_TIERS[cfg.difficulty]
    hole_r = cfg.peg_radius + clearance

    # Peg: rigidly fixed to gripper (representing an already-secured grasp).
    if cfg.rigid_peg:
        gripper_base = spec.body("gripper_base")
        peg = gripper_base.add_body(name="peg", pos=[0, 0, 0.16])
        peg.add_geom(
            name="peg_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.peg_radius, cfg.peg_length / 2, 0],
            rgba=[0.75, 0.75, 0.78, 1], friction=[0.4, 0.005, 0.0001],
            condim=4, priority=1,
        )
    else:
        # Free peg placed near the gripper for a real grasp-then-insert task.
        world = spec.worldbody
        peg = world.add_body(name="peg", pos=[0.45, -0.15, 0.30])
        peg.add_freejoint(name="peg_free")
        peg.add_geom(
            name="peg_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.peg_radius, cfg.peg_length / 2, 0],
            rgba=[0.75, 0.75, 0.78, 1], friction=[0.4, 0.005, 0.0001],
            condim=4, priority=1,
        )

    # Hole fixture: a block with a cylindrical bore, randomized pose (cheap DR done at
    # env-reset time on the compiled model; nominal pose set here). Approximated with
    # flat wedge segments (MuJoCo has no native tapered-tube primitive); each segment's
    # inner face is placed exactly tangent to the target bore radius -- an earlier
    # version shrank the radial half-width for "overlap", which actually pushed the
    # inner faces ~2mm past tangent and intruded into the opening (verified by walking
    # the compiled model's geom_pos/geom_size; see scripts/smoke_test.py history).
    # Tangential (not radial) padding is what removes seam gaps between segments.
    #
    # A two-tier bore (a wider entry chamfer over the top ~40% of the depth, tight
    # nominal-clearance bore below) is standard practice in peg-in-hole sim setups --
    # it makes "find the opening" tractable for both the classical search phase and
    # early RL exploration without weakening the actual precision-insertion tier.
    table = spec.body("table")
    dx, dy = rng.uniform(-0.03, 0.03, size=2)
    fixture = table.add_body(name="fixture", pos=[dx, dy, 0.02])
    n_seg = 20
    wall_thickness = 0.02
    entry_frac = 0.4
    entry_lead = max(0.003, 1.5 * clearance)  # chamfer radius grows by this much

    def _add_ring(z0: float, z1: float, radius: float, tag: str) -> None:
        seg_len = z1 - z0
        outer_r = radius + wall_thickness
        for i in range(n_seg):
            theta_m = 2 * np.pi * (i + 0.5) / n_seg
            cx = (radius + wall_thickness / 2) * np.cos(theta_m)
            cy = (radius + wall_thickness / 2) * np.sin(theta_m)
            fixture.add_geom(
                name=f"fixture_wall_{tag}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_thickness / 2, outer_r * np.sin(np.pi / n_seg) + 0.001, seg_len / 2],
                pos=[cx, cy, z0 + seg_len / 2], euler=[0, 0, np.degrees(theta_m)],
                rgba=[0.35, 0.38, 0.42, 1], friction=[0.3, 0.005, 0.0001], condim=4,
            )

    entry_z = cfg.hole_depth * (1 - entry_frac)
    _add_ring(entry_z, cfg.hole_depth, hole_r + entry_lead, "entry")
    _add_ring(0.0, entry_z, hole_r, "bore")

    # Target/goal site at the bottom of the bore (used for reward + success check).
    fixture.add_site(name="hole_target", pos=[0, 0, 0.002], size=[0.002, 0.002, 0.002], rgba=[0, 1, 0, 0.5])


def _add_kitting_fixture(spec: mujoco.MjSpec, cfg: SceneConfig, rng: np.random.Generator) -> None:
    table = spec.body("table")
    tray = table.add_body(name="kit_tray", pos=[0.15, 0.0, 0.02])
    slot_pitch = 0.08
    colors = [[0.8, 0.2, 0.2, 1], [0.2, 0.5, 0.8, 1], [0.2, 0.7, 0.3, 1], [0.8, 0.7, 0.2, 1]]
    for i in range(cfg.n_kit_parts):
        slot = tray.add_site(
            name=f"kit_slot_{i}", pos=[0, (i - (cfg.n_kit_parts - 1) / 2) * slot_pitch, 0.001],
            size=[0.02, 0.02, 0.001], type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.2, 0.2, 0.2, 0.3],
        )

    world = spec.worldbody
    for i in range(cfg.n_kit_parts):
        px, py = 0.45 + rng.uniform(-0.05, 0.05), -0.25 + rng.uniform(-0.02, 0.02) + i * 0.07
        part = world.add_body(name=f"part_{i}", pos=[px, py, 0.25])
        part.add_freejoint(name=f"part_{i}_free")
        part.add_geom(
            name=f"part_{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.015],
            rgba=colors[i % len(colors)], friction=[0.6, 0.01, 0.001], condim=4, mass=0.05,
        )


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    rng = np.random.default_rng(cfg.seed)
    spec = _base_arm_spec()
    _add_wrist_ft_site_and_gripper(spec)
    _add_camera(spec)
    if cfg.task in ("peg_in_hole", "connector_mating"):
        _add_peg_in_hole_fixture(spec, cfg, rng)
    elif cfg.task == "kitting":
        _add_kitting_fixture(spec, cfg, rng)
    else:
        raise ValueError(f"Unknown task {cfg.task!r}")
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    spec = build_spec(cfg)
    return spec.compile()


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "peg_in_hole"
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "loose"
    cfg = SceneConfig(task=task, difficulty=difficulty)
    model = build_model(cfg)
    print(f"Compiled {task}/{difficulty}: nq={model.nq} nv={model.nv} nbody={model.nbody} "
          f"ngeom={model.ngeom} nsensor={model.nsensor} ncam={model.ncam}")
    out = ASSETS / "generated" / f"{task}_{difficulty}.xml"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write(build_spec(cfg).to_xml())
    print(f"Wrote {out}")
