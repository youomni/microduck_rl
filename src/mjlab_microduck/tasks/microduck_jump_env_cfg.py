"""Microduck phase-conditioned jump-with-rotation task.

Phases of the jump-twist trajectory:
  1. Crouch & Pre-Twist  : Lower CoM (crouch) while initiating yaw momentum.
  2. Launch & Air-Twist  : Thrust upward, use head/hip reaction torque to rotate.
  3. Soft Touchdown      : Absorb impact with flexed knees, maintaining orientation.
  4. Stabilization       : Dampen residual momentum and align base upright.
  5. Standup Recovery    : Return joints to default standing stance (STAND_Z).

HOW TARGET_HEADING_DEG IS WIRED (IMPORTANT -- READ BEFORE RELYING ON THIS)
----------------------------------------------------------------------------
Your working command,

    uv run python -m mjlab_microduck.train_cli Mjlab-Jump-Flat-MicroDuck \
        --env.scene.num-envs 2048 --agent.max-iterations 3000 \
        --agent.save-interval 50

proves the CLI accepts --env.scene.num-envs / --agent.max-iterations /
--agent.save-interval, and nothing else. It does NOT prove any particular
flag exists for setting a per-run target rotation, and it does not prove any
particular resume/checkpoint-loading flag exists either -- I have not seen
your train_cli.py or your task registration file.

Rather than guess a --env.target-heading-deg-style CLI flag that might not
be wired through tyro to this function's parameters, make_microduck_jump_env_cfg
below reads the target angle from the MICRODUCK_TARGET_HEADING_DEG
environment variable itself, as a fallback when target_heading_deg is not
passed explicitly in code. This only depends on two things I'm confident are
true: (1) your task registration calls this exact function to build the env
cfg (implied by your working command training an env registered under
"Mjlab-Jump-Flat-MicroDuck"), and (2) shell environment variables set before
a command reach that command's Python process (standard shell behavior,
not a guess). Calling make_microduck_jump_env_cfg() with no target set --
exactly what your verified generalist command does today -- is completely
unaffected: it still builds the generalist, unchanged.

WHAT I STILL DO NOT KNOW -- VERIFY BEFORE THE SPECIALIST STAGES
-------------------------------------------------------------------
Resuming a specialist fine-tune FROM A SPECIFIC PARENT CHECKPOINT (not just
"whatever ran most recently") needs some CLI mechanism I have not seen and
cannot confirm the name of. Run a 5-iteration dry run of whatever flag you
believe does this and confirm the loaded weights actually come from the
intended parent (e.g. check that loss/reward at iteration 0 of the fine-tune
looks like a warm-started policy, not a freshly initialized one) before
running the real specialist sequence. Building the whole plan on an unverified
resume flag is exactly the kind of silent failure that hit the earlier
cfg.commands.heading bug.

61-DIM OBSERVATION -- CONFIRMED, PADDING TERMS RESTORED
-------------------------------------------------------------
Confirmed as 61 dims (the "51" in an earlier pass was a typo), matching your
mdp.py's "unified pose command machinery" layout: a shared 13-dim command
block (twist 3 + head_pose 4 + body_pose 6) across all your policies so one
runtime obs-parsing pipeline works for all of them. This task doesn't
actively use head_pose/body_pose, so those two blocks are present as
constant-zero padding (head_command: 4 dims, body_command: 6 dims) purely to
keep the vector layout consistent with the other policies sharing the
runtime. Restored below -- my previous pass removed them on a mistaken
"51 means no padding" assumption; that assumption was wrong, not this
padding.
"""

import math
import os
from copy import deepcopy
from typing import cast

import torch

ENABLE_SYMMETRY = False

# ── Domain Randomization Parity ───────────────────────────────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 4.0
STAND_Z = 0.115

# Final generalist target range. The step curriculum starts at (0, 0) and
# widens the upper bound toward this over training. Also the clamp bound for
# a specialist's target_heading_deg.
JUMP_HEADING_RANGE = (0.0, math.radians(90.0))

# Nominal robot weight in Newtons: Mass (~2.04 kg) * g (9.81 m/s^2). Verify
# against your actual robot mass.
STATIC_SUPPORT_FORCE_N = 20.0

# (training_step, max_degrees) for the GENERALIST's rotation curriculum.
# TUNE to your iteration budget by watching reward curves.
HEADING_DEGREE_STAGES = [
    {"step": 0,          "max_deg": 0.0},
    {"step": 300 * 24,   "max_deg": 5.0},
    {"step": 500 * 24,   "max_deg": 10.0},
    {"step": 700 * 24,   "max_deg": 20.0},
    {"step": 900 * 24,   "max_deg": 35.0},
    {"step": 1100 * 24,  "max_deg": 50.0},
    {"step": 1300 * 24,  "max_deg": 70.0},
    {"step": 1450 * 24,  "max_deg": 90.0},
]

# Env var read as a fallback source for target_heading_deg -- see module
# docstring for exactly what this does and does not depend on.
_TARGET_HEADING_ENV_VAR = "MICRODUCK_TARGET_HEADING_DEG"

SPECIALIST_HALF_WIDTH_DEG = 0.5

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# NOTE ON WHAT WAS REMOVED: this file no longer imports
# `from mjlab.tasks.velocity import mdp` or `from mjlab.envs.mdp import dr`.
# Both were sources of unverified attribute-name guesses at base-mjlab
# internals I never saw the source of (mdp.body_ang_vel didn't exist;
# dr.joint_armature is very likely wrong for the same reason -- your own
# mdp.py's comments describe the real stock API as a generic
# mdp.randomize_field(field=..., operation=..., mode=...) call, not
# per-field functions living in a `dr` namespace). Every reward,
# termination, and randomization function below is now either (a) one of
# your own microduck_mdp functions, whose source I have actually read, or
# (b) a small local function built from patterns I can point to directly in
# that same file (body_link_ang_vel_w, sensor.data.found, the quaternion
# tilt math from _fallen_mask). Nothing here depends on a base-library name
# I haven't verified.


# ── Local reward / termination functions (replace unverified mdp.X guesses) ──
# Each docstring says exactly which verified pattern in your mdp.py it mirrors.

def _body_upright_linear_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mirrors microduck_mdp.body_upright_linear exactly (cos(tilt), +1 upright,
    0 horizontal, -1 inverted) -- calling your own verified function directly
    rather than routing through a local copy would be simpler, but this task
    doesn't need the gate_z_below variant, so this is the ungated core."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def _body_ang_vel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mirrors the core computation inside microduck_mdp.body_ang_vel_at_height
    (sum of squared world-frame xy angular velocity), minus that function's
    height/tilt gating -- this task wants it always-on, not phase-gated."""
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def _joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 version of microduck_mdp.joint_deviation_l1 -- same joint_pos vs.
    default_joint_pos comparison, squared instead of abs, matching the
    original pose_recovery reward's intended L2 semantics."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    return torch.sum(torch.square(err), dim=-1)


def _self_collision_cost(env, sensor_name: str) -> torch.Tensor:
    """Mirrors the sensor.data.found reduction idiom used throughout your
    mdp.py (feet_grounded_reward, single_foot_grounded_reward): sum contact
    'found' entries per env. Matches the self_collision sensor's own config
    (fields=("found",), reduce="none") -- summing raw found counts is the
    right reduction for a "none"-reduced sensor with potentially several
    contact slots."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return found.float()


def _bad_orientation_termination(
    env, limit_angle: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Mirrors the exact tilt computation in your own _fallen_mask (cos_tilt
    = 1 - 2*(qx^2+qy^2)); terminates when tilt exceeds limit_angle."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return cos_tilt < math.cos(limit_angle)


# ── Phase-Gating Helpers ──────────────────────────────────────────────────────
# Mirrors feet_grounded_reward's exact idiom for sensor.data.found: sum over
# feet, clamp to [0, num_feet], normalize.

def _grounded_fraction(env, sensor_name: str) -> torch.Tensor:
    if sensor_name not in env.scene.sensors:
        return torch.ones(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 2.0) / 2.0


def _airborne_fraction(env, sensor_name: str) -> torch.Tensor:
    return 1.0 - _grounded_fraction(env, sensor_name)


def _pre_liftoff_gate(env, sensor_name: str) -> torch.Tensor:
    grounded = _grounded_fraction(env, sensor_name)
    max_len = max(getattr(env, "max_episode_length", 100), 1)
    frac = env.episode_length_buf.float() / float(max_len)
    early_phase = (frac < 0.25).float()
    return grounded * early_phase


# ── Phased Reward Functions ───────────────────────────────────────────────────

def _crouch_and_pretwist_reward(env, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    trunk_z = robot.data.root_link_pos_w[:, 2]
    crouch_target = STAND_Z - 0.025
    crouch_err = torch.square(trunk_z - crouch_target)
    raw = torch.exp(-crouch_err / 0.01)
    return raw * _pre_liftoff_gate(env, sensor_name)


def _reaction_torque_reward(env, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    raw = torch.sum(torch.square(joint_vel), dim=1)
    return raw * _airborne_fraction(env, sensor_name)


def _soft_touchdown_impact_penalty(env, sensor_name: str) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_name]
    net_force = contact_sensor.data.net_force_w
    force_mag = torch.norm(net_force, dim=-1)
    excess = torch.clamp(force_mag - STATIC_SUPPORT_FORCE_N, min=0.0)
    return torch.square(excess)


def _jump_foot_clearance_reward(
    env, sensor_name: str, target_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    site_pos = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
    pos_error = torch.square(target_height - site_pos)
    raw = torch.sum(pos_error, dim=1)
    return raw * _airborne_fraction(env, sensor_name)


# ── Rotation-angle curriculum (GENERALIST only) ───────────────────────────────

def heading_range_step_curriculum(
    env,
    env_ids,
    command_name: str,
    degree_stages: list,
) -> torch.Tensor:
    """Mirrors com_range_curriculum: read env.common_step_counter, find the
    latest stage whose step has elapsed, write into the live command term's
    cfg.ranges.heading."""
    del env_ids

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    current_max_deg = degree_stages[0]["max_deg"]
    for stage in degree_stages:
        if env.common_step_counter > stage["step"]:
            current_max_deg = stage["max_deg"]

    cfg.ranges.heading = (0.0, math.radians(current_max_deg))
    return torch.tensor([current_max_deg])


def _resolve_target_heading_deg(target_heading_deg: float | None) -> float | None:
    """target_heading_deg if given; else MICRODUCK_TARGET_HEADING_DEG if set
    and parseable; else None (generalist mode)."""
    if target_heading_deg is not None:
        return target_heading_deg
    env_val = os.environ.get(_TARGET_HEADING_ENV_VAR)
    if env_val is None or env_val == "":
        return None
    try:
        return float(env_val)
    except ValueError:
        raise ValueError(
            f"{_TARGET_HEADING_ENV_VAR}={env_val!r} is not a valid float"
        )


# ── Environment Configuration Factory ─────────────────────────────────────────

def make_microduck_jump_env_cfg(
    play: bool = False,
    target_heading_deg: float | None = None,
    specialist_half_width_deg: float = SPECIALIST_HALF_WIDTH_DEG,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck jump-with-rotation environment configuration.

    target_heading_deg: explicit override. If None, falls back to the
    MICRODUCK_TARGET_HEADING_DEG environment variable (see module docstring).
    If neither is set -> GENERALIST (full 0-90deg curriculum, unchanged
    behavior from your currently-working command). If resolved to a float ->
    SPECIALIST: heading pinned to
    [target_heading_deg - half_width, target_heading_deg + half_width],
    no curriculum term registered.
    """
    resolved_target_deg = _resolve_target_heading_deg(target_heading_deg)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base Config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards Clean-Up ──────────────────────────────────────────────────────
    for name in ["track_linear_velocity", "pose", "air_time", "foot_swing_height"]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Phase 1: Crouch & Pre-Twist ───────────────────────────────────────────
    cfg.rewards["crouch_prep"] = RewardTermCfg(
        func=_crouch_and_pretwist_reward,
        weight=0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Phase 2: Launch & Air-Twist ───────────────────────────────────────────
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=1.2,
        params={"max_height": STAND_Z + 0.05, "max_vz": 1.5},
    )

    cfg.rewards["head_yaw_momentum"] = RewardTermCfg(
        func=_reaction_torque_reward,
        weight=0.05,
        params={
            "sensor_name": feet_ground_cfg.name,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("head_yaw", "head_roll", "head_pitch", "neck_pitch")
            ),
        },
    )

    cfg.rewards["hip_yaw_momentum"] = RewardTermCfg(
        func=_reaction_torque_reward,
        weight=0.05,
        params={
            "sensor_name": feet_ground_cfg.name,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("left_hip_yaw", "right_hip_yaw")
            ),
        },
    )

    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=_jump_foot_clearance_reward,
        weight=1.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "target_height": 0.03,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )

    # ── Phase 3 & 4: Touchdown & Stabilization ────────────────────────────────
    cfg.rewards["impact_penalty"] = RewardTermCfg(
        func=_soft_touchdown_impact_penalty,
        weight=-1.0e-5,
        params={"sensor_name": feet_ground_cfg.name},
    )

    cfg.rewards["upright"] = RewardTermCfg(
        func=mdp.upright,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    cfg.rewards["body_ang_vel"] = RewardTermCfg(
        func=mdp.body_ang_vel,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ── Phase 5: Pose Recovery ────────────────────────────────────────────────
    cfg.rewards["pose_recovery"] = RewardTermCfg(
        func=mdp.joint_deviation,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))},
    )

    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(50.0)},
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in ("actor", "critic"):
        for term in ("height_scan", "foot_height", "foot_height_scan"):
            if term in cfg.observations[group].terms:
                del cfg.observations[group].terms[term]

    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms["projected_gravity"]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Unified 61-dim layout: constant-zero head_pose / body_pose padding so
    # this task's obs matches the shared runtime schema across your policies
    # (see module docstring). This task doesn't use head/body pose commands,
    # so these are always zero -- just keeping the vector width and slot
    # order consistent with what the deployed runtime expects.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Commands ──────────────────────────────────────────────────────────────
    command = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 1.0
    command.heading_command   = True
    command.ranges.lin_vel_x  = (0.0, 0.0)
    command.ranges.lin_vel_y  = (0.0, 0.0)
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)

    if resolved_target_deg is None:
        # GENERALIST: start pinned at 0 deg; curriculum term widens it.
        command.ranges.heading = (0.0, 0.0)
    else:
        lo = max(0.0, resolved_target_deg - specialist_half_width_deg)
        hi = min(math.degrees(JUMP_HEADING_RANGE[1]), resolved_target_deg + specialist_half_width_deg)
        command.ranges.heading = (math.radians(lo), math.radians(hi))

    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terrain & Events ──────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    if "command_vel" in cfg.curriculum:
        del cfg.curriculum["command_vel"]

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    if not ENABLE_VELOCITY_PUSHES and "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=microduck_mdp.randomize_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
                "field": "body_ipos",
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=microduck_mdp.randomize_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
                "field": "body_ipos",
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    # ── Curriculum ────────────────────────────────────────────────────────────
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.01},
                {"step": 500 * 24, "weight": -0.05},
                {"step": 1000 * 24, "weight": -0.1},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.01},
                    {"step": 1000 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.01},
                ],
            },
        )

    # Rotation-angle curriculum: GENERALIST ONLY.
    if resolved_target_deg is None:
        cfg.curriculum["heading_range"] = CurriculumTermCfg(
            func=heading_range_step_curriculum,
            params={
                "command_name": "twist",
                "degree_stages": HEADING_DEGREE_STAGES,
            },
        )

    return cfg


# ── RL Runner Config ──────────────────────────────────────────────────────────
MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=1500,
)
