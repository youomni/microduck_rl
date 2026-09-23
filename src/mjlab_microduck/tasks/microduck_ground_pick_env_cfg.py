"""
microduck_jump_env_cfg.py — Jump with target rotation environment configuration for MicroDuck.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp import actions, observations
from mjlab.managers import (
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.rl_cfg import RlCfg
from mjlab.tasks.velocity.mdp import (
    base_angle_penalty,
    base_angular_velocity_penalty,
    joint_pos_limits_penalty,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
    CommandCfg,
    EventCfg,
    ObservationCfg,
    RewardCfg,
    SceneCfg,
    TerminationCfg,
    VelocityEnvCfg,
)
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_GROUNDCONTACT_ROBOT_CFG,
)
import mjlab_microduck.tasks.mdp as mdp


# -----------------------------------------------------------------------------
# MDP Terms Specific to the Rotational Jump Task
# -----------------------------------------------------------------------------

def yaw_rotation_progress(
    env: ManagerBasedRlEnv,
    target_yaw_rad: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Potential-based shaping for body yaw rotation toward target_yaw_rad."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Compute minimal angular distance wrapped to [-pi, pi]
    yaw_diff = torch.atan2(torch.sin(yaw - target_yaw_rad), torch.cos(yaw - target_yaw_rad))
    potential = -torch.abs(yaw_diff)

    if not hasattr(env, "_yaw_jump_potential_prev"):
        env._yaw_jump_potential_prev = potential.clone()

    fresh = env.episode_length_buf <= 1
    env._yaw_jump_potential_prev[fresh] = potential[fresh]

    delta = potential - env._yaw_jump_potential_prev
    env._yaw_jump_potential_prev = potential.clone()
    return delta


def jump_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward peak CoM height during mid-air leap phase."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    
    # Reward positive vertical speed when below peak, or closeness to height
    height_err = torch.clamp(target_height - z, min=0.0)
    return torch.exp(-torch.square(height_err) / 0.005) + 0.1 * torch.clamp(vz, min=0.0)


def soft_landing_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize impact force upon landing to protect actuators."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.net_forces_w  # (N, feet, 3)
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
        
    impact_mag = torch.norm(forces, dim=-1).sum(dim=-1)
    return torch.square(impact_mag / 100.0)


# -----------------------------------------------------------------------------
# Task Configurations
# -----------------------------------------------------------------------------

class MicroduckJumpObservationCfg(ObservationCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Fixed 61-dimension observation vector configuration:
        # - base_lin_vel (3)
        # - base_ang_vel (3)
        # - projected_gravity (3)
        # - joint_pos (14)
        # - joint_vel (14)
        # - actions / history (10 + 14 = 24)
        # Total active actor features = 61
        self.actor = ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=observations.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=observations.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=observations.projected_gravity),
                "joint_pos": ObservationTermCfg(func=observations.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=observations.joint_vel_rel),
                "last_action": ObservationTermCfg(func=observations.last_action),
            },
            concatenate_terms=True,
        )


class MicroduckJumpRewardCfg(RewardCfg):
    def __post_init__(self):
        super().__post_init__()

        # Jump phase trajectory & rotation shaping
        self.yaw_progress = RewardTermCfg(
            func=yaw_rotation_progress,
            weight=10.0,
            params={"target_yaw_rad": math.radians(0.0)},
        )
        self.jump_height = RewardTermCfg(
            func=jump_height_reward,
            weight=5.0,
            params={"target_height": 0.20},
        )
        self.upright_progress = RewardTermCfg(
            func=mdp.upright_progress,
            weight=3.0,
        )
        
        # Balance and Landing Stabilization
        self.upright_at_landing = RewardTermCfg(
            func=mdp.body_upright_linear,
            weight=2.0,
        )
        self.stand_composite = RewardTermCfg(
            func=mdp.standing_composite_score,
            weight=4.0,
            params={
                "target_height": 0.117,
                "height_std": 0.02,
                "upright_std": 0.1,
                "pose_std": 0.3,
                "joint_indices": list(range(14)),
            },
        )

        # Smoothness & Hardware Protection
        self.action_rate = RewardTermCfg(
            func=mdp.leg_action_rate_l2,
            weight=-0.01,
        )
        self.joint_torques = RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-0.0001,
        )
        self.torque_rate = RewardTermCfg(
            func=mdp.joint_torque_rate_l2,
            weight=-0.00001,
        )
        self.soft_landing = RewardTermCfg(
            func=soft_landing_penalty,
            weight=-0.05,
        )


class MicroduckJumpTerminationCfg(TerminationCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot_nan = TerminationTermCfg(func=mdp.robot_state_is_nan)
        self.fallen = TerminationTermCfg(
            func=mdp.fallen_too_long,
            params={
                "gate_z_below": 0.06,
                "gate_tilt_above_deg": 60.0,
                "max_duration_s": 1.5,
            },
        )


class MicroduckJumpEnvCfg(VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MICRODUCK_GROUNDCONTACT_ROBOT_CFG
        self.observations = MicroduckJumpObservationCfg()
        self.rewards = MicroduckJumpRewardCfg()
        self.terminations = MicroduckJumpTerminationCfg()
        self.episode_length_s = 3.0


class MicroduckJumpRlCfg(RlCfg):
    def __post_init__(self):
        super().__post_init__()
        self.runner.max_iterations = 3000
        self.runner.save_interval = 50
        self.runner.experiment_name = "microduck_jump"


def make_microduck_jump_env_cfg(play: bool = False, target_yaw_deg: float = 0.0) -> MicroduckJumpEnvCfg:
    cfg = MicroduckJumpEnvCfg()
    cfg.rewards.yaw_progress.params["target_yaw_rad"] = math.radians(target_yaw_deg)
    if play:
        cfg.scene.num_envs = 16
        cfg.episode_length_s = 10.0
    return cfg
