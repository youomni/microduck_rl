"""Jump environment configuration for MicroDuck."""

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import BuiltinMotorCfg, PDMotorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg as EventTerm
from mjlab.managers import ObservationGroupCfg as ObsGroup
from mjlab.managers import ObservationTermCfg as ObsTerm
from mjlab.managers import RewardTermCfg as RewardTerm
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTerm
from mjlab.scene import InteractiveSceneCfg
from mjlab.sensors import ContactSensorCfg
from mjlab import mdp
from rsl_rl.runners import OnPolicyRunnerCfg as RslRlOnPolicyRunnerCfg

from mjlab_microduck import mdp as microduck_mdp
from mjlab_microduck.actuator import BamActuatorCfg
from mjlab_microduck.robots.microduck import (
    MICRODUCK_FEET_GEOMS,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks.roulade.roulade_env_cfg import make_roulade_env_cfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

JUMP_HEADING_RANGE = (0.0, math.radians(90.0))


def _jump_foot_clearance_reward(
    env: "ManagerBasedRlEnv",
    target_height: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    left_foot_pos = robot.data.site_xpos[:, 0, 2]
    right_foot_pos = robot.data.site_xpos[:, 1, 2]
    left_err = torch.clamp(target_height - left_foot_pos, min=0.0)
    right_err = torch.clamp(target_height - right_foot_pos, min=0.0)
    return torch.exp(-left_err / 0.02) + torch.exp(-right_err / 0.02)


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_roulade_env_cfg(play=play)

    # 1. Commands: Focus on yaw spin during jump
    cfg.commands.twist.ranges.heading = JUMP_HEADING_RANGE
    cfg.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.twist.ranges.ang_vel_z = (0.0, math.radians(180.0))

    # 2. Rewards: Add jump launch force and foot clearance
    cfg.rewards.jump_launch = RewardTerm(
        func=microduck_mdp.com_upward_velocity,
        weight=2.0,
    )
    cfg.rewards.foot_clearance = RewardTerm(
        func=_jump_foot_clearance_reward,
        weight=1.5,
        params={"target_height": 0.08},
    )
    cfg.rewards.track_angular_velocity = RewardTerm(
        func=microduck_mdp.track_yaw_velocity,
        weight=2.0,
    )
    cfg.rewards.gentle_landing = RewardTerm(
        func=microduck_mdp.gentle_landing_penalty,
        weight=0.002,
    )

    return cfg


class MicroduckJumpRlCfg(RslRlOnPolicyRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPO"
        self.num_steps_per_env = 24
        self.max_iterations = 1500
        self.save_interval = 100
        self.experiment_name = "microduck_jump"
        self.run_name = ""
        self.logger = "wandb"
        self.wandb_project = "microduck_rl"
