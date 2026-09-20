import gymnasium as gym
from mjlab_microduck.tasks.jump.jump_env_cfg import make_microduck_jump_env_cfg, MicroduckJumpRlCfg

gym.register(
    id="Mjlab-Jump-Flat-MicroDuck",
    entry_point="mjlab.envs:ManagerBasedRlEnvCfg",
    kwargs={
        "env_cfg_entry_point": make_microduck_jump_env_cfg,
        "rsl_rl_cfg_entry_point": MicroduckJumpRlCfg,
    },
)
