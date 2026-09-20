# Import jump env cfg
from .microduck_jump_env_cfg import (
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
)

# Register Jump task
register_mjlab_task(
    task_id="Mjlab-Jump-Flat-MicroDuck",
    env_cfg=make_microduck_jump_env_cfg(),
    play_env_cfg=make_microduck_jump_env_cfg(play=True),
    rl_cfg=MicroduckJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Add Jump backlash task variant
_BACKLASH_TASKS = (
    # ... existing backlash tasks ...
    ("Mjlab-Jump-Flat-Backlash-MicroDuck", make_microduck_jump_env_cfg, {}, MicroduckJumpRlCfg, _BL_GROUNDCONTACT),
)
