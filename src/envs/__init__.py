from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env, StarCraft2MTEnv
from .particle import Particle
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2mt"] = partial(env_fn, env=StarCraft2MTEnv)
REGISTRY["particle"] = partial(env_fn, env=Particle)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "/nfs4-p1/lsy/download/StarCraftII")

# sc2
# state:
#   n_agents:
#     health (1), energy/cooldown (1), rel_x (1), rel_y (1)
#     shield (1): abandon when there is no Protoss
#     unit_type (n_units): abandon when there is one type
#   n_enemies:
#     health (1)
#     placeholder (1): abandon when state_align is False
#     rel_x (1), rel_y (1)
#     shield (1): abandon when there is no Protoss
#     unit_type (n_units): abandon when there is one type
#   n_agents:
#     last_action (n_actions)

# obs:
#   avail_move (4)
#   n_enemies:
#     avail_attack (1), distance (1), rel_x (1), rel_y (1), health (1)
#     shield (1): abandon when there is no Protoss
#     unit_type (n_units): abandon when there is one type
#   n_agents - 1:
#     visible (1), distance (1), rel_x (1), rel_y (1), health (1)
#     shield (1): abandon when there is no Protoss
#     unit_type (n_units): abandon when there is one type
#   health (1)
#   shield (1): abandon when there is no Protoss
#   unit_type (n_units): abandon when there is one type
#   grid_min_health (grid_size): abandon when action_grid_attack is False
#   grid_avail_attack (grid_size): abandon when action_grid_attack is False
#   last_action (n_actions): add in basic_controller
#   agent_id (n_agents): add in basic_controller

# 智能体的可视范围是9，可攻击范围是6，图中网格大小为 1.5*1.5

# action:
#   no-op (1): valid only when dead
#   stop (1)
#   move[direction] (4): north, south, east, or west
#   attack[enemy_id] (n_enemies) / attack[grid_id] (grid_size): action_grid_attack is False / True
#
# reward:
#   reward  = reward / (max_reward / reward_scale_rate)
#     max_reward = n_enemies * reward_death_value + reward_win
#     reward_death_value=10
#     reward_win=200
#     reward_scale_rate=20


# Name            Agents  Enemies Limit
# 3m              3       3       60
# 8m              8       8       120
# 25m             25      25      150
# 5m_vs_6m        5       6       70
# 8m_vs_9m        8       9       120
# 10m_vs_11m      10      11      150
# 27m_vs_30m      27      30      180
# MMM             10      10      150
# MMM2            10      12      180
# 2s3z            5       5       120
# 3s5z            8       8       150
# 3s5z_vs_3s6z    8       9       170
# 3s_vs_3z        3       3       150
# 3s_vs_4z        3       4       200
# 3s_vs_5z        3       5       250
# 1c3s5z          9       9       180
# 2m_vs_1z        2       1       150
# corridor        6       24      400
# 6h_vs_8z        6       8       150
# 2s_vs_1sc       2       1       300
# so_many_baneling 7       32      100
# bane_vs_bane    24      24      200
# 2c_vs_64zg      2       64      400

# https://github.com/oxwhirl/smac/blob/master/docs/smac.md
