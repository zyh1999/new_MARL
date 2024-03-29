from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..multiagentenv import MultiAgentEnv
from .maps import get_mt_scenario_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
from numpy.random import RandomState
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib.units import Protoss, Terran, Zerg

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

type_name2health = {'Baneling': 30.0, 'Colossus': 200.0, 'Hydralisk': 80.0,
                    'Marauder': 125.0, 'Marine': 45.0, 'Medivac': 150.0,
                    'Stalker': 80.0, 'Zealot': 100.0, 'Zergling': 35.0}


type_name2shield = {'Baneling': 0.0, 'Colossus': 150.0, 'Hydralisk': 0.0,
                    'Marauder': 0.0, 'Marine': 0.0, 'Medivac': 0.0,
                    'Stalker': 80.0, 'Zealot': 50.0, 'Zergling': 0.0}

# Linux
# Version: B75689 (SC2.4.10)
type_id2name = {1970: 'Baneling_RL', 1971: 'Colossus_RL', 1972: 'Hydralisk_RL',
                1973: 'Marauder_RL', 1974: 'Marine_RL', 1975: 'Medivac_RL',
                1976: 'Stalker_RL', 1977: 'Zealot_RL', 1978: 'Zergling_RL',
                9: 'Baneling', 4: 'Colossus', 107: 'Hydralisk',
                51: 'Marauder', 48: 'Marine', 54: 'Medivac',
                74: 'Stalker', 73: 'Zealot', 105: 'Zergling'}

custom_type_name2id = {'Baneling': 1970, 'Colossus': 1971, 'Hydralisk': 1972,
                       'Marauder': 1973, 'Marine': 1974, 'Medivac': 1975,
                       'Stalker': 1976, 'Zealot': 1977, 'Zergling': 1978}

# Mac
# Version: B84643 (SC2.5.0.7)
# type_id2name = {2005: 'Baneling_RL', 2006: 'Colossus_RL', 2007: 'Hydralisk_RL',
#                 2008: 'Marauder_RL', 2009: 'Marine_RL', 2010: 'Medivac_RL',
#                 2011: 'Stalker_RL', 2012: 'Zealot_RL', 2013: 'Zergling_RL',
#                 9: 'Baneling', 4: 'Colossus', 107: 'Hydralisk',
#                 51: 'Marauder', 48: 'Marine', 54: 'Medivac',
#                 74: 'Stalker', 73: 'Zealot', 105: 'Zergling'}
#
# custom_type_name2id = {'Baneling': 2005, 'Colossus': 2006, 'Hydralisk': 2007,
#                        'Marauder': 2008, 'Marine': 2009, 'Medivac': 2010,
#                        'Stalker': 2011, 'Zealot': 2012, 'Zergling': 2013}

standard_type_name2id = {'Baneling': 9, 'Colossus': 4, 'Hydralisk': 107,
                         'Marauder': 51, 'Marine': 48, 'Medivac': 54,
                         'Stalker': 74, 'Zealot': 73, 'Zergling': 105}


def get_type_name_by_id(type_id):
    return type_id2name[type_id]


def get_type_id_by_name(type_name, custom=False):
    if custom:
        return custom_type_name2id[type_name]
    else:
        return standard_type_name2id[type_name]


def get_race_by_type_name(type_name):
    for race in zip(["P", "T", "Z"], [Protoss, Terran, Zerg]):
        if getattr(race[1], type_name, None) is not None:
            return race[0]
    raise ValueError("Bad unit type {}".format(type_name))


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2MTEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            map_name='3-8m_symmetric',
            step_mul=8,
            move_amount=2,
            difficulty="7",
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            action_grid_attack=False,
            action_grid_segment=8,
            action_tag_attack=True,
            entity_scheme=True,
            max_reward_reset=False,
            debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.scenario_name = map_name
        scenario_params = get_mt_scenario_params(self.scenario_name)
        self.n_agents = 0
        self.n_enemies = 0
        self.max_n_agents = 0
        self.max_n_enemies = 0
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Scenario arguments
        self.scenarios = scenario_params['scenarios']
        self.max_types_and_units_scenario = scenario_params['max_types_and_units_scenario']
        self.pos_ally_centered = scenario_params['ally_centered']
        self.pos_rotate = scenario_params['rotate']
        self.pos_separation = scenario_params['separation']
        self.pos_jitter = scenario_params['jitter']
        self.episode_limit = scenario_params["episode_limit"]
        self.n_extra_tags = scenario_params['n_extra_tags']
        self.map_name = scenario_params['map_name']
        self.map_type = scenario_params["map_type"]

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.rs = RandomState(seed)
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        self.unit_types = set()
        self.standard2custom = {}
        self._agent_race = self._bot_race = None
        ally_army, enemy_army = self.max_types_and_units_scenario

        for num, utype_name in ally_army:
            self.n_agents += num
            custom_utype_id = get_type_id_by_name(utype_name, custom=True)
            standard_utype_id = get_type_id_by_name(utype_name)
            race = get_race_by_type_name(utype_name)
            self.standard2custom[standard_utype_id] = custom_utype_id
            if self._agent_race is None:
                self._agent_race = race
            elif self._agent_race != race:
                raise ValueError("Army spec implies multiple races {}".format(ally_army))
            self.unit_types.add(standard_utype_id)

        for num, utype_name in enemy_army:
            self.n_enemies += num
            utype_id = get_type_id_by_name(utype_name)
            race = get_race_by_type_name(utype_name)
            if self._bot_race is None:
                self._bot_race = race
            elif self._bot_race != race:
                raise ValueError("Army spec implies multiple races {}".format(enemy_army))
            self.unit_types.add(utype_id)

        self.max_n_agents = self.n_agents
        self.max_n_enemies = self.n_enemies

        self.action_tag_attack = action_tag_attack
        self.enemy_tags = None
        self.ally_tags = None
        self.entity_scheme = entity_scheme

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4

        self.action_grid_attack = action_grid_attack  # attack[enemy_id] -> attack[grid_id]
        self.action_grid_segment = action_grid_segment
        if self.action_grid_attack:
            self.action_grid_size = action_grid_segment * action_grid_segment
            # Whether the enemy can be attacked
            self.action_grid_avail_t_id = np.zeros(self.n_enemies)
            # Health of the enemy with the minimum health in the grid
            self.action_grid_min_health = np.zeros(self.action_grid_size)
            # Maximum health of the enemy with the minimum health in the grid
            self.action_grid_min_health_max = np.ones(self.action_grid_size)
            # Id of the enemy with the minimum health in the grid
            self.action_grid_min_health_t_id = np.zeros(self.action_grid_size) - 1
            # Actions: no-op, stop, move[direction], attack[grid_id]
            self.n_actions = self.n_actions_no_attack + self.action_grid_size
        else:
            if self.map_type == "MMM":
                # medivacs can heal friendly agents
                self.n_actions = self.n_actions_no_attack + self.max_n_enemies + self.max_n_agents + 2 * self.n_extra_tags
            else:
                self.n_actions = self.n_actions_no_attack + self.max_n_enemies + self.n_extra_tags

        # Map info
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = len(self.unit_types) if len(self.unit_types) > 1 else 0

        self.unit_type_ids = {}
        for i, utype_id in enumerate(sorted(list(self.unit_types))):
            self.unit_type_ids[utype_id] = i
        for standard_utype_id, custom_utype_id in self.standard2custom.items():
            self.unit_type_ids[custom_utype_id] = self.unit_type_ids[standard_utype_id]

        self.max_reward_reset = max_reward_reset
        self.max_reward = (
                self.max_n_enemies * self.reward_death_value + self.reward_win +
                scenario_params["reward_health_shield_max"]
        )

        # create lists containing the names of attributes returned in states
        self.ally_state_attr_names = [
            "health",
            "energy/cooldown",
            "rel_x",
            "rel_y",
        ]
        self.enemy_state_attr_names = ["health", "rel_x", "rel_y"]

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ["shield"]
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ["shield"]

        if self.unit_type_bits > 0:
            bit_attr_names = [
                "type_{}".format(bit) for bit in range(self.unit_type_bits)
            ]
            self.ally_state_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.baneling_id = 2005
        self.colossus_id = 2006
        self.hydralisk_id = 2007
        self.marauder_id = 2008
        self.marine_id = 2009
        self.medivac_id = 2010
        self.stalker_id = 2011
        self.zealot_id = 2012
        self.zergling_id = 2013

        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        self.map_center = (self.map_x // 2, self.map_y // 2)

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=np.bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )

        self.terrain_height = (
                np.flip(
                    np.transpose(
                        np.array(list(map_info.terrain_height.data)).reshape(
                            self.map_x, self.map_y
                        )
                    ),
                    1,
                )
                / 255
        )

    def reset(self, index=None):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        try:
            self.init_units(index)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            self.reset(index)

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.action_grid_attack:
            self.action_grid_avail_t_id = np.zeros(self.n_enemies)
            self.action_grid_min_health = np.zeros(self.action_grid_size)
            self.action_grid_min_health_max = np.ones(self.action_grid_size)
            self.action_grid_min_health_t_id = np.zeros(self.action_grid_size) - 1

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        self._calc_distance_mtx()

        if self.entity_scheme:
            return self.get_entities(), self.get_masks()
        return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one."""
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        if self.entity_scheme:
            actions = actions[:self.n_agents]

        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        self._calc_distance_mtx()

        return reward, terminated, info

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
                avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            if self.action_grid_attack:
                target_id = self.action_grid_min_health_t_id[action - self.n_actions_no_attack]
            else:
                target_tag = action - self.n_actions_no_attack
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    target_id = np.where(self.ally_tags == target_tag)[0].item()
                else:
                    target_id = np.where(self.enemy_tags == target_tag)[0].item()

            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    # TODO: 未修改适应 random tag
    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (
                    target is None
                    or self.agents[target].health == 0
                    or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (
                            al_unit.health != 0
                            and al_unit.health != al_unit.health_max
                    ):
                        dist = self.distance(
                            unit.pos.x,
                            unit.pos.y,
                            al_unit.pos.x,
                            al_unit.pos.y,
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["heal"]
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (
                            unit.unit_type == self.marauder_id
                            and e_unit.unit_type == self.medivac_id
                    ):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["attack"]
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        action_available = True
        if self.action_grid_attack:
            if self.action_grid_avail_t_id[self.heuristic_targets[a_id]] == 0:
                action_available = False
        else:
            if self.get_avail_agent_actions(a_id)[action_num] == 0:
                action_available = False

        # Check if the action is available
        if self.heuristic_rest and not action_available:

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y
                    )
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y
                    )
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount
                    )
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False,
            )
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(),
            replay_dir=replay_dir,
            prefix=prefix,
        )
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_masks(self):
        """
        Returns:
        1) per agent observability mask over all entities (unobservable = 1, else 0)
        3) mask of inactive entities (including enemies) over all possible entities
        """
        sight_range = np.array([self.unit_sight_range(a_i) for a_i in range(self.n_agents + self.n_enemies)]).reshape(-1, 1)
        obs_mask = (self.dist_mtx > sight_range).astype(np.bool8)
        obs_mask_pad = np.ones((self.max_n_agents + self.max_n_enemies,
                                self.max_n_agents + self.max_n_enemies), dtype=np.bool8)
        obs_mask_pad[:self.n_agents, :self.n_agents] = obs_mask[:self.n_agents, :self.n_agents]
        obs_mask_pad[:self.n_agents, self.max_n_agents:self.max_n_agents + self.n_enemies] = (
            obs_mask[:self.n_agents, self.n_agents:]
        )
        obs_mask_pad[self.max_n_agents:self.max_n_agents + self.n_enemies, :self.n_agents] = obs_mask[self.n_agents:, :self.n_agents]
        obs_mask_pad[self.max_n_agents:self.max_n_agents + self.n_enemies, self.max_n_agents:self.max_n_agents + self.n_enemies] = (
            obs_mask[self.n_agents:, self.n_agents:]
        )
        entity_mask = np.ones(self.max_n_agents + self.max_n_enemies, dtype=np.bool8)
        entity_mask[:self.n_agents] = 0
        entity_mask[self.max_n_agents:self.max_n_agents + self.n_enemies] = 0

        return obs_mask_pad, entity_mask

    def get_entities(self):
        """
        Returns list of agent entities and enemy entities in the map (all entities are a fixed size)
        All entities together form the global state
        For decentralized execution agents should only have access to the
        entities specified by get_masks()
        """
        all_units = list(self.agents.values()) + list(self.enemies.values())

        nf_entity = self.get_entity_size()

        center_x = self.map_x / 2
        center_y = self.map_y / 2
        mean_x = sum(unit.pos.x for unit in all_units) / len(all_units)
        mean_y = sum(unit.pos.y for unit in all_units) / len(all_units)
        max_dist_com = max(self.distance(unit.pos.x, unit.pos.y, mean_x, mean_y) for unit in all_units)

        entities = []
        avail_actions = self.get_avail_actions()
        for u_i, unit in enumerate(all_units):
            entity = np.zeros(nf_entity, dtype=np.float32)
            ind = 0

            if self.action_tag_attack:
                # entity tag
                if u_i < self.n_agents:
                    tag = self.ally_tags[u_i]
                else:
                    tag = self.enemy_tags[u_i - self.n_agents]
                entity[tag] = 1
                ind = self.max_n_agents + self.max_n_enemies + 2 * self.n_extra_tags

            if self.action_grid_attack or self.action_tag_attack:
                # available actions (if user controlled entity)
                if u_i < self.n_agents:
                    for ac_i in range(self.n_actions - 2):
                        entity[ind + ac_i] = avail_actions[u_i][2 + ac_i]
                ind += self.n_actions - 2

            # unit type
            if self.unit_type_bits > 0:
                type_id = self.unit_type_ids[unit.unit_type]
                entity[ind + type_id] = 1
                ind += self.unit_type_bits

            if unit.health > 0:  # otherwise dead, return all zeros
                # health and shield
                if self.obs_all_health or self.obs_own_health:
                    entity[ind] = unit.health / unit.health_max
                    if ((self.shield_bits_ally > 0 and u_i < self.n_agents) or
                            (self.shield_bits_enemy > 0 and u_i >= self.n_agents)):
                        entity[ind + 1] = unit.shield / unit.shield_max
                    ind += 1 + int(self.shield_bits_ally or self.shield_bits_enemy)

                # energy and cooldown (for ally units only)
                if u_i < self.n_agents:
                    if unit.energy_max > 0.0:
                        entity[ind] = unit.energy / unit.energy_max
                    entity[ind + 1] = unit.weapon_cooldown / self.unit_max_cooldown(unit)
                ind += 2

                # x-y positions
                entity[ind] = (unit.pos.x - center_x) / self.max_distance_x
                entity[ind + 1] = (unit.pos.y - center_y) / self.max_distance_y
                entity[ind + 2] = (unit.pos.x - mean_x) / max_dist_com
                entity[ind + 3] = (unit.pos.y - mean_y) / max_dist_com
                ind += 4

                if self.obs_pathing_grid:
                    entity[ind:ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
                    ind += self.n_obs_pathing
                if self.obs_terrain_height:
                    entity[ind:] = self.get_surrounding_height(unit)

            entities.append(entity)

            # pad entities to fixed number across episodes (for easier batch processing)
            if u_i == self.n_agents - 1:
                entities += [np.zeros(nf_entity, dtype=np.float32)
                             for _ in range(self.max_n_agents - self.n_agents)]
            elif u_i == self.n_agents + self.n_enemies - 1:
                entities += [np.zeros(nf_entity, dtype=np.float32)
                             for _ in range(self.max_n_enemies - self.n_enemies)]

        return entities

    def get_entity_size(self):
        nf_entity = 0

        if self.action_tag_attack:
            nf_entity += self.max_n_agents + self.max_n_enemies + 2 * self.n_extra_tags  # tag

        if self.action_grid_attack or self.action_tag_attack:
            nf_entity += self.n_actions - 2   # available actions minus those that are always available

        nf_entity += self.unit_type_bits  # unit type

        # below are only observed for alive units (else zeros)
        if self.obs_all_health or self.obs_own_health:
            nf_entity += 1 + int(self.shield_bits_ally or self.shield_bits_enemy)  # health and shield

        nf_entity += 2  # energy and cooldown for ally units
        nf_entity += 4  # global x-y coords + rel x-y to center of mass of all agents (normalized by furthest agent's distance)

        if self.obs_pathing_grid:
            nf_entity += self.n_obs_pathing  # local pathing
        if self.obs_terrain_height:
            nf_entity += self.n_obs_height   # local terrain

        return nf_entity

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information
            and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y,
            shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield,
            unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if self.action_grid_attack:
            grid_feats_dim = self.get_obs_grid_feats_size()
            grid_feats = np.zeros(grid_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                ind : ind + self.n_obs_pathing  # noqa
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            if self.action_grid_attack:
                grid_feats[0, :] = self.action_grid_min_health / self.action_grid_min_health_max
                grid_feats[1, :] = avail_actions[self.n_actions_no_attack:]

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range

                    # available
                    if self.action_grid_attack:
                        enemy_feats[e_id, 0] = self.action_grid_avail_t_id[e_id]
                    else:
                        enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + self.enemy_tags[e_id]]

                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                                                   e_x - x
                                           ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                                                   e_y - y
                                           ) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.unit_type_ids[e_unit.unit_type]
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                        dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.unit_type_ids[al_unit.unit_type]
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.unit_type_ids[unit.unit_type]
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.action_grid_attack:
            agent_obs = np.concatenate((agent_obs, grid_feats.flatten()))

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))
            if self.action_grid_attack:
                logging.debug("Grid feats {}".format(grid_feats))

        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        state_dict = self.get_state_dict()

        state = np.append(
            state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())
        if "timestep" in state_dict:
            state = np.append(state, state_dict["timestep"])

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(state_dict["allies"]))
            logging.debug("Enemy state {}".format(state_dict["enemies"]))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_ally_num_attributes(self):
        return len(self.ally_state_attr_names)

    def get_enemy_num_attributes(self):
        return len(self.enemy_state_attr_names)

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()
        nf_en = self.get_enemy_num_attributes()

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                        al_unit.health / al_unit.health_max
                )  # health
                if (
                        self.map_type == "MMM"
                        and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                            al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                                               x - center_x
                                       ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                                               y - center_y
                                       ) / self.max_distance_y  # relative Y

                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, 4] = (
                            al_unit.shield / max_shield
                    )  # shield

                if self.unit_type_bits > 0:
                    type_id = self.unit_type_ids[al_unit.unit_type]
                    ally_state[al_id, type_id - self.unit_type_bits] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                        e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                                               x - center_x
                                       ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                                               y - center_y
                                       ) / self.max_distance_y  # relative Y

                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, 3] = e_unit.shield / max_shield  # shield

                if self.unit_type_bits > 0:
                    type_id = self.unit_type_ids[e_unit.unit_type]
                    enemy_state[e_id, type_id - self.unit_type_bits] = 1

        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = self.last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit

        return state

    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """
        Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-
        related features.
        """
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_grid_feats_size(self):
        """Returns the size of the vector containing the grid features.
        """
        n_grid_feats = self.action_grid_size
        n_grids = 2

        return n_grids, n_grid_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        if self.action_grid_attack:
            n_grids, n_grid_feats = self.get_obs_grid_feats_size()
            grid_feats = n_grids * n_grid_feats
            return move_feats + enemy_feats + ally_feats + own_feats + grid_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    def _calc_distance_mtx(self):
        # Calculate distances of all agents to all agents and enemies (for visibility calculations)
        dist_mtx = 1000 * np.ones((self.n_agents + self.n_enemies, self.n_agents + self.n_enemies))
        for i in range(self.n_agents + self.n_enemies):
            for j in range(self.n_agents + self.n_enemies):
                if j < i:
                    continue
                elif j == i:
                    dist_mtx[i, j] = 0.0
                else:
                    if i >= self.n_agents:
                        unit_a = self.enemies[i - self.n_agents]
                    else:
                        unit_a = self.agents[i]
                    if j >= self.n_agents:
                        unit_b = self.enemies[j - self.n_agents]
                    else:
                        unit_b = self.agents[j]

                    if unit_a.health > 0 and unit_b.health > 0:
                        dist = self.distance(unit_a.pos.x, unit_a.pos.y,
                                             unit_b.pos.x, unit_b.pos.y)
                        dist_mtx[i, j] = dist
                        dist_mtx[j, i] = dist

        self.dist_mtx = dist_mtx

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            dist_offset = self.n_agents
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]
                dist_offset = 0

            if self.action_grid_attack:
                self.action_grid_avail_t_id = np.zeros(self.n_enemies)
                self.action_grid_min_health = np.zeros(self.action_grid_size)
                self.action_grid_min_health_max = np.ones(self.action_grid_size)
                self.action_grid_min_health_t_id = np.zeros(self.action_grid_size) - 1

                grid_unit_size = (shoot_range * 2.0) / self.action_grid_segment

                for t_id, t_unit in target_items:
                    if t_unit.health > 0:
                        x_dist = t_unit.pos.x - unit.pos.x + shoot_range
                        y_dist = t_unit.pos.y - unit.pos.y + shoot_range

                        x_grid = x_dist // grid_unit_size
                        y_grid = y_dist // grid_unit_size

                        if 0 <= x_grid < self.action_grid_segment and 0 <= y_grid < self.action_grid_segment:
                            grid_id = int(x_grid * self.action_grid_segment + y_grid)
                            avail_actions[self.n_actions_no_attack + grid_id] = 1
                            self.action_grid_avail_t_id[t_id] = 1
                            if t_unit.health < self.action_grid_min_health[grid_id] or self.action_grid_min_health[grid_id] == 0:
                                self.action_grid_min_health[grid_id] = t_unit.health
                                self.action_grid_min_health_max[grid_id] = t_unit.health_max
                                self.action_grid_min_health_t_id[grid_id] = t_id
            else:
                for t_id, t_unit in target_items:
                    if t_unit.health > 0:
                        dist = self.dist_mtx[agent_id, t_id + dist_offset]
                        if dist <= shoot_range:
                            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                                tag = self.ally_tags[t_id]
                            else:
                                tag = self.enemy_tags[t_id]
                            avail_actions[tag + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        if not self.entity_scheme:
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
        else:
            for agent_id in range(self.max_n_agents):
                if agent_id < self.n_agents:
                    avail_agent = self.get_avail_agent_actions(agent_id)
                else:
                    avail_agent = [1] + [0] * (self.n_actions - 1)
                avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self, mode="human"):
        if self.renderer is None:
            from .render import StarCraft2Renderer

            self.renderer = StarCraft2Renderer(self, mode)
        assert (
                mode == self.renderer.mode
        ), "mode must be consistent across render calls"
        return self.renderer.render(mode)

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
                          unit.tag for unit in self.agents.values() if unit.health > 0
                      ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def _assign_pos(self, scenario):
        if self.pos_rotate:
            theta = self.rs.random() * 2 * np.pi
        else:
            theta = np.pi

        if self.pos_ally_centered:
            r = self.pos_separation
            ally_pos = (0, 0)
            enemy_pos = (r * np.cos(theta), r * np.sin(theta))
        else:
            r = self.pos_separation / 2
            ally_pos = (r * np.cos(theta), r * np.sin(theta))
            enemy_pos = (-ally_pos[0], -ally_pos[1])

        ally_army, enemy_army = scenario
        return ([(num, unit, ally_pos + (self.rs.rand(2) - 0.5) * 2 * self.pos_jitter) for (num, unit) in ally_army],
                [(num, unit, enemy_pos + (self.rs.rand(2) - 0.5) * 2 * self.pos_jitter) for (num, unit) in enemy_army])

    def init_units(self, index=None):
        """Initialise the units."""
        while True:
            if index is not None:
                scenario = self.scenarios[index]
            else:
                scenario = self.scenarios[self.rs.randint(len(self.scenarios))]

            ally_army, enemy_army = self._assign_pos(scenario)

            cmds = []

            self.n_agents = 0
            for num, utype_name, pos in ally_army:
                sc_pos = sc_common.Point2D(x=self.map_center[0] + pos[0],
                                           y=self.map_center[1] + pos[1])
                utype_id = get_type_id_by_name(utype_name, custom=True)
                cmd = d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=utype_id,
                        owner=1,
                        pos=sc_pos,
                        quantity=num))
                cmds.append(cmd)
                self.n_agents += num

            self.n_enemies = 0
            reward_health_shield = 0
            for num, utype_name, pos in enemy_army:
                sc_pos = sc_common.Point2D(x=self.map_center[0] + pos[0],
                                           y=self.map_center[1] + pos[1])
                utype_id = get_type_id_by_name(utype_name)
                cmd = d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=utype_id,
                        owner=2,
                        pos=sc_pos,
                        quantity=num))
                cmds.append(cmd)
                self.n_enemies += num
                reward_health_shield += (type_name2health[utype_name] + type_name2shield[utype_name]) * num

            self._controller.debug(cmds)
            self.try_controller_step()

            if self.max_reward_reset:
                self.max_reward = (
                        self.n_enemies * self.reward_death_value + self.reward_win + reward_health_shield
                )

            if self.entity_scheme and self.action_tag_attack:
                # assign random tags to agents (used for identifying entities as well as targets for actions)
                self.enemy_tags = self.rs.choice(np.arange(self.max_n_enemies + self.n_extra_tags),
                                                 size=self.n_enemies,
                                                 replace=False)
                self.ally_tags = self.rs.choice(np.arange(self.max_n_enemies + self.n_extra_tags,
                                                          self.max_n_enemies + self.max_n_agents + 2 * self.n_extra_tags),
                                                size=self.n_agents,
                                                replace=False)
            else:  # TODO
                self.enemy_tags = np.arange(self.n_enemies)
                self.ally_tags = np.arange(self.max_n_enemies + self.n_extra_tags,
                                           self.max_n_enemies + self.n_extra_tags + self.n_agents)

            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit

            self._init_enemy_strategy()

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            if all_agents_created and all_enemies_created:  # all good
                return

    def try_controller_step(self):
        try:
            self._controller.step(2)
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            self.reset()

    def _init_enemy_strategy(self):
        # TODO: 是否需要单独定义治疗动作，确定 controller step 次数
        cmd = d_pb.DebugCommand(
            game_state=d_pb.DebugGameState.control_enemy
        )
        self._controller.debug([cmd])
        self.try_controller_step()

        tags = [u.tag for u in self.enemies.values()]
        ally_spawn_center = sc_common.Point2D(
            x=sum(al.pos.x for al in self.agents.values()) / len(self.agents),
            y=sum(al.pos.y for al in self.agents.values()) / len(self.agents)
        )
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=actions["attack"],
            target_world_space_pos=ally_spawn_center,
            unit_tags=tags,
            queue_command=False
        )
        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        req_actions = sc_pb.RequestAction(actions=[sc_action])
        self._controller.actions(req_actions)

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
                n_ally_alive == 0
                and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)
        ):
            return -1  # lost
        if (
                n_ally_alive > 0
                and n_enemy_alive == 0
                or self.only_medivac_left(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        env_info["n_agents"] = self.n_agents
        env_info["n_enemies"] = self.n_enemies
        if self.entity_scheme:
            env_info["n_agents"] = self.max_n_agents
            env_info["n_enemies"] = self.max_n_agents
            env_info["n_entities"] = self.max_n_agents + self.max_n_enemies
            env_info["entity_shape"] = self.get_entity_size()
        if self.action_grid_attack or self.action_tag_attack:
            env_info["n_fixed_actions"] = self.get_total_actions()
            env_info["action_enemy_wise"] = False
            env_info["action_ally_wise"] = False
        else:
            env_info["n_fixed_actions"] = self.n_actions_no_attack
            env_info["action_enemy_wise"] = True
            env_info["action_ally_wise"] = True
        return env_info
