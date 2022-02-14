import numpy as np

from envs.multiagentenv import MultiAgentEnv
from envs.particle.environment import MultiAgentEnv as OpenAIMultiAgentEnv
from envs.particle import scenarios
from utils.dict2namedtuple import convert


class Particle(MultiAgentEnv):

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # load scenario from script
        self._episode_steps = 0
        self.episode_limit = self.args.episode_limit
        self.scenario = scenarios.load(self.args.map_name.split("_")[0] + ".py").Scenario()
        self.world = self.scenario.make_world(self.args)
        self.n_agents = len(self.world.policy_agents)
        self.n_enemies = len(self.world.agents) - len(self.world.policy_agents) + len(self.world.landmarks)

        if self.args.benchmark:
            self.env = OpenAIMultiAgentEnv(world=self.world,
                                           reset_callback=self.scenario.reset_world,
                                           reward_callback=self.scenario.reward,
                                           observation_callback=self.scenario.observation,
                                           state_callback=self.scenario.state,
                                           info_callback=self.scenario.benchmark_data,
                                           done_callback=self.scenario.get_done)
        else:
            self.env = OpenAIMultiAgentEnv(world=self.world,
                                           reset_callback=self.scenario.reset_world,
                                           reward_callback=self.scenario.reward,
                                           observation_callback=self.scenario.observation,
                                           state_callback=self.scenario.state,
                                           done_callback=self.scenario.get_done)

    def step(self, actions):
        actions = [int(a) for a in actions]

        obs_n, reward, terminated, info = self.env.step(actions)

        self._episode_steps += 1
        if self._episode_steps >= self.episode_limit:
            terminated = True

        min_dists = []
        for i, a in enumerate(self.world.agents):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in self.world.landmarks]
            min_dist = min(dists)
            info["min_dist_agent_{}".format(i)] = min_dist
            min_dists.append(min_dist)

        info["min_dists_mean"] = np.mean(min_dists)

        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        agent_obs = self.env._get_obs(self.world.policy_agents[agent_id])
        return agent_obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.observation_space[0].shape[0]

    def get_state(self):
        state = self.env._get_state()
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_state())
        return state_size

    def get_avail_actions(self):
        avail_actions = np.ones((self.n_agents, self.get_total_actions()))
        avail_actions[:, 0] = 0
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.env.action_space[0].n

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.env.reset()

        return self.get_obs(), self.get_state()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        return

    def seed(self):
        return self.args.seed

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_stats(self):
        stats = {}
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["n_fixed_actions"] = self.get_total_actions()
        env_info["action_enemy_wise"] = False
        env_info["action_ally_wise"] = False
        env_info["n_agents"] = self.n_agents
        env_info["n_enemies"] = self.n_enemies
        env_info["unit_dim"] = int(self.get_state_size() / (self.n_agents + self.n_enemies))
        env_info["n_tokens"] = self.n_agents + self.n_enemies
        env_info["obs_token_shape"] = int(self.get_obs_size() / (self.n_agents + self.n_enemies))
        env_info["state_token_shape"] = int(self.get_state_size() / (self.n_agents + self.n_enemies))
        env_info["obs_mask_bit"] = 0
        env_info["state_mask_bit"] = 0
        return env_info
