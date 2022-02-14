import numpy as np
from envs.particle.core import World, Agent, Landmark
from envs.particle.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = getattr(args, "num_agents", 3)
        num_landmarks = getattr(args, "num_landmarks", 3)
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.dead = False
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def get_done(self, world):
        return False

    def reward(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        min_dists = []
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists.append(min(dists))
        rew = -np.mean(min_dists)
        return rew

    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius == -1 or dist <= agent.view_radius:
                other_pos.append(np.hstack([np.array([1.0]), other.state.p_pos, other.state.p_vel, np.array([1.0, 0.0])]))
            else:
                other_pos.append(np.array([0., 0., 0., 0., 0., 0., 0.0]))

        entity_pos = []
        for entity in world.landmarks:
            dist = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if agent.view_radius == -1 or dist <= agent.view_radius:
                entity_pos.append(np.hstack([np.array([1.0]), entity.state.p_pos, entity.state.p_vel, np.array([0.0, 1.0])]))
            else:
                entity_pos.append(np.array([0., 0., 0., 0., 0., 0., 0.0]))

        return np.concatenate([np.array([1.0])] + [agent.state.p_pos] + [agent.state.p_vel] + [np.array([1.0, 0.0])] + entity_pos + other_pos)

    def state(self, world):
        other_pos = []
        for other in world.agents:
            other_pos.append(np.hstack([np.array([1.0]), other.state.p_pos, other.state.p_vel, np.array([1.0, 0.0])]))

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(np.hstack([np.array([1.0]), entity.state.p_pos, entity.state.p_vel, np.array([0.0, 1.0])]))

        return np.concatenate(other_pos + entity_pos)
