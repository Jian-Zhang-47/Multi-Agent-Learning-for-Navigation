from copy import copy

import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import random
import math

import config as cfg

from model import ReplayBuffer
from model import D3QL


def dict_to_arr(dict_var):
    row = col = 2 * cfg.M + 1
    array_3d = np.full((cfg.N, row, col), '  ', dtype=object)

    for a_id, sub_dict in dict_var.items():
        for (i, j), value in sub_dict.items():
            array_3d[a_id - 1][i, j] = value

    array_3d_num = np.vectorize(replace_element)(array_3d).astype(float)

    return array_3d_num


def replace_element(elem):
    return cfg.replacement_dict.get(elem)


def return_one_hot_vector(value):
    one_hot_vector = [0 for _ in range(4)]
    one_hot_vector[value] = 1
    return one_hot_vector


class GridNavigationEnv(gym.Env):
    def __init__(self, L=6, P=0.3, N=1, T=100, M=6):
        super(GridNavigationEnv, self).__init__()
        self.L = L  # Grid size
        self.P = P  # Number of obstacles as a percentage of grid size
        self.N = N  # Number of agents
        self.T = T  # Maximum episode length
        self.M = M  # Agent FoV size
        self.grid = np.zeros((L, L), dtype=int)
        self.obstacles = int(P * L * L)
        self.agents = []  # Coordinates of agents
        self.agents_id_list = []  # List of agent IDs
        self.agents_route_dict = {}  # Route coordinates of agents
        self.destination = None
        self.steps = 0  # Number of steps
        self.rewards = np.zeros(self.N)  # Rewards of agents
        self.distance_dict = {i + 1: [] for i in range(self.N)}  # Distance between agents and destination
        self.fov = {i + 1: [] for i in range(self.N)}  # FoV of agents
        self.fov_rel = {i + 1: [] for i in range(self.N)}  # FoV of agents, agent as the origin
        self.view_angle = 90  # View angle of agent
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.init_environment()

    def init_environment(self):
        self.place_obstacles()
        self.place_destination()
        self.place_agents()

    def place_obstacles(self):
        for _ in range(self.obstacles):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = -1  # Obstacle grid value is -1
                    break

    def place_destination(self):
        while True:
            x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
            if self.grid[x, y] == 0:
                self.destination = (x, y)  # Destination grid value is 0
                break

    def place_agents(self):
        for i in range(self.N):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.agents_id_list.append(i + 1)  # Agent ID
                    self.grid[x, y] = self.agents_id_list[i]
                    self.agents.append((x, y))
                    self.agents_route_dict[self.agents_id_list[i]] = [
                        self.agents[i]]  # Record the original position of agents
                    break
            self.fov[i + 1] = (self.get_fov(self.agents[i], -1, 0))  # Get FoV at step 0. The default view is up.
            self.fov_rel[i + 1] = self.relative_coordinates(self.fov[i + 1], self.agents[i])

    def relative_coordinates(self, original_dict, agent_pos):  # Set the top left grid of FoV as the origin
        return {(k[0] - agent_pos[0] + self.M, k[1] - agent_pos[1] + self.M): v for k, v in original_dict.items()}

    def calculate_gradient(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_fov(self, agent_pos, dx, dy):
        fov = []
        fov_circle = []
        x, y = agent_pos
        for i in range(-self.M, self.M + 1):
            for j in range(-self.M, self.M + 1):
                xi, yj = x + i, y + j
                ij_pos = (xi, yj)
                d = self.calculate_distance(agent_pos, ij_pos)  # Distance between grid and agent
                if 0 <= xi < self.L and 0 <= yj < self.L and d <= self.M:
                    fov_circle.append(ij_pos)

        if self.view_angle < 180:
            g1 = round(-1 / math.tan(math.radians(self.view_angle / 2)), 4)  # Gradient of the 1st view field boundary
            g2 = round(1 / math.tan(math.radians(self.view_angle / 2)), 4)  # Gradient of the 2nd view field boundary
            for p in fov_circle:
                g_p = self.calculate_gradient(agent_pos, p)  # Gradient of the line between grid and agent
                if dx == -1 and dy == 0:  # Move to up
                    if g1 <= g_p <= g2 and p[0] <= x:
                        fov.append(p)
                elif dx == 1 and dy == 0:  # Move to down
                    if g1 <= g_p <= g2 and p[0] >= x:
                        fov.append(p)
                elif dx == 0 and dy == -1:  # Move to left
                    if (g_p <= g1 or g_p >= g2 or g_p == float('inf')) and p[1] <= y:
                        fov.append(p)
                elif dx == 0 and dy == 1:  # Move to right
                    if (g_p <= g1 or g_p >= g2 or g_p == float('inf')) and p[1] >= y:
                        fov.append(p)
        elif self.view_angle >= 180:
            g1 = -1 / math.tan(math.radians((360 - self.view_angle) / 2))
            g2 = 1 / math.tan(math.radians((360 - self.view_angle) / 2))
            for p in fov_circle:
                g_p = self.calculate_gradient(agent_pos, p)  # Gradient of the line between grid and agent
                if dx == -1 and dy == 0:  # Move to up
                    if p[0] <= x:
                        fov.append(p)
                    elif p[0] > x and (g_p >= g2 or g_p <= g1):
                        fov.append(p)
                elif dx == 1 and dy == 0:  # Move to down
                    if p[0] >= x:
                        fov.append(p)
                    elif p[0] < x and (g_p >= g2 or g_p <= g1):
                        fov.append(p)
                elif dx == 0 and dy == -1:  # Move to left
                    if p[1] <= y:
                        fov.append(p)
                    elif p[1] > y and (g1 <= g_p <= g2):
                        fov.append(p)
                elif dx == 0 and dy == 1:  # Move to right
                    if p[1] >= y:
                        fov.append(p)
                    elif p[1] < y and (g1 <= g_p <= g2):
                        fov.append(p)
        if agent_pos not in fov:
            fov.append(agent_pos)

        # Update the FoV state
        state_map = {pos: 'S-' for pos in fov}
        obstacles = [pos for pos in fov if self.grid[pos] == -1]
        for obs in obstacles:
            if state_map[obs] == 'S-':
                state_map[obs] = 'S+'
                self.blocked_fov_by_obstacle(agent_pos, obs, fov, state_map)
        return state_map

    def update_S0(self, corner1, corner2, agent_pos, d_ao, obs, fov, state_map):
        g1 = self.calculate_gradient(agent_pos, corner1)
        g2 = self.calculate_gradient(agent_pos, corner2)
        for p in fov:
            g = self.calculate_gradient(p, agent_pos)
            d_ap = self.calculate_distance(p, agent_pos)  # The distance between agent and p
            d_op = self.calculate_distance(p, obs)  # The distance between p and obs
            if min(g1, g2) < g < max(g1, g2) and d_ap > d_ao and d_ap > d_op:
                state_map[p] = 'S0'

    def update_S0_inf(self, corner1, corner2, agent_pos, d_ao, obs, fov, state_map):
        g1 = self.calculate_gradient(agent_pos, corner1)
        g2 = self.calculate_gradient(agent_pos, corner2)
        for p in fov:
            g = self.calculate_gradient(p, agent_pos)
            d_ap = self.calculate_distance(p, agent_pos)
            d_op = self.calculate_distance(p, obs)
            if (min(g1, g2) > g or g > max(g1, g2) or g == float('inf')) and d_ap > d_ao and d_ap > d_op:
                state_map[p] = 'S0'

    def blocked_fov_by_obstacle(self, agent_pos, obs, fov, state_map):
        xb, yb = obs
        x, y = agent_pos
        d_ao = self.calculate_distance(agent_pos, obs)  # The distance between agent and obs

        if xb < x and yb < y:  # The obstacle is on the 'Up Left' of the agent.
            corner1 = (xb + 0.5, yb - 0.5)
            corner2 = (xb - 0.5, yb + 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb < x and yb > y:  # ...Up Right...
            corner1 = (xb + 0.5, yb + 0.5)
            corner2 = (xb - 0.5, yb - 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb > x and yb < y:  # ...Down Left...
            corner1 = (xb + 0.5, yb + 0.5)
            corner2 = (xb - 0.5, yb - 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb > x and yb > y:  # ...Down Right...
            corner1 = (xb + 0.5, yb - 0.5)
            corner2 = (xb - 0.5, yb + 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb == x and yb < y:  # ...Left...
            corner1 = (xb + 0.5, yb + 0.5)
            corner2 = (xb - 0.5, yb + 0.5)
            self.update_S0_inf(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb == x and yb > y:  # ...Right...
            corner1 = (xb + 0.5, yb - 0.5)
            corner2 = (xb - 0.5, yb - 0.5)
            self.update_S0_inf(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb < x and yb == y:  # ...Up...
            corner1 = (xb + 0.5, yb - 0.5)
            corner2 = (xb + 0.5, yb + 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

        elif xb > x and yb == y:  # ...Down...
            corner1 = (xb - 0.5, yb + 0.5)
            corner2 = (xb - 0.5, yb - 0.5)
            self.update_S0(corner1, corner2, agent_pos, d_ao, obs, fov, state_map)

    def move_agent(self, agent_id, action):
        x, y = self.agents[agent_id - 1]
        move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = move_offsets[action]
        print(f'Agent{agent_id} move:({dx, dy})')
        nx, ny = x + dx, y + dy
        self.grid[self.destination] = 0
        if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
            self.grid[x, y] = 0
            self.grid[nx, ny] = agent_id
            self.agents[agent_id - 1] = (nx, ny)
            self.agents_route_dict[agent_id].append(self.agents[agent_id - 1])  # Record the new position of agents
        else:
            self.agents_route_dict[agent_id].append((x, y))
        self.fov[agent_id] = (self.get_fov(self.agents[agent_id - 1], dx, dy))  # Get FoV after moving
        self.fov_rel[agent_id] = self.relative_coordinates(self.fov[agent_id], self.agents[agent_id - 1])

    def reset(self, **kwargs):
        self.grid.fill(0)
        self.agents.clear()
        self.agents_id_list.clear()
        self.agents_route_dict.clear()
        self.destination = None
        self.steps = 0
        self.rewards = np.zeros(self.N)
        self.distance_dict = {i + 1: [] for i in range(self.N)}
        self.fov = {i + 1: [] for i in range(self.N)}
        self.init_environment()

        observation = dict_to_arr(self.fov_rel)

        return observation

    def step(self, actions):
        for a_id in self.agents_id_list[:]:
            if self.agents[a_id - 1] == self.destination:
                self.agents_id_list.remove(a_id)
            else:
                self.move_agent(a_id, actions[a_id - 1])
                distance = self.calculate_distance(self.agents[a_id - 1], self.destination)
                self.distance_dict[a_id].append(distance)  # Record the distance in dict
                if distance == 0:
                    self.rewards[a_id - 1] = 1  # Reward for reaching destination
                    self.agents_id_list.remove(a_id)
                else:
                    self.rewards[a_id - 1] = - 0.01  # Penalty for each move

        self.steps += 1
        done = all(self.agents[agent_id - 1] == self.destination for agent_id in self.agents_id_list)
        terminate = self.steps >= self.T
        observation = dict_to_arr(self.fov_rel)
        info = {}
        info['grid_map'] = self.grid
        info['agents_route'] = self.agents_route_dict
        info['steps_number'] = self.steps
        info['distances'] = self.distance_dict
        info['destination'] = self.destination
        return observation, self.rewards, terminate, info, done

    def render(self, mode='human'):
        for row in self.grid:
            print('   '.join(str(x) for x in row))

    def render_fov(self, fov):
        for key, coordinates in fov.items():
            if not coordinates:
                print(f"Fov map for Agent {key} is empty.")
                continue
            FoV_map = [['  ' for _ in range(0, 2 * self.M + 1)] for _ in range(0, 2 * self.M + 1)]
            for (x, y), state in coordinates.items():
                FoV_map[x][y] = state
            print(f"Fov map for Agent {key}:")
            for row in FoV_map:
                print('  '.join(str(x) for x in row))


if __name__ == "__main__":
    env = GridNavigationEnv(L=cfg.L, P=cfg.P, N=cfg.N, T=cfg.T, M=cfg.M)
    observation = env.reset()
    done = False
    terminate = False
    print(f'In 0 step')
    env.render()
    # env.render_fov(env.fov_rel)
    print(dict_to_arr(env.fov_rel))
    print()

    dimension = (2 * cfg.M) + 1
    buffer = ReplayBuffer(cfg.memory_size, cfg.N, dimension)
    d3ql_algorithm = D3QL(cfg.N, dimension)

    is_successful = np.zeros(cfg.episode_num)
    average_rewards = np.zeros(cfg.episode_num)
    epsilon_history = []
    for ep in range(cfg.episode_num):
        print(f'starting episode {ep}')

        rewards_this_episode = []
        epsilon = 1
        epsilon_decay = 0.99
        epsilon_min = 0.05

        while not done and not terminate:
            old_observation = copy(observation)

            # epsilon-greedy algorithm
            if np.random.random() < epsilon:
                actions = np.array([env.action_space.sample() for _ in range(env.N)])  # Random actions
            else:
                np.array([d3ql_algorithm.get_model_output(observation[i, :, :].flatten(), i)
                          for i in range(env.N)])  # Intelligent actions

            observation, rewards, terminate, info, done = env.step(actions)
            print(f'done is {done}')
            print(f"In {info['steps_number']} steps")
            env.render()
            print('')
            env.render_fov(env.fov_rel)
            print(env.fov_rel)
            print(observation)
            print('')

            rewards_this_episode.append(rewards.mean())

            actions_encoded = np.array([return_one_hot_vector(a) for a in actions])
            buffer.store_experience(old_observation.reshape(cfg.N, -1),
                                    observation.reshape(cfg.N, -1),
                                    actions_encoded, rewards, done or terminate)

            # update epsilon
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)
            epsilon_history.append(epsilon)

            # train models independently
            if buffer.mem_counter > cfg.batch_size:
                state, next_state, action, reward, dones = buffer.sample_buffer()
                d3ql_algorithm.train_independent(state, next_state, action, reward, dones)

            print(f"Destination: {info['destination']}")
            print(f"Reward: {rewards}")
            print(f"All agents have reached the destination: {done}")
            print(f"Some agents are stuck somewhere: {terminate}")
            print(f"Route: {info['agents_route']}\n")

        is_successful[ep] = done
        average_rewards[ep] = np.array(rewards_this_episode).mean() if len(rewards_this_episode) > 0 else 0

    print('*****************************************')
    print(f'Success rate for {cfg.episode_num} episodes was {is_successful.mean() * 100}%')
    print(f'Average reward for {cfg.episode_num} episodes was {round(average_rewards.mean(), 3)}')
    print('*****************************************')

    plt.figure(1)
    sum_rewards = np.zeros(cfg.episode_num)
    for x in range(cfg.episode_num):
        sum_rewards[x]=np.sum(rewards_this_episode[max(0,x-100):(x+1)])
    fig, ax1 = plt.subplots()
    x = np.arange(cfg.episode_num)
    if len(epsilon_history) < cfg.episode_num:
        epsilon_history = np.pad(epsilon_history, (0, cfg.episode_num - len(epsilon_history)), 'constant', constant_values=np.nan)

    ax1.plot(x, sum_rewards, 'b-', label='Reward', marker='o')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, epsilon_history, 'r-', label='Epsilon', marker='x')
    ax2.set_ylabel('Epsilon', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Reward & Epsilon')
    fig.tight_layout() 

    plt.savefig('GridNavigationEnv_dqn.png')

    plt.figure(2)



    # Graph to show the distance between agent and destination
    fig1, ax1 = plt.subplots()
    x = list(range(cfg.T))
    for i in range(cfg.N):
        y = info['distances'][i+1]
        if len(y)<len(x):
            y.extend([None] * (len(x) - len(y)))
        ax1.plot(x,y,label=f'Agent {i+1}')
    ax1.set_xlabel('t')
    ax1.set_ylabel('d')
    ax1.set_title('Distance between agent and destination')
    ax1.legend()
    plt.savefig('Distance.png')

    # Graph to show the numbers of step
    fig2, ax2 = plt.subplots()
    x = []
    y = []
    for i in range(cfg.N):
        x.append(f'{i+1}')
        y.append(len(info['agents_route'][i+1]))
    ax2.bar(x,y)
    ax2.set_xlabel('agent ID')
    ax2.set_ylabel('Numbers of step')
    ax2.set_title('Numbers of agents\' step')
    plt.savefig('Step.png')