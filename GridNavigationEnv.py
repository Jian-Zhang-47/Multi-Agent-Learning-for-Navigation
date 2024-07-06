from copy import copy,deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import config as cfg
import utils


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

        # Save the initial state
        self.initial_state = None

        # Initialize the environment
        self.init_environment()
        self.save_initial_state()

    def save_initial_state(self):
        self.initial_state = deepcopy((self.grid, self.agents, self.agents_id_list, self.agents_route_dict, 
                                       self.destination, self.steps, self.rewards, self.distance_dict, 
                                       self.fov, self.fov_rel))

    def restore_initial_state(self):
        (self.grid, self.agents, self.agents_id_list, self.agents_route_dict, 
         self.destination, self.steps, self.rewards, self.distance_dict, 
         self.fov, self.fov_rel) = deepcopy(self.initial_state)



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
                    self.agents_route_dict[self.agents_id_list[i]] = [self.agents[i]]  # Record the original position of agents
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

    def reset(self):
        #self.grid.fill(0)
        #self.agents.clear()
        #self.agents_id_list.clear()
        #self.agents_route_dict.clear()
        #self.destination = None
        #self.steps = 0
        #self.rewards = np.zeros(self.N)
        #self.distance_dict = {i + 1: [] for i in range(self.N)}
        #self.fov = {i + 1: [] for i in range(self.N)}
        #self.init_environment()
        #observation = utils.dict_to_arr(self.fov_rel)
        #return observation
    
        self.restore_initial_state()
        observation = utils.dict_to_arr(self.fov_rel)
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
                    self.rewards[a_id - 1] = 1.5*self.L  # Reward for reaching destination
                    self.agents_id_list.remove(a_id)
                else:
                    self.rewards[a_id - 1] = 1.5*self.L/distance # Penalty for each move

        self.steps += 1
        done = all(self.agents[agent_id - 1] == self.destination for agent_id in self.agents_id_list)
        terminate = self.steps >= self.T
        observation = utils.dict_to_arr(self.fov_rel)
        rewards = self.rewards
        info = {}
        info['grid_map'] = self.grid
        info['agents_route'] = self.agents_route_dict
        info['steps_number'] = self.steps
        info['distances'] = self.distance_dict
        info['destination'] = self.destination
        return observation, rewards, terminate, info, done

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
    env.init_environment()
    observation = env.reset()
    done = False
    terminate = False
    print(f'In 0 step')
    env.render()
    #env.render_fov(env.fov_rel)
    print(utils.dict_to_arr(env.fov_rel))
    print()

    while not done and not terminate:
        actions = [env.action_space.sample() for _ in range(env.N)]  # Random actions
        observation, rewards, terminate, info, done = env.step(actions)
        print(f'In {info['steps_number']} steps')
        print(f"Reward: {rewards}")
        env.render()
        print('')
        env.render_fov(env.fov_rel)
        print(env.fov_rel)
        print(observation)
        print('')
    print(f"Destination: {info['destination']}")
    print(f"Reward: {rewards}")
    print(f"All agents have reached the destination: {done}")
    print(f"Some agents are stuck somewhere: {terminate}")
    print(f"Route: {info['agents_route']}\n")