import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Tuple, Dict

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
        self.rewards_dict = {i + 1: [] for i in range(self.N)}  # Rewards of agents
        self.distance_dict = {i + 1: [] for i in range(self.N)}  # Distance between agents and destination
        self.fov = {i + 1: [] for i in range(self.N)}  # FoV of agents
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
                    self.agents_route_dict[self.agents_id_list[i]] = [self.agents[i]]  # Record the original position of agents
                    break
            self.fov[i+1]=(self.get_fov(self.agents[i])) # Get FoV at step 0

    def calculate_gradient(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    def get_fov(self, agent_pos: Tuple[int, int]) -> Dict[Tuple[int, int], str]:
        fov_radius = self.M // 2
        x, y = agent_pos
        fov = []

        for i in range(-fov_radius, fov_radius + 1):
            for j in range(-fov_radius, fov_radius + 1):
                xi, yj = x + i, y + j
                if 0 <= xi < self.L and 0 <= yj < self.L:
                    fov.append((xi, yj))

        # Update the FoV state
        state_map = {pos: 'S-' for pos in fov}
        obstacles = [pos for pos in fov if self.grid[pos] == -1]
        for obs in obstacles:
            if state_map[obs] == 'S-':
                state_map[obs] = 'S+'
                self.blocked_fov_by_obstacle(agent_pos, obs, fov, state_map)

        return state_map

    def blocked_fov_by_obstacle(self, agent_pos: Tuple[int, int], obs: Tuple[int, int], fov: List[Tuple[int, int]], state_map: Dict[Tuple[int, int], str]):
        xb, yb = obs
        x, y = agent_pos

        if xb < x and yb < y:   # Top left
            low = (xb + 0.5, yb - 0.5)
            high = (xb - 0.5, yb + 0.5)
            g1 = self.calculate_gradient(obs,high)
            g2 = self.calculate_gradient(obs,low)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[0]<x:
                    state_map[p] = 'S0'
        elif xb < x and yb > y:    # Top right
            low = (xb + 0.5, yb + 0.5)
            high = (xb - 0.5, yb - 0.5)
            g1 = self.calculate_gradient(obs,low)
            g2 = self.calculate_gradient(obs,high)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[1]>y:
                    state_map[p] = 'S0'
        elif xb > x and yb < y:     # Down left
            low = (xb + 0.5, yb + 0.5)
            high = (xb - 0.5, yb - 0.5)
            g1 = self.calculate_gradient(obs,high)
            g2 = self.calculate_gradient(obs,low)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[0]>x:
                    state_map[p] = 'S0'
        elif xb > x and yb > y:     # Down right
            low = (xb + 0.5, yb - 0.5)
            high = (xb - 0.5, yb + 0.5)
            g1 = self.calculate_gradient(obs,low)
            g2 = self.calculate_gradient(obs,high)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[1]>y:
                    state_map[p] = 'S0'
        elif xb == x and yb < y:    # Left
            low = (xb + 0.5, yb + 0.5)
            high = (xb - 0.5, yb + 0.5)
            g1 = self.calculate_gradient(obs,high)
            g2 = self.calculate_gradient(obs,low)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[1]<y:
                    state_map[p] = 'S0'
        elif xb == x and yb > y:    # Right
            low = (xb + 0.5, yb - 0.5)
            high = (xb - 0.5, yb - 0.5)
            g1 = self.calculate_gradient(obs,low)
            g2 = self.calculate_gradient(obs,high)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[1]>y:
                    state_map[p] = 'S0'
        elif xb < x and yb == y:    # Top
            low = (xb + 0.5, yb - 0.5)
            high = (xb + 0.5, yb + 0.5)
            g1 = self.calculate_gradient(obs,high)
            g2 = self.calculate_gradient(obs,low)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g or g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[0]<x:
                    state_map[p] = 'S0'
        elif xb > x and yb == y:    # Down
            low = (xb - 0.5, yb + 0.5)
            high = (xb - 0.5, yb - 0.5)
            g1 = self.calculate_gradient(obs,high)
            g2 = self.calculate_gradient(obs,low)
            for p in fov:
                g = self.calculate_gradient(p,agent_pos)
                if g1<g or g<g2 and (x-p[0])**2+(y-p[1])**2 > (x-xb)**2+(y-yb)**2 and p[0]>x:
                    state_map[p] = 'S0'


    def move_agent(self, agent_id: int, action: int):
        x, y = self.agents[agent_id - 1]
        move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = move_offsets[action]

        nx, ny = x + dx, y + dy
        self.grid[self.destination] = 0
        if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
            self.grid[x, y] = 0
            self.grid[nx, ny] = agent_id
            self.agents[agent_id - 1] = (nx, ny)
            self.agents_route_dict[agent_id].append(self.agents[agent_id - 1])  # Record the new position of agents
        else:
            self.agents_route_dict[agent_id].append((x, y))
        self.fov[agent_id]=(self.get_fov(self.agents[agent_id - 1]))  # Get FoV after moving

    def reset(self):
        self.grid.fill(0)
        self.agents.clear()
        self.agents_id_list.clear()
        self.agents_route_dict.clear()
        self.destination = None
        self.steps = 0
        self.rewards_dict = {i + 1: [] for i in range(self.N)}
        self.distance_dict = {i + 1: [] for i in range(self.N)}
        self.fov = {i + 1: [] for i in range(self.N)}
        self.init_environment()
        return self.grid

    def step(self, actions: List[int]):
        for a_id in self.agents_id_list[:]:
            if self.agents[a_id - 1] == self.destination:
                self.agents_id_list.remove(a_id)
            else:
                self.move_agent(a_id, actions[a_id - 1])
                x, y = self.agents[a_id - 1]
                dest_x, dest_y = self.destination
                distance = abs(x - dest_x) + abs(y - dest_y)
                self.distance_dict[a_id].append(distance)  # Record the distance in dict
                if distance == 0:
                    self.rewards_dict[a_id].append(self.T)  # Reward for reaching destination
                    self.agents_id_list.remove(a_id)
                else:
                    self.rewards_dict[a_id].append(round(1 / (distance + 1), 4))  # Penalty for each move

        self.steps += 1
        done = all(self.agents[agent_id - 1] == self.destination for agent_id in self.agents_id_list)
        terminate = self.steps >= self.T
        return self.grid, self.rewards_dict, done, terminate, self.agents_route_dict, self.steps, self.destination, self.distance_dict, self.fov

    def render(self, mode='human'):
        for row in self.grid:
            print('   '.join(str(x) for x in row))
    
    def render_fov(self, fov):
        for key, coordinates in fov.items():
            max_x = max([coord[0] for coord in coordinates]) + 1
            max_y = max([coord[1] for coord in coordinates]) + 1
            grid_map = [[' ' for _ in range(max_y)] for _ in range(max_x)]
            for (x, y), state in coordinates.items():
                grid_map[x][y] = state
            print(f"Grid map for key {key}:")
            for row in grid_map:
                print(' '.join(str(x) for x in row))


if __name__ == "__main__":
    env = GridNavigationEnv(L=10, P=0.3, N=3, T=5, M=3)
    grid_map = env.reset()
    done = False
    terminate = False
    print(f'In 0 step')
    env.render()
    env.render_fov(env.fov)
    print()

    while not done and not terminate:
        actions = [env.action_space.sample() for _ in range(env.N)]  # Random actions
        grid_map, rewards, done, terminate, routes, steps, destination, distance, fov = env.step(actions)
        print(fov)
        print(f'In {steps} steps')
        env.render()
        env.render_fov(fov)
        print('')
    print(f"Destination: {destination}")
    print(f"Reward: {rewards}")
    print(f"All agents have reached the destination: {done}")
    print(f"Some agents are stuck somewhere: {terminate}")
    print(f"Route: {routes}\n")
